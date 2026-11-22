from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


@dataclass
class BESSConfig:
    """Technical and economic parameters for a co-located Solar PV + BESS asset."""
    power_mw: float = 5.0                 # Max charge/discharge power (MW)
    energy_mwh: float = 20.0              # Energy storage capacity (MWh, 4-hr battery)
    eta_charge: float = 0.94              # Charging efficiency
    eta_discharge: float = 0.94           # Discharging efficiency
    soc_min_frac: float = 0.10            # Minimum State of Charge (fraction)
    soc_max_frac: float = 0.95            # Maximum State of Charge (fraction)
    soc_init_frac: float = 0.50           # Initial State of Charge (fraction)
    degradation_cost_usd_mwh: float = 8.0 # Cycling degradation cost ($/MWh discharged)
    grid_export_limit_mw: float = 10.0    # Max grid interconnection export limit (MW)
    allow_grid_charging: bool = True      # Allow charging from the wholesale grid


def _solve_window(
    df_window: pd.DataFrame,
    config: BESSConfig,
    initial_soc_mwh: float,
) -> tuple[pd.DataFrame, float]:
    """Solve the MILP dispatch problem for a single time window using SciPy HiGHS."""
    n = len(df_window)
    prices = df_window["price_usd_mwh"].to_numpy(dtype=float)
    pv_gen = df_window["pv_generation_mw"].to_numpy(dtype=float)

    soc_min = config.soc_min_frac * config.energy_mwh
    soc_max = config.soc_max_frac * config.energy_mwh

    # Variable block offsets (7 * n variables total):
    # 0: pv_to_grid, 1: pv_to_bess, 2: pv_curtail, 3: grid_to_bess,
    # 4: bess_dis, 5: soc, 6: is_charging (binary)
    idx_pv_grid = np.arange(0, n)
    idx_pv_bess = np.arange(n, 2 * n)
    idx_pv_curt = np.arange(2 * n, 3 * n)
    idx_grid_bess = np.arange(3 * n, 4 * n)
    idx_bess_dis = np.arange(4 * n, 5 * n)
    idx_soc = np.arange(5 * n, 6 * n)
    idx_is_chg = np.arange(6 * n, 7 * n)

    # Objective: Minimize negative net revenue (equivalent to maximizing net revenue)
    c = np.zeros(7 * n, dtype=float)
    c[idx_pv_grid] = -prices
    c[idx_grid_bess] = prices
    c[idx_bess_dis] = -prices + config.degradation_cost_usd_mwh

    # Variable bounds
    lb = np.zeros(7 * n, dtype=float)
    ub = np.full(7 * n, np.inf, dtype=float)
    ub[idx_grid_bess] = config.power_mw if config.allow_grid_charging else 0.0
    ub[idx_bess_dis] = config.power_mw
    lb[idx_soc] = soc_min
    ub[idx_soc] = soc_max
    ub[idx_is_chg] = 1.0

    # Integrality: only is_charging variables are binary (1)
    integrality = np.zeros(7 * n, dtype=int)
    integrality[idx_is_chg] = 1

    # Build sparse constraint matrix (5 * n constraints total)
    rows, cols, vals = [], [], []
    con_lb = np.empty(5 * n, dtype=float)
    con_ub = np.empty(5 * n, dtype=float)

    def add_terms(r_idx, c_idx, v):
        rows.append(r_idx)
        cols.append(c_idx)
        vals.append(np.full_like(r_idx, v, dtype=float) if np.isscalar(v) else v)

    t_arr = np.arange(n)

    # 1. Solar PV power balance: pv_grid + pv_bess + pv_curt == pv_gen
    r1 = t_arr
    add_terms(r1, idx_pv_grid, 1.0)
    add_terms(r1, idx_pv_bess, 1.0)
    add_terms(r1, idx_pv_curt, 1.0)
    con_lb[r1] = pv_gen
    con_ub[r1] = pv_gen

    # 2. Charging power limit: pv_bess + grid_bess - power_mw * is_chg <= 0
    r2 = n + t_arr
    add_terms(r2, idx_pv_bess, 1.0)
    add_terms(r2, idx_grid_bess, 1.0)
    add_terms(r2, idx_is_chg, -config.power_mw)
    con_lb[r2] = -np.inf
    con_ub[r2] = 0.0

    # 3. Discharging power limit: bess_dis + power_mw * is_chg <= power_mw
    r3 = 2 * n + t_arr
    add_terms(r3, idx_bess_dis, 1.0)
    add_terms(r3, idx_is_chg, config.power_mw)
    con_lb[r3] = -np.inf
    con_ub[r3] = config.power_mw

    # 4. Grid export limit: pv_grid + bess_dis <= grid_export_limit_mw
    r4 = 3 * n + t_arr
    add_terms(r4, idx_pv_grid, 1.0)
    add_terms(r4, idx_bess_dis, 1.0)
    con_lb[r4] = -np.inf
    con_ub[r4] = config.grid_export_limit_mw

    # 5. Battery SoC dynamics:
    # soc[t] - soc[t-1] - eta_c * (pv_bess + grid_bess) + (1/eta_d) * bess_dis == 0 (or initial_soc at t=0)
    r5 = 4 * n + t_arr
    add_terms(r5, idx_soc, 1.0)
    add_terms(r5[1:], idx_soc[:-1], -1.0)
    add_terms(r5, idx_pv_bess, -config.eta_charge)
    add_terms(r5, idx_grid_bess, -config.eta_charge)
    add_terms(r5, idx_bess_dis, 1.0 / config.eta_discharge)
    con_lb[r5] = 0.0
    con_ub[r5] = 0.0
    con_lb[r5[0]] = initial_soc_mwh
    con_ub[r5[0]] = initial_soc_mwh

    A = csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(5 * n, 7 * n),
    )

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(A, con_lb, con_ub),
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    x = result.x
    res = df_window.copy()
    res["pv_to_grid_mw"] = x[idx_pv_grid]
    res["pv_to_bess_mw"] = x[idx_pv_bess]
    res["pv_curtail_mw"] = x[idx_pv_curt]
    res["grid_to_bess_mw"] = x[idx_grid_bess]
    res["bess_discharge_mw"] = x[idx_bess_dis]
    res["soc_mwh"] = x[idx_soc]

    res["net_grid_export_mw"] = (
        res["pv_to_grid_mw"] + res["bess_discharge_mw"] - res["grid_to_bess_mw"]
    )
    res["revenue_usd"] = (
        res["price_usd_mwh"] * res["net_grid_export_mw"]
        - config.degradation_cost_usd_mwh * res["bess_discharge_mw"]
    )
    res["pv_only_revenue_usd"] = np.maximum(res["price_usd_mwh"], 0.0) * res["pv_generation_mw"]

    final_soc = float(res["soc_mwh"].iloc[-1])
    return res, final_soc


def optimize_dispatch(
    df: pd.DataFrame,
    config: Optional[BESSConfig] = None,
    chunk_days: int = 30,
) -> pd.DataFrame:
    """
    Run hourly MILP dispatch optimization across the full dataset using sequential windows.
    """
    if config is None:
        config = BESSConfig()

    chunk_hours = chunk_days * 24
    current_soc = config.soc_init_frac * config.energy_mwh
    results = []

    for start in range(0, len(df), chunk_hours):
        window = df.iloc[start : start + chunk_hours]
        res_window, current_soc = _solve_window(window, config, current_soc)
        results.append(res_window)

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    from data_loader import load_market_data

    market_df = load_market_data()
    cfg = BESSConfig()
    dispatch_df = optimize_dispatch(market_df, cfg)

    pv_only_rev = dispatch_df["pv_only_revenue_usd"].sum()
    hybrid_rev = dispatch_df["revenue_usd"].sum()
    bess_uplift = hybrid_rev - pv_only_rev
    total_discharged = dispatch_df["bess_discharge_mw"].sum()
    equiv_cycles = total_discharged / cfg.energy_mwh

    print("=== Annual Optimization Summary ===")
    print(f"PV-Only Revenue:       ${pv_only_rev:,.2f}")
    print(f"PV + BESS Net Revenue: ${hybrid_rev:,.2f}")
    print(f"BESS Value Uplift:     ${bess_uplift:,.2f} (+{bess_uplift / pv_only_rev * 100:.1f}%)")
    print(f"Total BESS Discharged: {total_discharged:,.1f} MWh ({equiv_cycles:.1f} full cycles/yr)")