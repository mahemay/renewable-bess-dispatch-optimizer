from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from optimizer import BESSConfig


@dataclass
class FinancialConfig:
    """Financial assumptions for incremental BESS investment analysis."""
    bess_capex_usd_kwh: float = 180.0      # Utility-scale BESS turnkey CapEx ($/kWh)
    bess_opex_usd_kw_yr: float = 10.0      # Annual fixed O&M ($/kW-year)
    itc_rate: float = 0.30                 # Investment Tax Credit (e.g., 30% US IRA)
    project_lifetime_yrs: int = 15         # Economic lifetime (years)
    discount_rate: float = 0.08            # WACC / Discount rate
    annual_revenue_fade: float = 0.02      # Annual capacity/revenue degradation (2%/yr)


def calculate_npv(cash_flows: np.ndarray, discount_rate: float) -> float:
    """Calculate Net Present Value (NPV) for a series of annual cash flows starting at t=0."""
    years = np.arange(len(cash_flows))
    return float(np.sum(cash_flows / ((1.0 + discount_rate) ** years)))


def calculate_irr(cash_flows: np.ndarray) -> Optional[float]:
    """Calculate Internal Rate of Return (IRR) using Brent's root-finding method."""
    try:
        return float(brentq(lambda r: calculate_npv(cash_flows, r), -0.95, 5.0))
    except ValueError:
        return None


def calculate_payback(cash_flows: np.ndarray) -> Optional[float]:
    """Calculate simple payback period in years with linear interpolation."""
    cumulative = np.cumsum(cash_flows)
    positive_idx = np.where(cumulative >= 0)[0]
    if len(positive_idx) == 0:
        return None
    first_pos = int(positive_idx[0])
    if first_pos == 0:
        return 0.0
    prev_cum = cumulative[first_pos - 1]
    curr_cf = cash_flows[first_pos]
    return float((first_pos - 1) + abs(prev_cum) / curr_cf)


def evaluate_bess_investment(
    annual_bess_uplift_usd: float,
    bess_config: BESSConfig,
    fin_config: Optional[FinancialConfig] = None,
) -> dict:
    """
    Evaluate incremental financial viability of adding BESS to an existing Solar PV plant.
    """
    if fin_config is None:
        fin_config = FinancialConfig()

    gross_capex = bess_config.energy_mwh * 1000.0 * fin_config.bess_capex_usd_kwh
    net_capex = gross_capex * (1.0 - fin_config.itc_rate)
    annual_opex = bess_config.power_mw * 1000.0 * fin_config.bess_opex_usd_kw_yr

    cash_flows = np.zeros(fin_config.project_lifetime_yrs + 1, dtype=float)
    cash_flows[0] = -net_capex

    for yr in range(1, fin_config.project_lifetime_yrs + 1):
        degraded_uplift = annual_bess_uplift_usd * ((1.0 - fin_config.annual_revenue_fade) ** (yr - 1))
        cash_flows[yr] = degraded_uplift - annual_opex

    npv = calculate_npv(cash_flows, fin_config.discount_rate)
    irr = calculate_irr(cash_flows)
    payback = calculate_payback(cash_flows)

    return {
        "gross_capex_usd": gross_capex,
        "net_capex_usd": net_capex,
        "annual_opex_usd": annual_opex,
        "year1_uplift_usd": annual_bess_uplift_usd,
        "npv_usd": npv,
        "irr_pct": irr * 100.0 if irr is not None else None,
        "payback_years": payback,
        "cash_flows": cash_flows,
    }


def run_sensitivity_analysis(
    annual_bess_uplift_usd: float,
    bess_config: BESSConfig,
    fin_config: Optional[FinancialConfig] = None,
    capex_range: tuple[float, ...] = (140.0, 160.0, 180.0, 200.0, 220.0),
    wacc_range: tuple[float, ...] = (0.06, 0.08, 0.10, 0.12),
) -> pd.DataFrame:
    """
    Generate a 2D sensitivity matrix of BESS NPV ($k) across CapEx ($/kWh) and WACC.
    """
    if fin_config is None:
        fin_config = FinancialConfig()

    matrix = pd.DataFrame(
        index=[f"${c:.0f}/kWh" for c in capex_range],
        columns=[f"WACC {w*100:.0f}%" for w in wacc_range],
        dtype=float,
    )

    for c in capex_range:
        for w in wacc_range:
            cfg = FinancialConfig(
                bess_capex_usd_kwh=c,
                bess_opex_usd_kw_yr=fin_config.bess_opex_usd_kw_yr,
                itc_rate=fin_config.itc_rate,
                project_lifetime_yrs=fin_config.project_lifetime_yrs,
                discount_rate=w,
                annual_revenue_fade=fin_config.annual_revenue_fade,
            )
            res = evaluate_bess_investment(annual_bess_uplift_usd, bess_config, cfg)
            matrix.loc[f"${c:.0f}/kWh", f"WACC {w*100:.0f}%"] = round(res["npv_usd"] / 1000.0, 1)

    return matrix


if __name__ == "__main__":
    from data_loader import load_market_data
    from optimizer import optimize_dispatch

    market_df = load_market_data()
    bess_cfg = BESSConfig()
    dispatch_df = optimize_dispatch(market_df, bess_cfg)

    annual_uplift = (
        dispatch_df["revenue_usd"].sum() - dispatch_df["pv_only_revenue_usd"].sum()
    )

    fin_res = evaluate_bess_investment(annual_uplift, bess_cfg)
    print("=== BESS Incremental Financial Evaluation ===")
    print(f"Gross CapEx:          ${fin_res['gross_capex_usd']:,.0f}")
    print(f"Net CapEx (30% ITC):  ${fin_res['net_capex_usd']:,.0f}")
    print(f"Annual Fixed OpEx:    ${fin_res['annual_opex_usd']:,.0f}")
    print(f"Year-1 Value Uplift:  ${fin_res['year1_uplift_usd']:,.0f}")
    print(f"Project NPV (8%):     ${fin_res['npv_usd']:,.0f}")
    if fin_res["irr_pct"] is not None:
        print(f"Project IRR:          {fin_res['irr_pct']:.2f}%")
    if fin_res["payback_years"] is not None:
        print(f"Simple Payback:       {fin_res['payback_years']:.2f} years")

    print("\n=== NPV Sensitivity Matrix ($ Thousands) ===")
    sens_df = run_sensitivity_analysis(annual_uplift, bess_cfg)
    print(sens_df)