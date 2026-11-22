import numpy as np
import pytest

from src.data_loader import generate_synthetic_market_data
from src.optimizer import BESSConfig, optimize_dispatch


@pytest.fixture(scope="module")
def sample_dispatch():
    """Run dispatch optimization on a 14-day sample for fast unit testing."""
    df = generate_synthetic_market_data(year=2025, pv_capacity_mw=10.0, seed=42).iloc[: 14 * 24]
    cfg = BESSConfig()
    res = optimize_dispatch(df, cfg, chunk_days=14)
    return res, cfg


def test_no_simultaneous_charge_and_discharge(sample_dispatch):
    res, _ = sample_dispatch
    total_charge = res["pv_to_bess_mw"] + res["grid_to_bess_mw"]
    product = total_charge * res["bess_discharge_mw"]
    assert np.allclose(product, 0.0, atol=1e-5), "Battery charged and discharged simultaneously."


def test_soc_within_bounds(sample_dispatch):
    res, cfg = sample_dispatch
    soc_min = cfg.soc_min_frac * cfg.energy_mwh
    soc_max = cfg.soc_max_frac * cfg.energy_mwh
    assert (res["soc_mwh"] >= soc_min - 1e-5).all(), "SoC dropped below minimum bound."
    assert (res["soc_mwh"] <= soc_max + 1e-5).all(), "SoC exceeded maximum bound."


def test_pv_power_balance(sample_dispatch):
    res, _ = sample_dispatch
    reconstructed_pv = res["pv_to_grid_mw"] + res["pv_to_bess_mw"] + res["pv_curtail_mw"]
    assert np.allclose(reconstructed_pv, res["pv_generation_mw"], atol=1e-5), "PV balance violated."


def test_grid_export_limit(sample_dispatch):
    res, cfg = sample_dispatch
    total_export = res["pv_to_grid_mw"] + res["bess_discharge_mw"]
    assert (total_export <= cfg.grid_export_limit_mw + 1e-5).all(), "Grid export limit exceeded."


def test_bess_adds_positive_value(sample_dispatch):
    res, _ = sample_dispatch
    pv_only_rev = res["pv_only_revenue_usd"].sum()
    hybrid_rev = res["revenue_usd"].sum()
    assert hybrid_rev >= pv_only_rev, "Hybrid PV+BESS revenue should not be lower than PV-only."