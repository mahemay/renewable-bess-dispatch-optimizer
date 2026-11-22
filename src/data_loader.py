from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def generate_synthetic_market_data(
    year: int = 2025,
    pv_capacity_mw: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic hourly wholesale electricity prices and solar PV generation
    for a full calendar year (8,760 hours).

    Parameters
    ----------
    year : int
        Calendar year for the hourly timestamp index.
    pv_capacity_mw : float
        Nameplate capacity of the solar PV plant in MW.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, price_usd_mwh, pv_generation_mw.
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start=f"{year}-01-01 00:00:00",
        end=f"{year}-12-31 23:00:00",
        freq="h",
    )
    n_hours = len(timestamps)
    hour = timestamps.hour.to_numpy()
    day_of_year = timestamps.day_of_year.to_numpy()
    is_weekend = (timestamps.day_of_week >= 5).astype(float)
    # 1. Solar PV generation profile (MW)
    # Seasonal factor peaks in summer (around day 172)
    seasonal_solar = 0.65 + 0.35 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    # Daylight curve centered at 13:00
    solar_angle = np.clip(np.sin(np.pi * (hour - 6) / 14.0), 0.0, None)
    # Cloud cover factor (Beta distribution skewed toward clear skies)
    cloud_factor = rng.beta(a=5.0, b=1.5, size=n_hours)

    pv_generation_mw = pv_capacity_mw * seasonal_solar * (solar_angle ** 1.2) * cloud_factor
    pv_generation_mw = np.round(np.clip(pv_generation_mw, 0.0, pv_capacity_mw), 4)

    # 2. Wholesale electricity price ($/MWh) with Duck Curve dynamics
    base_price = 55.0
    # Seasonal demand effect (higher in summer and winter)
    seasonal_price = 12.0 * np.cos(4 * np.pi * (day_of_year - 15) / 365.0)
    # Diurnal pattern: morning ramp, midday solar dip, sharp evening peak
    morning_ramp = 15.0 * np.exp(-0.5 * ((hour - 8.5) / 2.0) ** 2)
    midday_dip = -25.0 * np.exp(-0.5 * ((hour - 13.0) / 2.2) ** 2)
    evening_peak = 45.0 * np.exp(-0.5 * ((hour - 19.0) / 2.0) ** 2)
    weekend_discount = -10.0 * is_weekend

    # Stochastic noise + occasional scarcity spikes
    noise = rng.normal(loc=0.0, scale=8.0, size=n_hours)
    spike_mask = rng.random(size=n_hours) < 0.015
    spikes = spike_mask * rng.uniform(60.0, 180.0, size=n_hours)

    price_usd_mwh = (
        base_price
        + seasonal_price
        + morning_ramp
        + midday_dip
        + evening_peak
        + weekend_discount
        + noise
        + spikes
    )
    # Allow occasional mild negative prices during extreme midday oversupply
    price_usd_mwh = np.round(np.clip(price_usd_mwh, -15.0, 350.0), 2)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price_usd_mwh": price_usd_mwh,
            "pv_generation_mw": pv_generation_mw,
        }
    )
    return df


def load_market_data(
    filepath: Optional[str] = "data/processed/market_data.csv",
    pv_capacity_mw: float = 10.0,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """
    Load hourly market and PV data from CSV, or generate and save it if missing.
    """
    path = Path(filepath) if filepath else None

    if path and path.exists() and not force_regenerate:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        required_cols = {"timestamp", "price_usd_mwh", "pv_generation_mw"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")
        return df

    df = generate_synthetic_market_data(pv_capacity_mw=pv_capacity_mw)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    data = load_market_data(force_regenerate=True)
    print(f"Dataset generated successfully: {len(data):,} hourly records.")
    print("\nSummary Statistics:")
    print(data[["price_usd_mwh", "pv_generation_mw"]].describe().round(2))