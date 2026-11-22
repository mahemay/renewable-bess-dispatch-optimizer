from pathlib import Path
from typing import Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _save_figure(
    fig: go.Figure,
    html_path: str,
    png_path: Optional[str] = None,
    width: int = 1200,
    height: int = 700,
    scale: int = 2,
) -> None:
    """Save a Plotly figure as a lightweight CDN-backed HTML and optionally a high-res PNG."""
    html_file = Path(html_path)
    html_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_file, include_plotlyjs="cdn")

    if png_path:
        png_file = Path(png_path)
        png_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(png_file, width=width, height=height, scale=scale)
        except Exception as exc:
            print(
                f"Warning: Could not export PNG to '{png_file}' ({exc}). "
                "Ensure 'kaleido' is installed (pip install kaleido)."
            )


def plot_dispatch_window(
    dispatch_df: pd.DataFrame,
    start_day: int = 180,
    num_days: int = 3,
    output_path: str = "outputs/dispatch_profile.html",
    png_path: Optional[str] = "assets/dispatch_profile.png",
) -> go.Figure:
    """
    Create an interactive 2-panel chart showing hourly PV/BESS dispatch,
    wholesale electricity prices, and battery State of Charge (SoC).
    """
    start_idx = start_day * 24
    end_idx = start_idx + (num_days * 24)
    sub = dispatch_df.iloc[start_idx:end_idx].copy()
    sub["bess_charge_mw"] = -(sub["pv_to_bess_mw"] + sub["grid_to_bess_mw"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.65, 0.35],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(
            "Hourly Power Dispatch vs. Wholesale Electricity Price",
            "Battery State of Charge (SoC)",
        ),
    )

    # Panel 1: Power flows (MW)
    fig.add_trace(
        go.Bar(
            x=sub["timestamp"],
            y=sub["pv_to_grid_mw"],
            name="PV to Grid (MW)",
            marker_color="#F4B400",
            opacity=0.85,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=sub["timestamp"],
            y=sub["bess_discharge_mw"],
            name="BESS Discharge (MW)",
            marker_color="#0F9D58",
            opacity=0.9,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=sub["timestamp"],
            y=sub["bess_charge_mw"],
            name="BESS Charge (MW)",
            marker_color="#4285F4",
            opacity=0.85,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # Panel 1 Secondary Axis: Wholesale Price ($/MWh)
    fig.add_trace(
        go.Scatter(
            x=sub["timestamp"],
            y=sub["price_usd_mwh"],
            name="Wholesale Price ($/MWh)",
            line=dict(color="#DB4437", width=2.5),
            mode="lines",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Panel 2: Battery State of Charge (MWh)
    fig.add_trace(
        go.Scatter(
            x=sub["timestamp"],
            y=sub["soc_mwh"],
            name="Battery SoC (MWh)",
            fill="tozeroy",
            line=dict(color="#673AB7", width=2),
            fillcolor="rgba(103, 58, 183, 0.2)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Co-Located Solar PV + BESS Optimal Dispatch (3-Day Summer Sample)",
        barmode="relative",
        template="plotly_white",
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Price ($/MWh)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Energy (MWh)", row=2, col=1)

    _save_figure(fig, html_path=output_path, png_path=png_path, width=1200, height=700)
    return fig


def plot_sensitivity_heatmap(
    sens_df: pd.DataFrame,
    output_path: str = "outputs/npv_sensitivity.html",
    png_path: Optional[str] = "assets/npv_sensitivity.png",
) -> go.Figure:
    """
    Create an annotated heatmap of BESS Project NPV ($ Thousands) across CapEx and WACC.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=sens_df.values,
            x=sens_df.columns.tolist(),
            y=sens_df.index.tolist(),
            colorscale="RdYlGn",
            zmid=0.0,
            text=sens_df.values,
            texttemplate="$%{text:.1f}k",
            textfont={"size": 13},
            colorbar=dict(title="NPV ($k)"),
        )
    )

    fig.update_layout(
        title="BESS Investment NPV Sensitivity ($ Thousands): CapEx vs. WACC",
        xaxis_title="Discount Rate (WACC)",
        yaxis_title="Turnkey BESS CapEx",
        template="plotly_white",
        height=500,
    )

    _save_figure(fig, html_path=output_path, png_path=png_path, width=900, height=550)
    return fig


if __name__ == "__main__":
    from data_loader import load_market_data
    from optimizer import BESSConfig, optimize_dispatch
    from financials import run_sensitivity_analysis

    market_df = load_market_data()
    bess_cfg = BESSConfig()
    dispatch_df = optimize_dispatch(market_df, bess_cfg)

    annual_uplift = (
        dispatch_df["revenue_usd"].sum() - dispatch_df["pv_only_revenue_usd"].sum()
    )
    sens_df = run_sensitivity_analysis(annual_uplift, bess_cfg)

    plot_dispatch_window(dispatch_df)
    plot_sensitivity_heatmap(sens_df)
    print("Charts saved successfully:")
    print(" - HTML (lightweight): outputs/dispatch_profile.html, outputs/npv_sensitivity.html")
    print(" - PNG  (for README):  assets/dispatch_profile.png, assets/npv_sensitivity.png")