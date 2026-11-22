# Renewable + BESS Dispatch Optimizer

**MILP-Based Co-Located Solar PV & Battery Energy Storage Dispatch and Financial Valuation Framework**

[Overview](#overview) •
[Key Results](#key-results) •
[Visualizations](#visualizations) •
[Architecture](#system-architecture) •
[Math Model](#mathematical-formulation) •
[Quick Start](#quick-start) •
[Roadmap](#roadmap)

## Overview

**Renewable + BESS Dispatch Optimizer** is a production-structured operations research and energy economics toolkit designed to optimize hourly dispatch and evaluate the bankability of co-located **Solar PV + Battery Energy Storage Systems (BESS)** participating in wholesale electricity markets.

Using a sparse **Mixed-Integer Linear Programming (MILP)** formulation solved via **SciPy's HiGHS** backend, the engine quantifies energy arbitrage, solar curtailment mitigation, cycling degradation, and incremental investment metrics (**NPV**, **IRR**, **Simple Payback**, and **CapEx vs. WACC sensitivity**).

> \[!IMPORTANT\]
> This framework enforces strict physical and market realism—including binary mutual exclusion (no simultaneous charging and discharging), round-trip efficiency losses, State of Charge (SoC) boundaries, and shared Point of Interconnection (POI) export limits—verified by an automated `pytest` suite.

## Key Results

Baseline simulation for a **10 MW Solar PV** plant co-located with a **5 MW / 20 MWh (4-hour) BESS** over an 8,760-hour wholesale market year featuring duck-curve diurnal spreads, weekend price softening, seasonal dynamics, and scarcity spikes:

| Financial & Operational Metric | Baseline Result | Notes | 
| ----- | ----- | ----- | 
| **PV-Only Annual Revenue** | `$763,795` | Counterfactual benchmark without storage | 
| **PV + BESS Net Annual Revenue** | `$1,154,731` | Net of grid charging & degradation costs | 
| **BESS Annual Value Uplift** | **`+$390,935 (+51.2%)`** | Incremental arbitrage & curtailment capture | 
| **Annual Battery Throughput** | `9,958.9 MWh` | `497.9` full equivalent cycles/year | 
| **Net BESS CapEx (after 30% ITC)** | `$2,520,000` | `$180/kWh` turnkey capital cost | 
| **Incremental Project NPV (8% WACC)** | **`$51,172`** | Positive net value under base assumptions | 
| **Incremental Project IRR** | **`8.35%`** | Exceeds 8.0% hurdle rate | 
| **Simple Payback Period** | `8.01 years` | Based on incremental annual cash flows | 

| Parameter | Value | Description | 
| ----- | ----- | ----- | 
| Solar PV Nameplate Capacity | `10.0 MW` | Hourly synthetic generation profile (8,760 hrs) | 
| BESS Power Rating ($P_{\max}$) | `5.0 MW` | Symmetric charge/discharge inverter rating | 
| BESS Energy Capacity ($E_{\max}$) | `20.0 MWh` | 4-hour duration lithium-ion system | 
| Operating SoC Window | `10% – 95%` | `2.0 MWh` min to `19.0 MWh` max usable range | 
| Charge / Discharge Efficiency | `94% / 94%` | $\eta_{\text{RTE}} = 0.94^2 \approx 88.36\%$ | 
| Cycling Degradation Cost | `$8.00 / MWh` | Applied per MWh discharged | 
| Grid Export Limit (POI) | `10.0 MW` | Shared interconnection cap for PV + BESS | 
| Investment Tax Credit (ITC) | `30%` | Applied to gross turnkey BESS CapEx | 
| Rolling Optimization Horizon | `30 days` | Sequential MILP chunks with SoC continuity | 

## Visualizations

### 1. Optimal Hourly Dispatch & Battery State of Charge (3-Day Summer Window)

The optimizer absorbs low-priced midday solar peaks (and negative-price intervals) to charge the battery, then dispatches stored energy into high-priced evening ramps while strictly honoring power, SoC, and POI export limits.

> \[!TIP\]
> Running `python src/visualizer.py` also generates lightweight, CDN-backed interactive Plotly dashboards (`outputs/dispatch_profile.html` and `outputs/npv_sensitivity.html`) with zoom, pan, and unified hover tooltips.

### 2. Financial Sensitivity Heatmap (Turnkey CapEx vs. WACC)

Evaluates incremental BESS project NPV (`$ Thousands`) across battery CapEx trajectories (`$140–$220/kWh`) and Weighted Average Cost of Capital (`6%–12% WACC`), highlighting the exact break-even frontier for investment committees.

## System Architecture

```
flowchart LR
    A[src/data_loader.py<br/>8,760-hr Market & PV Engine] --> B[data/processed/<br/>Hourly Time-Series CSV]
    B --> C[src/optimizer.py<br/>Sparse MILP Dispatch<br/>SciPy HiGHS]
    C --> D[src/financials.py<br/>NPV, IRR, Payback &<br/>Sensitivity Matrix]
    C --> E[src/visualizer.py<br/>Plotly Exporter]
    D --> E
    E --> F[outputs/*.html<br/>Interactive Dashboards]
    E --> G[assets/*.png<br/>High-Res Report Charts]
    C -.-> H[tests/test_optimizer.py<br/>5 Physical Constraint Tests]

```

### Repository Structure

```
renewable-bess-dispatch-optimizer/
├── assets/
│   ├── dispatch_profile.png     # High-resolution dispatch chart (300-equivalent DPI)
│   └── npv_sensitivity.png      # High-resolution CapEx vs. WACC heatmap
├── data/
│   └── processed/               # Generated 8,760-hour market & PV dataset
├── outputs/
│   ├── dispatch_profile.html    # Lightweight interactive Plotly dispatch dashboard
│   └── npv_sensitivity.html     # Lightweight interactive Plotly sensitivity heatmap
├── src/
│   ├── data_loader.py           # Synthetic wholesale price & solar generation engine
│   ├── optimizer.py             # Sparse MILP dispatch model (SciPy / HiGHS)
│   ├── financials.py            # Incremental NPV, IRR, Payback & sensitivity analysis
│   └── visualizer.py            # Interactive HTML & automated Kaleido PNG generator
├── tests/
│   └── test_optimizer.py        # Automated pytest suite verifying physical constraints
├── requirements.txt             # Pinned Python dependencies
└── README.md

```

## Mathematical Formulation

The dispatch engine (`src/optimizer.py`) solves a Mixed-Integer Linear Program over sequential rolling windows with exact State of Charge continuity between chunks.

### Objective Function

Maximize net wholesale market revenue minus cycling degradation cost over horizon $t \in \{1, \dots, T\}$ with hourly step $\Delta t = 1\text{ h}$:

$$
\max \sum_{t=1}^{T} \left[ p_t \left( P_t^{\text{pv}\to\text{grid}} + P_t^{\text{dis}} - P_t^{\text{grid}\to\text{bess}} \right) - c_{\text{deg}} \, P_t^{\text{dis}} \right] \Delta t
$$

Where:

* $p_t$: Wholesale locational marginal price (`$/MWh`) at hour $t$

* $P_t^{\text{pv}\to\text{grid}}$, $P_t^{\text{pv}\to\text{bess}}$, $P_t^{\text{curt}}$: Solar power sent to grid, battery, or curtailed (`MW`)

* $P_t^{\text{grid}\to\text{bess}}$, $P_t^{\text{dis}}$: Grid-to-battery charging and battery discharging power (`MW`)

* $c_{\text{deg}}$: Cycling degradation cost (`$8/MWh` discharged)

### Physical & Market Constraints

1. **Solar PV Power Balance:**
   ```math
   P_t^{\text{pv}\to\text{grid}} + P_t^{\text{pv}\to\text{bess}} + P_t^{\text{curt}} = P_t^{\text{pv,avail}} \quad \forall t
   ```

2. **State of Charge (SoC) Dynamics:**
   ```math
   E_t = E_{t-1} + \eta_c \left( P_t^{\text{pv}\to\text{bess}} + P_t^{\text{grid}\to\text{bess}} \right) \Delta t - \frac{1}{\eta_d} P_t^{\text{dis}} \Delta t \quad \forall t
   ```

3. **SoC Operating Bounds:**
   ```math
   E_{\min} \le E_t \le E_{\max} \quad \forall t
   ```

4. **No Simultaneous Charge & Discharge (Binary Mutual Exclusion):** Using binary operating-state variable $u_t \in \lbrace 0, 1 \rbrace$ ($u_t = 1$ when charging, $0$ when discharging):
   ```math
   P_t^{\text{pv}\to\text{bess}} + P_t^{\text{grid}\to\text{bess}} \le P_{\max} \cdot u_t \quad \forall t
   ```
   ```math
   P_t^{\text{dis}} \le P_{\max} \cdot (1 - u_t) \quad \forall t
   ```

5. **Shared Point of Interconnection (POI) Export Limit:**
   ```math
   P_t^{\text{pv}\to\text{grid}} + P_t^{\text{dis}} \le P_{\text{grid,max}} \quad \forall t
   ```
## Quick Start

### 1. Clone & Set Up Environment

```
git clone https://github.com/mahemay/renewable-bess-dispatch-optimizer.git
cd renewable-bess-dispatch-optimizer

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

```

### 2. Run the Full Pipeline

```
python src/data_loader.py   # Generate 8,760-hour synthetic market & PV data
python src/optimizer.py     # Solve annual MILP dispatch optimization
python src/financials.py    # Compute NPV, IRR, Payback & sensitivity matrix
python src/visualizer.py    # Export interactive HTML & high-res PNG charts

```

## Testing & Verification

Run the automated test suite to validate all physical battery constraints and economic sanity checks:

```
python -m pytest -v

```

```
tests/test_optimizer.py::test_no_simultaneous_charge_and_discharge PASSED [ 20%]
tests/test_optimizer.py::test_soc_within_bounds PASSED                    [ 40%]
tests/test_optimizer.py::test_pv_power_balance PASSED                     [ 60%]
tests/test_optimizer.py::test_grid_export_limit PASSED                    [ 80%]
tests/test_optimizer.py::test_bess_adds_positive_value PASSED             [100%]

============================== 5 passed in 1.82s ==============================

```

## Roadmap

* \[x\] Synthetic 8,760-hour duck-curve wholesale market & PV generator

* \[x\] Sparse MILP dispatch model with binary mutual exclusion (`scipy.optimize.milp`)

* \[x\] Incremental financial valuation (NPV, IRR, Payback, CapEx $\times$ WACC sensitivity)

* \[x\] Automated interactive HTML & Kaleido PNG chart generation

* \[x\] Automated `pytest` unit testing suite for physical constraints

* \[ \] Real historical ISO market data connectors (CAISO, ERCOT, EPEX SPOT)

* \[ \] Co-optimization with ancillary services (frequency regulation / spinning reserve)

* \[ \] Interactive Streamlit web dashboard for live parameter tuning

## License & Author

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

Developed and maintained by [**@mahemay**](https://github.com/mahemay).
