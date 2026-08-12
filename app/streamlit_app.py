"""
Streamlit demo for the battery revenue-stacking model.

Pick one of the synthetic sample price series (or upload your own), tune the
battery and the deployment fraction, and see the optimal schedule alongside
what the arbitrage-only version of the same battery would have earned.

Run with:  streamlit run app/streamlit_app.py
"""

from pathlib import Path

import streamlit as st

from bess_opt.data.loaders import load_price_series_csv, load_price_series_text
from bess_opt.data.schema import Battery, System
from bess_opt.model.builder import build_bess_model
from bess_opt.solve import solve_bess
from bess_opt.viz import plot_price_and_dispatch, plot_revenue_stack, plot_soc_trajectory

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_price_series"
SAMPLE_SERIES = {
    "Sample day (24 hourly periods)": "day_hourly.csv",
    "Sample week (168 hourly periods)": "week_hourly.csv",
}

st.set_page_config(page_title="Battery Revenue Stacking", layout="wide")

st.title("Battery revenue stacking — energy arbitrage + frequency regulation")
st.caption(
    "A single LP co-optimizes energy arbitrage against regulation capacity. "
    "The sample price series are synthetic and are not calibrated to any real market."
)

with st.sidebar:
    st.header("Price series")
    source = st.radio("Source", ["Sample series", "Upload CSV"], label_visibility="collapsed")

    if source == "Sample series":
        choice = st.selectbox("Series", list(SAMPLE_SERIES))
        series_path = SAMPLE_DIR / SAMPLE_SERIES[choice]
        uploaded = None
    else:
        uploaded = st.file_uploader(
            "CSV with an `energy_price` column "
            "(`reg_up_price` and `reg_down_price` optional)",
            type="csv",
        )
        series_path = None

    delta_t = st.number_input(
        "Period duration (hours)",
        min_value=0.05,
        max_value=24.0,
        value=1.0,
        step=0.05,
        help="Not inferred from the file: 24 rows could be hourly or five-minute.",
    )

    st.header("Battery")
    p_charge_max = st.slider("Max charge power (MW)", 1.0, 50.0, 10.0, step=1.0)
    p_discharge_max = st.slider("Max discharge power (MW)", 1.0, 50.0, 10.0, step=1.0)
    energy_max = st.slider("Usable energy (MWh)", 1.0, 200.0, 40.0, step=1.0)
    initial_soc = st.slider(
        "Initial state of charge (MWh)",
        0.0,
        float(energy_max),
        min(20.0, float(energy_max)),
        step=1.0,
        help="The horizon is energy-neutral: the schedule must end here too.",
    )
    charge_efficiency = st.slider("Charge efficiency", 0.5, 1.0, 0.94, step=0.01)
    discharge_efficiency = st.slider("Discharge efficiency", 0.5, 1.0, 0.94, step=0.01)

    st.header("Regulation")
    include_regulation = st.checkbox("Sell regulation capacity", value=True)
    phi = st.slider(
        "Deployment fraction φ",
        0.0,
        1.0,
        0.5,
        step=0.05,
        help=(
            "Assumed share of committed capacity that gets called. Sizes the energy "
            "reserved for regulation. An assumption, not any ISO's actual rules."
        ),
    )


def build_system(prices, *, with_regulation: bool) -> System:
    battery = Battery(
        name="Batt1",
        p_charge_max=p_charge_max,
        p_discharge_max=p_discharge_max,
        energy_max=energy_max,
        initial_soc=initial_soc,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
    )
    return System(
        battery=battery, prices=prices, phi=phi, include_regulation=with_regulation
    )


if source == "Upload CSV" and uploaded is None:
    st.info("Upload a CSV to run the model, or switch back to a sample series.")
    st.stop()

try:
    if series_path is not None:
        prices = load_price_series_csv(series_path, delta_t=delta_t)
    else:
        prices = load_price_series_text(
            uploaded.getvalue().decode("utf-8"), label=uploaded.name, delta_t=delta_t
        )

    system = build_system(prices, with_regulation=include_regulation)
    result = solve_bess(build_bess_model(system))

    # Always solve the arbitrage-only counterfactual so the value of stacking
    # is visible rather than asserted.
    baseline_system = build_system(prices, with_regulation=False)
    baseline = solve_bess(build_bess_model(baseline_system))
except ValueError as exc:
    st.error(f"Invalid input: {exc}")
    st.stop()
except RuntimeError as exc:
    st.error(f"Solve failed: {exc}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total profit", f"${result.total_profit:,.0f}")
col2.metric("Arbitrage", f"${result.arbitrage_revenue:,.0f}")
col3.metric("Regulation", f"${result.regulation_revenue:,.0f}")
col4.metric(
    "vs. arbitrage only",
    f"${result.total_profit:,.0f}",
    delta=f"${result.total_profit - baseline.total_profit:,.0f}",
)

if include_regulation:
    # Dollar signs are escaped: Streamlit's markdown treats a matched pair of
    # them as inline LaTeX, which turns money into equations.
    st.caption(
        rf"Arbitrage alone would earn \${baseline.total_profit:,.0f} on this series. "
        rf"Stacking usually *lowers* the arbitrage component "
        rf"(\${result.arbitrage_revenue:,.0f} here) because power and energy are held "
        rf"in reserve — the capacity payment more than covers it."
    )

st.pyplot(plot_price_and_dispatch(system, result))

left, right = st.columns(2)
with left:
    st.pyplot(plot_soc_trajectory(system, result))
with right:
    st.pyplot(plot_revenue_stack(system, result))

with st.expander("What this model does not do"):
    st.markdown(
        """
- **No degradation or cycling cost.** Cycling is free here, so the schedule is
  harder on the battery than an operator watching warranty terms would allow.
- **Perfect foresight.** The whole price series is known up front; a real
  operator re-optimizes against a forecast and earns less.
- **Price taker.** The battery's own bids are assumed not to move prices.
- **No specific ISO's regulation rules.** φ is a labeled simplification, not a
  tariff — real products have published mileage and performance-score rules
  this model does not implement.

Full list with rationale in `docs/formulation.md`.
"""
    )
