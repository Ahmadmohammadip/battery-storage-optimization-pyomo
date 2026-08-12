"""Phase 5: plotting helpers. These check that each figure builds and that
the revenue split it draws matches the solved result — not how it looks.

The Agg backend is selected in conftest.py so these run headless.
"""

from pathlib import Path

import pytest

from bess_opt.data.loaders import load_price_series_csv
from bess_opt.data.schema import Battery, PriceSeries, System
from bess_opt.model.builder import build_bess_model
from bess_opt.solve import solve_bess
from bess_opt.viz import (
    period_revenue,
    plot_price_and_dispatch,
    plot_revenue_stack,
    plot_soc_trajectory,
)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_price_series"


@pytest.fixture
def solved():
    system = System(
        battery=Battery(
            name="Batt1",
            p_charge_max=10.0,
            p_discharge_max=10.0,
            energy_max=40.0,
            initial_soc=10.0,
            charge_efficiency=0.94,
            discharge_efficiency=0.94,
        ),
        prices=load_price_series_csv(SAMPLE_DIR / "day_hourly.csv"),
    )
    return system, solve_bess(build_bess_model(system))


@pytest.mark.parametrize(
    "plot_fn", [plot_price_and_dispatch, plot_soc_trajectory, plot_revenue_stack]
)
def test_plots_build(solved, plot_fn):
    system, result = solved
    fig = plot_fn(system, result)
    assert fig.axes  # produced at least one populated axis


def test_period_revenue_sums_to_the_reported_totals(solved):
    system, result = solved
    arbitrage, regulation = period_revenue(system, result)

    assert sum(arbitrage) == pytest.approx(result.arbitrage_revenue, abs=1e-6)
    assert sum(regulation) == pytest.approx(result.regulation_revenue, abs=1e-6)
    assert sum(arbitrage) + sum(regulation) == pytest.approx(result.total_profit, abs=1e-6)


def test_period_revenue_is_negative_while_charging():
    system = System(
        battery=Battery(
            name="Batt1",
            p_charge_max=10.0,
            p_discharge_max=10.0,
            energy_max=20.0,
            initial_soc=0.0,
        ),
        prices=PriceSeries(energy=[10.0, 50.0]),
    )
    result = solve_bess(build_bess_model(system))

    arbitrage, _ = period_revenue(system, result)

    assert arbitrage[0] < 0  # buying energy
    assert arbitrage[1] > 0  # selling it back
