"""End-to-end runs on the full sample series, plus the economic invariants
that must hold whatever the prices happen to be."""

from pathlib import Path

import pytest

from bess_opt.data.loaders import load_price_series_csv
from bess_opt.data.schema import Battery, PriceSeries, System
from bess_opt.model.builder import build_bess_model
from bess_opt.solve import solve_bess

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_price_series"


def _battery(**overrides) -> Battery:
    defaults = dict(
        name="Batt1",
        p_charge_max=10.0,
        p_discharge_max=10.0,
        energy_max=40.0,
        initial_soc=20.0,
        charge_efficiency=0.94,
        discharge_efficiency=0.94,
    )
    defaults.update(overrides)
    return Battery(**defaults)


@pytest.fixture(scope="module")
def week_prices():
    return load_price_series_csv(SAMPLE_DIR / "week_hourly.csv")


def test_full_week_solves_and_is_internally_consistent(week_prices):
    system = System(battery=_battery(), prices=week_prices, phi=0.5)

    result = solve_bess(build_bess_model(system))

    assert len(result.soc) == 168
    assert sum(result.revenue_breakdown().values()) == pytest.approx(
        result.total_profit, abs=1e-6
    )
    assert result.total_profit > 0


def test_full_week_respects_every_constraint(week_prices):
    battery = _battery()
    phi = 0.5
    system = System(battery=battery, prices=week_prices, phi=phi)

    result = solve_bess(build_bess_model(system))

    prev = battery.initial_soc
    for t in range(1, 169):
        expected_soc = (
            prev
            + battery.charge_efficiency * result.charge[t] * week_prices.delta_t
            - (result.discharge[t] / battery.discharge_efficiency) * week_prices.delta_t
        )
        assert result.soc[t] == pytest.approx(expected_soc, abs=1e-6)
        assert battery.energy_min - 1e-6 <= result.soc[t] <= battery.energy_max + 1e-6

        assert result.charge[t] <= battery.p_charge_max + 1e-6
        assert result.discharge[t] <= battery.p_discharge_max + 1e-6
        assert result.discharge[t] + result.reg_up[t] <= battery.p_discharge_max + 1e-6
        assert result.charge[t] + result.reg_down[t] <= battery.p_charge_max + 1e-6

        headroom_up = result.soc[t] - phi * result.reg_up[t] * week_prices.delta_t
        headroom_down = result.soc[t] + phi * result.reg_down[t] * week_prices.delta_t
        assert headroom_up >= battery.energy_min - 1e-6
        assert headroom_down <= battery.energy_max + 1e-6

        prev = result.soc[t]

    assert result.soc[168] == pytest.approx(battery.initial_soc, abs=1e-6)


def test_stacking_never_loses_to_arbitrage_alone(week_prices):
    # The arbitrage-only schedule is a feasible point of the stacked problem
    # (set all capacity to zero), so co-optimizing cannot do worse. If this
    # ever fails, the coupling constraints are wrong.
    battery = _battery()
    stacked = System(battery=battery, prices=week_prices, phi=0.5)
    arbitrage_only = System(
        battery=battery, prices=week_prices, phi=0.5, include_regulation=False
    )

    stacked_profit = solve_bess(build_bess_model(stacked)).total_profit
    arbitrage_profit = solve_bess(build_bess_model(arbitrage_only)).total_profit

    assert stacked_profit >= arbitrage_profit - 1e-6
    # On this series regulation is worth having, so the inequality is strict.
    assert stacked_profit > arbitrage_profit


def test_tighter_deployment_assumption_never_pays_more(week_prices):
    # Raising phi only shrinks the feasible set, so profit must be
    # non-increasing in phi. Monotonicity is the cheapest check that the SOC
    # headroom constraints point the right way.
    battery = _battery()
    profits = []
    for phi in (0.0, 0.25, 0.5, 0.75, 1.0):
        system = System(battery=battery, prices=week_prices, phi=phi)
        profits.append(solve_bess(build_bess_model(system)).total_profit)

    for tighter, looser in zip(profits[1:], profits[:-1], strict=True):
        assert tighter <= looser + 1e-6


def test_no_simultaneous_charge_and_discharge_when_lossy(week_prices):
    # The model has no exclusivity binaries by design (PROJECT_BRIEF.md 1.6).
    # With a round trip below 1 the objective penalizes doing both at once, so
    # the LP optimum avoids it without being told to.
    system = System(
        battery=_battery(charge_efficiency=0.9, discharge_efficiency=0.9),
        prices=week_prices,
        phi=0.5,
    )

    result = solve_bess(build_bess_model(system))

    for t in range(1, 169):
        assert min(result.charge[t], result.discharge[t]) == pytest.approx(0.0, abs=1e-6)


def test_sub_hourly_periods_run_end_to_end():
    # Same day, read as five-minute intervals instead of hours: a twelfth of
    # the energy moves per period, so the profit must be far smaller.
    hourly = load_price_series_csv(SAMPLE_DIR / "day_hourly.csv", delta_t=1.0)
    five_minute = load_price_series_csv(SAMPLE_DIR / "day_hourly.csv", delta_t=1 / 12)

    battery = _battery()
    hourly_profit = solve_bess(
        build_bess_model(System(battery=battery, prices=hourly, include_regulation=False))
    ).total_profit
    short_profit = solve_bess(
        build_bess_model(System(battery=battery, prices=five_minute, include_regulation=False))
    ).total_profit

    assert 0 < short_profit < hourly_profit


def test_single_period_horizon_earns_nothing():
    # The degenerate horizon: one period that must end where it started leaves
    # no room to move energy, however attractive the price. Worth pinning
    # because it is the edge case where the terminal constraint fully
    # determines the answer.
    system = System(
        battery=_battery(energy_max=40.0, initial_soc=20.0),
        prices=PriceSeries(energy=[50.0]),
    )

    result = solve_bess(build_bess_model(system))

    # Nothing can be earned in a single period that must end where it started.
    assert result.total_profit == pytest.approx(0.0, abs=1e-6)
    assert result.soc[1] == pytest.approx(20.0, abs=1e-6)
