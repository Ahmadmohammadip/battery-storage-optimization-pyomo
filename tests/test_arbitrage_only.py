"""Phase 1: energy arbitrage behavior with no regulation market."""

import pytest

from bess_opt.data.schema import Battery, PriceSeries, System
from bess_opt.model.builder import build_bess_model
from bess_opt.solve import solve_bess


def _battery(**overrides) -> Battery:
    defaults = dict(
        name="Batt1",
        p_charge_max=10.0,
        p_discharge_max=10.0,
        energy_max=20.0,
        initial_soc=0.0,
        energy_min=0.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    defaults.update(overrides)
    return Battery(**defaults)


def test_charges_low_discharges_high():
    system = System(
        battery=_battery(),
        prices=PriceSeries(energy=[10.0, 50.0]),
    )

    result = solve_bess(build_bess_model(system))

    # Fill at $10/MWh, empty at $50/MWh: 10 MW for one hour each way.
    assert result.charge[1] == pytest.approx(10.0, abs=1e-6)
    assert result.discharge[1] == pytest.approx(0.0, abs=1e-6)
    assert result.charge[2] == pytest.approx(0.0, abs=1e-6)
    assert result.discharge[2] == pytest.approx(10.0, abs=1e-6)

    # -10 MWh * $10 + 10 MWh * $50
    assert result.total_profit == pytest.approx(400.0, abs=1e-6)
    assert result.arbitrage_revenue == pytest.approx(result.total_profit, abs=1e-6)


def test_flat_prices_with_losses_produce_no_cycling():
    # Round-trip efficiency of 0.81 means any cycle at a constant price
    # strictly loses money, so the optimal schedule is to sit still.
    system = System(
        battery=_battery(charge_efficiency=0.9, discharge_efficiency=0.9),
        prices=PriceSeries(energy=[20.0, 20.0, 20.0]),
    )

    result = solve_bess(build_bess_model(system))

    assert result.total_profit == pytest.approx(0.0, abs=1e-6)
    for t in (1, 2, 3):
        assert result.charge[t] == pytest.approx(0.0, abs=1e-6)
        assert result.discharge[t] == pytest.approx(0.0, abs=1e-6)


def test_negative_prices_are_exploited():
    # Being paid to consume is a real market condition, not bad data:
    # charging through a negative price is itself revenue.
    system = System(
        battery=_battery(),
        prices=PriceSeries(energy=[-10.0, 30.0]),
    )

    result = solve_bess(build_bess_model(system))

    assert result.charge[1] == pytest.approx(10.0, abs=1e-6)
    assert result.discharge[2] == pytest.approx(10.0, abs=1e-6)
    # +$100 for absorbing energy at -$10/MWh, +$300 for selling it at $30/MWh.
    assert result.total_profit == pytest.approx(400.0, abs=1e-6)


def test_efficiency_losses_shrink_the_spread_captured():
    lossless = System(battery=_battery(), prices=PriceSeries(energy=[10.0, 50.0]))
    lossy = System(
        battery=_battery(charge_efficiency=0.9, discharge_efficiency=0.9),
        prices=PriceSeries(energy=[10.0, 50.0]),
    )

    profit_lossless = solve_bess(build_bess_model(lossless)).total_profit
    profit_lossy = solve_bess(build_bess_model(lossy)).total_profit

    assert profit_lossy < profit_lossless
    assert profit_lossy > 0  # a 5x spread still clears an 0.81 round trip


def test_power_limit_caps_throughput():
    system = System(
        battery=_battery(p_charge_max=4.0, p_discharge_max=4.0),
        prices=PriceSeries(energy=[10.0, 50.0]),
    )

    result = solve_bess(build_bess_model(system))

    assert result.charge[1] == pytest.approx(4.0, abs=1e-6)
    assert result.discharge[2] == pytest.approx(4.0, abs=1e-6)
    assert result.total_profit == pytest.approx(160.0, abs=1e-6)
