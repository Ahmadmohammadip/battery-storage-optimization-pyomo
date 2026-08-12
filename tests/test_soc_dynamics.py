"""Phase 1: state-of-charge recursion, energy bounds, and the terminal
energy-neutrality constraint."""

import pytest

from bess_opt.data.schema import Battery, PriceSeries, System
from bess_opt.model.builder import build_bess_model
from bess_opt.solve import solve_bess


def _check_recursion(result, battery, delta_t, periods):
    """SOC must follow e_t = e_{t-1} + eta_ch * p_ch * dt - (p_dis / eta_dis) * dt."""
    prev = battery.initial_soc
    for t in periods:
        expected = (
            prev
            + battery.charge_efficiency * result.charge[t] * delta_t
            - (result.discharge[t] / battery.discharge_efficiency) * delta_t
        )
        assert result.soc[t] == pytest.approx(expected, abs=1e-6)
        prev = result.soc[t]


def test_soc_recursion_holds_with_asymmetric_efficiencies():
    battery = Battery(
        name="Batt1",
        p_charge_max=5.0,
        p_discharge_max=5.0,
        energy_max=20.0,
        initial_soc=5.0,
        charge_efficiency=0.9,
        discharge_efficiency=0.8,
    )
    system = System(battery=battery, prices=PriceSeries(energy=[10.0, 30.0, 20.0]))

    result = solve_bess(build_bess_model(system))

    _check_recursion(result, battery, delta_t=1.0, periods=[1, 2, 3])


def test_soc_recursion_scales_with_sub_hourly_periods():
    # delta_t = 0.25 means a 5 MW charge moves only 1.25 MWh of energy.
    battery = Battery(
        name="Batt1",
        p_charge_max=5.0,
        p_discharge_max=5.0,
        energy_max=20.0,
        initial_soc=2.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )
    system = System(
        battery=battery,
        prices=PriceSeries(energy=[5.0, 80.0, 5.0, 80.0], delta_t=0.25),
    )

    result = solve_bess(build_bess_model(system))

    _check_recursion(result, battery, delta_t=0.25, periods=[1, 2, 3, 4])


def test_soc_stays_within_the_usable_band():
    battery = Battery(
        name="Batt1",
        p_charge_max=50.0,   # deliberately oversized relative to the energy band
        p_discharge_max=50.0,
        energy_max=15.0,
        energy_min=3.0,
        initial_soc=9.0,
    )
    system = System(
        battery=battery,
        prices=PriceSeries(energy=[5.0, 90.0, 5.0, 90.0, 40.0]),
    )

    result = solve_bess(build_bess_model(system))

    for t in result.soc:
        assert result.soc[t] >= battery.energy_min - 1e-6
        assert result.soc[t] <= battery.energy_max + 1e-6


def test_terminal_soc_returns_to_initial():
    battery = Battery(
        name="Batt1",
        p_charge_max=10.0,
        p_discharge_max=10.0,
        energy_max=40.0,
        initial_soc=12.0,
    )
    # A rising price series is exactly the case that would otherwise end with
    # the battery drained: without the terminal constraint the last period
    # always dumps everything, because end-of-horizon energy has no value.
    system = System(battery=battery, prices=PriceSeries(energy=[10.0, 20.0, 30.0, 60.0]))

    result = solve_bess(build_bess_model(system))

    last = max(result.soc)
    assert result.soc[last] == pytest.approx(battery.initial_soc, abs=1e-6)


def test_energy_capacity_limits_arbitrage_volume():
    # 10 MW of power but only 4 MWh of usable energy: the battery can only
    # sell what it can store, regardless of how wide the price spread is.
    #
    # Assert on *net* power, not on p_ch / p_dis individually. With lossless
    # efficiency the LP is degenerate — charging 6 and discharging 10 in the
    # same period nets the same profit as discharging 4 and costs nothing, so
    # the solver may return either. That degeneracy is the flip side of having
    # no exclusivity binaries (PROJECT_BRIEF.md section 1.6); it disappears as
    # soon as efficiency is below 1, which is covered in test_integration.py.
    battery = Battery(
        name="Batt1",
        p_charge_max=10.0,
        p_discharge_max=10.0,
        energy_max=4.0,
        initial_soc=0.0,
    )
    system = System(battery=battery, prices=PriceSeries(energy=[10.0, 50.0]))

    result = solve_bess(build_bess_model(system))

    assert result.net_power(1) == pytest.approx(-4.0, abs=1e-6)  # charging
    assert result.net_power(2) == pytest.approx(4.0, abs=1e-6)   # discharging
    assert result.total_profit == pytest.approx(160.0, abs=1e-6)
