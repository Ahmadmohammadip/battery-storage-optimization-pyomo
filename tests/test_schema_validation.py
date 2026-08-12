"""Validation is a design feature here: a malformed system must fail at
construction with a readable message, not as an opaque solver infeasibility
three layers down. These tests pin that behavior."""

import re

import pytest

from bess_opt.data.schema import Battery, PriceSeries, System


def _battery(**overrides) -> Battery:
    defaults = dict(
        name="Batt1",
        p_charge_max=10.0,
        p_discharge_max=10.0,
        energy_max=20.0,
        initial_soc=5.0,
    )
    defaults.update(overrides)
    return Battery(**defaults)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"p_charge_max": 0.0}, "p_charge_max must be > 0"),
        ({"p_discharge_max": -1.0}, "p_discharge_max must be > 0"),
        ({"energy_min": -1.0}, "energy_min must be >= 0"),
        ({"energy_max": 5.0, "energy_min": 5.0}, "must be > energy_min"),
        ({"charge_efficiency": 0.0}, "charge_efficiency must be in (0, 1]"),
        ({"charge_efficiency": 1.5}, "charge_efficiency must be in (0, 1]"),
        ({"discharge_efficiency": -0.1}, "discharge_efficiency must be in (0, 1]"),
        ({"initial_soc": 25.0}, "must be within the usable band"),
        ({"initial_soc": 1.0, "energy_min": 4.0}, "must be within the usable band"),
    ],
)
def test_invalid_battery_is_rejected(overrides, expected):
    # `match` is a regex and these messages contain parentheses and brackets,
    # so the expected fragment has to be escaped.
    with pytest.raises(ValueError, match=re.escape(expected)):
        _battery(**overrides)


def test_battery_accepts_a_valid_configuration():
    battery = _battery(charge_efficiency=0.95, discharge_efficiency=0.9)
    assert battery.usable_energy == pytest.approx(20.0)
    assert battery.round_trip_efficiency == pytest.approx(0.855)


def test_empty_price_series_is_rejected():
    with pytest.raises(ValueError, match="at least one period"):
        PriceSeries(energy=[])


def test_non_positive_delta_t_is_rejected():
    with pytest.raises(ValueError, match="delta_t must be > 0"):
        PriceSeries(energy=[10.0], delta_t=0.0)


def test_regulation_series_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="must span the same horizon"):
        PriceSeries(energy=[10.0, 20.0, 30.0], reg_up=[5.0, 5.0])


def test_negative_capacity_price_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        PriceSeries(energy=[10.0, 20.0], reg_down=[1.0, -1.0])


def test_negative_energy_price_is_accepted():
    # Negative energy prices are real market behavior, not bad data.
    prices = PriceSeries(energy=[-15.0, 40.0])
    assert prices.energy[0] == -15.0


def test_absent_regulation_prices_normalize_to_zeros():
    prices = PriceSeries(energy=[10.0, 20.0, 30.0])
    assert prices.reg_up == [0.0, 0.0, 0.0]
    assert prices.reg_down == [0.0, 0.0, 0.0]
    assert prices.has_regulation is False


def test_has_regulation_detects_non_zero_prices():
    prices = PriceSeries(energy=[10.0, 20.0], reg_up=[0.0, 3.0])
    assert prices.has_regulation is True


@pytest.mark.parametrize("phi", [-0.1, 1.5])
def test_phi_outside_unit_interval_is_rejected(phi):
    with pytest.raises(ValueError, match=r"phi .* must be in \[0, 1\]"):
        System(battery=_battery(), prices=PriceSeries(energy=[10.0]), phi=phi)


def test_system_reports_period_count():
    system = System(battery=_battery(), prices=PriceSeries(energy=[1.0, 2.0, 3.0, 4.0]))
    assert system.n_periods == 4
