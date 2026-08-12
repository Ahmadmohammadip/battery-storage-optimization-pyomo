"""Regulation capacity and the two constraint families that couple the
energy and regulation markets: power headroom (phase 2) and SOC headroom,
the deployment-fraction logic (phase 3)."""

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
    )
    defaults.update(overrides)
    return Battery(**defaults)


def test_idle_battery_sells_full_capacity_to_regulation():
    # Flat energy prices leave nothing to arbitrage, so every MW of the power
    # rating is free to be committed as regulation capacity in both directions.
    #
    # The energy band is deliberately oversized (100 MWh against a 10 MW
    # rating, sitting half full) so the SOC headroom constraints stay slack
    # and this test isolates the power headroom. The SOC-limited case is
    # covered separately below.
    system = System(
        battery=_battery(energy_max=100.0, initial_soc=50.0),
        prices=PriceSeries(
            energy=[20.0, 20.0],
            reg_up=[5.0, 5.0],
            reg_down=[3.0, 3.0],
        ),
    )

    result = solve_bess(build_bess_model(system))

    for t in (1, 2):
        assert result.reg_up[t] == pytest.approx(10.0, abs=1e-6)
        assert result.reg_down[t] == pytest.approx(10.0, abs=1e-6)
        assert result.charge[t] == pytest.approx(0.0, abs=1e-6)
        assert result.discharge[t] == pytest.approx(0.0, abs=1e-6)

    assert result.arbitrage_revenue == pytest.approx(0.0, abs=1e-6)
    assert result.regulation_revenue == pytest.approx(160.0, abs=1e-6)


def test_energy_dispatch_consumes_regulation_headroom():
    # A 10x price spread is worth far more than the capacity payment, so the
    # battery spends its full power rating on energy and the headroom
    # constraints squeeze regulation to zero in the direction being used.
    system = System(
        battery=_battery(),
        prices=PriceSeries(
            energy=[10.0, 100.0],
            reg_up=[2.0, 2.0],
            reg_down=[2.0, 2.0],
        ),
    )

    result = solve_bess(build_bess_model(system))

    # Period 1 charges at full power: no charge headroom left for reg-down.
    assert result.charge[1] == pytest.approx(10.0, abs=1e-6)
    assert result.reg_down[1] == pytest.approx(0.0, abs=1e-6)
    # ...but the discharge path is idle, so reg-up is free to take it.
    assert result.reg_up[1] == pytest.approx(10.0, abs=1e-6)

    # Period 2 mirrors it: full discharge, no reg-up headroom, reg-down free.
    assert result.discharge[2] == pytest.approx(10.0, abs=1e-6)
    assert result.reg_up[2] == pytest.approx(0.0, abs=1e-6)
    assert result.reg_down[2] == pytest.approx(10.0, abs=1e-6)

    assert result.arbitrage_revenue == pytest.approx(900.0, abs=1e-6)
    assert result.regulation_revenue == pytest.approx(40.0, abs=1e-6)


def test_headroom_constraints_are_never_violated():
    battery = _battery(p_charge_max=7.0, p_discharge_max=9.0, energy_max=15.0, initial_soc=4.0)
    system = System(
        battery=battery,
        prices=PriceSeries(
            energy=[12.0, 45.0, 8.0, 60.0],
            reg_up=[4.0, 1.0, 6.0, 0.5],
            reg_down=[2.0, 3.0, 1.0, 4.0],
        ),
    )

    result = solve_bess(build_bess_model(system))

    for t in (1, 2, 3, 4):
        assert result.discharge[t] + result.reg_up[t] <= battery.p_discharge_max + 1e-6
        assert result.charge[t] + result.reg_down[t] <= battery.p_charge_max + 1e-6


def test_regulation_capacity_cannot_exceed_power_rating():
    # Capacity prices far above any plausible energy value: the binding limit
    # must be the power rating, not the price signal. Energy band oversized
    # again so SOC headroom stays slack.
    system = System(
        battery=_battery(
            p_charge_max=6.0, p_discharge_max=8.0, energy_max=100.0, initial_soc=50.0
        ),
        prices=PriceSeries(
            energy=[20.0, 20.0],
            reg_up=[999.0, 999.0],
            reg_down=[999.0, 999.0],
        ),
    )

    result = solve_bess(build_bess_model(system))

    for t in (1, 2):
        assert result.reg_up[t] == pytest.approx(8.0, abs=1e-6)
        assert result.reg_down[t] == pytest.approx(6.0, abs=1e-6)


def test_include_regulation_false_pins_capacity_to_zero():
    prices = PriceSeries(
        energy=[10.0, 50.0],
        reg_up=[5.0, 5.0],
        reg_down=[5.0, 5.0],
    )
    stacked = System(battery=_battery(), prices=prices)
    arbitrage_only = System(battery=_battery(), prices=prices, include_regulation=False)

    stacked_result = solve_bess(build_bess_model(stacked))
    arbitrage_result = solve_bess(build_bess_model(arbitrage_only))

    for t in (1, 2):
        assert arbitrage_result.reg_up[t] == pytest.approx(0.0, abs=1e-9)
        assert arbitrage_result.reg_down[t] == pytest.approx(0.0, abs=1e-9)

    assert arbitrage_result.regulation_revenue == pytest.approx(0.0, abs=1e-9)
    assert arbitrage_result.total_profit == pytest.approx(400.0, abs=1e-6)
    # Stacking can only add value — the arbitrage-only case is a feasible
    # point of the stacked problem.
    assert stacked_result.total_profit >= arbitrage_result.total_profit - 1e-6


# --- SOC headroom (phase 3): the deployment-fraction constraints ---


def test_stored_energy_caps_reg_up_below_the_power_rating():
    # 10 MW of power but only 1 MWh of usable energy. With phi = 0.5 and
    # hourly periods, honoring a reg-up call needs 0.5 * r_up MWh in reserve,
    # so a full battery can back at most 1.0 / 0.5 = 2 MW — a fifth of what
    # the power rating alone would allow.
    system = System(
        battery=_battery(energy_max=1.0, initial_soc=1.0),
        prices=PriceSeries(energy=[20.0, 20.0], reg_up=[50.0, 50.0]),
        phi=0.5,
    )

    result = solve_bess(build_bess_model(system))

    for t in (1, 2):
        assert result.reg_up[t] == pytest.approx(2.0, abs=1e-6)
        assert result.reg_down[t] == pytest.approx(0.0, abs=1e-6)  # battery is full


def test_empty_battery_can_still_sell_reg_down():
    # The mirror case: an empty battery has no energy to give but plenty of
    # room to absorb, so reg-down is available and reg-up is not.
    system = System(
        battery=_battery(energy_max=1.0, initial_soc=0.0),
        prices=PriceSeries(energy=[20.0, 20.0], reg_down=[50.0, 50.0]),
        phi=0.5,
    )

    result = solve_bess(build_bess_model(system))

    for t in (1, 2):
        assert result.reg_down[t] == pytest.approx(2.0, abs=1e-6)
        assert result.reg_up[t] == pytest.approx(0.0, abs=1e-6)  # battery is empty


def test_phi_zero_recovers_the_power_only_limit():
    # phi = 0 assumes committed capacity is never called, which switches the
    # SOC headroom constraints off and leaves the power rating binding.
    system = System(
        battery=_battery(energy_max=1.0, initial_soc=1.0),
        prices=PriceSeries(energy=[20.0, 20.0], reg_up=[50.0, 50.0]),
        phi=0.0,
    )

    result = solve_bess(build_bess_model(system))

    for t in (1, 2):
        assert result.reg_up[t] == pytest.approx(10.0, abs=1e-6)


def test_higher_phi_commits_less_capacity():
    # A more conservative deployment assumption reserves more energy per MW
    # committed, so it can only ever support less capacity.
    def profit_at(phi):
        system = System(
            battery=_battery(energy_max=4.0, initial_soc=2.0),
            prices=PriceSeries(energy=[20.0, 20.0, 20.0], reg_up=[8.0, 8.0, 8.0]),
            phi=phi,
        )
        result = solve_bess(build_bess_model(system))
        return sum(result.reg_up.values()), result.total_profit

    capacity_low, profit_low = profit_at(0.25)
    capacity_high, profit_high = profit_at(0.75)

    assert capacity_high < capacity_low
    assert profit_high < profit_low


def test_soc_headroom_holds_across_a_varied_schedule():
    battery = _battery(
        p_charge_max=8.0,
        p_discharge_max=8.0,
        energy_max=12.0,
        energy_min=2.0,
        initial_soc=6.0,
        charge_efficiency=0.93,
        discharge_efficiency=0.93,
    )
    phi = 0.4
    delta_t = 0.5
    system = System(
        battery=battery,
        prices=PriceSeries(
            energy=[9.0, 41.0, 15.0, 70.0, 12.0, 33.0],
            reg_up=[3.0, 2.0, 5.0, 1.0, 4.0, 2.0],
            reg_down=[2.0, 4.0, 1.0, 3.0, 2.0, 5.0],
            delta_t=delta_t,
        ),
        phi=phi,
    )

    result = solve_bess(build_bess_model(system))

    for t in range(1, 7):
        assert result.soc[t] - phi * result.reg_up[t] * delta_t >= battery.energy_min - 1e-6
        assert result.soc[t] + phi * result.reg_down[t] * delta_t <= battery.energy_max + 1e-6


def test_revenue_breakdown_sums_to_total_profit():
    system = System(
        battery=_battery(charge_efficiency=0.92, discharge_efficiency=0.92),
        prices=PriceSeries(
            energy=[15.0, 55.0, 20.0],
            reg_up=[3.0, 1.0, 4.0],
            reg_down=[2.0, 2.0, 2.0],
        ),
    )

    result = solve_bess(build_bess_model(system))

    breakdown = result.revenue_breakdown()
    assert sum(breakdown.values()) == pytest.approx(result.total_profit, abs=1e-6)
    assert breakdown["arbitrage"] == pytest.approx(result.arbitrage_revenue, abs=1e-6)
    assert breakdown["regulation"] == pytest.approx(result.regulation_revenue, abs=1e-6)
