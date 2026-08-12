"""
Builds a Pyomo ConcreteModel for battery revenue stacking: energy arbitrage
co-optimized with frequency regulation capacity.

Phase 2 scope — arbitrage plus regulation capacity and the power headroom
constraints that couple the two markets. The SOC headroom constraints
(the deployment-fraction logic) arrive in phase 3; docs/formulation.md
carries the full derivation.

    max   sum_t dt * price_t * (p_dis_t - p_ch_t)          # energy
        + sum_t (price_ru_t * r_up_t + price_rd_t * r_dn_t)  # capacity
    s.t.  p_dis_t + r_up_t <= P_dis_max
          p_ch_t  + r_dn_t <= P_ch_max
          e_t = e_{t-1} + eta_ch * p_ch_t * dt - (p_dis_t / eta_dis) * dt
          E_min <= e_t <= E_max
          e_T = SOC_0

Implemented as `minimize(-profit)` for consistency with the sibling
`economic-dispatch-pyomo` repo, which minimizes cost.

Two modeling points worth stating plainly:

* Regulation is paid per MW of capacity *committed*, not per MWh delivered,
  so its revenue term carries no `dt` factor while the energy term does.
* Committing regulation-up capacity is a promise that the battery *could*
  discharge that much more if called. That promise has to fit inside the
  same power rating as energy-market discharge, which is what makes this a
  single co-optimization rather than two independent problems.

The terminal SOC constraint (`e_T = SOC_0`) is not in PROJECT_BRIEF.md
section 1.5 — it was added afterwards. Without it, perfect foresight drains
the battery to E_min in the final period because end-of-horizon energy has
no value, which makes the tail of every schedule an artifact. Requiring the
horizon to be energy-neutral is always feasible (holding still satisfies
it), so it cannot make a well-formed system infeasible.
"""

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    Reals,
    Set,
    Var,
    minimize,
)

from bess_opt.data.schema import System


def build_bess_model(system: System) -> ConcreteModel:
    m = ConcreteModel(name="BESS_RevenueStacking")

    battery = system.battery
    prices = system.prices
    periods = list(range(1, system.n_periods + 1))

    # --- Sets ---
    m.T = Set(initialize=periods, ordered=True)

    # --- Parameters ---
    # Energy price is Reals, not NonNegativeReals: negative prices are real
    # market behavior and the battery earns by charging through them.
    # Capacity prices are non-negative (enforced in the schema).
    m.dt = Param(initialize=prices.delta_t)
    m.price_energy = Param(
        m.T, initialize={t: prices.energy[t - 1] for t in periods}, within=Reals
    )
    m.price_reg_up = Param(
        m.T,
        initialize={t: prices.reg_up[t - 1] for t in periods},
        within=NonNegativeReals,
    )
    m.price_reg_down = Param(
        m.T,
        initialize={t: prices.reg_down[t - 1] for t in periods},
        within=NonNegativeReals,
    )

    m.eta_ch = Param(initialize=battery.charge_efficiency)
    m.eta_dis = Param(initialize=battery.discharge_efficiency)
    m.initial_soc = Param(initialize=battery.initial_soc)

    # --- Variables ---
    m.p_ch = Var(m.T, domain=NonNegativeReals, bounds=(0, battery.p_charge_max))
    m.p_dis = Var(m.T, domain=NonNegativeReals, bounds=(0, battery.p_discharge_max))
    m.r_up = Var(m.T, domain=NonNegativeReals, bounds=(0, battery.p_discharge_max))
    m.r_dn = Var(m.T, domain=NonNegativeReals, bounds=(0, battery.p_charge_max))
    m.soc = Var(m.T, domain=NonNegativeReals, bounds=(battery.energy_min, battery.energy_max))

    # An arbitrage-only case runs through this same builder with the
    # regulation variables pinned to zero, so there is one model to maintain.
    if not system.include_regulation:
        for t in periods:
            m.r_up[t].fix(0.0)
            m.r_dn[t].fix(0.0)

    # --- Objective ---
    def _negative_profit_rule(m):
        arbitrage = sum(m.dt * m.price_energy[t] * (m.p_dis[t] - m.p_ch[t]) for t in m.T)
        regulation = sum(
            m.price_reg_up[t] * m.r_up[t] + m.price_reg_down[t] * m.r_dn[t] for t in m.T
        )
        return -(arbitrage + regulation)

    m.negative_profit = Objective(rule=_negative_profit_rule, sense=minimize)

    # --- Constraints ---

    # Power headroom: energy-market dispatch and committed regulation capacity
    # share one power rating in each direction.
    def _discharge_headroom_rule(m, t):
        return m.p_dis[t] + m.r_up[t] <= battery.p_discharge_max

    m.discharge_headroom_con = Constraint(m.T, rule=_discharge_headroom_rule)

    def _charge_headroom_rule(m, t):
        return m.p_ch[t] + m.r_dn[t] <= battery.p_charge_max

    m.charge_headroom_con = Constraint(m.T, rule=_charge_headroom_rule)

    # SOC dynamics. Charging adds eta_ch * p_ch of energy per hour; discharging
    # p_dis to the grid drains p_dis / eta_dis from the cells. Only energy-market
    # flows move the SOC — regulation is a capacity commitment, not a guaranteed
    # energy delivery (see the phi discussion in docs/formulation.md).
    def _soc_rule(m, t):
        prev_soc = m.initial_soc if t == m.T.first() else m.soc[m.T.prev(t)]
        return m.soc[t] == prev_soc + m.eta_ch * m.p_ch[t] * m.dt - (m.p_dis[t] / m.eta_dis) * m.dt

    m.soc_con = Constraint(m.T, rule=_soc_rule)

    # Energy-neutral horizon — see module docstring.
    def _terminal_soc_rule(m):
        return m.soc[m.T.last()] == m.initial_soc

    m.terminal_soc_con = Constraint(rule=_terminal_soc_rule)

    return m
