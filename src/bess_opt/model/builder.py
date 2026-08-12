"""
Builds a Pyomo ConcreteModel for battery energy arbitrage.

Phase 1 scope — energy arbitrage only. Regulation capacity variables and
the constraints that couple the two markets arrive in phases 2 and 3; the
full formulation is in docs/formulation.md.

    max   sum_t dt * price_t * (p_dis_t - p_ch_t)
    s.t.  0 <= p_ch_t  <= P_ch_max
          0 <= p_dis_t <= P_dis_max
          e_t = e_{t-1} + eta_ch * p_ch_t * dt - (p_dis_t / eta_dis) * dt
          E_min <= e_t <= E_max
          e_T = SOC_0

Implemented as `minimize(-profit)` for consistency with the sibling
`economic-dispatch-pyomo` repo, which minimizes cost.

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
    m = ConcreteModel(name="BESS_Arbitrage")

    battery = system.battery
    prices = system.prices
    periods = list(range(1, system.n_periods + 1))

    # --- Sets ---
    m.T = Set(initialize=periods, ordered=True)

    # --- Parameters ---
    # Energy price is Reals, not NonNegativeReals: negative prices are real
    # market behavior and the battery earns by charging through them.
    m.dt = Param(initialize=prices.delta_t)
    m.price_energy = Param(
        m.T, initialize={t: prices.energy[t - 1] for t in periods}, within=Reals
    )

    m.eta_ch = Param(initialize=battery.charge_efficiency)
    m.eta_dis = Param(initialize=battery.discharge_efficiency)
    m.initial_soc = Param(initialize=battery.initial_soc)

    # --- Variables ---
    m.p_ch = Var(m.T, domain=NonNegativeReals, bounds=(0, battery.p_charge_max))
    m.p_dis = Var(m.T, domain=NonNegativeReals, bounds=(0, battery.p_discharge_max))
    m.soc = Var(m.T, domain=NonNegativeReals, bounds=(battery.energy_min, battery.energy_max))

    # --- Objective ---
    def _negative_profit_rule(m):
        arbitrage = sum(m.dt * m.price_energy[t] * (m.p_dis[t] - m.p_ch[t]) for t in m.T)
        return -arbitrage

    m.negative_profit = Objective(rule=_negative_profit_rule, sense=minimize)

    # --- Constraints ---

    # SOC dynamics. Charging adds eta_ch * p_ch of energy per hour; discharging
    # p_dis to the grid drains p_dis / eta_dis from the cells.
    def _soc_rule(m, t):
        prev_soc = m.initial_soc if t == m.T.first() else m.soc[m.T.prev(t)]
        return m.soc[t] == prev_soc + m.eta_ch * m.p_ch[t] * m.dt - (m.p_dis[t] / m.eta_dis) * m.dt

    m.soc_con = Constraint(m.T, rule=_soc_rule)

    # Energy-neutral horizon — see module docstring.
    def _terminal_soc_rule(m):
        return m.soc[m.T.last()] == m.initial_soc

    m.terminal_soc_con = Constraint(rule=_terminal_soc_rule)

    return m
