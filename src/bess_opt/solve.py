"""Solver interface and result extraction.

This repo is a pure LP end to end — no quadratic terms, no binaries — so
HiGHS is the only solver needed. (The sibling `economic-dispatch-pyomo`
repo needs Ipopt as well for its quadratic cost curves; this one does not.)
"""

from dataclasses import dataclass, field

from pyomo.environ import ConcreteModel, SolverFactory, value
from pyomo.opt import SolverStatus, TerminationCondition

DEFAULT_SOLVER = "appsi_highs"


@dataclass
class ScheduleResult:
    """A solved battery schedule, keyed by period (1-indexed).

    Revenue components are reported separately so the value of stacking
    services is visible rather than buried in a single profit number.
    """

    charge: dict[int, float]
    discharge: dict[int, float]
    soc: dict[int, float]
    arbitrage_revenue: float
    total_profit: float
    reg_up: dict[int, float] = field(default_factory=dict)
    reg_down: dict[int, float] = field(default_factory=dict)
    regulation_revenue: float = 0.0

    def revenue_breakdown(self) -> dict[str, float]:
        """Revenue by service. Sums to `total_profit`."""
        return {
            "arbitrage": self.arbitrage_revenue,
            "regulation": self.regulation_revenue,
        }

    def net_power(self, t: int) -> float:
        """Net power to the grid in period t (MW): discharge minus charge."""
        return self.discharge[t] - self.charge[t]


def solve_bess(model: ConcreteModel, solver_name: str = DEFAULT_SOLVER) -> ScheduleResult:
    """Solve the model and extract the schedule. Raises RuntimeError if the
    solver does not reach optimality."""
    solver = SolverFactory(solver_name)
    results = solver.solve(model)
    _require_optimal(results, solver_name)

    periods = list(model.T)
    charge = {t: value(model.p_ch[t]) for t in periods}
    discharge = {t: value(model.p_dis[t]) for t in periods}
    soc = {t: value(model.soc[t]) for t in periods}
    reg_up = {t: value(model.r_up[t]) for t in periods}
    reg_down = {t: value(model.r_dn[t]) for t in periods}

    dt = value(model.dt)
    arbitrage_revenue = sum(
        dt * value(model.price_energy[t]) * (discharge[t] - charge[t]) for t in periods
    )
    # No dt factor: regulation pays per MW of capacity committed for the
    # period, not per MWh delivered.
    regulation_revenue = sum(
        value(model.price_reg_up[t]) * reg_up[t] + value(model.price_reg_down[t]) * reg_down[t]
        for t in periods
    )

    # Profit is the negated objective — the model minimizes -profit.
    total_profit = -value(model.negative_profit)

    return ScheduleResult(
        charge=charge,
        discharge=discharge,
        soc=soc,
        arbitrage_revenue=arbitrage_revenue,
        total_profit=total_profit,
        reg_up=reg_up,
        reg_down=reg_down,
        regulation_revenue=regulation_revenue,
    )


def _require_optimal(results, solver_name: str) -> None:
    """Raise unless the solve reached optimality.

    An infeasible model is the expected failure mode here — a battery whose
    terminal SOC cannot be reached, for instance — and it should surface as a
    clear error rather than silently returning whatever values the variables
    happen to hold.
    """
    if (
        results.solver.status != SolverStatus.ok
        or results.solver.termination_condition != TerminationCondition.optimal
    ):
        raise RuntimeError(
            f"Solve failed with {solver_name}: status={results.solver.status}, "
            f"termination={results.solver.termination_condition}"
        )
