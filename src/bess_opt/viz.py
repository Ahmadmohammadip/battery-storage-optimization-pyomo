"""
Plotting helpers for a solved battery schedule.

matplotlib only, kept dependency-light so these work headless in CI and
inside the Streamlit app. Each function takes a `System` and a
`ScheduleResult` (see solve.py) and returns a matplotlib Figure — callers
decide whether to show(), save(), or hand it to st.pyplot().
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from bess_opt.data.schema import System
from bess_opt.solve import ScheduleResult


def _periods(system: System) -> list[int]:
    return list(range(1, system.n_periods + 1))


def period_revenue(system: System, result: ScheduleResult) -> tuple[list[float], list[float]]:
    """Per-period (arbitrage, regulation) revenue.

    Arbitrage is an energy payment and carries the period duration;
    regulation is a capacity payment and does not.
    """
    prices = system.prices
    dt = prices.delta_t

    arbitrage = [
        dt * prices.energy[t - 1] * (result.discharge[t] - result.charge[t])
        for t in _periods(system)
    ]
    regulation = [
        prices.reg_up[t - 1] * result.reg_up[t] + prices.reg_down[t - 1] * result.reg_down[t]
        for t in _periods(system)
    ]
    return arbitrage, regulation


def plot_price_and_dispatch(system: System, result: ScheduleResult):
    """Charge/discharge power against the energy price that drove it.

    Charging is drawn below the axis: it is power leaving the grid.
    """
    periods = _periods(system)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        periods,
        [result.discharge[t] for t in periods],
        color="seagreen",
        label="Discharge (to grid)",
    )
    ax.bar(
        periods,
        [-result.charge[t] for t in periods],
        color="indianred",
        label="Charge (from grid)",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Period")
    ax.set_ylabel("Power (MW)")

    price_ax = ax.twinx()
    price_ax.plot(
        periods,
        system.prices.energy,
        color="black",
        linewidth=2,
        linestyle="--",
        label="Energy price",
    )
    price_ax.set_ylabel("Energy price ($/MWh)")

    handles, labels = ax.get_legend_handles_labels()
    price_handles, price_labels = price_ax.get_legend_handles_labels()
    ax.legend(handles + price_handles, labels + price_labels, loc="upper left", fontsize="small")

    ax.set_title("Dispatch against energy price")
    fig.tight_layout()
    return fig


def plot_soc_trajectory(system: System, result: ScheduleResult):
    """State of charge over the horizon, with the usable energy band marked.

    The horizon is energy-neutral by construction, so the trace ends where
    it started — see the terminal SOC constraint in model/builder.py.
    """
    battery = system.battery
    periods = _periods(system)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        [0, *periods],
        [battery.initial_soc, *(result.soc[t] for t in periods)],
        marker="o",
        markersize=4,
        color="steelblue",
        label="State of charge",
    )

    ax.axhline(battery.energy_max, color="grey", linestyle="--", linewidth=1)
    ax.axhline(battery.energy_min, color="grey", linestyle="--", linewidth=1)
    ax.axhspan(battery.energy_min, battery.energy_max, color="steelblue", alpha=0.07)
    ax.axhline(
        battery.initial_soc,
        color="darkorange",
        linestyle=":",
        linewidth=1.5,
        label="Initial / required final SOC",
    )

    ax.set_xlabel("Period")
    ax.set_ylabel("Energy (MWh)")
    ax.set_title(f"State of charge — {battery.name}")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


def plot_revenue_stack(system: System, result: ScheduleResult):
    """Where the money comes from, period by period.

    Arbitrage revenue is negative while charging — that is the cost of the
    energy being bought, and the point of the plot is that regulation
    capacity keeps paying through those periods.
    """
    periods = _periods(system)
    arbitrage, regulation = period_revenue(system, result)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(periods, arbitrage, color="steelblue", label="Arbitrage")
    # Stack regulation on top of positive arbitrage, but from zero where
    # arbitrage is negative, so neither bar hides the other.
    bottoms = [max(a, 0.0) for a in arbitrage]
    ax.bar(periods, regulation, bottom=bottoms, color="darkorange", label="Regulation capacity")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Period")
    ax.set_ylabel("Revenue ($)")
    ax.set_title(
        f"Revenue by service — total ${result.total_profit:,.0f} "
        f"(arbitrage ${result.arbitrage_revenue:,.0f}, "
        f"regulation ${result.regulation_revenue:,.0f})"
    )
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig
