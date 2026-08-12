"""
Typed, validated data structures for the battery storage optimization model.

Design intent (same as the model layer's contract): a `Battery`,
`PriceSeries`, or `System` object fails loudly at construction time —
efficiency outside (0, 1], a regulation price series whose length doesn't
match the energy price series, an initial SOC outside the usable energy
band — rather than surfacing three layers down as an opaque solver
infeasibility.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Battery:
    """A single battery energy storage system.

    Fidelity is deliberately simple: efficiency losses plus power and
    energy limits. Degradation, cycling cost, and capacity fade are out of
    scope (see PROJECT_BRIEF.md section 4).

    Energy limits are expressed as an absolute usable band
    [energy_min, energy_max] in MWh, not as a percentage depth-of-discharge.
    """

    name: str
    p_charge_max: float       # MW
    p_discharge_max: float    # MW
    energy_max: float         # MWh
    initial_soc: float        # MWh
    energy_min: float = 0.0   # MWh
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0

    def __post_init__(self):
        if self.p_charge_max <= 0:
            raise ValueError(
                f"{self.name}: p_charge_max must be > 0, got {self.p_charge_max}"
            )
        if self.p_discharge_max <= 0:
            raise ValueError(
                f"{self.name}: p_discharge_max must be > 0, got {self.p_discharge_max}"
            )
        if self.energy_min < 0:
            raise ValueError(f"{self.name}: energy_min must be >= 0, got {self.energy_min}")
        if self.energy_max <= self.energy_min:
            raise ValueError(
                f"{self.name}: energy_max ({self.energy_max}) must be > "
                f"energy_min ({self.energy_min})"
            )
        if not (0 < self.charge_efficiency <= 1):
            raise ValueError(
                f"{self.name}: charge_efficiency must be in (0, 1], "
                f"got {self.charge_efficiency}"
            )
        if not (0 < self.discharge_efficiency <= 1):
            raise ValueError(
                f"{self.name}: discharge_efficiency must be in (0, 1], "
                f"got {self.discharge_efficiency}"
            )
        if not (self.energy_min <= self.initial_soc <= self.energy_max):
            raise ValueError(
                f"{self.name}: initial_soc ({self.initial_soc}) must be within the usable "
                f"band [{self.energy_min}, {self.energy_max}]"
            )

    @property
    def usable_energy(self) -> float:
        """Width of the usable energy band (MWh)."""
        return self.energy_max - self.energy_min

    @property
    def round_trip_efficiency(self) -> float:
        return self.charge_efficiency * self.discharge_efficiency


@dataclass(frozen=True)
class PriceSeries:
    """Market prices over the optimization horizon.

    `energy` is the energy price in $/MWh and **may be negative** — negative
    prices are real market behavior (oversupply), not bad data, and a
    battery earns by charging through them.

    `reg_up` / `reg_down` are regulation *capacity* prices in $/MW, paid for
    committing capacity whether or not it is called. Capacity prices must be
    non-negative. Both default to None, which is normalized to a series of
    zeros — that is how an arbitrage-only case is expressed.

    `delta_t` is the period duration in hours (1.0 for hourly data, 1/12 for
    5-minute intervals).
    """

    energy: list[float]
    reg_up: list[float] | None = None
    reg_down: list[float] | None = None
    delta_t: float = 1.0

    def __post_init__(self):
        if not self.energy:
            raise ValueError("PriceSeries.energy must have at least one period")
        if self.delta_t <= 0:
            raise ValueError(f"PriceSeries.delta_t must be > 0, got {self.delta_t}")

        n = len(self.energy)
        for label in ("reg_up", "reg_down"):
            series = getattr(self, label)
            if series is None:
                continue
            if len(series) != n:
                raise ValueError(
                    f"PriceSeries.{label} covers {len(series)} periods but energy covers "
                    f"{n} — all price series must span the same horizon"
                )
            if any(p < 0 for p in series):
                raise ValueError(
                    f"PriceSeries.{label} values must be >= 0 — a regulation capacity "
                    f"price cannot be negative (unlike the energy price, which can be)"
                )

        # Normalize the optional series to explicit zeros so the model layer
        # never has to branch on None.
        object.__setattr__(
            self, "reg_up", list(self.reg_up) if self.reg_up is not None else [0.0] * n
        )
        object.__setattr__(
            self, "reg_down", list(self.reg_down) if self.reg_down is not None else [0.0] * n
        )
        object.__setattr__(self, "energy", list(self.energy))

    def __len__(self) -> int:
        return len(self.energy)

    @property
    def has_regulation(self) -> bool:
        """True if any regulation capacity price is non-zero."""
        return any(p > 0 for p in self.reg_up) or any(p > 0 for p in self.reg_down)


@dataclass(frozen=True)
class System:
    """A battery plus the prices it optimizes against.

    `phi` is the assumed regulation deployment fraction used to size the SOC
    headroom constraints. It is a clearly-labeled simplifying assumption, NOT
    an implementation of any ISO's actual regulation rules — see
    docs/formulation.md.

    `include_regulation=False` forces all regulation capacity to zero, giving
    a pure energy-arbitrage model through the same builder.
    """

    battery: Battery
    prices: PriceSeries
    phi: float = 0.5
    include_regulation: bool = True

    def __post_init__(self):
        if not (0 <= self.phi <= 1):
            raise ValueError(
                f"phi (regulation deployment fraction) must be in [0, 1], got {self.phi}"
            )

    @property
    def n_periods(self) -> int:
        return len(self.prices)
