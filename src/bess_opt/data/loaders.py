"""Load validated data objects from CSV and JSON.

Price data is flat and time-indexed, which is exactly what CSV is good at,
so price series live in CSV. A full case — battery parameters plus the
prices to run them against — is nested, so that lives in JSON and may point
at a CSV for its price series.

Everything returned here has already passed through the schema's validation
(see data/schema.py); a malformed file fails at load, not at solve.
"""

import csv
import json
from pathlib import Path

from bess_opt.data.schema import Battery, PriceSeries, System

ENERGY_COLUMN = "energy_price"
REG_UP_COLUMN = "reg_up_price"
REG_DOWN_COLUMN = "reg_down_price"


def load_price_series_csv(path: str | Path, delta_t: float = 1.0) -> PriceSeries:
    """Read a price series from CSV.

    Requires an `energy_price` column. `reg_up_price` and `reg_down_price`
    are optional — a file without them loads as an arbitrage-only series.
    Any `period` column is ignored: row order defines the horizon.

    `delta_t` is the period duration in hours and is not inferred from the
    file, because nothing in a bare CSV distinguishes 24 hourly rows from 24
    five-minute rows.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return _price_series_from_rows(rows, path.name, delta_t)


def load_price_series_text(
    text: str, label: str = "uploaded.csv", delta_t: float = 1.0
) -> PriceSeries:
    """Same as `load_price_series_csv`, for CSV content already in memory —
    an uploaded file in the Streamlit app, for instance. `label` only appears
    in error messages."""
    rows = list(csv.DictReader(text.lstrip("﻿").splitlines()))
    return _price_series_from_rows(rows, label, delta_t)


def _price_series_from_rows(rows: list[dict], label: str, delta_t: float) -> PriceSeries:
    if not rows:
        raise ValueError(f"{label}: file has a header but no data rows")

    columns = rows[0].keys()
    if ENERGY_COLUMN not in columns:
        raise ValueError(
            f"{label}: missing required column '{ENERGY_COLUMN}' "
            f"(found: {sorted(c for c in columns if c)})"
        )

    energy = _column(rows, ENERGY_COLUMN, label)
    reg_up = _column(rows, REG_UP_COLUMN, label) if REG_UP_COLUMN in columns else None
    reg_down = _column(rows, REG_DOWN_COLUMN, label) if REG_DOWN_COLUMN in columns else None

    return PriceSeries(energy=energy, reg_up=reg_up, reg_down=reg_down, delta_t=delta_t)


def load_system_json(path: str | Path) -> System:
    """Read a full case from JSON.

    Expected shape — `prices` may be given inline or as a path to a CSV,
    resolved relative to the JSON file:

        {
          "battery": {"name": "Batt1", "p_charge_max": 10, ...},
          "prices": {"energy": [...], "reg_up": [...], "delta_t": 1.0},
          "phi": 0.5,
          "include_regulation": true
        }

        {
          "battery": {...},
          "price_series_csv": "../sample_price_series/day_hourly.csv",
          "delta_t": 1.0
        }
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if "battery" not in data:
        raise ValueError(f"{path.name}: missing required key 'battery'")
    battery = Battery(**data["battery"])

    has_inline = "prices" in data
    has_csv = "price_series_csv" in data
    if has_inline == has_csv:
        raise ValueError(
            f"{path.name}: provide exactly one of 'prices' or 'price_series_csv', "
            f"not both and not neither"
        )

    if has_inline:
        prices = PriceSeries(**data["prices"])
    else:
        csv_path = (path.parent / data["price_series_csv"]).resolve()
        prices = load_price_series_csv(csv_path, delta_t=data.get("delta_t", 1.0))

    return System(
        battery=battery,
        prices=prices,
        phi=data.get("phi", 0.5),
        include_regulation=data.get("include_regulation", True),
    )


def _column(rows: list[dict], name: str, label: str) -> list[float]:
    values = []
    for line_number, row in enumerate(rows, start=2):  # row 1 is the header
        raw = row.get(name)
        if raw is None or raw.strip() == "":
            raise ValueError(f"{label} line {line_number}: empty value in column '{name}'")
        try:
            values.append(float(raw))
        except ValueError as exc:
            raise ValueError(
                f"{label} line {line_number}: column '{name}' is not a number ({raw!r})"
            ) from exc
    return values
