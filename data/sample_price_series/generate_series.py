"""Regenerates the sample price series in this directory.

THESE PRICES ARE SYNTHETIC. They are shaped to look like a plausible day in
a market with meaningful solar penetration — an overnight trough, a morning
ramp, a midday dip that goes briefly negative, and a sharp evening peak —
but they are not taken from, calibrated to, or validated against any real
ISO's published data. Use them to exercise the model, not to draw
conclusions about any actual market.

The generator is committed alongside the CSVs so the shape of the data is
inspectable rather than magic. It is deterministic: no RNG, no seed.

Run with:  python data/sample_price_series/generate_series.py
"""

import csv
from pathlib import Path

# One representative day, hour by hour ($/MWh). Hour 14 dips negative:
# oversupply that pays a battery to charge.
DAY_ENERGY = [
    22.0, 20.0, 19.0, 18.0, 19.0, 24.0,
    38.0, 52.0, 60.0, 45.0, 32.0, 26.0,
    18.0, -5.0, 12.0, 28.0, 55.0, 95.0,
    130.0, 120.0, 88.0, 62.0, 44.0, 30.0,
]

# Regulation-up capacity is worth most when the system is ramping hard.
DAY_REG_UP = [
    4.0, 3.5, 3.0, 3.0, 3.5, 5.0,
    9.0, 12.0, 11.0, 7.0, 5.0, 4.0,
    4.0, 5.0, 7.0, 9.0, 12.0, 14.0,
    13.0, 10.0, 8.0, 6.0, 5.0, 4.5,
]

# Regulation-down is worth most when the system is long on energy.
DAY_REG_DOWN = [
    3.0, 3.0, 3.5, 4.0, 4.0, 3.5,
    3.0, 3.0, 4.0, 7.0, 9.0, 10.0,
    12.0, 14.0, 11.0, 8.0, 4.0, 3.0,
    2.5, 2.5, 3.0, 3.5, 4.0, 4.0,
]

# Monday through Sunday. Midweek runs hot; the weekend is soft.
DAY_SCALES = [1.00, 1.05, 0.98, 1.02, 1.10, 0.85, 0.78]

HERE = Path(__file__).parent
HEADER = ["period", "energy_price", "reg_up_price", "reg_down_price"]


def _write(path: Path, energy, reg_up, reg_down) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        rows = zip(energy, reg_up, reg_down, strict=True)
        for period, (e, ru, rd) in enumerate(rows, start=1):
            writer.writerow([period, round(e, 2), round(ru, 2), round(rd, 2)])
    print(f"wrote {path.name} ({len(energy)} periods)")


def main() -> None:
    _write(HERE / "day_hourly.csv", DAY_ENERGY, DAY_REG_UP, DAY_REG_DOWN)

    week_energy, week_reg_up, week_reg_down = [], [], []
    for scale in DAY_SCALES:
        week_energy.extend(e * scale for e in DAY_ENERGY)
        # Capacity prices move with the day's tightness, but less sharply
        # than energy — they are an availability payment, not a scarcity one.
        capacity_scale = 1 + (scale - 1) * 0.5
        week_reg_up.extend(r * capacity_scale for r in DAY_REG_UP)
        week_reg_down.extend(r * capacity_scale for r in DAY_REG_DOWN)

    _write(HERE / "week_hourly.csv", week_energy, week_reg_up, week_reg_down)


if __name__ == "__main__":
    main()
