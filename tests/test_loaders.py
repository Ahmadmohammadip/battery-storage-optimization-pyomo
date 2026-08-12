"""Phase 4: CSV and JSON loading, including the failure messages."""

import json
from pathlib import Path

import pytest

from bess_opt.data.loaders import load_price_series_csv, load_system_json

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_price_series"


def test_loads_the_sample_day():
    prices = load_price_series_csv(SAMPLE_DIR / "day_hourly.csv")

    assert len(prices) == 24
    assert prices.delta_t == 1.0
    assert prices.has_regulation
    # The sample day deliberately includes a negative hour (midday oversupply).
    assert min(prices.energy) < 0


def test_loads_the_sample_week():
    prices = load_price_series_csv(SAMPLE_DIR / "week_hourly.csv")

    assert len(prices) == 168
    assert len(prices.reg_up) == 168
    assert len(prices.reg_down) == 168


def test_delta_t_is_explicit_not_inferred():
    prices = load_price_series_csv(SAMPLE_DIR / "day_hourly.csv", delta_t=0.25)
    assert prices.delta_t == 0.25


def test_regulation_columns_are_optional(tmp_path):
    path = tmp_path / "energy_only.csv"
    path.write_text("period,energy_price\n1,10\n2,50\n", encoding="utf-8")

    prices = load_price_series_csv(path)

    assert prices.energy == [10.0, 50.0]
    assert prices.has_regulation is False


def test_missing_energy_column_is_rejected(tmp_path):
    path = tmp_path / "no_energy.csv"
    path.write_text("period,reg_up_price\n1,4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column 'energy_price'"):
        load_price_series_csv(path)


def test_non_numeric_value_names_the_line(tmp_path):
    path = tmp_path / "bad_value.csv"
    path.write_text("period,energy_price\n1,10\n2,not-a-price\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 3: column 'energy_price' is not a number"):
        load_price_series_csv(path)


def test_empty_value_names_the_line(tmp_path):
    path = tmp_path / "empty_value.csv"
    path.write_text("period,energy_price\n1,10\n2,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 3: empty value"):
        load_price_series_csv(path)


def test_header_only_file_is_rejected(tmp_path):
    path = tmp_path / "header_only.csv"
    path.write_text("period,energy_price\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no data rows"):
        load_price_series_csv(path)


def _battery_config() -> dict:
    return {
        "name": "Batt1",
        "p_charge_max": 10.0,
        "p_discharge_max": 10.0,
        "energy_max": 20.0,
        "initial_soc": 5.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }


def test_loads_system_with_inline_prices(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "battery": _battery_config(),
                "prices": {"energy": [10.0, 50.0], "reg_up": [3.0, 3.0], "delta_t": 1.0},
                "phi": 0.4,
            }
        ),
        encoding="utf-8",
    )

    system = load_system_json(path)

    assert system.n_periods == 2
    assert system.phi == 0.4
    assert system.battery.name == "Batt1"
    assert system.include_regulation is True


def test_loads_system_pointing_at_a_csv(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "battery": _battery_config(),
                "price_series_csv": str(SAMPLE_DIR / "day_hourly.csv"),
            }
        ),
        encoding="utf-8",
    )

    system = load_system_json(path)

    assert system.n_periods == 24
    assert system.phi == 0.5  # default


def test_system_json_requires_exactly_one_price_source(tmp_path):
    both = tmp_path / "both.json"
    both.write_text(
        json.dumps(
            {
                "battery": _battery_config(),
                "prices": {"energy": [10.0]},
                "price_series_csv": "day_hourly.csv",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one of 'prices' or 'price_series_csv'"):
        load_system_json(both)

    neither = tmp_path / "neither.json"
    neither.write_text(json.dumps({"battery": _battery_config()}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one of 'prices' or 'price_series_csv'"):
        load_system_json(neither)


def test_system_json_requires_a_battery(tmp_path):
    path = tmp_path / "no_battery.json"
    path.write_text(json.dumps({"prices": {"energy": [10.0]}}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required key 'battery'"):
        load_system_json(path)


def test_invalid_battery_in_json_fails_at_load(tmp_path):
    # Validation lives in the schema, so a bad file fails at load time rather
    # than surfacing later as an odd solver result.
    config = _battery_config()
    config["charge_efficiency"] = 1.5
    path = tmp_path / "bad_battery.json"
    path.write_text(
        json.dumps({"battery": config, "prices": {"energy": [10.0]}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="charge_efficiency"):
        load_system_json(path)
