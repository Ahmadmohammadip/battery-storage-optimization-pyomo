# Battery Energy Storage Optimization with Pyomo

A battery energy storage (BESS) dispatch model built with
[Pyomo](https://www.pyomo.org/) that co-optimizes **energy price arbitrage**
with **frequency regulation capacity** — revenue stacking — as a single linear
program.

> Built incrementally, phase by phase — see `PROJECT_BRIEF.md` for the scope
> agreed before any code was written, and `docs/formulation.md` for the full
> mathematical formulation and the assumptions behind it.

## Features

- **Revenue stacking in one LP** — energy arbitrage and regulation capacity
  are co-optimized, not solved separately. The power headroom constraints
  (`p_dis + r_up <= P_dis_max`) are what make it a single problem: a MW sold
  as regulation capacity cannot also be sold as energy.
- **SOC headroom for deliverability** — committing capacity requires the
  *energy* to honor a call, not just spare power rating, sized by a
  deployment fraction `phi`.
- **Energy-neutral horizon** — the schedule must end at the state of charge
  it started from, so reported profit is not inflated by selling off the
  opening charge in the final period.
- **Pure LP** — no quadratic terms, no binaries. HiGHS solves it; no Ipopt or
  conda needed anywhere in this repo.
- **Validated data model** — a `Battery`, `PriceSeries`, or `System` fails
  loudly at construction (efficiency outside `(0, 1]`, a regulation series
  whose length doesn't match the energy series, an initial SOC outside the
  usable band) rather than surfacing as an opaque solver infeasibility.
- **Negative energy prices supported** — they are real market behavior, and a
  battery earns by charging through them. The sample day includes one.

## Install

```bash
pip install -e ".[dev,solvers,viz]"
```

Requires [HiGHS](https://highs.dev/), installed via the `solvers` extra
(`highspy`). That is the only solver this project needs.

## Quickstart

```python
from bess_opt.data.loaders import load_price_series_csv
from bess_opt.data.schema import Battery, System
from bess_opt.model.builder import build_bess_model
from bess_opt.solve import solve_bess

prices = load_price_series_csv("data/sample_price_series/day_hourly.csv")

battery = Battery(
    name="Batt1",
    p_charge_max=10.0,      # MW
    p_discharge_max=10.0,   # MW
    energy_max=40.0,        # MWh
    initial_soc=20.0,       # MWh
    charge_efficiency=0.94,
    discharge_efficiency=0.94,
)

system = System(battery=battery, prices=prices, phi=0.5)
result = solve_bess(build_bess_model(system))

print(result.total_profit)
print(result.revenue_breakdown())   # {'arbitrage': ..., 'regulation': ...}
print(result.soc[24])               # back where it started
```

Set `include_regulation=False` on the `System` to run the same battery as a
pure arbitrage case through the same builder.

## Repo layout

```
battery-storage-optimization-pyomo/
├── src/bess_opt/
│   ├── data/schema.py      # Battery, PriceSeries, System (validated dataclasses)
│   ├── data/loaders.py     # CSV price series, JSON cases
│   ├── model/builder.py    # Pyomo ConcreteModel construction
│   ├── solve.py            # solver interface, ScheduleResult
│   └── viz.py              # dispatch, SOC, and revenue-stack plots
├── data/sample_price_series/   # synthetic day (24h) and week (168h) series
├── notebooks/01_walkthrough.ipynb
├── app/streamlit_app.py    # interactive demo
├── tests/
├── docs/formulation.md
└── .github/workflows/ci.yml
```

## Interactive demo

```bash
streamlit run app/streamlit_app.py
```

Pick a sample series or upload your own CSV, tune the battery and `phi`, and
compare against the arbitrage-only counterfactual, which is solved on every
run.

## Tests

```bash
pytest -v
```

## A note on the sample data

The price series in `data/sample_price_series/` are **synthetic**. They are
shaped to look like a plausible day in a market with meaningful solar
penetration — overnight trough, morning ramp, a midday dip that goes briefly
negative, sharp evening peak — but they are not taken from or calibrated
against any real ISO. The generator is committed alongside them so the shape
is inspectable rather than magic.

## Known simplifications

No degradation or cycling cost, perfect foresight over a known price series,
price-taker assumption, no unit commitment or charge/discharge exclusivity
binaries, and — most importantly — **no specific ISO's regulation market
rules**. The deployment fraction `phi` is a clearly-labeled assumption used to
size energy reserves, not an implementation of any real tariff. Full list with
rationale in `docs/formulation.md`.

## License

MIT — see `LICENSE`.
