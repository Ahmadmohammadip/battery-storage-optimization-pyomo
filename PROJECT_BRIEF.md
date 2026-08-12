# Battery Energy Storage Optimization with Pyomo — Handoff Brief

## Purpose of this document
This is a handoff brief for Claude Code (or any engineer) to build this
project from scratch. It captures the locked scope, full mathematical
formulation, repo architecture, and phased build plan agreed on before any
code was written. No code exists yet — this brief is the starting point.

## Goal
Public GitHub repo: a battery energy storage system (BESS) optimization
model in Pyomo that co-optimizes energy arbitrage with frequency
regulation capacity ("revenue stacking"), built and committed phase by
phase, each phase tested and working before moving to the next. Repo
should be public from the first commit. This is a companion/portfolio
piece to an existing repo, `economic-dispatch-pyomo` (multi-period DC-OPF
Economic Dispatch) — same conventions, same level of polish, but built as
a fully independent, standalone repo (no shared code between them).

## Scope (locked decisions)
- **Use case**: multi-service — energy price arbitrage *and* frequency
  regulation capacity, co-optimized in a single LP.
- **Battery model fidelity**: simple — efficiency losses + power/energy
  limits only. No degradation or capacity-fade modeling in this scope
  (see "Explicitly out of scope" below).
- **Repo structure**: installable package (`src/bess_opt`) + notebook +
  Streamlit demo app, same shape as `economic-dispatch-pyomo`.
- **Horizon**: perfect-foresight optimization over a known price series
  (not a rolling/receding-horizon re-optimization — that's a documented
  future extension, not this scope).
- **Solver**: pure LP — no quadratic terms anywhere in this formulation.
  HiGHS is the default solver; no need for Ipopt in this repo at all
  (unlike `economic-dispatch-pyomo`, which needs both).

## 1. Mathematical Formulation

### 1.1 Sets and indices

| Symbol | Description |
|---|---|
| $T$ | Time periods, index $t = 1, \dots, T$ (e.g. hourly or 5-minute intervals) |

### 1.2 Parameters

| Symbol | Description |
|---|---|
| $\pi^{E}_t$ | Energy price ($/MWh) at period $t$ |
| $\pi^{RU}_t, \pi^{RD}_t$ | Regulation-up / regulation-down capacity price ($/MW) at period $t$ |
| $\Delta t$ | Period duration (hours) |
| $\eta^{ch}, \eta^{dis}$ | Charge / discharge efficiency, each in $(0, 1]$ |
| $\overline{P}^{ch}, \overline{P}^{dis}$ | Max charge / discharge power (MW) |
| $\overline{E}, \underline{E}$ | Max / min usable state of charge (MWh) |
| $SOC_0$ | Initial state of charge (MWh) |
| $\phi$ | Assumed regulation "deployment fraction" — see Section 1.5 |

### 1.3 Decision variables

| Symbol | Description |
|---|---|
| $p^{ch}_t, p^{dis}_t$ | Charge / discharge power in the energy market (MW), both $\ge 0$ |
| $r^{up}_t, r^{dn}_t$ | Regulation-up / regulation-down capacity committed (MW), both $\ge 0$ |
| $e_t$ | State of charge (MWh) |

### 1.4 Objective — maximize profit

$$
\max \sum_{t} \Delta t \cdot \pi^E_t \left( p^{dis}_t - p^{ch}_t \right) \;+\; \sum_t \left( \pi^{RU}_t \, r^{up}_t + \pi^{RD}_t \, r^{dn}_t \right)
$$

Implement as **minimize negative profit** (`sense=minimize` on the negated
expression), for consistency with the sibling repo's convention.

### 1.5 Constraints

**Power headroom for regulation deliverability** (the constraint that
actually couples the two markets — committing regulation-up capacity
means promising you *could* discharge that much more if called, so it
must fit inside the same power rating as energy-market discharge):

$$
p^{dis}_t + r^{up}_t \le \overline{P}^{dis}, \qquad p^{ch}_t + r^{dn}_t \le \overline{P}^{ch} \quad \forall t
$$

**SOC dynamics** (only energy-market flows affect actual SOC — regulation
is a capacity *commitment*, not a guaranteed energy flow, per the
deployment-fraction note below):

$$
e_t = e_{t-1} + \eta^{ch} p^{ch}_t \Delta t - \frac{p^{dis}_t}{\eta^{dis}} \Delta t, \qquad \underline{E} \le e_t \le \overline{E} \quad \forall t
$$

with $e_0 = SOC_0$.

**SOC headroom for regulation** (the battery must have enough *energy*
margin to actually deliver regulation if called, not just enough power):

$$
e_t + \phi \, r^{dn}_t \, \Delta t \le \overline{E}, \qquad e_t - \phi \, r^{up}_t \, \Delta t \ge \underline{E} \quad \forall t
$$

**Non-negativity**: $p^{ch}_t, p^{dis}_t, r^{up}_t, r^{dn}_t \ge 0$

### 1.6 Modeling notes and assumptions to document explicitly in `docs/formulation.md`

- **Regulation deployment fraction ($\phi$)**: real ISO markets (e.g.
  PJM RegD/RegA, CAISO) have specific, published "mileage" and
  performance-score rules governing how much of committed regulation
  capacity is actually called, and how it is paid. This project does
  **not** claim to implement any specific ISO's rules — no verified,
  current rule text for a named market was used. $\phi$ is a simple,
  clearly-labeled assumption (e.g. 0.5 — "on average, half of committed
  capacity is deployed") used only to size the SOC headroom constraint.
  This must be stated plainly in `docs/formulation.md`, not implied to be
  market-accurate. If this is later extended to match a specific ISO
  product, that requires sourcing the actual tariff/manual — flag as a
  possible next step, not a current claim.
- **No simultaneous charge/discharge exclusivity constraint**: with
  $\eta < 1$, using both flows at once to game the objective is
  self-penalizing in nearly all cases. Deliberately left as an LP (no
  binaries) with a targeted test case rather than enforced structurally —
  same reasoning used in `economic-dispatch-pyomo` for its storage model.
- **Price-taker assumption**: the battery is assumed small enough that
  its bids don't move market prices. State this explicitly.
- **Perfect foresight**: the model optimizes against a known price
  series. A rolling/receding-horizon variant (re-optimizing as new price
  forecasts arrive) is out of scope for this build — listed as a
  possible next step.

## 2. Repo architecture

```
battery-storage-optimization-pyomo/
├── README.md
├── LICENSE (MIT)
├── pyproject.toml
├── PROJECT_BRIEF.md              # this document, or a version of it
├── src/
│   └── bess_opt/
│       ├── __init__.py
│       ├── data/
│       │   ├── schema.py         # Battery, PriceSeries, System (validated dataclasses)
│       │   └── loaders.py        # CSV/JSON -> validated data objects
│       ├── model/
│       │   ├── __init__.py
│       │   └── builder.py        # Pyomo ConcreteModel construction
│       ├── solve.py              # solver interface, result dataclass, result extraction
│       └── viz.py                # SOC trajectory, price/dispatch overlay, revenue stack plots
├── data/
│   └── sample_price_series/      # example day/week price + regulation price CSVs
├── notebooks/
│   └── 01_walkthrough.ipynb
├── app/
│   └── streamlit_app.py          # pick/upload a price series, tune battery params, see optimal schedule
├── tests/
│   ├── test_soc_dynamics.py
│   ├── test_regulation_headroom.py
│   ├── test_arbitrage_only.py
│   └── test_integration.py
├── .github/workflows/ci.yml      # ruff + pytest, HiGHS only (no conda/Ipopt needed)
└── docs/
    └── formulation.md            # Section 1 above, rendered, with the assumptions called out
```

**Design rationale** (carried over from `economic-dispatch-pyomo`):
separate `data/schema.py` (validated dataclasses) from `model/builder.py`
(pure Pyomo construction) — the model layer should never touch raw
CSV/JSON directly. A `System`/`Battery`/`PriceSeries` object should fail
loudly at construction time (e.g. `p_min > p_max`, efficiency outside
$(0,1]$, price series length mismatch with regulation price series)
rather than surfacing as an opaque solver infeasibility three layers down.

## 3. Build plan (phased)

| Phase | Scope | Output |
|---|---|---|
| 1 | Arbitrage-only, single battery, no regulation | Working LP; sanity-checked against a hand-picked price series (charge on low prices, discharge on high prices) |
| 2 | Add regulation-up/down capacity variables + power headroom constraints (Section 1.5, first constraint block) | Revenue-stacking model, still LP |
| 3 | Add SOC headroom constraint (the $\phi$ deployment-fraction logic) | Full formulation from Section 1 |
| 4 | Data loaders + sample price series (synthetic but realistically-shaped day/week) | `data/sample_price_series/`, `loaders.py` |
| 5 | `viz.py` + notebook walkthrough | Dispatch, SOC, and revenue-stack plots |
| 6 | Streamlit app | Interactive demo: pick/upload a price series, tune battery params, see optimal schedule |
| 7 | Tests, CI, README, `docs/formulation.md` polish | GitHub-ready |

Each phase should leave `main` green (tests passing) before moving to the
next, and ideally corresponds to its own commit(s) — same convention as
`economic-dispatch-pyomo`.

## 4. Explicitly out of scope (do not build unless asked)
- Battery degradation / cycling cost or capacity fade over time
- Rolling/receding-horizon re-optimization (perfect foresight only)
- Any specific ISO's actual regulation market rules (mileage,
  performance score, real tariff structure) — the $\phi$ parameter is a
  clearly-labeled simplification, not a claim of market accuracy
- Multi-battery fleets or portfolio-level optimization
- Unit commitment / binary charge-discharge exclusivity (kept as LP)

## 5. Git conventions
- One phase per commit (or a few commits per phase if large), each
  commit should leave `main` green
- Commit message prefixes: `feat` / `test` / `docs` / `ci` / `chore` / `fix`
- Public repo from commit 1
- Suggested repo name: `battery-storage-optimization-pyomo`

## 6. Provenance note
This brief was authored directly in this conversation (not reconstructed
or transcribed from an external source) as a planning document, before
any code was written. It is the complete specification agreed on so far —
nothing here should be treated as already-implemented.
