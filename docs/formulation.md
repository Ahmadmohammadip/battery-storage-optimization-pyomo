# Formulation

A single linear program co-optimizing energy arbitrage against frequency
regulation capacity for one battery over a known price series. This document
matches the implementation in `src/bess_opt/`.

## 1. Sets and indices

| Symbol | Description |
|---|---|
| $T$ | Time periods, index $t = 1, \dots, T$ |

A single battery, so there is no unit index — multi-battery portfolios are
out of scope.

## 2. Parameters

| Symbol | Description | Code |
|---|---|---|
| $\pi^{E}_t$ | Energy price ($/MWh) | `PriceSeries.energy` |
| $\pi^{RU}_t, \pi^{RD}_t$ | Regulation-up / -down capacity price ($/MW) | `PriceSeries.reg_up`, `.reg_down` |
| $\Delta t$ | Period duration (hours) | `PriceSeries.delta_t` |
| $\eta^{ch}, \eta^{dis}$ | Charge / discharge efficiency, each in $(0, 1]$ | `Battery.charge_efficiency`, `.discharge_efficiency` |
| $\overline{P}^{ch}, \overline{P}^{dis}$ | Max charge / discharge power (MW) | `Battery.p_charge_max`, `.p_discharge_max` |
| $\overline{E}, \underline{E}$ | Max / min usable state of charge (MWh) | `Battery.energy_max`, `.energy_min` |
| $SOC_0$ | Initial state of charge (MWh) | `Battery.initial_soc` |
| $\phi$ | Assumed regulation deployment fraction — see §6 | `System.phi` |

## 3. Decision variables

| Symbol | Description | Code |
|---|---|---|
| $p^{ch}_t, p^{dis}_t$ | Charge / discharge power in the energy market (MW), $\ge 0$ | `m.p_ch`, `m.p_dis` |
| $r^{up}_t, r^{dn}_t$ | Regulation-up / -down capacity committed (MW), $\ge 0$ | `m.r_up`, `m.r_dn` |
| $e_t$ | State of charge (MWh) | `m.soc` |

## 4. Objective — maximize profit

$$
\max \sum_{t} \Delta t \cdot \pi^E_t \left( p^{dis}_t - p^{ch}_t \right) \;+\; \sum_t \left( \pi^{RU}_t \, r^{up}_t + \pi^{RD}_t \, r^{dn}_t \right)
$$

Implemented as **minimize negative profit** (`sense=minimize` on the negated
expression), matching the sibling `economic-dispatch-pyomo` repo, which
minimizes cost.

Note the asymmetry in $\Delta t$: energy is paid per MWh, so the energy term
scales with period duration. Regulation is paid per MW of capacity
*committed* for the period, so its term does not.

## 5. Constraints

### 5.1 Power headroom

$$
p^{dis}_t + r^{up}_t \le \overline{P}^{dis}, \qquad p^{ch}_t + r^{dn}_t \le \overline{P}^{ch} \quad \forall t
$$

This is what couples the two markets. Committing regulation-up capacity is a
promise that the battery *could* discharge that much more if called, and that
promise has to fit inside the same power rating as energy-market discharge.
Without it, the two markets would decouple into independent problems and the
battery could sell the same MW twice.

### 5.2 SOC dynamics

$$
e_t = e_{t-1} + \eta^{ch} p^{ch}_t \Delta t - \frac{p^{dis}_t}{\eta^{dis}} \Delta t, \qquad \underline{E} \le e_t \le \overline{E} \quad \forall t
$$

with $e_0 = SOC_0$. Only energy-market flows move the state of charge:
regulation is a capacity commitment, not a guaranteed energy delivery.

### 5.3 SOC headroom for regulation

$$
e_t + \phi \, r^{dn}_t \, \Delta t \le \overline{E}, \qquad e_t - \phi \, r^{up}_t \, \Delta t \ge \underline{E} \quad \forall t
$$

Power headroom alone is not enough — the battery also needs the *energy* to
honor a call. A nearly-empty battery has plenty of spare discharge rating but
nothing to discharge, and without these constraints it would happily sell
reg-up it could not deliver.

### 5.4 Terminal state of charge

$$
e_T = SOC_0
$$

**This constraint is not in `PROJECT_BRIEF.md` §1.5.** It was added after the
brief was written, for a reason worth stating: under perfect foresight, energy
left in the battery at the end of the horizon has no value, so an
unconstrained model always drains to $\underline{E}$ in the final periods.
That inflates reported profit by the value of the starting charge and makes
the tail of every schedule an artifact of where the horizon happens to stop.

Requiring the horizon to be energy-neutral is always feasible — holding still
satisfies it — so it cannot make a well-formed system infeasible. The cost is
that a genuinely profitable end-of-horizon drawdown is now forbidden; for a
day or week against a repeating price shape, that is the right trade.

### 5.5 Non-negativity

$p^{ch}_t, p^{dis}_t, r^{up}_t, r^{dn}_t \ge 0$.

## 6. Assumptions, stated plainly

### The deployment fraction $\phi$ is an assumption, not a market rule

Real ISO regulation markets (PJM RegD/RegA, CAISO, and others) have specific
published rules governing how much committed capacity is actually called and
how it is paid — mileage, performance scores, tariff structures. **This model
implements none of them.** No verified, current rule text for any named market
was consulted in building it.

$\phi$ is a single number expressing "on average, this share of committed
capacity gets deployed," used only to size the SOC headroom in §5.3. The
default of 0.5 is a round number, not a calibrated estimate.

The honest way to use a parameter like this is to sweep it and report how much
the answer moves — the walkthrough notebook does exactly that. Extending this
model to a specific ISO product would require sourcing that market's actual
tariff or manual, and is a possible next step, not a current claim.

### No simultaneous charge/discharge exclusivity constraint

The model can, in principle, charge and discharge at once. Enforcing otherwise
requires binaries, which would make this a MILP.

It is left as an LP because with $\eta^{ch} \eta^{dis} < 1$, doing both at once
strictly loses energy and money, so the optimum avoids it unprompted —
`test_integration.py` verifies this holds across a full week. The exception is
a perfectly lossless battery, where the two are a tie and the solver may return
either; `test_soc_dynamics.py` documents that degeneracy. Net power is the
meaningful quantity in that case.

### Price taker

The battery is assumed small enough that its bids do not move market prices.
Prices are exogenous inputs, unaffected by what the model decides to do.

### Perfect foresight

The entire price series is known when the schedule is chosen. A real operator
re-optimizes against a forecast as the horizon rolls forward and earns less
than this model reports. A rolling/receding-horizon variant is out of scope
here and listed as a possible next step.

### Simple battery physics

Efficiency losses and power/energy limits only. No degradation, no cycling
cost, no capacity fade, no temperature or state-of-health effects. Cycling is
free in this model, so it will cycle harder than an operator watching warranty
terms would allow.

## 7. Out of scope

Carried from `PROJECT_BRIEF.md` §4:

- Battery degradation / cycling cost / capacity fade
- Rolling or receding-horizon re-optimization
- Any specific ISO's actual regulation market rules
- Multi-battery fleets or portfolio optimization
- Unit commitment or binary charge/discharge exclusivity
