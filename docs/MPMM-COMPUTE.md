# What One Onboard Propagation Costs

**155 mm M107 HE · step 2 · Task E**

Step 3's guidance law will call the MPMM in a loop: from the current state,
propagate to ground impact, compare the predicted impact point with the target,
steer. At roughly 10 Hz, on a microcontroller. This measures what one such
propagation costs, what coarsening the step buys and loses, and which terms are
worth their compute.

Reproduce with `python -m analysis.mpmm_compute`.

## Read the absolute times with care

Everything below is **CPython on the development machine**. That is not the
target and the microsecond figures do not transfer. Repeated runs of this
script vary by ±25 % depending on what else the machine is doing — the
per-derivative cost measured across runs during this session ranged from 9.6 to
12.6 µs.

**What transfers is the derivative count and the relative cost of each term.**
Those are properties of the algorithm, and they are what step 7 needs in order
to size the C implementation. The wall-clock columns are here to show the
shape of the trade, not to be quoted as a budget.

---

## The reference engagement

Charge 8, QE 525.3 mils — the longest shot in the firing-table envelope,
15.8 km, and therefore the worst case for propagation cost.

| | |
|---|---|
| Initialisation | true 6-DOF state at apogee, t = 22.20 s, 3096.8 m |
| Remaining flight | **26.29 s** |
| Reference solution | dt = 0.001 s → range 15 836.561 m, drift 327.172 m |

Every "Δ" below is against that dt = 0.001 s reference.

---

## 1. Cost and accuracy versus step size

| dt (s) | steps | **derivative calls** | wall (ms) | µs/deriv | ΔRange (m) | ΔDrift (m) |
|---|---|---|---|---|---|---|
| 0.001 | 26 275 | 105 100 | 1320.1 | 12.6 | — | — |
| 0.005 | 5 255 | 21 020 | 249.5 | 11.9 | −0.000016 | −0.0000002 |
| **0.01** | 2 628 | 10 512 | 126.7 | 12.1 | −0.000116 | −0.000004 |
| 0.02 | 1 314 | 5 256 | 63.0 | 12.0 | −0.00042 | −0.000011 |
| 0.05 | 526 | 2 104 | 25.3 | 12.0 | −0.0031 | −0.000089 |
| **0.1** | 263 | 1 052 | 12.7 | 12.1 | −0.0095 | −0.00029 |
| 0.2 | 132 | 528 | 7.0 | 13.3 | −0.047 | −0.0014 |
| 0.5 | 53 | 212 | 3.5 | 16.6 | −0.316 | −0.0086 |

**Step size is essentially free accuracy here.** A 500× coarsening from
0.001 s to 0.5 s costs **32 cm of range** on a 15.8 km shot — 20 parts per
million. Against a model error of metres (see
[MODEL-ERROR.md](MODEL-ERROR.md)) and a CEP budget of tens of metres, every
step size in this table is exact.

### Where the step-size error actually comes from

Not from RK4. The convergence order measured across the table is **≈ 2**, not
4:

```
0.5 -> 0.2 : error ratio 6.8  (2.5^2.1)
0.2 -> 0.1 : error ratio 4.9  (2^2.3)
0.1 -> 0.05: error ratio 3.0  (2^1.6)
```

That is the **linear interpolation across the final step to z = 0**, not the
integration. The check is quantitative: for a chord of duration dt through a
trajectory of curvature a, the interpolation error is about a·dt²/8. At
dt = 0.5 s with a ≈ 9.8 m/s² that is **0.31 m**, against a measured 0.316 m.

Two consequences:

1. **Shrinking dt to buy impact accuracy is the wrong lever** — it improves a
   4th-order term that is already negligible while the 2nd-order interpolation
   dominates. If sub-centimetre impact prediction is ever needed, solve the
   final step properly (one Newton iteration on z(t) using the known
   derivative) rather than halving the step.
2. It is not needed. The interpolation error at any usable step size is orders
   of magnitude below the model error.

### Recommended step size: **dt = 0.1 s**

1052 derivative evaluations for a full 26-second propagation, at a cost of
1 cm of range. Even dt = 0.2 s (528 evaluations, 5 cm) is defensible. The
recommendation is 0.1 s only because there is no reason to push further —
the saving is 7 ms of a 12.7 ms budget on a machine that is not the target.

**Note:** the validation runs in [MPMM-VALIDATION.md](MPMM-VALIDATION.md) and
[MODEL-ERROR.md](MODEL-ERROR.md) use dt = 0.01 s, ten times finer than this
recommendation. That was deliberate: it puts the integration error four orders
of magnitude below the model error being measured, so those documents report
model-structure error uncontaminated by step size. This table is what
retrospectively justifies that claim.

---

## 2. Which terms are worth their compute

### Accuracy cost of dropping a term

| Term dropped | ΔRange (m) | ΔDrift (m) |
|---|---|---|
| Magnus force | **20.53** | 1.07 |
| lift (the yaw-of-repose force) | 3.21 | **−100.78** |
| yaw drag (α² increment to C_D) | 2.29 | 0.03 |
| *Coriolis, **added** rather than dropped* | *−0.41* | *−11.72* |

### Compute cost of dropping a term

The end-to-end propagation timings for these ablations came out **inside the
measurement noise** — one of them repeatedly timed *slower* with a term
removed. That is not a result, it is a null measurement, and it is reported as
one. Timing the derivative directly over 20 000 calls resolves what the
propagation timing cannot:

| Variant | µs/deriv | vs full |
|---|---|---|
| **full model** | **12.09** | — |
| no Magnus | 11.00 | −1.09 (−9 %) |
| no yaw drag | 11.31 | −0.78 (−6 %) |
| no lift | 11.49 | −0.60 (−5 %) |
| **no yaw terms at all** (pure 3-DOF) | **10.53** | **−1.57 (−13 %)** |
| — of which ISA atmosphere | 0.84 | 7 % of the total |
| — of which aero table lookup | 1.08 | 9 % of the total |

### The answer to "what is the largest term you could drop, and what does it cost?"

**None of them, and the question has a structural answer rather than a
numerical one.**

Dropping *all three* yaw-dependent forces saves only 13 % of the derivative,
because the expensive part is not the forces — it is the **yaw of repose
itself**, which has to be formed before any of the three can be evaluated and
is shared by all of them. Once α_e exists, each force is a handful of
multiply-adds. So the terms are individually near-free (5–9 % each) and
collectively still cheap (13 %), while their accuracy contributions are
20.5 m, 100.8 m and 2.3 m respectively.

If one had to be named, it is the **yaw-drag term**: 6 % of the derivative for
2.3 m of range and 3 cm of deflection — the only term whose accuracy
contribution is of the same order as the model error itself. It is kept
anyway, because 6 % of a 12 µs derivative is not a saving worth 2.3 m.

The genuinely expensive shared costs are the **ISA atmosphere (7 %)** and the
**aero table lookup (9 %)** — 16 % between them, and both are trivially
cacheable in the C port if the pressure/density evaluation is replaced with a
table and the Mach search is made incremental. That is where step 7 should
look, not at the force terms.

**Coriolis is worth its 12 m of deflection** and should be on in flight. Note
that these validation runs have it *off*, to match step 1's firing-table
convention; that is a comparison convention, not a recommendation.

---

## 3. The yaw-of-repose iteration — the one refinement that pays

STANAG's α_e is driven by dv/dt. The default implementation evaluates that
dv/dt from drag, gravity and Coriolis only, excluding the lift and Magnus
accelerations that α_e itself produces — which keeps the derivative a
closed-form function of state with no inner loop.

`MpmmModel(iterate_yaw=True)` performs **one fixed-point pass**: form the
yaw-dependent forces from the base acceleration, fold them into dv/dt, form
them again. Still a pure function of state, still no free parameter, still the
same formula and the same coefficients — only a more faithful evaluation of
what the standard writes.

| | plain | one pass |
|---|---|---|
| α_e at apogee | 0.698893° | 0.703758° (**+0.70 %**) |
| cost | 12.09 µs/deriv | 16.98 µs/deriv (**+40 %**) |
| **model error, range RMS** (15 engagements, from apogee) | 2.29 m | **0.65 m** |
| **model error, deflection RMS** | 0.47 m | **0.06 m** |
| difference from 6-DOF, range RMS (from launch) | 6.60 m | **1.58 m** |
| difference from 6-DOF, deflection RMS (from launch) | 1.18 m | **0.11 m** |

**A 40 % increase in derivative cost buys a 3.5× reduction in range model
error and an 8× reduction in deflection model error.** Nothing else in this
document comes close to that exchange rate — coarsening the step by 500× saves
compute but buys nothing, and dropping every yaw term saves 13 % at a cost of
a hundred metres.

**This is the recommended configuration for step 3.** See
[MODEL-ERROR.md](MODEL-ERROR.md) §"Two configurations" for the budget entries.

### Why this is a refinement and not a fit

It has to be said explicitly, because "a change that made the numbers better"
is exactly the shape of a fitting error.

1. **It has no free parameter.** All four STANAG fitting factors remain 1.0
   and the test still enforces it. There is nothing here to tune.
2. **It is independently justified.** The yaw of repose is the balance struck
   as the spin axis lags the turning velocity vector. The velocity vector's
   turn rate is set by the *total* transverse acceleration, which includes
   lift and Magnus. Excluding them is an approximation made for
   implementation convenience; including them is simply more correct.
3. **It makes agreement with the firing table slightly *worse*, not better.**
   The plain MPMM sits +13.57 % above the FT drift column; the iterated one
   sits at **+14.38 %**, moving *away* from the reference data and onto the
   6-DOF's +14.37 %. A fit would have done the opposite. See
   [DRIFT-RESOLUTION.md](DRIFT-RESOLUTION.md) §1.

---

## 4. Duty cycle, and sizing the embedded port

At 10 Hz the budget is 100 ms per cycle, and one full propagation to impact
must fit inside it with room for everything else the flight computer does.

| dt (s) | derivative calls | CPython, dev machine | % of a 100 ms budget |
|---|---|---|---|
| 0.01 | 10 512 | 126.7 ms | 127 % |
| 0.05 | 2 104 | 25.3 ms | 25 % |
| **0.1** | **1 052** | **12.7 ms** | **13 %** |
| 0.2 | 528 | 7.0 ms | 7 % |

**Even in CPython, at the recommended step, a worst-case propagation is 13 %
of the duty cycle.** With the iteration on it is roughly 18 %.

### What that implies for the Cortex-M7 port (an estimate, not a measurement)

The derivative is one barometric `pow`, one `sqrt`, a Mach table search with
eight linear interpolations, and on the order of 150 floating-point
operations. On a 300 MHz Cortex-M7 with a single-precision FPU, a plausible
cost is a few hundred cycles plus the `pow` — call it **1–3 µs per
derivative**, so **1–3 ms per propagation** at dt = 0.1 s, or 2–5 ms with the
iteration.

That is a sizing estimate from an operation count, with its assumptions
stated. It has not been measured and will not be until step 7. What can be
stated without qualification is the part that does not depend on the target:
**a full worst-case impact prediction is about 1000 derivative evaluations**,
and both the shared costs worth optimising (atmosphere 7 %, table lookup 9 %)
are known.

The margin is large enough that the interesting question for step 3 is not
whether the propagation fits, but whether to spend the headroom on the
yaw-of-repose iteration (recommended — see §3) or on more frequent
re-prediction.
