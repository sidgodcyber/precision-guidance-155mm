# The Step-1 Drift Residual: Hypothesis Tested

**155 mm M107 HE · step 2 · Task D**

Step 1 closed with a +14.4 % lateral-deflection bias against the FT 155-AM-2
drift column, every coefficient candidate eliminated, and one structural
hypothesis left open:

> This model forms drift by integrating the full 6-DOF epicyclic motion,
> whereas the FT drift column was produced by a modified point-mass model of
> the STANAG 4355 family. Two models can agree on every coefficient and still
> differ on drift if they differ in how the yaw of repose is formed.
> — STEP1-CLOSEOUT.md §3

Step 2 built exactly such a model. The hypothesis is now testable, and it has
been tested.

## Result: **the hypothesis is refuted.** Both mechanisms proposed for it fail.

---

## 1. The MPMM does not reproduce the FT drift column — it reproduces the 6-DOF

A STANAG 4355 modified point-mass model, driven by the same coefficients, with
all fitting factors at unity:

| Chg | QE (mils) | FT drift (m) | 6-DOF (m) | vs FT | **MPMM (m)** | **vs FT** | MPMM vs 6-DOF |
|---|---|---|---|---|---|---|---|
| 4 | 97.2 | 3.2 | 3.99 | +24.84 % | 3.97 | **+23.95 %** | −0.71 % |
| 4 | 152.0 | 7.8 | 9.18 | +17.74 % | 9.13 | **+17.03 %** | −0.60 % |
| 4 | 211.6 | 15.2 | 17.02 | +11.96 % | 16.92 | **+11.32 %** | −0.57 % |
| 5 | 118.1 | 7.5 | 9.27 | +23.63 % | 9.22 | **+22.89 %** | −0.60 % |
| 5 | 280.4 | 36.6 | 41.19 | +12.53 % | 40.95 | **+11.87 %** | −0.58 % |
| 5 | 420.6 | 79.2 | 83.38 | +5.28 % | 82.84 | **+4.60 %** | −0.64 % |
| 6 | 258.4 | 45.5 | 53.64 | +17.88 % | 53.31 | **+17.16 %** | −0.61 % |
| 6 | 378.6 | 88.2 | 99.26 | +12.54 % | 98.58 | **+11.77 %** | −0.68 % |
| 6 | 539.9 | 169.4 | 177.88 | +5.01 % | 176.44 | **+4.16 %** | −0.81 % |
| 7 | 177.6 | 37.1 | 43.24 | +16.54 % | 42.97 | **+15.82 %** | −0.62 % |
| 7 | 319.8 | 94.0 | 108.91 | +15.86 % | 108.11 | **+15.01 %** | −0.73 % |
| 7 | 520.7 | 211.9 | 231.88 | +9.43 % | 229.79 | **+8.44 %** | −0.90 % |
| 8 | 141.6 | 37.6 | 43.27 | +15.07 % | 42.98 | **+14.32 %** | −0.65 % |
| 8 | 248.4 | 92.4 | 106.36 | +15.11 % | 105.53 | **+14.21 %** | −0.78 % |
| 8 | 525.3 | 292.8 | 328.58 | +12.22 % | 325.13 | **+11.04 %** | −1.05 % |

**Mean: 6-DOF +14.37 % vs FT; MPMM +13.57 % vs FT; MPMM within 0.70 % of the
6-DOF.**

The two model classes agree with each other to under one percent and both sit
about fourteen percent above the firing table. Swapping integrated attitude
dynamics for STANAG's algebraic yaw of repose moves the deflection by less
than a percent — nowhere near the fourteen percent that would be needed.

**The disagreement with the FT column is not a model-class difference.** If it
were, the point-mass model would have landed on the FT column and the 6-DOF
would have been the outlier. Instead they are on the same side, close
together, and the firing table is elsewhere.

### And the residual 0.7 % was the MPMM's own error, not a hint

The remaining gap between the two models is not evidence of anything about the
firing table either. It is an artefact of how the default MPMM evaluates its
own formula: the acceleration driving alpha_e excludes the lift and Magnus
terms that alpha_e itself produces. Turning on a single fixed-point pass to
include them — no new parameter, all four fitting factors still 1.0, see
[MPMM-COMPUTE.md](MPMM-COMPUTE.md) §3 — removes it:

| | vs the 6-DOF drift | vs the FT drift column |
|---|---|---|
| MPMM, default | −0.70 % mean, 0.72 % RMS | +13.57 % |
| **MPMM, iterated** | **+0.004 % mean, 0.115 % RMS** | **+14.38 %** |
| 6-DOF | — | +14.37 % |

**The better the reduced model gets, the more exactly it reproduces the 6-DOF
and the further it moves from the firing table.** A seven-state point-mass
model carrying no attitude at all now agrees with a full rigid-body
integration on lateral deflection to **0.115 % RMS across an 8:1 span of
range**, and the two of them agree with each other on the FT disagreement to
0.01 percentage points.

This closes off the last way the hypothesis could have survived. One could
have argued that the default MPMM's small tilt towards the FT column (+13.57 %
against +14.37 %) was a trace of the model-class effect, too weak to explain
14 % but pointing the right way. It was not. It was the MPMM's own
approximation error, and correcting it moves the number the other way.

---

## 2. The proposed mechanism also fails, on its own precondition

The specific mechanism offered for testing was:

> The MPMM's algebraic α_e assumes a steady-state yaw of repose; the 6-DOF
> superimposes epicyclic motion. Those agree only if the epicyclic motion
> damps. Step 1 found this shell is dynamically unstable below about Mach 0.7.
> Undamped epicyclic yaw would give the 6-DOF a larger effective angle of
> attack than the steady-state assumption predicts.

**The precondition is never met. None of these trajectories reaches Mach 0.7.**

| Chg | QE (mils) | minimum Mach reached | fraction of flight below Mach 0.7 |
|---|---|---|---|
| 4 | 97.2 | 0.888 | **0.000** |
| 4 | 152.0 | 0.854 | **0.000** |
| 4 | 211.6 | 0.824 | **0.000** |
| 5 | 118.1 | 0.918 | **0.000** |
| 5 | 280.4 | 0.834 | **0.000** |
| 5 | 420.6 | 0.795 | **0.000** |
| 6 | 258.4 | 0.870 | **0.000** |
| 6 | 378.6 | 0.840 | **0.000** |
| 6 | 539.9 | 0.794 | **0.000** |
| 7 | 177.6 | 0.928 | **0.000** |
| 7 | 319.8 | 0.878 | **0.000** |
| 7 | 520.7 | 0.837 | **0.000** |
| 8 | 141.6 | 1.004 | **0.000** |
| 8 | 248.4 | 0.907 | **0.000** |
| 8 | 525.3 | 0.874 | **0.000** |

The slowest point of the slowest trajectory in the envelope is **Mach 0.794**.
A shell that never goes below Mach 0.79 cannot be affected by a dynamic
instability that sets in below Mach 0.7. The step-1 observation about subsonic
dynamic instability is correct as a property of the shell, but it is not
reachable in this engagement envelope.

### The epicyclic motion damps, and damps early

Comparing the 6-DOF's actual total yaw against the MPMM's algebraic α_e
evaluated **at the same states** (same position, velocity and spin):

| Chg | QE | mean α_e alg (°) | mean 6-DOF yaw (°) | ratio, whole flight | ratio, first 25 % | ratio, after 50 % |
|---|---|---|---|---|---|---|
| 4 | 97.2 | 0.2526 | 0.2802 | 1.109 | **1.169** | 1.072 |
| 4 | 211.6 | 0.3022 | 0.3164 | 1.047 | **1.131** | 1.012 |
| 5 | 420.6 | 0.3997 | 0.4061 | 1.016 | **1.091** | 1.003 |
| 6 | 539.9 | 0.4719 | 0.4758 | 1.008 | **1.040** | 1.003 |
| 7 | 520.7 | 0.4572 | 0.4605 | 1.007 | **1.035** | 1.004 |
| 8 | 141.6 | 0.2051 | 0.2091 | 1.020 | **1.113** | 1.005 |
| 8 | 525.3 | 0.4716 | 0.4749 | 1.007 | **1.018** | 1.006 |

(Full table in `docs/mpmm_compare.log`.)

The 6-DOF's excess yaw over the steady-state value is **1.8 % to 19.7 % in the
first quarter of flight and 0.2 % to 7.2 % after the halfway point**. It is a
launch transient that decays — the opposite of a growing instability. And it is
largest on the *shortest* flights, where the transient occupies the biggest
fraction of the flight, not on the longest ones where the most time is spent
slow.

### Why even that excess yaw does not produce proportional drift

Note the anti-correlation: the engagement with the largest mean-yaw excess
(charge 4, QE 97.2: +10.9 %) has the *smallest* deflection difference
(−0.71 %), while the smallest yaw excess (charge 8, QE 525.3: +0.7 %) has the
*largest* deflection difference (−1.05 %).

That is expected once the vector nature of the problem is taken seriously.
With a linear normal-force coefficient, the lift is proportional to the
angle-of-attack **vector**, so the time-integrated side force depends on the
**vector mean** of the yaw. Epicyclic motion is a roughly circular excursion
about the repose point: it raises the mean *magnitude* of the yaw while
contributing close to nothing to the vector mean. So a 10 % excess in mean
|yaw| does not buy 10 % more drift.

This is why the mechanism could not have delivered a 14 % deflection
difference even if the epicyclic motion had failed to damp.

---

## 3. What is now established, and what is not

**Established:**

- The FT-column disagreement is **not** a 6-DOF-versus-point-mass artefact.
  Two model classes, same coefficients, no fitting: they agree with each other
  to 0.7 % and both sit ~14 % above the firing table.
- The subsonic-epicyclic mechanism is **not** operating: zero time below
  Mach 0.7 in every engagement, and the epicyclic yaw demonstrably damps.
- The small residual difference between the two models *is* explained, and it
  is not about the epicyclic transient at all: it is the acceleration used to
  form alpha_e. Including the lift and Magnus contributions to dv/dt takes the
  model-to-model deflection difference from 0.72 % RMS to **0.115 % RMS**.

**Not established, and not pursued further** (step 1's stopping rule stands,
and this was characterisation rather than another correction pass):

- Why the firing table's drift column sits ~14 % below both models. One
  observation is recorded without a hypothesis attached: within each charge,
  the disagreement *shrinks* as elevation rises — charge 6 runs +17.9 %,
  +12.5 %, +5.0 % as QE goes 258 → 379 → 540 mils, and charge 5 runs +23.6 %,
  +12.5 %, +5.3 %. The FT drift grows with elevation faster than either model
  does. Whatever the cause, it is common to both model classes and therefore
  lies in the inputs or in the reference data, not in the trajectory
  integration.

**What would settle it** is a source that either states how the FT 155-AM-2
drift column was generated, or provides measured deflection from instrumented
firings of this projectile. Neither is in hand, and neither is reachable from
the documents this project has been able to obtain.

---

## 4. Consequence for the step-1 fitness statement

Step 1's caveat — *"absolute lateral deflection carries a known +14 % bias and
must not be treated as truth in an absolute sense"* — **stands unchanged, and
now applies equally to the MPMM**, which inherits the bias almost exactly
(+14.38 % iterated, +13.57 % default, against the 6-DOF's +14.37 %).

What has changed is that the bias is now *characterised* rather than merely
bounded: it is common-mode across model classes, it is not caused by the model
reduction, and it therefore **cancels** in the model-error measurement of
[MODEL-ERROR.md](MODEL-ERROR.md). That measurement is a comparison of two
models that share the bias, so the bias does not contaminate it.

It does mean the CEP budget needs a **separate line for absolute deflection
accuracy** which this session has not measured and cannot measure from the
data available. That is flagged in MODEL-ERROR.md §"Assumptions and
limitations" item 2.
