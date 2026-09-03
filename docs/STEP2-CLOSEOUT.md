# Step 2 Closeout — Reduced-Order Model (MPMM)

**155 mm M107 HE**

Step 2 built the model that runs on the flight computer: a modified point-mass
model in STANAG 4355 form, validated against the step-1 6-DOF, with the model
error priced for the CEP budget and the compute cost measured.

Everything here is reproducible from `python -m analysis.mpmm_compare` and
`python -m analysis.mpmm_compute`. 90 tests pass (66 from step 1, 24 new).

---

## 1. What was built

`models/mpmm.py` — a **7-state** model (position, velocity, axial spin) with
no attitude at all. STANAG's algebraic yaw of repose replaces the 6-DOF's
integrated rotational dynamics:

```
alpha_e = -( 8 Ix p (v x dv/dt) ) / ( pi rho d^3 C_Malpha |v|^4 )
```

`derivative()` is pure, returns plain floats, and does no I/O — the same
discipline as `sim/dynamics.py`, for the same reason: step 7 transcribes it
into C.

**It imports its aerodynamics rather than owning them.** Every coefficient
comes from `sim.aerodata`, the atmosphere from `sim.atmosphere`, the
projectile from `sim.projectile`.
`test_no_hardcoded_coefficients_in_the_module` parses the module and fails on
any high-precision float literal in executable code. Both models are therefore
driven by byte-identical aerodynamics, which is what makes every comparison
below mean something.

**No fitting factors.** The four STANAG ballistic fitting parameters — form
factor `i`, lift factor `fL`, Magnus factor `QM`, yaw-drag factor `QD` — are
present as named constants, all 1.0. `test_all_fitting_factors_are_unity`
fails if any default moves and also if the field set grows, so a fifth cannot
be added quietly. Every number in this document was produced with all four at
unity.

---

## 2. Results

### Task B — the reduction is faithful

15 engagements, 5 charges, QE 97–540 mils, 2–16 km, TOF 6.4–48.5 s.
Differences are MPMM minus 6-DOF, from identical muzzle conditions:

| Quantity | RMS, default | RMS, `iterate_yaw=True` |
|---|---|---|
| Range | 0.079 % (6.60 m) | **0.030 % (1.58 m)** |
| Deflection | 0.715 % (1.18 m) | **0.115 % (0.11 m)** |
| Time of flight | 0.028 s | — |
| Impact velocity | 0.058 m/s | — |
| Impact angle | 0.006° | — |

A seven-state point-mass model reproduces a full rigid-body integration to
**0.03 % in range and 0.12 % in deflection** across an 8:1 span of range, with
nothing fitted. Full per-engagement tables in
[MPMM-VALIDATION.md](MPMM-VALIDATION.md).

### Task C — the model-error term, measured

The CEP budget carried an **estimated 10 m 1σ per axis** as its largest single
contributor. Initialising the MPMM from a true 6-DOF state at apogee and
propagating to impact — which is how step 3 will actually use it:

| Axis | 1σ | **RMS** | Worst |
|---|---|---|---|
| Range | 0.60 m | **0.65 m** | 1.24 m |
| Deflection | 0.06 m | **0.06 m** | 0.09 m |

**The placeholder was pessimistic by a factor of roughly 15 in range and 150
in deflection.** Under a stated assumption about how the 18.9 m CEP was
composed, it would fall to about **14.8 m**, and trajectory model error would
stop being a meaningful contributor. Details, method and the default-config
figures (2.29 m / 0.47 m) in [MODEL-ERROR.md](MODEL-ERROR.md).

### Task D — step 1's last open question, answered

Step 1 closed with a +14.4 % lateral-deflection bias against the FT 155-AM-2
drift column and exactly one hypothesis left standing: that the FT column came
from a point-mass model of a different class, and the disagreement was
structural.

**Refuted.** The MPMM does not reproduce the FT column — it reproduces the
6-DOF:

| | vs the FT drift column |
|---|---|
| 6-DOF | +14.37 % |
| MPMM, default | +13.57 % |
| **MPMM, iterated** | **+14.38 %** |

And the more accurate the reduced model is made, the *closer* it gets to the
6-DOF (0.115 % RMS on deflection) and the *further* it moves from the firing
table. The proposed mechanism — undamped epicyclic yaw below Mach 0.7 — fails
on its own precondition: **no engagement in the envelope spends any time below
Mach 0.7**, the slowest point reached being Mach 0.794, and the epicyclic yaw
demonstrably damps rather than growing.

Two model classes, identical coefficients, nothing fitted, agreeing with each
other to 0.1 % and both sitting 14 % above the reference column. **The cause
lies in the inputs or in the reference data, not in the trajectory
integration.** Per the stopping rule, no further hypothesis was opened.
[DRIFT-RESOLUTION.md](DRIFT-RESOLUTION.md).

### Task E — compute cost

Worst-case engagement, apogee to impact, 26.3 s of flight:

| | |
|---|---|
| **Derivative evaluations at the recommended dt = 0.1 s** | **1052** |
| Cost in CPython on the dev machine | 12.7 ms — 13 % of a 100 ms duty cycle |
| Accuracy lost by coarsening 0.001 s → 0.5 s | 0.32 m in range (20 ppm) |
| Largest term that could be dropped | none is worth dropping — see below |
| Sizing estimate for a 300 MHz Cortex-M7 | 1–3 ms per propagation *(estimate, unmeasured)* |

The step-size error is **not** RK4 truncation — the measured convergence order
is ≈ 2, and it is the linear interpolation across the final step to z = 0,
confirmed quantitatively against a·dt²/8. If sub-centimetre impact prediction
is ever wanted, solve the last step properly rather than shrinking dt.

On droppable terms: dropping **all three** yaw-dependent forces saves only
13 % of the derivative, because the shared cost is forming α_e, not the forces
that use it. Individually they are 5–9 % each, against accuracy contributions
of 20.5 m (Magnus), 100.8 m (lift) and 2.3 m (yaw drag). The real shared costs
worth attacking in the C port are the ISA atmosphere (7 %) and the aero table
lookup (9 %). [MPMM-COMPUTE.md](MPMM-COMPUTE.md).

---

## 3. What this pass changed, and one bug it found

### The yaw-of-repose iteration

The single most consequential finding of step 2 is not in the task list. The
default MPMM forms α_e from a dv/dt containing drag, gravity and Coriolis but
**not** the lift and Magnus accelerations that α_e itself produces. One
fixed-point pass to include them costs 40 % more per derivative and buys a
3.5× reduction in range model error and an 8× reduction in deflection model
error, and it removes both structural biases identified in Task B.

**It is a refinement, not a fit**, and the distinction was checked three ways:
it has no free parameter (all four fitting factors remain 1.0); it is
independently justified, since the velocity vector's turn rate is set by the
total transverse acceleration; and **it makes agreement with the firing table
slightly worse rather than better**, moving the MPMM from +13.57 % to
+14.38 %. A fit would have done the opposite.

`iterate_yaw` defaults to `False`, so the simplest closed-form derivative
remains the documented baseline for the C port and every §1–§2 figure in
MPMM-VALIDATION.md describes what the constructor does by default. **Step 3
should set it to `True`.**

### A latent Coriolis bug, found by trying to use the code

The Coriolis branch of `_base_acceleration` referred to an undefined name and
had **never executed**, because every comparison run in `docs/` uses Coriolis
off to match step 1's firing-table convention. It raised `NameError` the first
time it was switched on, during the Task E ablation study.

Two tests now cover it: one asserting that enabling Coriolis changes the MPMM
acceleration by exactly the vector it changes the 6-DOF's by, and one with a
40 m/s wind asserting that the term uses the **ground** velocity and not the
air-relative velocity — a distinction that is invisible in still air, which is
why it needed a test of its own. A third pins a genuine structural property
that the 6-DOF does not share: because α_e is driven by dv/dt, enabling
Coriolis in the MPMM perturbs the yaw of repose slightly.

The general lesson is recorded because it will recur: **a code path that no
validation case exercises is not validated, however much validation there is
around it.**

---

## 4. Remaining known limitations

1. **Linear overturning moment only.** STANAG writes the yaw-of-repose
   denominator as (C_Mα + C_Mα3·α_e²). C_Mα3 is not available for the M107 in
   either source this project uses. At the 0.2–0.7° yaw of repose seen here
   the cubic term would be a fraction of a percent, but it is an omission.
2. **One fixed-point pass, not convergence.** A second pass was not tested.
   The first changes α_e by 0.7–1.0 %, so a second should be negligible —
   expected, not measured.
3. **Everything in step 1's limitation list still applies**, because the MPMM
   consumes the same coefficients: four of eight coefficients rest on a single
   source, the reduced-rate convention pd/(2V) vs pd/V remains an unresolved
   assumption worth 7–11 % in drift, and the table stops at Mach 2.00.
4. **The +14 % deflection bias against the firing table is inherited, not
   fixed.** Step 2 characterised it and eliminated the last hypothesis; it did
   not explain it.
5. **No wind or Coriolis in the reported comparisons.** Both are implemented
   and shared with the 6-DOF, and Coriolis is now tested, but the validation
   tables have them off to match step 1's convention.
6. **The embedded cost is an estimate from an operation count**, not a
   measurement. It becomes a measurement in step 7.

---

## 5. Fitness for use by step 3

Step 3 is the impact-point-prediction guidance law. It needs a model it can
call in a loop, and it needs to know what that model's predictions are worth.

**Fit for purpose, with these specifics:**

- **Use `MpmmModel(iterate_yaw=True)` and `dt = 0.1 s`.** Budget about 1050
  derivative evaluations per full prediction from apogee.
- **Model error for the CEP budget: 0.7 m 1σ in range, 0.1 m 1σ in
  deflection.** This is now a small term. If the default configuration is used
  instead, 2.3 m and 0.5 m.
- **Relative and differential predictions are trustworthy across the whole
  firing-table envelope.** Range, TOF, impact velocity and impact angle all
  agree with the 6-DOF far more closely than any other error source in the
  system.
- **Sensitivities and gradients are trustworthy.** The MPMM tracks the 6-DOF's
  response to changes in state, not merely its absolute answer — that is what
  the 0.03 % range and 0.115 % deflection agreement over an 8:1 range span
  actually demonstrates, and it is the property a guidance law depends on.

**The one exception, carried forward unchanged from step 1:**

> **Absolute lateral deflection carries a known ≈ +14 % bias against the
> FT 155-AM-2 drift column and must not be treated as truth in an absolute
> sense.** It remains valid for sign, trends and sensitivities.

Step 2 has now established that this bias is **common-mode across model
classes** — it is not caused by the reduction, and it cancels in every
model-to-model comparison in this repository. Two consequences for step 3:

1. A guidance law that **nulls a predicted miss distance** is unaffected,
   because the same bias appears in the prediction and in the truth model it
   is validated against.
2. The CEP budget still needs a **separate line for absolute deflection
   accuracy**, which step 2 could not measure and which the Task C number does
   not cover.

**Out of scope for step 2 and not started**, per the scope fence: guidance
laws, control loops, the navigation filter, Monte Carlo, canard forces.

---

## 6. Document map

| Document | What it holds |
|---|---|
| [MPMM-VALIDATION.md](MPMM-VALIDATION.md) | Task B — per-engagement MPMM vs 6-DOF, and what the differences depend on |
| [MODEL-ERROR.md](MODEL-ERROR.md) | Task C — the measured model-error term for the CEP budget |
| [DRIFT-RESOLUTION.md](DRIFT-RESOLUTION.md) | Task D — the step-1 drift hypothesis, tested and refuted |
| [MPMM-COMPUTE.md](MPMM-COMPUTE.md) | Task E — step size, term ablation, duty cycle, embedded sizing |
| [STEP1-CLOSEOUT.md](STEP1-CLOSEOUT.md) | The 6-DOF this is all measured against |
| `models/README.md` | The module, its invariants and the tests that hold them |
