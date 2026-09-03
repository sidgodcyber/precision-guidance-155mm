# Trajectory Model Error (MPMM) — Measured

**155 mm M107 HE · step 2 · Task C**

The CEP budget carried **"trajectory model error (MPMM): 10 m, 1σ per axis"**
as an estimate. It was a placeholder, and it was the largest single
contributor to a predicted 18.9 m CEP. It has now been measured.

---

## The number, for direct insertion into the CEP budget

| Axis | Mean (bias) | 1σ | **RMS (bias ⊕ σ)** | Worst case |
|---|---|---|---|---|
| **Range** | −0.28 m | 0.60 m | **0.65 m** | 1.24 m |
| **Deflection** | −0.03 m | 0.06 m | **0.06 m** | 0.09 m |

**Recommended budget entries: 0.7 m 1σ in range, 0.1 m 1σ in deflection.**

Those are the **RMS** figures, which fold the residual bias in with the
scatter — the conservative choice, since nothing downstream removes the bias.

**The placeholder was pessimistic by a factor of about 15 in range and 150 in
deflection.** That is a genuine finding and it is worth stating plainly: the
reduced-order model is far better than the budget assumed. Trajectory model
error is no longer a meaningful contributor to CEP.

---

## Two configurations, and which one these numbers are for

The figures above are for `MpmmModel(iterate_yaw=True)` — **one fixed-point
pass on the yaw of repose**, which is the configuration recommended for step 3
and the reason the numbers are as small as they are. The default
(`iterate_yaw=False`, no inner pass) is several times worse:

| | default (`iterate_yaw=False`) | **recommended (`iterate_yaw=True`)** |
|---|---|---|
| Range, mean | −1.91 m | **−0.28 m** |
| Range, 1σ | 1.30 m | **0.60 m** |
| **Range, RMS** | **2.29 m** | **0.65 m** |
| Range, worst | 4.68 m | **1.24 m** |
| Deflection, mean | −0.31 m | **−0.03 m** |
| Deflection, 1σ | 0.37 m | **0.06 m** |
| **Deflection, RMS** | **0.47 m** | **0.06 m** |
| Deflection, worst | 1.41 m | **0.09 m** |
| Cost | 12.1 µs/derivative | 17.0 µs/derivative (**+40 %**) |

**Use the iterated form.** A 40 % increase in the cost of a derivative that
runs about a thousand times per prediction — a few milliseconds even in
CPython — buys a 3.5× reduction in range error and an 8× reduction in
deflection error. [MPMM-COMPUTE.md](MPMM-COMPUTE.md) §3 has the compute
argument in full, and §3 of that document also sets out why this refinement is
a more faithful evaluation of the STANAG formula rather than a fit: it has no
free parameter, all four fitting factors remain 1.0, and it moves the model
*away* from the firing table, not towards it.

**If step 3 uses the default instead, the budget entries are 2.3 m and
0.5 m.** Both configurations are reported throughout this document.

---

## What this changes in the CEP budget

This depends on how the 18.9 m figure was assembled, which is not recorded in
this repository. **Under the assumption** that it was a root-sum-square of
independent per-axis 1σ terms with equal axes and CEP = 1.1774 σ:

```
18.9 = 1.1774 * sqrt(10.0^2 + sigma_other^2)   ->   sigma_other = 12.56 m
```

Substituting the measured model term in the larger (range) axis:

```
iterated:  CEP = 1.1774 * sqrt(12.56^2 + 0.65^2) = 14.81 m
default:   CEP = 1.1774 * sqrt(12.56^2 + 2.29^2) = 15.03 m
```

**CEP would fall from 18.9 m to about 14.8 m**, and trajectory model error
would go from the largest single contributor to a negligible one — note that
the difference between the two configurations, 0.2 m of CEP, is itself
irrelevant. Once the term is this small, it stops mattering which version is
used; what matters is that the 10 m placeholder is retired.

That arithmetic is an inference about a budget this session has not seen, and
it is offered as such. The measured quantity stands on its own regardless of
how the rest of the budget is composed.

---

## Method

The onboard model predicts the impact point **from a mid-flight state**, not
from the muzzle. So the relevant error is not the difference between two
trajectories flown from launch; it is the difference in *predicted impact
point* when the reduced model is handed a true state at the moment guidance
would begin.

1. Fly a 6-DOF trajectory at dt = 2×10⁻⁴ s (the step-1 ground truth).
2. Take the true 6-DOF state at **apogee** — position, velocity, axial spin.
   The attitude quaternion and the two transverse body rates are **discarded**,
   because the reduced model has nowhere to put them. That discarded
   information is precisely what this measurement prices.
3. Initialise the MPMM from that state and propagate to ground impact.
4. Continue the 6-DOF to ground impact.
5. The difference between the two impact points is the model error.
6. Repeat across the firing-table envelope: 5 charges, 15 engagements,
   quadrant elevations 97–540 mils, ranges 2–16 km.

Apogee is used as the initialisation point because it is where step 3's
guidance would plausibly begin and because it is unambiguous to locate in both
models. Reproduce with `python -m analysis.mpmm_compare`.

**Both models are driven by identical coefficients, atmosphere, projectile and
environment, and no STANAG fitting factors are used** — all four are unity and
a test asserts it. A fitted MPMM would have made this number meaningless.

---

## Results per engagement

ΔRange and ΔDeflection are MPMM-from-apogee minus 6-DOF, at the impact point.

| Chg | QE (mils) | Apogee t (s) | Apogee alt (m) | Descent (s) | ΔRange default | **ΔRange iterated** | ΔDefl default | **ΔDefl iterated** |
|---|---|---|---|---|---|---|---|---|
| 4 | 97.2 | 3.20 | 50.0 | 3.18 | −0.53 | **−0.35** | −0.05 | **−0.04** |
| 4 | 152.0 | 4.90 | 119.3 | 4.94 | −0.59 | **−0.32** | −0.06 | **−0.05** |
| 4 | 211.6 | 6.70 | 225.2 | 6.82 | −0.83 | **−0.45** | −0.04 | **−0.02** |
| 5 | 118.1 | 4.30 | 96.3 | 4.50 | −0.25 | **+0.16** | −0.10 | **−0.09** |
| 5 | 280.4 | 9.50 | 479.5 | 10.08 | −0.91 | **−0.20** | −0.12 | **−0.06** |
| 5 | 420.6 | 13.70 | 994.4 | 14.52 | −1.96 | **−0.87** | −0.21 | **−0.06** |
| 6 | 258.4 | 9.90 | 548.9 | 10.87 | −1.45 | **−0.38** | −0.19 | **−0.08** |
| 6 | 378.6 | 13.80 | 1064.4 | 15.09 | −2.06 | **−0.67** | −0.16 | **+0.05** |
| 6 | 539.9 | 18.60 | 1936.3 | 20.47 | −3.31 | **−1.24** | −0.50 | **−0.05** |
| 7 | 177.6 | 8.30 | 382.3 | 9.19 | −1.46 | **+0.32** | −0.21 | **−0.08** |
| 7 | 319.8 | 13.40 | 1055.9 | 15.17 | −2.05 | **−0.06** | −0.24 | **+0.09** |
| 7 | 520.7 | 19.90 | 2345.9 | 22.66 | −3.97 | **−1.14** | −0.83 | **−0.09** |
| 8 | 141.6 | 7.90 | 349.0 | 8.91 | −1.78 | **+0.99** | −0.19 | **−0.04** |
| 8 | 248.4 | 12.50 | 922.0 | 14.37 | −2.79 | **+0.50** | −0.42 | **+0.05** |
| 8 | 525.3 | 22.20 | 3096.8 | 26.29 | −4.68 | **−0.53** | −1.41 | **−0.04** |

### The character of the error changes, not just its size

| correlation of \|error\| with descent time | default | iterated |
|---|---|---|
| Range | **r = 0.968** | r = 0.532 |
| Deflection | **r = 0.866** | **r = 0.114** |

In the default configuration the error is a **systematic accumulation over the
remaining flight** — it grows with descent time, one-signed, largest on the
longest shot (4.68 m with 26.3 s left) and smallest on the shortest (0.25 m
with 4.5 s left). It behaves like a small constant bias in the effective
forces, integrated twice.

With the iteration on, that structure is largely gone. The deflection error in
particular becomes unstructured scatter at the 6 cm level with essentially no
correlation to flight time (r = 0.11) and no consistent sign. There is no
longer an accumulating term to remove; what remains is the genuine residual of
discarding the attitude state.

**Consequences for step 3:**

1. **Later initialisation is cheaper**, in the default configuration, because
   the divergence accumulates roughly as the square of remaining time (see
   [MPMM-VALIDATION.md](MPMM-VALIDATION.md) §2). A guidance loop that
   re-predicts continuously will see this error shrink as it closes on the
   target. With the iteration on, the error is already small enough at apogee
   that this hardly matters.
2. **A single fixed 1σ across the envelope is now entirely reasonable.** With
   the iteration on, the worst engagement (1.24 m in range) and the best
   (0.06 m) both sit far below every other term in the budget.

---

## Assumptions and limitations

1. **The 6-DOF is treated as truth.** It is the best reference available and
   step 1 validated it to 0.48 % RMS in range against a firing table, but it
   is a model. This measures *model reduction* error, not error against a real
   projectile. Any error common to both models — a wrong coefficient, for
   instance — cancels here and is invisible.
2. **In particular, the 6-DOF's documented +14 % deflection bias against the
   firing-table drift column cancels.** The MPMM inherits it almost exactly —
   +14.38 % iterated, against the 6-DOF's +14.37 % (see
   [DRIFT-RESOLUTION.md](DRIFT-RESOLUTION.md)) — so the deflection figure here
   measures the reduction, not the absolute deflection accuracy. **A separate
   budget line is needed for absolute deflection accuracy; this number does
   not cover it.**
3. **No wind, no Coriolis, nominal atmosphere, one projectile lot.** All are
   common-mode between the two models and therefore invisible here. Real
   onboard prediction error will additionally include the state-estimation
   error from step 5 and the atmospheric-knowledge error, which are separate
   budget lines.
4. **Apogee-specific.** Initialising elsewhere gives a different number; the
   scaling with remaining flight time above is the guide.
5. **MPMM step size 0.01 s.** [MPMM-COMPUTE.md](MPMM-COMPUTE.md) §1 shows the
   step-size contribution at 0.01 s is 0.1 mm in range — four orders of
   magnitude below the figures reported here — so what this document reports
   is model-structure error and not integration error.
6. **The iterated figures are a single fixed-point pass, not convergence.** No
   second pass was tested. Since the first pass changes α_e by 0.7 % and a
   second would change it by a fraction of that, further passes are not
   expected to matter, but that expectation has not been measured.
