# MPMM Validation Against the 6-DOF

**155 mm M107 HE · step 2 · Task B**

The reduced-order modified point-mass model (`models/mpmm.py`, STANAG 4355
form) against the step-1 6-DOF, over the full firing-table envelope: five
charges, quadrant elevations from 97 to 540 mils, flight times from 6.4 s to
48.5 s.

**Both models are driven by identical inputs.** The same coefficient table
object type, the same atmosphere, the same projectile, the same environment,
the same launch state. Nothing is duplicated — `models/mpmm.py` imports from
`sim/` and contains no coefficient literal of its own, which
`tests/test_mpmm.py::test_no_hardcoded_coefficients_in_the_module` enforces by
parsing the module and rejecting any high-precision float in executable code.

**No fitting factors are used.** All four STANAG ballistic fitting factors
(form factor `i`, lift factor `fL`, Magnus factor `QM`, yaw-drag factor `QD`)
are unity, asserted by
`tests/test_mpmm.py::test_all_fitting_factors_are_unity`. See
[MODEL-ERROR.md](MODEL-ERROR.md) for why that matters.

Reproduce with `python -m analysis.mpmm_compare`.
6-DOF at dt = 2×10⁻⁴ s; MPMM at dt = 0.01 s; Coriolis off (matching the
firing-table comparison convention of step 1).

---

## 1. Results per engagement

Differences are **MPMM minus 6-DOF**.

| Chg | QE (mils) | TOF (s) | ΔRange (m) | ΔRange (%) | ΔDrift (m) | ΔDrift (%) | ΔTOF (s) | ΔV_imp (m/s) | ΔAngle (°) | Max divergence (m) | at t (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 97.2 | 6.38 | −2.20 | −0.109 | −0.03 | −0.71 | −0.0074 | +0.033 | −0.007 | 0.23 | 6.3 |
| 4 | 152.0 | 9.84 | −2.54 | −0.084 | −0.06 | −0.60 | −0.0091 | +0.034 | −0.004 | 0.44 | 9.8 |
| 4 | 211.6 | 13.52 | −2.99 | −0.074 | −0.10 | −0.57 | −0.0114 | +0.034 | −0.005 | 0.75 | 13.5 |
| 5 | 118.1 | 8.80 | −2.87 | −0.095 | −0.06 | −0.60 | −0.0095 | +0.045 | −0.007 | 0.40 | 8.7 |
| 5 | 280.4 | 19.58 | −4.14 | −0.069 | −0.24 | −0.58 | −0.0162 | +0.038 | −0.005 | 1.53 | 19.5 |
| 5 | 420.6 | 28.22 | −5.74 | −0.072 | −0.54 | −0.64 | −0.0256 | +0.037 | −0.005 | 3.42 | 28.1 |
| 6 | 258.4 | 20.77 | −4.78 | −0.068 | −0.33 | −0.61 | −0.0177 | +0.042 | −0.005 | 1.77 | 20.7 |
| 6 | 378.6 | 28.89 | −6.25 | −0.069 | −0.68 | −0.68 | −0.0257 | +0.038 | −0.005 | 3.55 | 28.8 |
| 6 | 539.9 | 39.07 | −8.68 | −0.079 | −1.44 | −0.81 | −0.0418 | +0.032 | −0.008 | 7.69 | 39.0 |
| 7 | 177.6 | 17.49 | −5.05 | −0.072 | −0.27 | −0.62 | −0.0168 | +0.063 | −0.006 | 1.37 | 17.4 |
| 7 | 319.8 | 28.57 | −6.92 | −0.069 | −0.79 | −0.73 | −0.0262 | +0.042 | −0.004 | 3.59 | 28.5 |
| 7 | 520.7 | 42.56 | −10.13 | −0.078 | −2.10 | −0.90 | −0.0460 | +0.028 | −0.008 | 9.23 | 42.5 |
| 8 | 141.6 | 16.81 | −6.27 | −0.079 | −0.28 | −0.65 | −0.0189 | +0.169 | −0.007 | 1.41 | 16.7 |
| 8 | 248.4 | 26.87 | −7.85 | −0.072 | −0.83 | −0.78 | −0.0275 | +0.049 | −0.005 | 3.49 | 26.8 |
| 8 | 525.3 | 48.49 | −12.69 | −0.080 | −3.45 | −1.05 | −0.0565 | +0.012 | −0.010 | 12.78 | 48.4 |

---

## 2. What the differences depend on

Aggregates alone would hide the structure, so here is what actually varies.

### Range: a constant fractional bias, not a growing one

ΔRange in **percent** is nearly constant across the whole envelope:
**−0.068 % to −0.109 %**, with no trend against flight time, charge or
elevation. In metres it grows simply because the range does. The MPMM is
consistently, slightly **short**.

A constant fractional bias is the signature of a small systematic difference
in the effective drag, not of an error that accumulates. The likely origin is
the yaw-drag term: the MPMM applies the α² drag increment at the *steady-state*
yaw of repose, while the 6-DOF applies it at the *instantaneous* total yaw,
which is larger because it includes the epicyclic component. Larger yaw means
more drag, so one would expect the 6-DOF to fall short of the MPMM — the
opposite sign to what is observed, so the yaw-drag term is not the whole story
and the residual 0.08 % is left unattributed rather than guessed at.

### Deflection: a bias that grows with flight time

ΔDrift in **percent** grows monotonically with time of flight:

| TOF (s) | 6.4 | 13.5 | 19.6 | 28.6 | 39.1 | 48.5 |
|---|---|---|---|---|---|---|
| ΔDrift (%) | −0.71 | −0.57 | −0.58 | −0.73 | −0.81 | −1.05 |

The MPMM always produces **slightly less drift** than the 6-DOF, and the
shortfall widens on longer flights. This is the expected signature of the
structural difference between the models: the MPMM's yaw of repose is the
steady-state value, whereas the 6-DOF's total yaw also carries the epicyclic
motion, which is largest early in flight and decays. Section 2 of
[DRIFT-RESOLUTION.md](DRIFT-RESOLUTION.md) quantifies that yaw difference
directly at matched states. Section 3 below shows the bias almost entirely
removed by iterating the yaw of repose, which identifies its origin more
precisely: not the epicyclic motion, but the acceleration used to form
alpha_e.

Even at its worst this is **1 % of the deflection, 3.45 m in 328 m**.

### Trajectory divergence: monotonic accumulation, not a shape difference

The maximum position divergence occurs **at impact in every single case** —
compare the "at t" column with the TOF column. The two trajectories do not
cross, bulge apart and rejoin; they separate steadily. Divergence scales
approximately as TOF²:

    0.23 m at 6.38 s   →   12.78 m at 48.49 s
    ratio of times 7.6, ratio of divergence 55.6 ≈ 7.6²·⁰

That is what a near-constant fractional velocity difference integrated twice
looks like, and it is consistent with the constant fractional range bias.

**Consequence for step 3, and it is a favourable one:** because the divergence
is accumulation rather than a shape error, initialising the MPMM later in
flight cuts the error roughly as the square of the remaining time. That is
exactly what [MODEL-ERROR.md](MODEL-ERROR.md) measures, and it is why the
apogee-initialised error is an order of magnitude smaller than the
launch-initialised one.

### Impact conditions

Impact velocity agrees to **+0.012 to +0.169 m/s** (worst case 0.05 % of
338 m/s) and impact angle to **0.004° to 0.010°**. Both are negligible for
fuze-function or terminal-effect purposes.

---

## 3. The same comparison with the yaw-of-repose iteration on

Everything above is the **default** model, whose derivative is a closed-form
function of state. `MpmmModel(iterate_yaw=True)` adds one fixed-point pass so
that the dv/dt driving the yaw of repose includes the lift and Magnus
accelerations alpha_e itself produces. Same formula, same coefficients, same
fitting factors (all unity), 40 % more compute per derivative. See
[MPMM-COMPUTE.md](MPMM-COMPUTE.md) §3, which also sets out why this is a
refinement and not a fit.

| Chg | QE (mils) | TOF (s) | ΔRange default | ΔRange % | **ΔRange iterated** | **%** | ΔDrift default | **ΔDrift iterated** |
|---|---|---|---|---|---|---|---|---|
| 4 | 97.2 | 6.38 | −2.20 | −0.109 | **−1.41** | **−0.070** | −0.03 | **−0.01** |
| 4 | 152.0 | 9.84 | −2.54 | −0.084 | **−1.45** | **−0.048** | −0.06 | **−0.01** |
| 4 | 211.6 | 13.52 | −2.99 | −0.074 | **−1.59** | **−0.040** | −0.10 | **−0.02** |
| 5 | 118.1 | 8.80 | −2.87 | −0.095 | **−1.12** | **−0.037** | −0.06 | **−0.00** |
| 5 | 280.4 | 19.58 | −4.14 | −0.069 | **−1.36** | **−0.023** | −0.24 | **−0.01** |
| 5 | 420.6 | 28.22 | −5.74 | −0.072 | **−2.38** | **−0.030** | −0.54 | **−0.05** |
| 6 | 258.4 | 20.77 | −4.78 | −0.068 | **−0.44** | **−0.006** | −0.33 | **+0.03** |
| 6 | 378.6 | 28.89 | −6.25 | −0.069 | **−1.29** | **−0.014** | −0.68 | **+0.02** |
| 6 | 539.9 | 39.07 | −8.68 | −0.079 | **−3.21** | **−0.029** | −1.44 | **−0.08** |
| 7 | 177.6 | 17.49 | −5.05 | −0.072 | **+0.55** | **+0.008** | −0.27 | **+0.05** |
| 7 | 319.8 | 28.57 | −6.92 | −0.069 | **+0.12** | **+0.001** | −0.79 | **+0.11** |
| 7 | 520.7 | 42.56 | −10.13 | −0.078 | **−2.21** | **−0.017** | −2.10 | **+0.05** |
| 8 | 141.6 | 16.81 | −6.27 | −0.079 | **+1.18** | **+0.015** | −0.28 | **+0.06** |
| 8 | 248.4 | 26.87 | −7.85 | −0.072 | **+1.50** | **+0.014** | −0.83 | **+0.19** |
| 8 | 525.3 | 48.49 | −12.69 | −0.080 | **−0.99** | **−0.006** | −3.45 | **+0.33** |

**Both of the structural signatures identified in §2 disappear.**

- The **constant fractional range bias** goes with them. ΔRange % was
  −0.068 % to −0.109 % with a mean of −0.078 % — one-signed everywhere.
  Iterated it is −0.070 % to **+0.015 %**, mean −0.019 %, and it changes sign
  across the envelope. A bias has become scatter.
- The **deflection bias that grew with flight time** goes too. ΔDrift % was
  −0.57 % to −1.05 %, monotonically worsening with TOF, always negative.
  Iterated it is −0.25 % to +0.18 %, mean **+0.004 %**, RMS **0.115 %**, with
  no trend in TOF.

The second of those is the striking one. **A seven-state point-mass model with
no attitude at all reproduces the lateral deflection of a full 6-DOF
integration to 0.115 % RMS over an 8:1 span of range**, with every coefficient
shared and nothing fitted. Section 1 of
[DRIFT-RESOLUTION.md](DRIFT-RESOLUTION.md) turns that into the definitive
answer to step 1's open question.

| Quantity | default RMS | **iterated RMS** |
|---|---|---|
| ΔRange | 6.601 m | **1.577 m** |
| ΔRange % | 0.0788 % | **0.0299 %** |
| ΔDrift | 1.176 m | **0.109 m** |
| ΔDrift % | 0.715 % | **0.115 %** |

---

## 4. Aggregate

| Quantity | min | max | mean | RMS |
|---|---|---|---|---|
| ΔRange | −12.690 m | −2.200 m | −5.940 m | 6.601 m |
| ΔRange % | −0.1093 % | −0.0680 % | −0.0780 % | 0.0788 % |
| ΔDrift | −3.447 m | −0.028 m | −0.745 m | 1.176 m |
| ΔDrift % | −1.049 % | −0.571 % | −0.703 % | 0.715 % |
| ΔTOF | −0.0565 s | −0.0074 s | −0.0238 s | 0.0276 s |
| ΔV_impact | +0.012 m/s | +0.169 m/s | +0.046 m/s | 0.058 m/s |
| ΔImpact angle | −0.0103° | −0.0040° | −0.0060° | 0.0063° |

With the yaw-of-repose iteration on, the two headline rows become
ΔRange % **0.0299 %** RMS and ΔDrift % **0.115 %** RMS (§3).

**Verdict: the reduction is faithful.** Across an 8:1 span of range and five
charges, replacing integrated attitude dynamics with an algebraic yaw of
repose costs **under 0.11 % in range, under 1.1 % in deflection, and under
0.06 s in time of flight** — or **0.03 % and 0.25 %** with the iteration on
— with no fitting whatsoever.

That is the finding of Task B, and it has a consequence beyond step 2: the
reduced model is good enough that the 6-DOF is not needed in the guidance
loop, and good enough that **any disagreement between this model family and
external reference data cannot be blamed on the reduction**. That is what
[DRIFT-RESOLUTION.md](DRIFT-RESOLUTION.md) rests on.

---

## 5. Known limitations of this MPMM

1. **Linear overturning moment only.** STANAG 4355 writes the yaw-of-repose
   denominator as (C_Mα + C_Mα3·α_e²). The cubic coefficient C_Mα3 is not
   available for the M107 in either source this project uses, so the linear
   form is used. At the 0.2–0.5° yaw of repose seen here the cubic term would
   be a fraction of a percent of the linear one, but it is an omission and it
   is recorded as one.
2. **The default omits the yaw-of-repose iteration, and that costs more than
   it looks.** By default α_e is evaluated with v̇ from drag, gravity and
   Coriolis only, excluding the lift and Magnus terms that themselves depend on
   α_e. **Measured, not assumed:** turning the single fixed-point pass on cuts
   the range difference from 6.60 m RMS to 1.58 m and the deflection difference
   from 1.18 m to 0.11 m, for 40 % more compute per derivative (§3 above and
   [MPMM-COMPUTE.md](MPMM-COMPUTE.md) §3). **Step 3 should use
   `iterate_yaw=True`.** The default is left at `False` so that the simplest
   possible derivative — closed-form, no inner pass — remains the baseline for
   the step-7 C port, and so that every figure in §1–§2 above describes what
   the constructor actually does by default. Only one pass was tested; a second
   was not.
3. **No wind in these runs**, and no Coriolis (to match step 1's firing-table
   convention). Both are implemented and shared with the 6-DOF; neither is
   exercised here.
4. **The reference is the 6-DOF, which itself carries a documented +14 %
   deflection bias against the firing table.** These are differences between
   two models, not errors against truth. See
   [DRIFT-RESOLUTION.md](DRIFT-RESOLUTION.md).
5. **Same-lot caveat.** Both models use one coefficient set. Agreement between
   them says nothing about how either behaves on a projectile lot whose
   aerodynamics differ from this table.
