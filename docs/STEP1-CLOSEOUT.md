# Step 1 Closeout — 6-DOF Ballistic Simulator

**155 mm M107 HE · one build and two correction passes**

Step 1 is closed. This document states what the model is, what it was measured
against, what remains unexplained, and whether it is fit to serve as ground
truth for step 2.

---

## 1. Final ladder results

`dt = 2×10⁻⁴ s`, latitude 45° N, ISA standard atmosphere, no wind.
Reproduce with `python run_validation.py`.

| Rung | Result |
|---|---|
| **1 · Vacuum vs analytic parabola** | error < 10⁻⁶ % at five elevations; drift identically 0 |
| **2 · Drag-only vs independent 3-DOF** | 0.000 m difference at four elevations — machine precision |
| **3 · Gyroscopic stability** | Sg = 2.584 → 13.086, `Sg > 1` and `Sg > 1/(Sd(2−Sd))` at every logged sample of all 15 runs |
| **4 · Yaw of repose and drift** | drift RIGHT in 15/15; reverses with the rifling; magnitude **+14.4 % mean** (see §3) |
| **5 · Firing table**, 15 points, 5 charges, 2–16 km | **range RMS 0.48 %, mean +0.00 %**; TOF RMS 0.52 %; max ordinate RMS 0.51 %; impact velocity RMS 1.13 % |
| **5b · ASAT-13 fully specified case** | axial deceleration **+0.40 %**, flight time **−0.71 %**, summit time −2.06 %, peak AoA −0.19 % |
| **6 · Numerical health** | \|q\|−1 = 2.22×10⁻¹⁶; no NaN; max total AoA 0.760°; spin 1386→1050 rad/s |
| **Timestep convergence** | halving dt from 2×10⁻⁴ to 10⁻⁴ moves range by **0.021 m in 15 841 m** (1.3 ppm) and drift by 0.07 m in 328.6 m |

Maximum range 17 797 m at QE 800 mils against the published M107 figure of
~18 100 m (−1.7 %). Unit tests: **66 passed**.

The convergence table also remains the sharpest demonstration in the ladder of
the spin-resolution requirement: at 5.7 samples per revolution the range is
wrong by 1053 m (−6.6 %) while the trajectory still looks entirely reasonable.

---

## 2. What this pass changed

**One physics change: the C_Nα correction was widened from subsonic-only to
the full measured curve.**

It is built from all 22 usable full-scale rows of BRL Tables II and III
(19 M101 over Mach 0.57–2.41, 3 M107 at Mach 0.784–0.791), grouped into the
six Mach clusters the firing programme produced, yaw-filtered at 25 deg².
Details in [COEFFICIENTS.md](COEFFICIENTS.md) §8.

**It was adopted on the centre-of-pressure test, and it makes drift worse.**

| C_Nα option | CP RMS vs measured rows | mean drift error |
|---|---|---|
| raw ASAT deck (computed) | 0.1891 cal | +13.28 % |
| subsonic-only splice (previous pass) | 0.1438 cal | **+10.16 %** |
| **full measured correction (adopted)** | **0.1227 cal** | +14.37 % |
| *irreducible row-to-row scatter* | *0.1166 cal* | — |

The adopted curve sits at the measurement noise floor: its CP residual is no
longer distinguishable from the scatter of the measurements themselves. The
option with the best drift is not the one chosen, because choosing a
coefficient curve by the answer it produces is the failure mode this project
has refused throughout.

**C_Mα was not changed.** The same 21 rows give measured/ASAT = 1.0186, sd
0.0386, no Mach trend. Measurement *confirms* the computed overturning moment.

**The figure digitisation was attempted and rejected.** 66 % (Fig 8) and 72 %
(Fig 10) of the plot-area ink lies in connected components spanning thousands
of pixels — rules, faired curves and markers are fused — so the four marker
series cannot be separated, and separating them is the whole problem. The
tables are strictly the better source anyway: BRL's own per-round errors
(±0.08–0.10) exceed the best achievable figure read (±0.03), and the tables
name the series instead of leaving it to a glyph. Full record in
[DIGITISATION.md](DIGITISATION.md).

Two things the figure work did settle: the review's figure page map was off by
one, and Figure 10 carries **four different CG stations** in its legend
(2.80, 2.84, 2.96, 3.20 cal) with its faired curve being *model* data.

---

## 3. The drift residual — closed as unexplained

Drift is **+14.4 % high on average** (range +5.0 % to +24.8 %, worst where the
absolute drift is only a few metres). It is one-signed and bounded, and it does
not contaminate range, time of flight, summit conditions or impact velocity.

### Everything that has been ruled out, and by what measurement

| Candidate | How tested | Verdict |
|---|---|---|
| Rifling twist 1/25 vs 1/20 | direct run | **Inverts** the error (−8 % to −10 %) rather than closing it. Three sources give 1/20 for the M185/M199 tube behind FT 155-AM-2: McCoy verbatim, the NPS study that supplied the FT data, and BRL's own statement that *its* firings used the older 1/25 tube |
| C_Nα too high | replaced with the measurement | Drift got **worse**. The measured value is *higher* than the computed one over most of the range |
| C_Mα error | 21 measured full-scale rows | ratio 1.019 ± 0.039, no Mach trend — **confirmed correct** |
| Uniform C_Nα rescale | BRL data across Mach | Contradicted: BRL is *above* ASAT at Mach 0.57 and above Mach 2.19 |
| Missing pitch-damping force (C_Nq + C_Nα̇) | magnitude computed along a charge-8 trajectory | **0.70 % of the modelled normal force** (median; 1.12 % max) — far too small |
| Coriolis | measured separately (+5.8 m at 8 km, +27.9 m at 16 km) | Excluded from the comparison by construction |
| Sign error | rung 4 | Drift right 15/15; reverses correctly with the rifling; convention pinned by tests |
| Timestep | convergence study | Converged to 2 cm in 15.8 km |
| Integrator / frames | rungs 1 and 2 | Exact to machine precision |

### On the pitch-damping force specifically

BRL **does** tabulate the damping force coefficient, but *only for the
semi-scaled models*, transferred to the M101 CG station — there is no
full-scale measurement of it in the report, and BRL separately warns that the
semi-scaled damping data disagree badly with full scale (damping *moment*
about +1 for the models against about −9 for the full-scale rounds).

Values recorded in `analysis/brl_figures.py`. Using them, the omitted force is
**0.70 % of the modelled normal force** at the median and **1.12 %** at
maximum along a charge-8 trajectory. Both the instantaneous body rate and the
steady trajectory turn rate give the same answer, because the fast epicyclic
mode has damped out by mid-flight and the body rate settles onto the
trajectory turn rate.

**So the model-completeness gap is not the leading explanation.** It is two
orders of magnitude too small to account for a 14 % drift excess. This
contradicts the expectation set for this task, and the measurement is reported
rather than the expectation.

### What is actually left

With C_Nα and C_Mα both anchored to measurement, the twist established from
three sources, Ix agreeing between two independent sources to 2 %, and the
pitch-damping force quantified at 0.7 %, **no candidate remains inside the
aerodynamic coefficient set.**

The one substantive lead is structural rather than aerodynamic, and it is
recorded without being pursued: this model forms drift by integrating the full
6-DOF epicyclic motion, whereas the FT 155-AM-2 drift column was produced by a
modified point-mass model of the STANAG 4355 family. The NPS study that
supplied these FT numbers reproduced them to about 2 % with such a model using
the *same* ASAT coefficient deck. Two models can agree on every coefficient
and still differ on drift if they differ in how the yaw of repose is formed and
integrated.

**That is a model-to-model comparison, and it is exactly what step 2 is for** —
step 2 builds the reduced-order model and validates it against this one. If the
reduced-order model reproduces the firing-table drift while the 6-DOF does not,
the difference is isolated and diagnosable there, with both models in hand. It
cannot be isolated from inside step 1.

Per the stopping rule: **no fifth hypothesis is opened.**

---

## 4. Remaining known limitations

1. **Drift +14.4 %.** Above.
2. **The reduced-rate convention is an assumption, and permanently so from this
   source set.** ASAT-13 §3 gives only the rigid-body equations of motion and
   defers the aerodynamic expansion to Etkin and to an unreproduced M.Sc.
   thesis, so it cannot settle whether the rate-dependent coefficients are
   normalised on pd/V or pd/(2V). Worth < 0.25 % in range but 7–11 % in drift
   and a factor 2–4 in the angle-of-attack transient. `REDUCED_RATE_FACTOR` is
   the single constant expressing the choice.
3. **C_Mpα disagrees between sources by 36 %** at the only Mach where both
   measured it (BRL −0.567, ASAT −0.364). Rated LOW.
4. **The coefficient table ends at Mach 2.00.** Charge 8 launches at Mach 2.01.
   The measured Mach 2.265 cluster (C_Nα = 2.953) therefore cannot be
   represented; the model holds 2.801, about 5 % low. No trajectory in the
   ladder spends meaningful time there.
5. **Mach coverage gaps in the measured set**: 0.879–1.182 (the whole transonic
   including the C_Nα peak), 1.192–1.596, 1.770–2.190. Across these the ASAT
   shape is doing the work and only the level is measured.
6. **Linear aerodynamics only.** C_Mpα at 0° yaw; the source's 2°, 5° and 10°
   columns are unused. Valid because nominal flight stays below 0.8° total yaw.
7. **Peak angle of attack is a maximum over logged samples**, not over every
   step. Fine at the default cadence; a coarse `log_every` under-reports it.
8. **~29 s per trajectory** in pure CPython. By design — the Monte Carlo
   belongs on step 2's reduced-order model.

---

## 5. Fitness for use as step-2 ground truth

**The model is fit for purpose as the ground-truth reference for step 2, with
one stated exception.**

Fit, and well characterised:

- **Trajectory kinematics.** Range RMS 0.48 % with zero mean bias across five
  charges and an 8:1 range span; TOF RMS 0.52 %; summit and impact conditions
  within ~1 %. Independently corroborated by a fully specified published case
  (rung 5b) whose two text-stated scalars are matched to 0.71 % and 2.06 %, and
  whose initial axial deceleration is matched to 0.40 %.
- **Numerics.** Exact against the analytic parabola and against an
  independently written 3-DOF integration; converged to 1.3 ppm in range under
  timestep halving; quaternion norm at machine precision; no NaN.
- **Attitude dynamics.** Gyroscopic stability reproduces BRL's independently
  measured band when evaluated at BRL's tube; the 6-DOF epicyclic angle of
  attack tracks the closed-form yaw of repose after the launch transient
  damps; motion is bounded and damping throughout.
- **Coefficients.** C_X0, C_Nα and C_Mα are each confirmed or set by
  measurement of the actual projectile family. C_Mα, which dominates
  gyroscopic stability, agrees between two independent determinations to
  1.9 % ± 3.9 %.

The exception, stated plainly:

- **Absolute lateral deflection carries a known +14 % bias.** Step 2 must not
  treat this model's drift as truth in an absolute sense. It remains valid for
  *relative* work — sensitivities, sign, trends with charge and elevation, and
  differential comparisons — because the bias is one-signed and stable across
  the envelope. Any step-2 or step-6 result that depends on absolute drift must
  carry this caveat, and the reduced-order model of step 2 should be compared
  against the firing table's drift column directly as well as against this
  model, since that comparison is the most likely route to isolating the cause.

Nothing in the model is fitted to make an answer come out right. No form factor
is applied; the source's own published drag form factor was tested and makes
range *worse* (−0.99 % → +4.7 %), so it is not used.

---

## 6. Document map

| Document | Contents |
|---|---|
| [VALIDATION.md](VALIDATION.md) | full ladder, rung by rung, with numbers |
| [COEFFICIENTS.md](COEFFICIENTS.md) | provenance and confidence per coefficient, per Mach band |
| [DIGITISATION.md](DIGITISATION.md) | figure digitisation method and why it was rejected |
| [REVIEW-RESPONSE.md](REVIEW-RESPONSE.md) | response to the external review, including the claim rejected |
| [DONE-CHECKLIST.md](DONE-CHECKLIST.md) | SIXDOFSPEC.md §14 definition of done, item by item |
| `analysis/brl_reference.py` | verified BRL transcription with per-digit provenance |
| `analysis/brl_figures.py` | figure page map, axis calibration, damping-force table |
| `analysis/coefficient_crosscheck.py` | reruns the source comparison and the CP test |
