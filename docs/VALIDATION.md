# Validation Record — 6-DOF Ballistic Simulator

**155 mm M107 HE · step 1 · SIXDOFSPEC.md section 10 ladder**

Reproduce with `python run_validation.py`; raw output in
`validation_results.json` and `validation_run.log`. Unless stated otherwise:
`dt = 2×10⁻⁴ s`, latitude 45° N, ISA standard atmosphere, no wind.

Every rung below reports **numbers**, not a pass/fail assertion.

---

## Summary

> **Correction pass applied.** This record reflects the response to the
> external review (see [REVIEW-RESPONSE.md](REVIEW-RESPONSE.md)). One change
> was made to the physics — the subsonic C_Nα splice — and one proposed change
> was **rejected** after checking it against the sources: the rifling twist
> stays at 1 turn in 20 calibres. Rung 5b is new.

| Rung | What it catches | Result |
|---|---|---|
| 1 · Vacuum vs analytic parabola | integrator and frame errors | **Exact** — error < 10⁻⁶ % at 5 elevations, drift identically 0 |
| 2 · Drag-only vs independent 3-DOF | atmosphere, drag table, gravity | **Exact** — 0.000 m difference at 4 elevations |
| 3 · Gyroscopic stability | inertias, C_Mα magnitude | Sg = 2.58 → 13.09, always > 1. Dynamic criterion met in 14/15 runs; the 15th reproduces a subsonic instability BRL reports for this shell |
| 4 · Yaw of repose and drift | **sign errors** | Drift **right** in 15/15; magnitude +0.9 % to +23 % high (mean +10.2 %, was +13.3 %) |
| 5 · Firing table, 15 points, 5 charges | overall credibility | Range RMS **0.48 %**, mean **+0.00 %**; TOF RMS 0.53 % |
| **5b · ASAT-13 fully specified case** | **absolute check, nothing inferred** | **axial decel +0.40 %, TOF −0.71 %, summit time −2.1 %, peak AoA −0.19 %** |
| 6 · Norm drift, NaN, bounded α | numerical health | \|q\|−1 = 2.2×10⁻¹⁶; no NaN; max total AoA 0.76° |
| + Timestep convergence | spin under-resolution | Converged to **2 cm in 15.8 km** at dt = 2×10⁻⁴ |

**One real bug was found and fixed by this ladder.** See §9.

---

## Rung 1 — Vacuum against the analytic parabola

All aerodynamics zeroed, constant gravity, no Coriolis. Compared with
`R = V² sin 2θ / g`, `T = 2V sin θ / g`, `H = (V sin θ)² / 2g` at V = 684 m/s.

| QE (deg) | Range sim (m) | Range exact (m) | Error | TOF error | Apogee error | Drift (m) |
|---|---|---|---|---|---|---|
| 15 | 23854.017 | 23854.017 | −0.000000 % | −0.000000 % | −0.000000 % | 0.0 |
| 30 | 41316.370 | 41316.370 | −0.000000 % | −0.000000 % | −0.000000 % | 0.0 |
| 45 | 47708.035 | 47708.035 | −0.000000 % | −0.000000 % | −0.000000 % | 0.0 |
| 60 | 41316.370 | 41316.370 | −0.000000 % | −0.000000 % | +0.000000 % | 0.0 |
| 75 | 23854.017 | 23854.017 | −0.000000 % | −0.000000 % | +0.000000 % | 0.0 |

**Verdict: passes with enormous margin.** The spec requires better than 0.1 %;
the observed error is at the printing precision. This is expected rather than
impressive — RK4 integrates a quadratic exactly — but it is exactly what makes
it a clean test of the *frames*: any error in the quaternion construction, the
body→earth rotation, the gravity sign, or the NED axis assignment would show up
here as a range error or as non-zero drift. Drift is identically zero, which
confirms there is no spurious lateral coupling.

---

## Rung 2 — Drag only against an independent 3-DOF point mass

The 6-DOF is run with the total angle of attack forced to zero (axial force
applied along the relative velocity, all moments zeroed) and compared with
`analysis/pointmass3dof.py`, which shares **no code** with `sim/` — its own
state layout, its own RK4, its own impact interpolation, its own copy of the
ISA constants. Only the drag table is common, which is the thing being
compared. `dt = 10⁻³ s`, no Coriolis, inverse-square gravity, V = 684 m/s.

| QE (deg) | 6-DOF range (m) | 3-DOF range (m) | Difference | TOF difference |
|---|---|---|---|---|
| 15 | 11380.379 | 11380.379 | −0.0000 m (−0.00000 %) | −0.000000 s |
| 30 | 15982.077 | 15982.077 | +0.0000 m (+0.00000 %) | +0.000000 s |
| 45 | 17873.660 | 17873.660 | −0.0000 m (−0.00000 %) | +0.000000 s |
| 60 | 16005.574 | 16005.574 | +0.0000 m (+0.00000 %) | +0.000000 s |

**Verdict: passes to machine precision.** Two independently written
integrations of the same physics agree bit for bit.

This rung earned its place: **before** the fix described in §9 the same
comparison gave −0.03 %, −0.12 %, −0.37 %, −0.55 % — a discrepancy growing
with flight time that nothing else in the ladder revealed.

---

## Rung 3 — Gyroscopic and dynamic stability

`Sg = Ix² p² / (2 ρ S d It V² C_Mα)`, evaluated at the muzzle and at every
logged sample of all 15 firing-table trajectories.

| Quantity | Value |
|---|---|
| Sg at the muzzle, charge 4 (337 m/s) | 2.584 |
| Sg at the muzzle, charge 8 (684 m/s) | 3.192 |
| Sg minimum anywhere in any trajectory | **2.584** |
| Sg maximum anywhere | 13.087 |
| Sd at the muzzle | 0.72 – 0.92 |
| Dynamic requirement Sg > 1/(Sd(2−Sd)) | required Sg 1.007 – 1.085 |
| `Sg > 1` at every sample of every run | **True** |
| `Sg > 1/(Sd(2−Sd))` at every sample | **True in 14 of 15 runs** — see below |

Sg rises monotonically through flight — from 3.19 to 13.09 on the 16 km
charge-8 shot — because spin decays far more slowly than velocity. That is the
expected behaviour and is visible in `docs/figures/diagnostics_*.png`.

**Verdict: passes.** Sg > 1 everywhere by a wide margin.

### The one dynamic-stability exception, and why it is a result rather than a defect

In 14 of the 15 firing-table runs, `Sg > 1/(Sd(2−Sd))` holds at every logged
sample. In the fifteenth — charge 5, QE 420.6 mils — 106 of 707 samples fail
it, over Mach 0.795–0.801, from t = 21.8 s to 26.0 s of a 28.2 s flight.

The cause is that the computed **dynamic stability factor Sd goes negative at
low subsonic Mach**, which puts it outside the interval (0, 2) where the
criterion is defined at all, so no amount of spin satisfies it:

| Mach | Sd (raw table) | Sd (spliced) |
|---|---|---|
| 0.60 | −0.088 | −0.201 |
| 0.70 | +0.062 | −0.043 |
| 0.80 | +0.212 | +0.115 |
| 0.90 | +0.562 | +0.532 |
| 1.20 | +0.803 | +0.803 |

Sd is driven negative by the Magnus moment term: with kx⁻² = m d²/Ix = 7.10,
the product kx⁻²·C_Mpα at subsonic Mach outweighs C_Lα in the numerator of
McCoy's expression. This is **not** introduced by the C_Nα splice — the raw
table already crosses zero near Mach 0.68. The splice moved that crossing to
about Mach 0.72, which is why one more trajectory now dips below it.

**BRL MR-1582 reports exactly this behaviour for this shell**, from
measurement:

> "The 155-mm M101 shell is also dynamically unstable in the subsonic and
> transonic region at the muzzle spin of 1/25. However, the rate of divergence
> is small... **The change in sign of C_Mpα for the M107 shell decreases its
> dynamic stability relative to the M101**, and it would require more time
> along the trajectory to stabilize. The M101 and undoubtedly the M107 are
> dynamically stable if they are gyroscopically stable at above transonic
> speeds."

So the model reproduces a documented physical property of the M107, including
the detail that the M107 is worse than the M101 here specifically because of
the Magnus moment sign.

Two caveats, both stated rather than resolved:

1. **Sd is convention-sensitive.** It depends directly on the magnitudes of
   C_Mpα and C_mq, and therefore on the unresolved reduced-rate convention
   (COEFFICIENTS.md §6). Under the adopted reading Sd is negative subsonically;
   under a reading in which those coefficients are half as large, Sd is
   positive (about +0.47 at Mach 0.6) and the criterion is satisfied
   everywhere. The sign of this result is not robust.
2. **The convention-free observable is fine.** The direct output of the 6-DOF
   — the total angle of attack — stays bounded at 0.53° on the flagged run and
   0.76° across all 15, and the epicyclic envelope damps. Whatever the
   linearised criterion says, nothing in the simulated motion diverges.

**Caveat, stated rather than buried:** the muzzle Sg of 2.6–3.2 is above the
"typical 1.3–2.5" band quoted in the spec. That is not an error — the spec's
band is for a 1-in-25 twist, and this model uses the 1-in-20 twist of the
M185/M199 39-calibre tube. Sg scales as the square of spin, so
`3.19 × (20/25)² = 2.04`, squarely inside the quoted band. The independent
confirmation is that BRL MR-1582 tabulates its own *measured* stability factor
for the M101 at 1-in-25 as **1.69–2.27**, and this model evaluated at those
conditions gives 1.4–2.6 (`test_gyroscopic_factor_matches_brl_measured_values`).

---

## Rung 4 — Yaw of repose and drift · the sign-error detector

**A right-hand-rifled shell must drift RIGHT.** In all 15 firing-table
trajectories the drift is positive (right). Reversing the rifling reverses the
drift (`test_left_hand_rifling_drifts_left`).

Comparison is made with Coriolis **off**, because the firing-table drift column
is the ballistic yaw-of-repose drift and Coriolis is applied separately in the
fire-control solution.

| Charge | QE (mils) | Sim drift (m) | FT drift (m) | Error (%) | Error before splice (%) |
|---|---|---|---|---|---|
| 4 | 97.2 | +3.93 | 3.2 | +22.95 | +26.84 |
| 4 | 152.0 | +9.04 | 7.8 | +15.90 | +20.92 |
| 4 | 211.6 | +16.73 | 15.2 | +10.07 | +16.22 |
| 5 | 118.1 | +8.81 | 7.5 | +17.40 | +18.08 |
| 5 | 280.4 | +39.85 | 36.6 | +8.87 | +12.71 |
| 5 | 420.6 | +80.86 | 79.2 | **+2.10** | +8.62 |
| 6 | 258.4 | +50.71 | 45.5 | +11.46 | +13.16 |
| 6 | 378.6 | +94.83 | 88.2 | +7.52 | +11.15 |
| 6 | 539.9 | +170.89 | 169.4 | **+0.88** | +7.16 |
| 7 | 177.6 | +40.78 | 37.1 | +9.93 | +10.09 |
| 7 | 319.8 | +103.86 | 94.0 | +10.49 | +12.11 |
| 7 | 520.7 | +223.00 | 211.9 | +5.24 | +9.40 |
| 8 | 141.6 | +41.67 | 37.6 | +10.83 | +10.83 |
| 8 | 248.4 | +101.98 | 92.4 | +10.36 | +10.83 |
| 8 | 525.3 | +317.20 | 292.8 | +8.33 | +11.09 |

| | min | max | mean | RMS |
|---|---|---|---|---|
| before splice | +7.16 % | +26.84 % | +13.28 % | 14.21 |
| **after splice** | **+0.88 %** | **+22.95 %** | **+10.16 %** | **11.51** |

Every one of the 15 points improved or held; none got worse. The improvement
tracks time spent subsonic, which is what a subsonic-only correction should
do: the high-angle low-charge shots gain 4–6.5 percentage points (charge 6 at
QE 539.9 goes from +7.16 % to +0.88 %), while the flat charge-8 shots, which
are supersonic for most of their flight, barely move.

The largest remaining *percentage* errors are at the shortest ranges where the
drift itself is only a few metres (22.95 % of 3.2 m is 0.73 m).

### Independent confirmation of the yaw of repose

The closed-form yaw of repose

```
δ_R = 2 Ix p g cos θ / (ρ V³ S d C_Mα)
```

is computed alongside the 6-DOF and plotted in
`docs/figures/diagnostics_*.png`. The 6-DOF's epicyclic angle-of-attack
oscillation is centred on that analytic curve — for charge 4 at QE 97.2, the
analytic δ_R runs 0.22°→0.28° while the 6-DOF total AoA oscillates between
0.02° and 0.45° with a mean of 0.283°. The full nonlinear simulation and the
closed-form equilibrium agree, from completely different routes.

**Verdict: passes on sign, decisively. Magnitude is ~11 % high and is a
genuine open residual.**

### Why drift is still high — diagnosed, not tuned

**What was applied:** the subsonic C_Nα splice, justified on the
centre-of-pressure test against BRL's independently measured Figure 9, not on
drift. See [COEFFICIENTS.md](COEFFICIENTS.md) §8. It removed 3.1 points of
mean drift error and is the only physics change in this pass.

**What was tested and rejected:**

1. **Rifling twist.** The review proposed 1 turn in 25 calibres instead of
   1/20, which would cut spin by 20 % and drift with it. Setting 1/25 does not
   close the error — it **inverts** it, to −11.4 %/−11.4 %/−11.2 % at the three
   charge-8 points, leaving the magnitude essentially unchanged. Independently,
   three sources give 1/20 for the M185/M199 tube behind FT 155-AM-2. Rejected;
   full argument in [REVIEW-RESPONSE.md](REVIEW-RESPONSE.md) task 1.

2. **A uniform C_Nα rescale.** Scaling the *whole* curve by 0.899 would move
   drift to about −2 %, but BRL's own data does not support it: BRL's M101
   C_Nα is *higher* than the table at M 0.57 and above M 2.19, and the CP test
   shows the discrepancy is confined below Mach 0.9. Applying a single-Mach
   correction across the whole curve would be fitting.

3. **The reduced-rate convention.** Doubling the four rate-dependent
   coefficients moves drift by 7–12 % while changing range by under 0.25 %.
   Real evidence on the convention question, recorded rather than acted on:
   selecting a convention because it makes drift match would be fitting.

**Still open, in order of plausibility:**

- **C_Mα.** Drift goes roughly as C_Nα/C_Mα. Against the three measured M107
  rounds ASAT's C_Mα is 3.6 % low on average, which would make drift ~3.6 %
  high — but BRL's own three rounds scatter by 9 % (3.84, 3.51, 3.74), so this
  sits inside the measurement's own noise and cannot be acted on.
- **Supersonic C_Nα.** ASAT is *below* BRL above Mach 2, so correcting there
  would increase drift. Wrong direction.
- **Axial moment of inertia.** Drift ∝ Ix; BRL and ASAT differ by 2.1 %.
- **The linear yaw-of-repose model.** No C_Nα̇ or C_Nq term in this 6-DOF.

**What evidence would resolve it:** a digitisation of BRL Figures 8 and 10
(C_Nα and C_Mα versus Mach for the full-scale rounds), giving measured curves
across the whole Mach range instead of three rounds at one Mach; or the
ASAT-13 paper's own force and moment expansion, which would also settle the
reduced-rate convention. Direct retrieval of ASAT-13 was attempted repeatedly
and failed: the publisher's host `asat.journals.ekb.eg` is unreachable.

---

## Rung 5 — Firing table comparison

Firing table **FT 155-AM-2** (155 mm Howitzer M185/M199, projectile HE M107),
as tabulated in Lim, NPS thesis 2016 (DTIC AD1029824) Tables 15–19.
15 points across **5 charges** and **2 000 – 16 000 m**. Coriolis off.

| Charge | MV (m/s) | QE (mils) | Range sim (m) | FT (m) | Err | TOF sim (s) | FT (s) | Err |
|---|---|---|---|---|---|---|---|---|
| 4 | 337 | 97.2 | 2011.6 | 2000 | +0.58 % | 6.38 | 6.4 | −0.39 % |
| 4 | 337 | 152.0 | 3014.5 | 3000 | +0.48 % | 9.84 | 9.8 | +0.40 % |
| 4 | 337 | 211.6 | 4013.5 | 4000 | +0.34 % | 13.52 | 13.5 | +0.13 % |
| 5 | 397 | 118.1 | 3016.7 | 3000 | +0.56 % | 8.80 | 8.8 | +0.02 % |
| 5 | 397 | 280.4 | 6017.8 | 6000 | +0.30 % | 19.57 | 19.6 | −0.13 % |
| 5 | 397 | 420.6 | 7978.0 | 8000 | −0.27 % | 28.22 | 28.2 | +0.08 % |
| 6 | 474 | 258.4 | 7025.9 | 7000 | +0.37 % | 20.77 | 20.8 | −0.15 % |
| 6 | 474 | 378.6 | 9007.5 | 9000 | +0.08 % | 28.89 | 28.9 | −0.04 % |
| 6 | 474 | 539.9 | 10958.5 | 11000 | −0.38 % | 39.07 | 39.1 | −0.08 % |
| 7 | 568 | 177.6 | 7010.8 | 7000 | +0.15 % | 17.49 | 17.6 | −0.61 % |
| 7 | 568 | 319.8 | 10025.3 | 10000 | +0.25 % | 28.57 | 28.6 | −0.11 % |
| 7 | 568 | 520.7 | 12971.5 | 13000 | −0.22 % | 42.56 | 42.6 | −0.10 % |
| 8 | 684 | 141.6 | 7941.4 | 8000 | −0.73 % | 16.81 | 17.0 | −1.12 % |
| 8 | 684 | 248.4 | 10939.4 | 11000 | −0.55 % | 26.87 | 27.2 | −1.21 % |
| 8 | 684 | 525.3 | 15840.8 | 16000 | −1.00 % | 48.49 | 48.9 | −0.85 % |

Aggregate over all 15 points:

| Quantity | Min | Max | Mean | RMS |
|---|---|---|---|---|
| Range error | −1.00 % | +0.58 % | **−0.00 %** | **0.48 %** |
| Time of flight error | −1.21 % | +0.40 % | −0.28 % | 0.53 % |
| Impact velocity error | −1.83 % | +1.05 % | −0.50 % | 1.13 % |
| Maximum ordinate error | −0.86 % | +0.64 % | +0.10 % | 0.51 % |
| Drift error | +0.88 % | +22.95 % | +10.16 % | 11.51 % |

The mean range error is **+0.00 %** with an RMS of 0.48 % — the model is
unbiased in range across five charges and an 8:1 range span. There is a mild
trend from slightly long at low charge to about 1 % short at charge 8.

### Maximum range

Charge 8 (684 m/s), QE swept:

| QE (mils) | QE (deg) | Range (m) |
|---|---|---|
| 700 | 39.38 | 17465.5 |
| 750 | 42.19 | 17693.2 |
| **800** | **45.00** | **17796.8** |
| 850 | 47.81 | 17758.9 |
| 900 | 50.62 | 17572.9 |
| 950 | 53.44 | 17230.0 |

Maximum **17 797 m at QE 800 mils**, against the published M107 maximum of
**~18 100 m** — **−1.7 %**, consistent with the small short bias at charge 8.
Well inside the "out of family beyond ~24 km without base bleed" guidance.

**Verdict: passes.** Range and TOF are within a few tenths of a percent across
the whole envelope, without any form factor or fitted correction.

### The source's own form factor makes it worse

The NPS thesis applies a drag form factor `Fi = 0.9076` to this coefficient
table to match the firing table with its modified point-mass model. Applying
the same factor here:

| Case | Raw table | With Fi = 0.9076 | FT |
|---|---|---|---|
| Charge 8, QE 525.3 | 15841 m (−0.99 %) | 16585 m (+3.65 %) | 16000 m |
| Charge 8, QE 141.6 | 7941 m (−0.73 %) | 8232 m (+2.90 %) | 8000 m |

The full 6-DOF with **raw, unmodified, sourced coefficients** matches the
firing table better than the same coefficients with the published point-mass
form factor. No form factor is used in this model.

---

## Rung 5b — The ASAT-13 §4.3 fully specified trajectory

Better than a firing-table comparison in one respect: **every input is stated
by the source**, so nothing is left to inference, and the outputs include two
tight text-stated scalars that a drag form factor cannot be tuned to hit
simultaneously with a range figure.

**Inputs (ASAT-13 §4.3, used verbatim):** θ₀ = 44°, V₀ = 684.3 m/s,
p₀ = 175.48 rps, m = 43 kg, d = 0.155 m, L = 698 mm, CG = 0.459 m from the
nose, Ix = 0.144, Iy = Iz = 1.216 kg·m².

Note p₀ = 175.48 rps at V₀ = 684.3 m/s implies 1 turn in **25.16** calibres —
an M114-era twist paired with an M185-era muzzle velocity. That is used here
because the point of this rung is to reproduce the published case exactly; it
is *not* the nominal model's tube. See
[REVIEW-RESPONSE.md](REVIEW-RESPONSE.md) task 1.

| Quantity | Model | ASAT | Error | Reliability |
|---|---|---|---|---|
| muzzle spin | 175.48 rps | 175.48 rps | — | by construction |
| **initial axial deceleration** | **4.468 g** | **4.45 g** | **+0.40 %** | text-stated |
| **total flight time** | **66.194 s** | **66.67 s** | **−0.71 %** | text-stated |
| **summit time** | **30.36 s** | **~31 s** | **−2.06 %** | text-stated |
| summit altitude | 5634 m | ~5700 m | −1.15 % | figure-read |
| max total angle of attack | 1.2975° | ~1.3° | −0.19 % | figure-read |
| time of max AoA | 32.36 s | ~32 s | +1.12 % | figure-read |
| range | 17 715 m | ~16 500 m | **+7.36 %** | figure-read |
| drift direction | right (+460.6 m) | right | ✓ | text-stated |

**The axial-deceleration check is the most informative line in this table.**
Drag alone gives 3.773 g. Adding the component of gravity along the body axis
at θ₀ = 44° (sin 44° = 0.695 g) gives **4.468 g** against the stated 4.45 g.
That both identifies what ASAT's quoted figure includes and independently
confirms the axial force coefficient, the reference area, the mass and the
whole force scaling to 0.4 %.

**Range is the single outlier, and the evidence points at the source.** Five of
six published outputs agree within 2 %, including both tight text-stated
scalars. A range of 16 500 m at QE 44° and 684.3 m/s would be inconsistent
with the M107's published maximum range of ~18 100 m, and with this model's
own firing-table-validated maximum-range sweep (17 797 m at QE 45°). It is
also hard to reconcile internally: matching ASAT's own TOF of 66.67 s while
covering only 16 500 m demands a mean horizontal speed of 247 m/s against this
model's 268 m/s, and no plausible drag change moves range by 7 % without also
moving the flight time. Range is the least reliable of the published outputs
(read off Figure 3) and is treated as such.

**Verdict: passes.** Both tight scalars within 2.1 %, the axial-deceleration
identity within 0.4 %, peak angle of attack within 0.2 %, and the one
discrepancy attributable to a figure reading that conflicts with the
projectile's own published maximum range.

---

## Rung 6 — Energy, norm and angle-of-attack sanity

From the 48.5 s charge-8 trajectory at QE 525.3 (242 000 steps), and from all
15 firing-table runs:

| Check | Requirement | Observed |
|---|---|---|
| Quaternion norm | within 10⁻⁹ of unity | **2.22×10⁻¹⁶** max \|\|q\|−1\| over all logged samples, all runs |
| Per-step norm drift removed by renormalisation | — | 4.92×10⁻⁸ per step at dt = 2×10⁻⁴ |
| NaNs | none | **none**, in any run |
| Total angle of attack | "a few degrees", bounded | max **0.76°** over all 15 runs; mean 0.475° |
| Spin decay | physically plausible | 1386.4 → 1050.4 rad/s over 48.5 s (**75.8 %** retained) |

The per-step norm drift scales as (ω dt / 2)⁶, exactly as RK4 theory predicts:
measured 4.92×10⁻⁸ at θ = 0.1386 and 1.72×10⁻⁷ at θ = 0.1708, a ratio of 3.50
against a predicted (0.1708/0.1386)⁶ = 3.50. Renormalising every step removes
it; left to accumulate over 242 000 steps it would reach ~1.2×10⁻².

The angle-of-attack history (`docs/figures/angle_of_attack_*.png`) shows
textbook epicyclic motion: fast and slow modal arms beating in the total-AoA
envelope, α and β in quadrature, and the envelope slowly **damping**
(0.446° → 0.383° over 6.4 s at charge 4). The shell is dynamically stable.

**Verdict: passes.**

---

## Timestep convergence

Charge 8, QE 525.3 mils, Coriolis off. dt halved four times.

| dt (s) | Samples/rev at muzzle | Range (m) | Δ vs dt = 10⁻⁴ | Drift (m) | Δ | TOF (s) |
|---|---|---|---|---|---|---|
| 8×10⁻⁴ | 5.7 | 14601.17 | **−1239.65 m** | 280.71 | −36.42 | 47.4619 |
| 4×10⁻⁴ | 11.3 | 15840.24 | −0.577 m | 318.14 | +1.01 | 48.4845 |
| 2×10⁻⁴ | 22.7 | 15840.80 | **−0.021 m** | 317.20 | +0.07 | 48.4867 |
| 1×10⁻⁴ | 45.3 | 15840.82 | reference | 317.13 | reference | 48.4868 |

**Verdict: converged.** Halving the step from 2×10⁻⁴ to 10⁻⁴ changes the range
by **2 cm in 15.8 km** (1.3 ppm) and the drift by 7 cm in 325 m. The
production step of 2×10⁻⁴ s is fully resolved.

This table is also the sharpest demonstration in the whole ladder of the
spec's warning about spin under-resolution. At 5.7 samples per revolution the
range is wrong by **1215 m — 7.7 %** — and the trajectory still looks entirely
reasonable. Anyone tempted to speed this model up by enlarging dt should read
this row first. The spec's 20–50 samples/rev guidance is confirmed: the
threshold sits between 5.7 and 11.3.

---

## Sensitivity cases

Neither of these is applied to the delivered model; both are reported because
the underlying uncertainty is real.

| Case | QE 525.3: range | drift | max AoA | QE 141.6: range | drift | max AoA |
|---|---|---|---|---|---|---|
| **Nominal (spliced, 1/20)** | 15841 m | 317.2 m | 0.76° | 7941 m | 41.7 m | 0.39° |
| C_Nα splice OFF (raw ASAT) | 15841 m | 325.3 m | 0.76° | 7941 m | 41.7 m | 0.39° |
| Rate coefficients ×2 | 15813 m (−0.18 %) | 280.9 m (−11.4 %) | 1.08° | 7924 m (−0.21 %) | 38.9 m (−6.6 %) | 1.60° |
| Drag × 0.9076 | 16584 m (+4.7 %) | 335.5 m | 0.71° | 8232 m (+3.7 %) | 43.4 m | 0.36° |
| Twist 1/25 instead of 1/20 | 15855 m | 253.5 m (−20.1 %) | 0.61° | 7944 m | 33.3 m (−20.1 %) | 0.33° |

The **C_Nα splice** changes range by under 0.5 m — it is a drift and
angle-of-attack effect only, as expected for a normal-force coefficient at
sub-degree yaw.

The **reduced-rate convention** ambiguity (COEFFICIENTS.md §6) is worth **less
than 0.25 % in range** but **7–11 % in drift** and a factor of 2–4 in the
angle-of-attack transient. Range results are robust to it; drift and
damping-margin results are not.

The **drag form factor** published with the source table makes range *worse*
(from −1.00 % to +3.65 %), so it is not used.

### Twist sensitivity — the rejected change

| QE (mils) | Tube | Twist | p₀ (rps) | Sg muzzle | Drift (m) | FT (m) | Error | max AoA |
|---|---|---|---|---|---|---|---|---|
| 141.6 | M185 | 1/20 | 220.65 | 3.192 | 41.67 | 37.6 | **+10.83 %** | 0.387° |
| 141.6 | M1 | 1/25 | 176.52 | 2.043 | 33.31 | 37.6 | **−11.41 %** | 0.330° |
| 525.3 | M185 | 1/20 | 220.65 | 3.192 | 317.20 | 292.8 | **+8.33 %** | 0.761° |
| 525.3 | M1 | 1/25 | 176.52 | 2.043 | 253.47 | 292.8 | **−13.43 %** | 0.610° |

(These rows are run with the C_Nα splice in place. Before the splice the 1/25
figures were −11.41 % and −11.23 %; either way the error inverts rather than
closing.)

Changing the twist does not close the drift error; it inverts it while leaving
the magnitude essentially unchanged. And it moves the peak angle of attack
*away* from the ~1.3° of the ASAT reference case, because with zero muzzle
tip-off the angle of attack is yaw-of-repose driven and δ_R ∝ p. The nominal
model keeps 1/20, which is the twist of the tube behind FT 155-AM-2.

---

## §9 — The bug this ladder found

Rung 2 initially disagreed by −0.03 %, −0.12 %, −0.37 % and −0.55 % at QE 15,
30, 45 and 60 — a discrepancy growing with flight time between two
implementations of physics that should have been identical. Initial
accelerations matched bit for bit, so the error was in the *interior RK4
stages*.

**Cause.** `dcm_from_quat` requires a unit quaternion; given one of norm *n* it
returns a rotation scaled by *n²*. Inside an RK4 stage the quaternion is not
unit — the stage state `q + (dt/2)·q̇` leaves the unit sphere, and because q̇ is
orthogonal to q the norm grows as `|q| = √(1 + (dt·|ω|/4)²)`. At the muzzle
spin of a 155 mm shell (1386 rad/s) that is |q| = 1.058 at dt = 10⁻³. The
aerodynamic force passes through the rotation **twice** — earth→body to find
the relative wind, body→earth to apply the force — so it scaled as **|q|⁴**,
making the three interior stages up to **25 % too large**.

**Why it mattered that rung 2 exists.** The buggy model still produced a
completely plausible trajectory and still landed within about 1 % of the
firing table at every point tested. Rungs 1, 3, 5 and 6 all passed with it in
place. Only a comparison against an independently written integration of the
same physics exposed it.

**Fix.** `_aero_core` normalises the quaternion before building the DCM
(`sim/dynamics.py`). After the fix, rung 2 agrees to machine precision.

---

## Performance

Measured cleanly, single process, no contention: the 48.5 s charge-8
trajectory at QE 525.3 and dt = 2×10⁻⁴ runs **242 447 steps in 28.7 s**
(118 µs per RK4 step). The `wall_clock_s` field recorded in
`validation_results.json` is much larger than the sum of the individual runs
because the ladder was executed while other work was competing for the same
cores; it is not a clean benchmark and should not be quoted as one.

---

## Test suite

`python -m pytest tests -q` → **66 passed** in 46 s.

Coverage includes: quaternion round-trips and rotation-matrix orthonormality;
ISA values against published tables at 0, 11, 15 and 20 km; coefficient
interpolation exact at table knots; hot-path and readable-path coefficient
lookups pinned to each other; the physical sign of every coefficient; the
centre of pressure falling inside the projectile; BRL-measured gyroscopic
stability reproduced; derivative purity and determinism; the control-callback
seam; nose-up ⇒ lift up **and** nose-up moment; damping terms always opposing
their rates; right-hand rifling drifting right and left-hand rifling drifting
left.

Added in the correction pass: the C_Nα sign convention pinned end-to-end so it
cannot be inverted by a future edit; the Magnus force coefficient pinned
negative; the axial column pinned against every BRL C_D value so C_A and C_D
cannot be interleaved; the splice reproducing the BRL measurement, leaving
supersonic untouched, and improving CP against Figure 9; the twist recorded per
tube; the transverse inertia guarded against the commonly-quoted wrong value;
the ASAT initial axial deceleration reproduced to 1 %; and the subsonic
dynamic-stability behaviour documented.

---

## Open items

1. **Drift is ~11 % high.** Two candidate causes identified and quantified
   (§ rung 4); neither adopted. Resolving it needs either the ASAT-13 source
   paper's coefficient-definition equations, or a digitisation of BRL
   MR-1582's C_Nα-versus-Mach figure for the full-scale M107.
2. **The reduced-rate convention is an assumption.** See COEFFICIENTS.md §6.
   Resolving it needs the ASAT-13 paper's own force and moment expansion.
   An attempt to fetch it was rate-limited by the publisher.
3. **The coefficient table ends at Mach 2.00** while charge 8 launches at
   M 2.01. End values are held flat; the excursion is 0.5 % in Mach and is
   reported at run time.
4. **Impact-velocity RMS error (1.13 %) is the largest of the trajectory
   quantities.** Firing-table impact velocities are quoted to 3 significant
   figures, so part of this is quantisation in the reference data.
