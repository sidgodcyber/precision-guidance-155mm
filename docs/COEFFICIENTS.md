# Aerodynamic Coefficient Provenance

**155 mm M107 HE projectile — step 1 ground-truth 6-DOF model**

Every coefficient used by `sim/aerodata.py` is traceable to a cited document.
**There are no placeholder coefficients in this model.** Four of the eight
rest on a single source, and one convention is an assumption; both facts are
recorded below and are printed by the program on every run.

---

## 1. Sources

| Tag | Document | Nature |
|---|---|---|
| **KHALIL** | M. Khalil, H. Abdalla, O. Kamal, *"Dispersion Analysis for Spinning Artillery Projectile"*, 13th Int. Conf. on Aerospace Sciences & Aviation Technology (ASAT-13), Military Technical College, Cairo, 26–28 May 2009, paper ASAT-13-FM-03, DOI 10.21608/asat.2009.23740. Obtained through W. Y. Lim, *"Predicting the Accuracy of Unguided Artillery Projectiles"*, M.S. thesis, Naval Postgraduate School, Sept 2016 (DTIC AD1029824), **Table 11**, p. 53. | Complete 6-DOF deck for the M107, PRODAS-style, semi-empirical |
| **BRL-1582** | B. G. Karpov & L. E. Schmidt, rev. K. Krial & L. C. MacAllister, *"The Aerodynamic Properties of the 155-mm Shell M101 from Free Flight Range Tests of Full Scale and 1/12 Scale Models"*, Ballistic Research Laboratories Memorandum Report No. 1582, Aberdeen Proving Ground, June 1964 (DTIC AD0454925). | **Primary measurement.** Spark-range free flight, full-scale M101 over M 0.57–2.41, plus three full-scale **M107** rounds at M 0.78–0.79 |

> **Transcription note.** BRL MR-1582's OCR text layer confuses 3/5 and 2/8
> routinely and must not be used for digits. Every BRL value used here was
> read from page images rendered at 900–2400 dpi, and the verified
> transcription with per-digit provenance lives in
> `analysis/brl_reference.py`. Three digits disputed by the external review
> were re-checked and resolved there: Table III C_Mpα is −0.56/−0.56/−0.58
> (not −0.36), the M107 mass is 95.8 lb (not 95.2), and rounds 1802/990 are at
> Mach 1.596/1.599 (not 1.396/1.399).
| **AIMT-BB** | R. Balon & J. Komenda, *"Analysis of the 155 mm ERFB/BB Projectile Trajectory"*, Advances in Military Technology 1/2006, Tab. 2 (PRODAS-derived). | Different 155 mm shell; used only to corroborate the magnitude of `C_lp` |

BRL-1582 states that the M107 "differs from the M101 only in the rotating
band", which is what makes the M101 data a legitimate cross-check on an M107
table.

**The table actually used is KHALIL.** BRL-1582 is *not* blended into it; it
is used exclusively as an independent check. Mixing a fitted curve from two
sources would have destroyed the ability to say "these two agree".

---

## 2. The table as implemented

Reference length **d = 0.155 m**, reference area **S = πd²/4 = 0.018869 m²**,
moments referenced to the **centre of gravity**, all angular derivatives
**per radian**.

The C_Nα column is **spliced**: below Mach 1.0 it is scaled onto the BRL
free-flight measurement of the M107 (§8 below). "raw" is the ASAT deck as
published; "used" is what the model runs.

| Mach | C_X0 | C_X2 | C_Nα raw | **C_Nα used** | C_Ypα | C_lp | C_Mα | C_mq | C_Mpα |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.144 | 2.343 | 1.763 | **1.582** | −0.767 | −0.023 | 3.355 | −5.1 | −0.500 |
| 0.60 | 0.144 | 2.343 | 1.763 | **1.582** | −0.767 | −0.023 | 3.378 | −5.1 | −0.500 |
| 0.80 | 0.146 | 2.847 | 1.783 | **1.600** | −0.767 | −0.022 | 3.571 | −5.1 | −0.355 |
| 0.90 | 0.167 | 3.372 | 1.827 | **1.733** | −0.857 | −0.021 | 3.957 | −7.4 | −0.112 |
| 0.95 | 0.221 | 3.730 | 2.038 | **1.986** | −1.082 | −0.020 | 3.886 | −9.9 | 0.085 |
| 1.00 | 0.327 | 4.180 | 2.153 | 2.153 | −0.992 | −0.020 | 3.682 | −13.8 | 0.198 |
| 1.05 | 0.383 | 4.691 | 2.207 | 2.207 | −0.902 | −0.020 | 3.415 | −13.3 | 0.293 |
| 1.10 | 0.381 | 5.209 | 2.255 | 2.255 | −0.857 | −0.019 | 3.384 | −14.6 | 0.334 |
| 1.20 | 0.370 | 5.702 | 2.325 | 2.325 | −0.767 | −0.020 | 3.424 | −15.8 | 0.352 |
| 1.35 | 0.353 | 5.130 | 2.442 | 2.442 | −0.767 | −0.020 | 3.278 | −15.6 | 0.366 |
| 1.50 | 0.338 | 4.561 | 2.556 | 2.556 | −0.767 | −0.020 | 3.264 | −15.3 | 0.373 |
| 1.75 | 0.314 | 3.970 | 2.692 | 2.692 | −0.767 | −0.020 | 3.201 | −15.3 | 0.381 |
| 2.00 | 0.294 | 3.460 | 2.747 | 2.747 | −0.767 | −0.021 | 3.013 | −15.3 | 0.388 |

Linear interpolation between knots; end values held flat outside, with the
excursion recorded and reported. `C_Mpα` is the source's 0°-yaw column; the
source also tabulates 2°, 5° and 10°, which a later nonlinear-Magnus model can
use.

### Which source each coefficient comes from, at which Mach

| Coefficient | Mach < 0.80 | 0.80 – 1.00 | Mach > 1.00 |
|---|---|---|---|
| C_X0 | ASAT | ASAT | ASAT |
| C_X2 | ASAT | ASAT | ASAT |
| **C_Nα** | **BRL-scaled ASAT** (k = 0.8974) | **taper, k 0.8974 → 1** | **ASAT** |
| C_Ypα | ASAT | ASAT | ASAT |
| C_lp | ASAT | ASAT | ASAT |
| C_Mα | ASAT | ASAT | ASAT |
| C_mq | ASAT | ASAT | ASAT |
| C_Mpα | ASAT (0° yaw) | ASAT (0° yaw) | ASAT (0° yaw) |

---

## 3. Provenance and confidence, per coefficient

| Coefficient | Source | Mach range | Per | Ref. station | Confidence | Basis for that rating |
|---|---|---|---|---|---|---|
| **C_X0** zero-yaw axial force | KHALIL | 0.01–2.0 | — | n/a | **HIGH** | BRL-1582 M107 free flight at M 0.79 gives 0.136–0.155 over 3 rounds (mean 0.143) against this table's 0.1455. M101 at M 0.81 gives 0.123–0.149 over 4 rounds against 0.1465. Agreement is within the round-to-round scatter. |
| **C_X2** yaw drag | KHALIL | 0.01–2.0 | rad⁻² | n/a | **MEDIUM** | Single source. BRL measured the *drag* yaw coefficient C_D2 = 5.9 (subsonic), 9.9 (M 1.2), 11.6 (M 1.6), 7.8 (M 2.1). Converting to axial via `C_X2 = C_D2 − C_Nα + C_X0/2` gives 4.2 / 7.8 / — / 5.2 against this table's 2.8 / 5.7 / — / 3.4: same shape, ≈1.4× larger. Contributes ≈3 % of axial force at the 0.4–0.8° yaw of nominal flight, so the spread is not range-critical. |
| **C_Nα** normal force slope | **BRL-scaled KHALIL below M 1.0**, KHALIL above; **sign flipped** | 0.01–2.0 | rad | n/a | **HIGH** | Below Mach 1.0 the column is scaled onto BRL's measured M107 value (§8). Above Mach 1.0 it is KHALIL unchanged, and agrees with BRL: interpolating BRL's bracketing rows to M 2.00 gives 2.762 against KHALIL's 2.747 (0.5 %). The splice is justified by the centre-of-pressure test against BRL Figure 9, not by the 11 % single-Mach gap on its own. |
| **C_Ypα** Magnus force | KHALIL | 0.01–2.0 | rad | n/a | **LOW** | Single source. BRL could not extract Magnus force from full-scale swerve at all and quotes semiscaled-model values of −0.15 to −0.55 (in its own pd/V normalisation). Same sign, same order. |
| **C_lp** spin damping | KHALIL | 0.01–2.0 | rad | n/a | **MEDIUM** | Single source. AIMT-BB gives STANAG-4355 `C_spin` = −0.0132 to −0.0107 for a 155 mm ERFB/BB, which converts to `C_lp = 2·C_spin` = −0.026 to −0.021 in this package's convention. Same magnitude, different shell. Model reproduces 86 % spin retention over a 17 s flight, which is physically ordinary. |
| **C_Mα** overturning moment | KHALIL | 0.01–2.0 | rad | **CG** | **HIGH** | The best-determined coefficient. BRL vs table: 3.28–3.36/3.378 @ 0.6; 3.59–3.62 (M101) and 3.51–3.84 (M107)/3.571 @ 0.8; 3.67/3.682 @ 1.0; 3.49/3.424 @ 1.2; 3.40–3.50/3.264 @ 1.6; 2.91–3.00/3.013 @ 2.2. **3–7 % across the whole Mach range, from two fully independent determinations.** |
| **C_mq** pitch/yaw damping | KHALIL | 0.01–2.0 | rad | **CG** | **LOW-MEDIUM** | Single source (−5.1 to −15.8). BRL measured the combined (C_mq + C_mα̇) as −4.1 to −21.9 with a stated standard error of 2.5 — the free-flight data itself does not pin this down better than a factor of two. |
| **C_Mpα** Magnus moment | KHALIL (0° yaw column) | 0.01–2.0 | rad | **CG** | **LOW** | Sources disagree in magnitude: BRL M107 free flight gives −0.56, −0.56, −0.58 at M 0.79; this table gives −0.36. Both are negative subsonic and both turn positive supersonically, and BRL notes the M107 and M101 differ here specifically because of the rotating band. |

### Physical properties (not aerodynamic, same rigour)

| Property | Value | Source |
|---|---|---|
| mass | 43.454 kg (95.8 lb) | BRL-1582 Table I |
| diameter | 0.155 m | BRL-1582 Table I |
| CG from nose | 0.4588 m (2.96 cal) | BRL-1582 Table I |
| length | 0.6975 m (4.5 cal) | BRL-1582 Table I |
| I_axial | 0.14581 kg·m² (from k_a⁻² = 7.10) | BRL-1582 Table I |
| I_transverse | 1.27810 kg·m² (from k_t⁻² = 0.81) | BRL-1582 Table I |
| twist | 1 turn in 20 cal, right hand | M185/M199 39-cal tube; Lim NPS Table 12 |

`I_axial` is independently confirmed: Lim NPS 2016 Table 12 gives
Ixx = 0.1461 kg·m² from an unrelated source, agreeing to **0.7 %**. That
agreement also confirms the two `k⁻²` columns were read the right way round —
swapping them would give I_axial = 1.29 kg·m², nine times too large, and the
shell would not spin-stabilise at all.

---

## 4. The three hazards, addressed

### Hazard 1 — per-degree vs per-radian

**Both sources are per radian. No conversion applied.**

The check that does not depend on either paper asserting it: with C_Mα ≈ 3.4,
the gyroscopic stability factor of the M101 at its spark-range conditions
(1 turn in 25 calibres) evaluates to **Sg = 1.4–2.6**, and BRL-1582 Table II
tabulates its own *separately measured* gyroscopic stability factor `s` for
those same rounds as **1.69–2.27**. A per-degree misreading would put C_Mα near
195 and Sg near 0.03, and those shells demonstrably did not tumble.

This is pinned by `tests/test_sim.py::test_gyroscopic_factor_matches_brl_measured_values`.

### Hazard 2 — moment reference station

BRL-1582 Table I places the M101/M107 CG at 2.96 calibres from the nose and
references its moment coefficients to the centre of mass. The KHALIL table is
a 6-DOF deck, likewise CG-referenced. `sim/projectile.py` places `x_cg` at that
same station, so the transfer term is **identically zero**:

```
C_Mα|CG = C_Mα|ref + (x_ref − x_cg)/d · C_Nα
        = 3.571 + (0.4588 − 0.4588)/0.155 × 1.783
        = 3.571 + 0.000 × 1.783
        = 3.571
```

`aerodata.transfer_moment_reference()` implements the general transfer anyway,
so the step-6 Monte Carlo — which will perturb `x_cg` — applies it correctly
rather than silently reusing a coefficient referenced to the wrong station.

### Hazard 3 — sign of C_Mα (stabilising vs destabilising)

The spec takes **positive C_Mα = destabilising** (centre of pressure ahead of
the CG). BRL-1582 Appendix I states its convention explicitly: *"A positive
C_Mα yields a moment which increases the total angle of attack."* Same
convention. KHALIL's values are positive and of the same magnitude as BRL's,
so it is the same convention too. **No sign flip applied.**

Geometric confirmation: `C_Mα/C_Nα` at M 0.8 is 3.571/1.783 = **2.00 calibres**,
putting the centre of pressure 2.00 calibres *ahead* of the CG, at
2.96 − 2.00 = **0.96 calibres from the nose** — inside the ogive, exactly where
the CP of an ogive-cylinder-boattail body belongs. Under a
stabilising-positive reading it would sit 2 calibres *behind* the CG, at
4.96 calibres from the nose of a **4.5 calibre** shell — off the back of the
projectile. Only one reading is geometrically possible.

Pinned by `tests/test_sim.py::test_centre_of_pressure_lies_inside_the_projectile`.

---

## 5. Two sign corrections applied to the source table

**`C_Nα` — flipped to positive.** The source tabulates this column negative
(−1.763 … −2.747), i.e. it is tabulating ∂C_z/∂α, which is −C_Nα in this
convention. In the spec's force model `[Y,Z] = −q̄·S·C_Nα·[v,w]/V`, a positive
C_Nα puts the normal force along the angle-of-attack direction (nose up → force
up). That is the physically correct sense and is what BRL tabulates (positive,
1.45–3.00, same magnitudes). Used unflipped, the normal force would act to
*increase* the angle of attack.

**`C_Ypα` — kept negative.** Deliberate, not an oversight. A negative Magnus
force coefficient is the expected result for a spin-stabilised shell (boundary
layer asymmetry reverses the naive inviscid ω × v direction), and BRL
independently measured −0.15 to −0.55 for this shell family.

---

## 6. The one unresolved convention risk

The four rate-dependent coefficients — `C_Ypα`, `C_Mpα`, `C_lp`, `C_mq` — are
multiplied in `dynamics.py` by the reduced rates **pd/(2V)** and **qd/(2V)**,
per spec sections 5–6. That is the PRODAS / aircraft convention.

The classical aeroballistic literature uses **pd/V** and **qd/V**. BRL-1582
Appendix I is explicit about this:

```
F_Y + i F_Z = (1/2)ρV²S  { −[C_Nα + i(pd/V) C_Npα] ξ − … }
M_Y + i M_Z = (1/2)ρV²Sd { [(pd/V) C_Mpα − i C_Mq] ξ + … }
```

A coefficient in the pd/V convention is **exactly half** the pd/(2V)
coefficient describing the same physics.

The KHALIL table is a PRODAS-style deck — it tabulates C_Mpα against yaw angle
at 0°, 2°, 5° and 10°, which is a PRODAS output format — so **pd/(2V) is the
reading adopted, and no factor is applied**. This is an *assumption*: neither
source states it in a form this work could verify, and an attempt to retrieve
the ASAT-13 paper directly was rate-limited by the publisher.

**Consequence, measured rather than asserted:** `run_validation.py` runs a
sensitivity case with all four coefficients doubled. See
[VALIDATION.md](VALIDATION.md) §8 for the numbers. Range and drift barely
move; the dynamic-stability margin and the damping of the yaw transient are
what change. The static coefficients that set range, stability and drift —
C_X0, C_Nα, C_Mα — carry no such ambiguity.

`aerodata.REDUCED_RATE_FACTOR` is the single constant that expresses this
choice; setting it to 1.0 switches the whole model to the aeroballistic
convention.

---

## 8. The subsonic C_Nα splice (added in the correction pass)

**What changed.** Below Mach 1.0, C_Nα is multiplied by

```
k = 0.8974                            Mach ≤ 0.80
k = linear, 0.8974 → 1.0              0.80 < Mach < 1.00
k = 1.0                               Mach ≥ 1.00
```

where **K_SUBSONIC = 1.600 / 1.783 = 0.8974** is the ratio of BRL's measured
M107 three-round mean to the ASAT value at the same Mach.

**Why.** ASAT's deck is *computed* (SPINNER-98). BRL Table III is *measured* —
three full-scale M107 rounds at Mach 0.784/0.786/0.791 giving C_Nα = 1.61,
1.62, 1.57. Where both exist, measurement outranks computation. The gap is
11.4 %, about four standard errors of the three-round mean.

**But the justification is the centre-of-pressure test, not that gap.** An
11 % difference at one Mach number is not on its own grounds to reshape a
curve. BRL plots the centre of pressure independently in its Figure 9, and CP
constrains the *ratio* of the two coefficients:

```
CP_from_nose [cal] = x_cg/d − C_Mα / C_Nα
```

Since ASAT and BRL agree on C_Mα throughout (ratio 0.92–1.04, mean 0.99), any
systematic CP disagreement is attributable to C_Nα. Measured residuals:

| | mean CP residual (ASAT − BRL) |
|---|---|
| Mach < 0.9 (n = 10) | **+0.186 cal** |
| Mach ≥ 0.9 (n = 11) | **−0.018 cal** |

Against Figure 9 directly, the splice moves the residual from +0.294 to +0.075
at Mach 0.60 and from +0.257 to +0.028 at Mach 0.80, and leaves Mach ≥ 1.0
untouched.

**Crossover choice.** Lower anchor Mach 0.80 — the only Mach with a full-scale
M107 measurement. Upper anchor Mach 1.00 — the CP residual has already changed
sign there (−0.100), so correcting at or above Mach 1.0 would make agreement
worse. Finer placement is not supportable: Figure 9 reads to ±0.1 cal and the
transonic CP gradient is steep (0.6 → 1.35 cal between Mach 0.85 and 1.00).

**Not fitted to drift.** The quantity justifying the splice (CP, from BRL
Figure 9) and the quantity testing the model (drift, from FT 155-AM-2) are
different measurements in different documents. The drift improvement — mean
firing-table drift error from +13.3 % to +10.2 %, and from +7.2 % to +0.9 % on
the shot that spends longest subsonic — is reported as a consequence in
[VALIDATION.md](VALIDATION.md), not used as the reason.

**Known weakness.** Below Mach 0.6 there is no full-scale M107 measurement.
The single BRL M101 round at Mach 0.570 (C_Nα = 1.81) sits *above* ASAT and
would argue against any correction there; it also implies CP = 1.148 cal, well
outside the 0.6–0.9 cal band of BRL's own faired Figure 9, so it is treated as
an outlier against the curve BRL drew. Holding k constant below Mach 0.80 is
an assumption, supported by the fact that it brings the model onto Figure 9 at
Mach 0.60 as well as at 0.80.

Reproduce with `python -m analysis.coefficient_crosscheck`.

---

## 9. Rifling twist is a property of the tube, not the shell

Raised in review and resolved against primary sources.

| Tube | Twist | Evidence |
|---|---|---|
| **M185 / M199**, 39 cal (M109A1/A2/A3, M198) — **the nominal model** | **1 turn in 20 cal** | McCoy, *Modern Exterior Ballistics* 2nd ed. ch. 13, verbatim: the M549 "fired at Charge 4 from the M109A1 Howitzer with a rifling twist rate of 1 turn in 20 calibers of travel". Also Lim NPS 2016 Table 12, "Twist Rate 20 Calibers/rev" — the study that produced this model's FT 155-AM-2 comparison data |
| M1 / M1A1, 23 cal (M114) | 1 turn in 25 cal | BRL MR-1582: its own firings used "standard 155-mm artillery pieces with a twist of one turn in 25 calibers"; later, "the muzzle spin of 1/25" |
| ASAT-13 §4.1 configuration | 1 turn in 25.16 cal | Implied by its stated V₀ = 684.3 m/s with p₀ = 175.48 rps |

FT 155-AM-2 is the M185/M199 table, so **1/20 is the twist consistent with the
reference data** and is what the nominal model uses. ASAT pairs an M185-class
muzzle velocity with an M114-class twist, which is not self-consistent for any
single US tube; its configuration is reproduced faithfully in rung 5b but not
adopted.

Consequence worth stating because it is easy to get wrong: substituting
p = 2πV/(nd) into the gyroscopic stability factor gives

```
Sg = Ix² (2π)² / (2 ρ Iy S d³ n² C_Mα)
```

so **at fixed twist Sg does not depend on muzzle velocity at all** — it varies
across charges only through C_Mα(Mach). Quoting an Sg without naming the tube
is meaningless, and Sg scales as the inverse square of the twist.

---

## 10. Known limitations of the table

1. **Upper Mach limit is 2.00.** Charge 8 launches at M 2.01 at sea level, so
   the first instants of flight sit marginally outside the table and are held
   flat. The excursion is 0.5 % in Mach and is reported at run time.
2. **Linear aerodynamics only.** `C_Mpα` is taken at 0° yaw. The source's 2°,
   5° and 10° columns are not used. Nominal flight stays below 0.8° total
   angle of attack, where the linear term dominates, but a limit-cycle or
   large-yaw study would need the nonlinear columns.
3. **No Reynolds-number or spin-rate dependence** beyond Mach.
4. **No fuze or lot variation.** BRL-1582 shows ±10 % round-to-round scatter
   in C_X0 at fixed Mach for nominally identical shells. That scatter is a
   Monte Carlo input for step 6, not a fixed-model quantity.
5. **`C_mq` is the pitch damping moment alone**, not the free-flight-measurable
   combination (C_mq + C_mα̇). The 6-DOF here has no α̇ term, so the source's
   `C_mq` is applied as the effective total damping.
