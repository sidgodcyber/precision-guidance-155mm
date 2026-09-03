# Response to the External Review

**155 mm M107 6-DOF simulator · step 1 · correction pass**

Reviewing documents: `BRL1582EXTRACT.md`, `ASAT13CROSSCHECK.md`.

Every claim below was checked against the primary sources before any code was
touched. Two claims were adopted, three were confirmed as already-correct, and
**one was rejected** — the twist rate, which was the change that would have
closed the drift error. It is rejected on the arithmetic in the sources and on
what a 155 mm tube actually is, not on whether it improves rung 4.

Three transcription errors in the review documents were found and are
documented in §8, along with one place where the review was right and this
project's earlier reading was wrong.

---

## Summary table

| # | Task | Review claim | Verdict | Change made |
|---|---|---|---|---|
| 1 | Twist rate | Twist is 1/25, model uses 1/20, spin 25.8 % high | **REJECTED** for the nominal model; correct for ASAT's own configuration | None to the value. Documented `TUBES`; added `M107_ASAT` for rung 5b |
| 2 | Independent Sg | Sg should be 2.05, model gives 2.58 | Arithmetic **CONFIRMED**; comparison target **wrong** | None |
| 3 | C_Nα splice | ASAT 11.4 % high subsonic, agrees supersonic | **PARTLY CONFIRMED**; splice **ADOPTED** on different evidence | Subsonic C_Nα spliced to BRL measurement |
| 4 | Sign audit | May be handled; verify not two cancelling errors | **CONFIRMED** already correct | Added 2 pinning tests |
| 5 | Inertias | Check for the wrong ~1.79 value | **DOES NOT APPLY** — model never used 1.79 | Added a guard test |
| 6 | Rung 5b | Add the ASAT fully-specified case | **ADOPTED** | Added rung 5b |
| 7 | C_A vs C_D | Check the two are not interleaved | **CONFIRMED** already clean | Added a pinning test |

---

## Task 1 — Twist rate · CLAIM REJECTED

### What the review claimed

ASAT §4.1 gives V₀ = 684.3 m/s and p₀ = 175.48 rps, implying 3.900 m per
revolution = 25.16 calibres. The spec's worked example used 1/20. If that
carried into the build, spin is 25.8 % high, which would make drift ~26 %
high, Sg ~58 % high, and the angle of attack too small — "all three symptoms
point the same direction."

### What I found

**The arithmetic is correct.** Reproduced independently:

```
684.3 / 175.48       = 3.8996 m per revolution
3.8996 / 0.155       = 25.159 calibres per revolution
684.3 / (20 × 0.155) = 220.74 rps      220.74 / 175.48 = 1.2579
```

**But twist is a property of the gun tube, not of the projectile.** The same
M107 shell is fired from different tubes and leaves with different spin. The
question is not "what is the M107's twist" — that question has no answer — but
"what tube produced the data this model is validated against."

That data is firing table **FT 155-AM-2**, which is the table for the **M185**
cannon (M109A1/A2/A3) and **M199** (M198). Three independent sources give that
tube as 1 turn in 20 calibres:

1. **R. L. McCoy, *Modern Exterior Ballistics*, 2nd ed., ch. 13**, verbatim:
   the M549 shell *"fired at Charge 4 from the M109A1 Howitzer with a rifling
   twist rate of 1 turn in 20 calibers of travel"*. The M109A1 mounts the M185.
2. **W. Y. Lim, NPS thesis 2016 (DTIC AD1029824), Table 12**: *"Twist Rate
   20 Calibers/rev"* — and this is the study that produced the FT 155-AM-2
   comparison numbers used in rung 5, so 20 is the value consistent with the
   reference data by construction.
3. **BRL MR-1582 itself** states its own firings used *"standard 155-mm
   artillery pieces with a twist of one turn in 25 calibers"* and later refers
   to *"the muzzle spin of 1/25"*. That is the **older M1/M114-era tube**, not
   the M185. It is precisely why BRL's own measured stability factors
   (1.69–2.27) are lower than this model's.

So both twists are real; they belong to different weapons. 1/20 is correct for
the configuration under validation.

**ASAT's own numbers are not self-consistent with any single US tube.** A
muzzle velocity of 684.3 m/s is a 39-calibre-tube figure (the M114's charge 7
is about 563 m/s). ASAT pairs that with a 25.16-calibre twist, which is an
M114-era twist. Whatever ASAT modelled, it is not an M185 firing charge 8.

### The three predicted symptoms, tested

| Symptom | Review predicted | Measured |
|---|---|---|
| Drift | ~26 % high if twist wrong | **+10.8 % — less than half the prediction** |
| Sg | ~58 % high | Sg is 58 % higher at 1/20 than 1/25, but see task 2 |
| Peak AoA | Too small because Sg too high | **Backwards for this model — see below** |

**Direct experiment.** Setting the twist to 1/25, changing nothing else:

| QE (mils) | drift 1/20 | drift 1/25 | FT drift | err 1/20 | err 1/25 |
|---|---|---|---|---|---|
| 141.6 | 41.67 | 33.31 | 37.6 | **+10.83 %** | **−11.41 %** |
| 248.4 | 102.41 | 81.86 | 92.4 | **+10.83 %** | **−11.41 %** |
| 525.3 | 325.28 | 259.93 | 292.8 | **+11.09 %** | **−11.23 %** |

Changing the twist does not close the drift error. It **inverts** it, leaving
the magnitude essentially unchanged. If excess spin were the cause of a +11 %
drift error, removing 20 % of the spin would land near zero, not at −11 %.
This is decisive: the twist is not the source of the residual.

**The AoA symptom is backwards.** The review reasoned that higher Sg means a
stiffer round and therefore smaller angle of attack. That is true of the
*epicyclic arm excited by muzzle tip-off*. This model launches with zero
tip-off (`initial_q = initial_r = 0`), so the angle of attack is entirely
yaw-of-repose driven, and the yaw of repose goes as

```
δ_R = 2 Ix p g cos θ / (ρ V³ S d C_Mα)     ∝ p
```

Reducing spin therefore *reduces* AoA. Measured: peak AoA falls from 0.387° to
0.330° at QE 141.6, and from 0.761° to 0.610° at QE 525.3, on going from 1/20
to 1/25 — moving **away** from ASAT's ~1.3°, not toward it.

**The "Sg too high" concern dissolves independently.** McCoy reports the
105 mm M1 fired from the M103 howitzer (1/18 twist) at a muzzle gyroscopic
stability factor of **3.1**. A muzzle Sg near 3 is ordinary service practice,
because Sg must stay above 1 at every charge, in cold dense air, and through
the transonic where C_Mα peaks. This model's 3.19 at charge 8 is unremarkable.

### Change made

None to the twist value. The model keeps 1/20.

Added `sim/projectile.py::TUBES` recording all three twists with their
sources, a comment on the `twist_calibers` field stating that twist belongs to
the tube, `M107_ASAT` reproducing ASAT's configuration for rung 5b, and
`test_twist_is_a_tube_property_and_the_nominal_tube_is_the_M185`.

---

## Task 2 — Independent Sg check · ARITHMETIC CONFIRMED, TARGET WRONG

Recomputed from first principles, from ASAT's own mass properties at ASAT's
own muzzle condition, without copying the review's working:

```
S   = π d²/4 = 0.0188692 m²
p   = 2π × 175.48 = 1102.57 rad/s
Ix p = 0.144 × 1102.57 = 158.771      (Ix p)² = 25 208
den  = 2(1.225)(1.216)(0.0188692)(0.155)(468 266)(3.013) = 12 294
Sg   = 25 208 / 12 294 = 2.0505
```

**The review's 2.05 is confirmed exactly.**

But it is not the right comparison target for this model, for two reasons.

**First, there is no single "muzzle Sg" to compare against.** Substituting
p = 2πV/(nd) into the Sg expression gives

```
Sg = Ix² (2π)² / (2 ρ Iy S d³ n² C_Mα)
```

which is **independent of muzzle velocity**. At fixed twist, Sg depends only on
the twist, the air density, and C_Mα(Mach). This model's muzzle Sg therefore
varies across charges only through C_Mα: 2.584 at charge 4 (Mach 0.99, where
C_Mα peaks near 3.9) up to 3.192 at charge 8 (Mach 2.01, C_Mα 3.01). The
review compared its 2.05 against **2.58**, which is this model's *minimum over
all charges*, not its charge-8 muzzle value of 3.19.

**Second, the entire remaining gap is the twist.** (25/20)² = 1.5625, and
3.192/2.043 = 1.562. Running this model at 1/25 gives **Sg = 2.043** against
the review's 2.0505 — a 0.3 % residual, which is the difference between BRL's
inertias (Ix 0.14704, Iy 1.28887) and ASAT's (0.144, 1.216).

**Cross-check that validates the formula rather than the assumption:** BRL
measured s = 1.69–2.27 for the M101 on its 1/25 tubes. This model evaluated at
1/25 reproduces that band. The Sg implementation and the C_Mα table are
therefore both correct; only the tube differs.

No change made.

---

## Task 3 — C_Nα splice · PARTLY CONFIRMED, ADOPTED ON DIFFERENT EVIDENCE

### Verifying the characterisation

| Claim | Verdict |
|---|---|
| ASAT 11.4 % high at M ≈ 0.8 vs measured M107 | **Confirmed.** ASAT 1.782, BRL M107 3-round mean 1.600, ratio 1.1136. BRL's stated per-round error 0.08 gives s.e. of the mean 0.046 (2.9 %), so the gap is about 4 standard errors — real, not scatter. |
| ASAT agrees with BRL to ~2 % at M 2.0 | **Confirmed.** Interpolating BRL's bracketing rows (M 1.770 → 2.62 and M 2.190 → 2.88) to M 2.00 gives 2.762; a least-squares fit through the four rows above M 1.7 gives 2.782; ASAT gives 2.747. That is 0.5–1.3 % apart. |
| "SPINNER-98 overestimates subsonic normal force" as a general statement | **Not supported as stated.** At M 0.570 ASAT is 2.6 % *below* BRL's M101; at M 0.867 it is only 1.8 % above. The 11.4 % figure is specific to the M107 rows at M 0.784–0.791. Against M101 rows at nearly the same Mach (0.809–0.817) the gap is 6–8 %. |

So the discrepancy is real but its size depends on which rows you compare, and
an 11 % gap at a single Mach number is not on its own grounds to reshape a
curve. The justification has to come from somewhere else.

### The centre-of-pressure test is what justifies the splice

CP is a third quantity, measured and plotted independently by BRL in its
Figure 9, and it constrains the *ratio* of the two coefficients:

```
CP_from_nose [cal] = x_cg/d − C_Mα / C_Nα
```

Because ASAT and BRL agree on C_Mα across the whole range (ratio 0.92–1.04,
mean 0.99), any systematic CP disagreement is attributable to C_Nα.

```
mean CP residual (ASAT − BRL tabulated rows), Mach < 0.9  : +0.186 cal  (n=10)
mean CP residual,                             Mach ≥ 0.9  : −0.018 cal  (n=11)
```

Against BRL Figure 9 directly:

| Mach | Fig 9 CP | ASAT CP | residual | after splice |
|---|---|---|---|---|
| 0.60 | ~0.75 | 1.044 | **+0.294** | **+0.075** |
| 0.80 | ~0.70 | 0.957 | **+0.257** | **+0.028** |
| 0.85 | ~0.60 | 0.875 | +0.275 | +0.102 |
| 1.00 | ~1.35 | 1.250 | −0.100 | −0.100 |
| 1.60 | ~1.65 | 1.719 | +0.069 | +0.069 |
| 2.00 | ~1.80 | 1.863 | +0.063 | +0.063 |

The discrepancy is systematic, one-signed, confined below Mach 0.9, and absent
from Mach 1.0 up. That is the signature of a real modelling error in the
subsonic normal-force computation, not of measurement scatter — and it is not
a quantity this model was fitted to.

### The splice as implemented

```
C_Nα ← C_Nα × k(M)
  k = 0.8974                                  M ≤ 0.80
  k = linear from 0.8974 to 1.0               0.80 < M < 1.00
  k = 1.0                                     M ≥ 1.00
```

with K_SUBSONIC = 1.600/1.783, the ratio of the BRL M107 three-round mean to
the ASAT value at the same Mach. The spliced table returns **1.5988** at
M 0.787, reproducing the measurement.

**Crossover justification, and why it is not tuned.** The lower anchor is
Mach 0.80 because that is where the only full-scale M107 measurement exists.
The upper anchor is Mach 1.00 because the CP residual has *already changed
sign* there (−0.100), so any correction at or above Mach 1.0 would make CP
agreement worse. The crossover therefore lies between Mach 0.85 (+0.275) and
Mach 1.00 (−0.100); placing it more finely is not supportable, because
Figure 9 reads to only about ±0.1 cal and the transonic CP gradient is steep
(0.6 → 1.35 cal across that interval). A linear taper spans the whole
plausible band with its midpoint at Mach 0.90.

**Known weakness, recorded not hidden.** Below Mach 0.6 there is no full-scale
M107 measurement, and the single BRL M101 round at Mach 0.570 (C_Nα = 1.81)
sits *above* ASAT, which taken alone would argue against any correction there.
That round also implies CP = 1.148 cal, well outside the 0.6–0.9 cal band of
BRL's own faired Figure 9, so it is treated as an outlier against the curve BRL
itself drew. Holding k constant below Mach 0.80 is an assumption; what
supports it is that it brings the model onto Figure 9 at Mach 0.60 as well as
at 0.80.

### What it did to drift — a real improvement, but it does not close the gap

Over all 15 firing-table points:

| | min | max | mean | RMS |
|---|---|---|---|---|
| before splice | +7.16 % | +26.84 % | +13.28 % | 14.21 |
| **after splice** | **+0.88 %** | **+22.95 %** | **+10.16 %** | **11.51** |

Every point improved or held; none got worse. The improvement tracks time
spent subsonic, which is exactly what a subsonic-only correction should do:

| Charge | QE (mils) | err before | err after | gain |
|---|---|---|---|---|
| 6 | 539.9 | +7.16 % | **+0.88 %** | −6.28 pp |
| 5 | 420.6 | +8.62 % | **+2.10 %** | −6.52 pp |
| 4 | 211.6 | +16.22 % | +10.07 % | −6.15 pp |
| 8 | 525.3 | +11.09 % | +8.33 % | −2.76 pp |
| 8 | 141.6 | +10.83 % | +10.83 % | 0.00 pp |

The flat charge-8 shots barely move because they are supersonic for most of
their flight (muzzle Mach 2.01, impact Mach ~0.99); only the high-angle,
low-charge shots spend appreciable time below Mach 1.

Range is unaffected — under 0.5 m across the whole table — as expected for a
normal-force coefficient at sub-degree yaw.

**The splice does not close the drift gap, and it was not adopted in order
to.** It is adopted because the CP test says the subsonic normal force is
wrong. The drift change is a consequence, and 10 % of residual remains.

### One side effect, reported rather than buried

The splice lowers subsonic C_Nα, which lowers C_Lα, which feeds the dynamic
stability factor Sd. The raw table already gives Sd < 0 below about Mach 0.68
— outside the (0, 2) interval where `Sg > 1/(Sd(2−Sd))` is defined at all —
and the splice moves that crossing to about Mach 0.72. One firing-table run
(charge 5, QE 420.6) consequently dips below the criterion for 4.2 s near the
end of its flight, where it previously did not.

This is not a defect introduced by the splice so much as a threshold crossed.
BRL MR-1582 reports the same behaviour for this shell from measurement: *"The
155-mm M101 shell is also dynamically unstable in the subsonic and transonic
region... The change in sign of C_Mpα for the M107 shell decreases its dynamic
stability relative to the M101."* Details, including the convention
sensitivity and the fact that the simulated angle of attack stays bounded at
0.53° on that run, are in [VALIDATION.md](VALIDATION.md) rung 3.

---

## Task 4 — Sign convention audit · CONFIRMED ALREADY CORRECT

ASAT lists C_Nα negative (it is tabulating ∂C_z/∂α); BRL lists it positive and
states the convention explicitly in Appendix I: *"A positive C_Nα yields a
normal force in the direction of the total angle of attack."* That is the
spec's convention.

Audit result: `C_Nalpha` and `C_Ypalpha` each appear **exactly once** in the
force law (`sim/dynamics.py` lines 341 and 346). The flip is applied once,
where the table is defined, with the reason documented. The force law then
carries the spec's own leading minus sign:

```python
cn = -qS * C_Nalpha * inv_V      # spec: [Y,Z] = -q̄·S·C_Nα·[v,w]/V
```

One flip, one minus sign, no cancelling pair. `C_Ypα` is deliberately *not*
flipped — a negative Magnus force coefficient is physically expected for a
spin-stabilised shell, and BRL independently measured −0.15 to −0.55.

Added `test_cnalpha_is_stored_positive_and_negated_exactly_once` (which also
asserts the end-to-end consequence: nose-up gives force up *and* a nose-up
moment) and `test_magnus_force_coefficient_is_stored_negative`.

---

## Task 5 — Inertias · CLAIM DOES NOT APPLY

The review asked me to check for a commonly-quoted but unsupported Iy ≈ 1.79.

The model has always used, and still uses:

```
Ix = 0.14704 kg·m²      Iy = 1.28887 kg·m²
```

derived from BRL Table I (k₁⁻² = 7.10, k₂⁻² = 0.81) with m = 95.8 lb. It has
never used 1.79. No correction was needed.

The review's own derived figures (0.1461 / 1.2807) came from m = 95.2 lb for
the M107. **That mass is a misreading** — see §8. With the correct 95.8 lb the
BRL-derived values are 0.14704 and 1.28887, which is what the model has.

Agreement between the two independent determinations of the same shell:

| | Ix | Iy |
|---|---|---|
| BRL Table I (this model) | 0.14704 | 1.28887 |
| ASAT §4.1 (Inventor + PRODAS) | 0.144 | 1.216 |
| ratio | 1.021 | 1.060 |

Kept the BRL-derived values for the nominal model, because mass, CG, length
and both radii of gyration then all come from one consistent source. ASAT's
exact values are used in rung 5b, where reproducing ASAT's case requires
ASAT's inputs. The effect of the choice is small: Sg moves 1.6 %, drift 2.1 %.

Added `test_transverse_inertia_is_not_the_commonly_quoted_wrong_value`.

---

## Task 6 — Rung 5b · ADOPTED

The ASAT §4.3 case is now a validation rung. Inputs are ASAT's own, including
p₀ = 175.48 rps (hence the 1/25.16 twist, used here because the point is to
reproduce the published case exactly, not because it is an M185).

| Quantity | Model | ASAT | Error | Source reliability |
|---|---|---|---|---|
| muzzle spin | 175.48 rps | 175.48 rps | — | by construction |
| **initial axial deceleration** | **4.468 g** | **4.45 g** | **+0.40 %** | text-stated |
| **total flight time** | **66.194 s** | **66.67 s** | **−0.71 %** | text-stated |
| **summit time** | **30.36 s** | **~31 s** | **−2.06 %** | text-stated |
| summit altitude | 5634 m | ~5700 m | −1.15 % | figure-read |
| max total AoA | 1.2975° | ~1.3° | −0.19 % | figure-read |
| time of max AoA | 32.36 s | ~32 s | +1.13 % | figure-read |
| range | 17 715 m | ~16 500 m | **+7.36 %** | figure-read |
| drift direction | right (+460.6 m) | right | ✓ | text-stated |

**The axial deceleration check is the most informative result here.** Drag
alone gives 3.773 g. Adding the component of gravity along the body axis at
θ₀ = 44° (sin 44° = 0.695 g) gives 4.468 g against ASAT's stated 4.45 g. That
identifies what ASAT's figure includes *and* independently confirms the axial
force coefficient and the whole force scaling to 0.4 %.

**Range is the one outlier, and the evidence points at the source, not the
model.** Five of six published outputs agree within 2 %, including both tight
text-stated scalars. A range of 16 500 m at QE 44° and 684.3 m/s would be
inconsistent with the M107's published maximum range of ~18 100 m and with
this model's own firing-table-validated maximum-range sweep (17 797 m at
QE 45°). It would also be hard to reconcile internally: matching ASAT's TOF of
66.67 s while travelling only 16 500 m requires a mean horizontal speed of
247 m/s against this model's 268 m/s, which no plausible drag change produces
without also moving the flight time. Range is the least reliable of ASAT's
published outputs (read from Figure 3), and it is treated as such.

---

## Task 7 — C_A versus C_D · CONFIRMED ALREADY CLEAN

ASAT tabulates **C_A**, axial force along the body axis — which is what the
spec's force model consumes. BRL tabulates **C_D**, drag along the velocity
vector, and BRL's C_D additionally contains yaw drag requiring C_Dδ²·δ̄² to be
removed first. The two must never be interleaved.

Audit result: the C_X0 column is the ASAT C_A column throughout. No BRL C_D
value appears anywhere in it. BRL's C_D is used only in the documentation, as
a cross-check, and only after the yaw-drag correction is applied.

`C_Aα²` maps directly onto the spec's `C_X2`, as the review notes.

Added `test_axial_column_is_C_A_not_BRL_C_D`, which checks the column against
the specific BRL C_D values so a future paste cannot go unnoticed.

---

## §8 — Errors found in the review documents

Verified by re-rendering the BRL scan at 900–2400 dpi. The OCR text layer of
this report confuses 3/5 and 2/8 routinely and cannot be used for digits.

**1. Table III C_Mpα is −0.56, −0.56, −0.58 — not −0.36, −0.36, −0.38.**
The glyph is the same "5" that appears in ".1575" and "3.51" on the same rows;
a "3" in this typeface has an open left side and no closed counter. This
matters: the review's §6 used −0.36 to claim *"Agreement to 2 %"* with ASAT's
−0.355 and called it *"the strongest validation in either document."* The true
comparison is −0.567 measured against −0.364 computed — **36 % apart**. That
claim is withdrawn; `C_Mpα` remains rated LOW confidence.

**2. Table I M107 mass is 95.8 lb, not 95.2 lb.** At 2400 dpi the final glyph
has two stacked closed counters — an 8. A "2" has no closed counter at all.
Both the M101 and M107 rows carry the same value. This propagated into the
review's Ix/Iy derivation (§5 above).

**3. Table II rounds 1802 and 990 are at Mach 1.596 and 1.599, not 1.396 and
1.399.** Confirmed by the glyph and independently by the report's own
statement that rounds are *"numbered in order of increasing Mach number"* —
1.396 would place them before round 858 at 1.435. This is what produced the
review's own note that *"Rounds 858 and 1802 appear out of Mach order."* With
the correct reading the table is monotonic and there is no anomaly.

**4. Where the review was right and this project was wrong:** round 1678 has
C_Mα = **3.58**. An earlier low-resolution reading in this project recorded
3.30. The review's value is correct and has been adopted.

The verified transcription now lives in `analysis/brl_reference.py`, with the
provenance of each disputed digit recorded in its docstring.

---

## §9 — The drift residual after all changes

Drift remains high: **mean +10.2 %** over the 15 firing-table points, range
+0.9 % to +23 %, improved from +13.3 % mean. This is reported, not closed with
a third correction.

**Ruled out:**

- **Twist rate.** Changing 1/20 → 1/25 inverts the error to −11 % rather than
  removing it, and 1/20 is what three sources give for the tube behind the
  reference data.
- **Subsonic C_Nα.** Correcting it to the measurement — which is justified
  independently by the CP test — removes only ~2.8 points on the 16 km shot
  and ~0 on the 8 km shot.
- **Coriolis.** Measured separately and excluded from the comparison, since
  firing-table drift is the ballistic drift only.
- **Sign errors.** Audited; drift is to the right in every case and reverses
  correctly when the rifling is reversed.

**Still open, in order of plausibility:**

1. **C_Mα.** Drift goes roughly as C_Nα/C_Mα. Against the three measured M107
   rounds, ASAT's C_Mα is 3.6 % low on average (ratios 0.926, 1.014, 0.952),
   which would make drift ~3.6 % high. But BRL's own three rounds scatter by
   9 % (3.84, 3.51, 3.74), so this is within the measurement's own noise and
   cannot be acted on.
2. **Supersonic C_Nα.** ASAT is *below* BRL above Mach 2, so correcting it
   would increase drift — the wrong direction.
3. **Axial moment of inertia.** Drift ∝ Ix; the BRL and ASAT determinations
   differ by 2.1 %.
4. **The linear yaw-of-repose model.** This 6-DOF has no C_Nα̇ or C_Nq term.

**What evidence would resolve it:** a digitisation of BRL Figure 8 (C_Nα versus
Mach) and Figure 10 (C_Mα versus Mach) for the full-scale rounds, which would
give a measured curve across the whole Mach range instead of three rounds at a
single Mach; or the ASAT-13 paper's own force and moment expansion, which
would also settle the reduced-rate convention. Direct retrieval of ASAT-13 was
attempted repeatedly and failed — the publisher's host
(`asat.journals.ekb.eg`) is unreachable, not merely rate-limiting.
