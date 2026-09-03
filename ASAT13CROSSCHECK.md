# ASAT-13 × BRL-1582 Cross-Check

Khalil, Abdalla & Kamal, *Dispersion Analysis for Spinning Artillery
Projectile*, ASAT-13-FM-03, 13th Int. Conf. on Aerospace Sciences & Aviation
Technology, Military Technical College, Cairo, 26–28 May 2009.

Read alongside `BRL1582-EXTRACT.md`. This locates the 11% drift discrepancy
and flags a probable second error.

---

## 0. Which source wins, and why

**ASAT Table 1 is computed, not measured.** The paper states the coefficients
"are computed using the analytical capability of the SPINNER-98 code". Mass
properties come from Inventor and PRODAS.

**BRL Table III is measured** — free-flight spark-range firings of actual
full-scale M107 rounds.

Where both exist, measurement outranks computation. That is a principled
ordering, not a preference tuned to make drift match — which matters, because
it is exactly what resolves your open item.

---

## 1. The 11% is real, located, and subsonic-only

At Mach ≈ 0.8:

| Source | C_Nα | Nature |
|---|---|---|
| ASAT Table 1 (M 0.80) | 1.783 | SPINNER-98 computed |
| BRL Table III (M 0.784) | 1.61 | measured, full-scale M107 |
| BRL Table III (M 0.786) | 1.62 | measured, full-scale M107 |
| BRL Table III (M 0.791) | 1.57 | measured, full-scale M107 |
| **BRL mean** | **1.60** | |

**1.783 / 1.60 = 1.114 — ASAT runs 11.4% high.** That is your 11%, and it now
has a cause rather than a coincidence.

At the other end:

| Source | Mach | C_Nα |
|---|---|---|
| ASAT | 2.00 | 2.747 |
| BRL M101 | 2.190 | 2.88 |
| BRL M101 | 2.196 | 2.98 |
| BRL M101 | 2.411 | 3.00 |

Interpolating BRL back to M 2.00 gives ≈ 2.80 against ASAT's 2.747 — about 2%
apart. **The two sources agree supersonically and diverge subsonically.**

That is precisely why a uniform rescale failed and why refusing to apply one
was correct. SPINNER-98 appears to overestimate subsonic normal force on this
geometry; the supersonic branch is fine.

**The non-fitting fix:** use BRL's measured M107 `C_Nα` below Mach ≈ 0.9,
ASAT's above, with a taper across the transonic where neither has full-scale
M107 data. Document it as "measured where measured data exists".

The subsonic branch is also where drift accumulates — long time, low speed,
terminal third of the trajectory — so an 11% subsonic error producing an 11%
drift error is dimensionally sensible.

### The CP test corroborates independently

CG = 0.459 m / 0.155 = **2.961 cal**, matching BRL's 2.96 exactly.

| At M ≈ 0.8 | C_mα | C_Nα | CP from nose (cal) |
|---|---|---|---|
| ASAT | 3.571 | 1.783 | **0.958** |
| BRL M107 mean | 3.70 | 1.60 | **0.649** |

ASAT's centre of pressure sits ~0.31 cal aft of BRL's measured position. Force
`C_Nα` down to BRL's 1.60 and ASAT's CP moves to 0.729 — most of the gap
closes. BRL's own pair is internally consistent with BRL's measured Figure 9.

Three independent quantities, one consistent story. No fitting.

---

## 2. ⚠ Sign convention differs between the two sources

**ASAT Table 1 lists `C_Nα` as NEGATIVE** (−1.763 to −2.747).
**BRL lists it POSITIVE** (1.57 to 3.00).

Same magnitudes, opposite sign. ASAT also lists `C_Ypα` negative throughout
(−0.767 to −1.082) while its `C_mα` is positive — so SPINNER-98's force
convention is not BRL's.

BRL's Appendix I states its convention explicitly (see `BRL1582-EXTRACT.md`
§1) and it matches `SIXDOF-SPEC.md`. **Use BRL's sign convention and take
magnitudes from ASAT.** Your rung 4 gives correct drift direction 15/15, so
you have probably already handled this — but verify it is handled
deliberately rather than by a cancelling pair of errors.

---

## 3. ⚠ Probable second bug: the twist rate

ASAT §4.1 gives, for the same round:

```
V₀ = 684.3 m/s
p₀ = 175.48 rps
```

Therefore

```
684.3 / 175.48 = 3.900 m per revolution
3.900 / 0.155   = 25.16 calibers per revolution
```

**The twist is 1 turn in 25 calibers, not 1 in 20.**

`SIXDOF-SPEC.md` §9 uses 1/20 as the worked example. If that carried into the
build:

```
p₀(1/20) = 684.3 / (20 × 0.155) = 220.74 rps
220.74 / 175.48 = 1.258
```

**Spin would be 25.8% high.** Consequences:

- Drift scales roughly with `Ix·p` → **drift ~26% high**
- `Sg` scales with `p²` → **Sg ~58% high**
- Higher `Sg` means a stiffer round and *smaller* angle of attack

You reported drift **+7 to +27% high**, `Sg` **2.58** at muzzle, and max AoA
**0.76°** — against ASAT's Figure 10, which peaks at **~1.3°** near summit for
the equivalent case. All three symptoms point the same direction.

This is separable from the `C_Nα` issue. Check it first — it is one number.

---

## 4. Independent Sg computation

From ASAT's own stated properties at its own stated muzzle condition:

```
Sg = Ix²p² / (2 ρ Iy S d V² C_Mα)

Ix = 0.144      Iy = 1.216      d = 0.155
S  = πd²/4 = 0.0188692          ρ = 1.225 (sea level)
V  = 684.3      p = 2π × 175.48 = 1102.6 rad/s
C_Mα = 3.013 (ASAT, M 2.00)

numerator   = (0.144 × 1102.6)² = 25 211
denominator = 2(1.225)(1.216)(0.0188692)(0.155)(468 266)(3.013) = 12 294

Sg = 2.05
```

**Expected Sg ≈ 2.05.** That sits comfortably inside BRL's measured range of
1.42–2.27. Your 2.58 is **26% high**.

Sensitivities for locating it: `Sg ∝ p²`, `∝ Ix²`, `∝ 1/Iy`, `∝ 1/C_Mα`.

---

## 5. ✅ The Iy question is resolved

| Source | Ix (kg·m²) | Iy (kg·m²) |
|---|---|---|
| ASAT §4.1 (Inventor + PRODAS) | 0.144 | 1.216 |
| BRL Table I, derived from k₁⁻²=7.1, k₂⁻²=0.81 | 0.1461 | 1.2807 |
| Commonly quoted "155 mm" value | ~0.1444 | ~1.79 |

Two independent sources agree on **Iy ≈ 1.22–1.28**. The scanned `0.81` was
read correctly — it is not `0.61`. The frequently quoted 1.79 does not belong
to this projectile.

Mass also agrees: ASAT 43 kg, BRL 95.2 lb = 43.18 kg. Length: ASAT 698 mm =
4.503 cal, BRL 4.5 cal. CG: both 2.96 cal.

**Use Ix = 0.144, Iy = 1.216.** Withdraw the flag I raised in
`BRL1582-EXTRACT.md` §2.

---

## 6. ✅ Magnus moment cross-validates

ASAT's `C_npα` is tabulated at four yaw angles — **it is strongly nonlinear in
yaw, and changes sign with yaw amplitude, not only with Mach:**

| M | 0° | 2° | 5° | 10° |
|---|---|---|---|---|
| 0.60 | −0.500 | 0.005 | 0.294 | 0.58 |
| 0.80 | −0.355 | 0.078 | 0.366 | 0.65 |
| 0.90 | −0.112 | 0.172 | 0.415 | 0.86 |
| 1.00 | 0.198 | 0.388 | 0.482 | 0.72 |
| 2.00 | 0.388 | 0.431 | 0.438 | 0.51 |

At your operating point (AoA < 1°) the **0° column** is the right one.

Cross-check at M ≈ 0.8, 0° yaw:

| Source | C_Mpα |
|---|---|
| ASAT (M 0.80, 0°) | −0.355 |
| BRL Table III (M 0.784) | −0.36 |
| BRL Table III (M 0.786) | −0.36 |
| BRL Table III (M 0.791) | −0.38 |

**Agreement to 2%.** Independently measured versus independently computed.
This is the strongest validation in either document and worth putting in your
submission.

Note BRL's M101 at the same Mach reads +0.05 to +0.31 — genuinely different
from the M107. The two shells really do differ here, and ASAT's M107 deck
matches BRL's M107 rounds rather than its M101 ones. Good sign for both.

---

## 7. C_A is not C_D

ASAT tabulates `C_A`, "total axial force coefficient" — along the **body
axis**. BRL tabulates `C_D`, "drag force coefficient" — along the **velocity
vector**. Related by `C_D ≈ C_A cos α + C_N sin α`.

At small α they nearly coincide (M 0.8: ASAT `C_A` 0.146; BRL M107 `C_D` mean
0.149), so this is a small effect at your 0.76° AoA. But the spec's force
model takes an **axial** coefficient, which is ASAT's `C_A`, not BRL's `C_D`.
Confirm you have not mixed the two in one table — and note BRL's `C_D`
additionally includes yaw drag, needing `C_Dδ²·δ̄²` subtracted first.

ASAT's `C_Aα2` maps directly onto the spec's `C_X2`.

---

## 8. A fully specified validation case — use this

ASAT §4.3 gives every input and several outputs for one trajectory. Better
than a firing-table comparison, because nothing is left to inference.

**Inputs**
```
θ₀ = 44°,  V₀ = 684.3 m/s,  p₀ = 175.48 rps
m = 43 kg,  d = 0.155 m,  L = 698 mm
CG = 0.459 m from nose,  Ix = 0.144,  Iy = Iz = 1.216 kg·m²
```

**Published outputs**
| Quantity | ASAT value | Source |
|---|---|---|
| Total flight time | **66.67 s** | text |
| Summit time | **~31 s** | text |
| Summit altitude | **~5 700 m** | Fig. 4 |
| Range | **~16 500 m** | Fig. 3 |
| Initial axial deceleration | **4.45 g** | text |
| Max total angle of attack | **~1.3°** at t ≈ 32 s | Fig. 10 |
| Drift direction | right | text |

Add this as rung 5b. Time of flight and summit time are tight, unambiguous
scalars — far better regression targets than a range figure that a form factor
can be tuned to hit.

**Note an internal inconsistency in the paper:** §4.1 and the §4.3 text both
give p₀ = 175 rps, but Figure 9's axis runs 80–120 rps and appears to start
near 117. The tabulated value is stated twice and yields a physically sensible
1/25 twist; the figure axis looks mislabelled. Go with 175.48.

---

## 9. ASAT Table 1 — full transcription

155 mm M107, computed with SPINNER-98. `C_Nα` and `C_Ypα` are **negative in
the source's convention** — convert to BRL/spec convention before use.

| M | C_A | C_Aα² | C_Nα | C_Ypα | C_lp | C_mα | C_mq | C_npα 0° | 2° | 5° | 10° |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | .144 | 2.343 | −1.763 | −0.767 | −.023 | 3.355 | −5.1 | −0.500 | 0.005 | 0.294 | 0.58 |
| 0.60 | .144 | 2.343 | −1.763 | −0.767 | −.023 | 3.378 | −5.1 | −0.500 | 0.005 | 0.294 | 0.58 |
| 0.80 | .146 | 2.847 | −1.783 | −0.767 | −.022 | 3.571 | −5.1 | −0.355 | 0.078 | 0.366 | 0.65 |
| 0.90 | .167 | 3.372 | −1.827 | −0.857 | −.021 | 3.957 | −7.4 | −0.112 | 0.172 | 0.415 | 0.86 |
| 0.95 | .221 | 3.730 | −2.038 | −1.082 | −.020 | 3.886 | −9.9 | 0.085 | 0.292 | 0.500 | 1.12 |
| 1.00 | .327 | 4.180 | −2.153 | −0.992 | −.020 | 3.682 | −13.8 | 0.198 | 0.388 | 0.482 | 0.72 |
| 1.05 | .383 | 4.691 | −2.207 | −0.902 | −.020 | 3.415 | −13.3 | 0.293 | 0.430 | 0.465 | 0.55 |
| 1.10 | .381 | 5.209 | −2.255 | −0.857 | −.019 | 3.384 | −14.6 | 0.334 | 0.432 | 0.456 | 0.54 |
| 1.20 | .370 | 5.702 | −2.325 | −0.767 | −.020 | 3.424 | −15.8 | 0.352 | 0.424 | 0.438 | 0.51 |
| 1.35 | .353 | 5.130 | −2.442 | −0.767 | −.020 | 3.278 | −15.6 | 0.366 | 0.424 | 0.438 | 0.51 |
| 1.50 | .338 | 4.561 | −2.556 | −0.767 | −.020 | 3.264 | −15.3 | 0.373 | 0.424 | 0.438 | 0.51 |
| 1.75 | .314 | 3.970 | −2.692 | −0.767 | −.020 | 3.201 | −15.3 | 0.381 | 0.431 | 0.438 | 0.51 |
| 2.00 | .294 | 3.460 | −2.747 | −0.767 | −.021 | 3.013 | −15.3 | 0.388 | 0.431 | 0.438 | 0.51 |

`C_mq` here is the pitch-damping derivative alone; BRL tabulates the combined
`C_Mq + C_Mα̇`. ASAT −5.1 to −15.3 versus BRL −4.8 to −21.9 — same order, but
they are **not the same quantity**. Don't interleave them.

---

## 10. Action list

1. **Check the twist rate.** 1/25 cal, giving p₀ = 175.48 rps at 684.3 m/s.
   One number; explains drift, Sg and AoA together if it is wrong.
2. **Diff Sg against 2.05** at ASAT's muzzle condition.
3. **Splice C_Nα**: BRL measured below M 0.9, ASAT above, taper transonic.
4. **Confirm the C_Nα sign flip** is handled deliberately.
5. **Adopt Ix = 0.144, Iy = 1.216.**
6. **Add ASAT §4.3 as rung 5b** — TOF 66.67 s, summit 31 s, max AoA 1.3°.
7. **Verify C_A vs C_D** are not interleaved.
8. Keep the `C_npα` 0° column; note the yaw-nonlinearity for the Monte Carlo.
