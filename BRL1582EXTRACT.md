# BRL MR-1582 — Extracted Data & Consistency Tests

Source: Karpov & Schmidt (rev. Krial & MacAllister), *The Aerodynamic
Properties of the 155-mm Shell M101 from Free Flight Range Tests of Full Scale
and 1/12 Scale Models*, BRL Memorandum Report No. 1582, Ballistic Research
Laboratories, Aberdeen Proving Ground, June 1964. DTIC AD0454925.

Transcribed from the scanned report. Tabulated values are read directly from
Tables I–III and are high confidence apart from noted OCR ambiguity. Figure
readings are approximate and flagged as such.

---

## 1. Sign and reference conventions — settled, from Appendix I

The report states these explicitly. Quote them in `aerodata.py`; they remove
every convention ambiguity you were reasoning around from geometry.

> (1) A positive `C_Nα` yields a normal force in the direction of the total
> angle of attack.
> (2) A positive `C_Npα` yields a Magnus force at 90° to the normal force in
> the direction of spin.
> **(3) A positive `C_Mα` yields a moment which increases the total angle of
> attack.**
> (4) A positive `C_Mpα` yields a moment which turns the missile nose about
> the flight path in the direction of spin.
> (5) A positive `C_Mq` yields a moment which increases the steady angular
> velocity.
> (6) A positive `C_Mα̇` yields a moment which increases the unsteady angular
> velocity.

> "In this report ℓ is the maximum body diameter, d, and S is the maximum
> cross-sectional area, πd²/4."

**Item (3) is decisive.** Positive `C_Mα` increases angle of attack — i.e.
destabilising, centre of pressure ahead of the CG. That is exactly the
convention `SIXDOF-SPEC.md` assumes. No flip needed, and now you have a
citable sentence rather than a geometric inference.

Reference length `d`, reference area `πd²/4`. Confirms your assumption.

Coordinate system: X along the axis of symmetry, Z̃ down, Ỹ by right-hand
rule. Positive angle of attack is nose-up; positive sideslip is nose-left
viewed from behind.

---

## 2. Physical properties — Table I

| Type | Mass (lb) | CG from nose (cal) | k₁⁻² (cal) | k₂⁻² (cal) | Length (cal) | Dia (mm) |
|---|---|---|---|---|---|---|
| 155-mm M101 | 95.8 | 2.96 | 7.1 | 0.81 | 4.5 | 155 |
| **155-mm M107** | **95.2** | **2.96** | **7.1** | **0.81** | **4.5** | **155** |
| Semiscaled Model 1 | 0.097 | 2.80 | 9.2 | 0.92 | 4.5 | 12.7 |
| Semiscaled Model 2 | 0.068 | 3.20 | 8.0 | 1.20 | 4.5 | 12.7 |
| Exact Scale | 0.095 | 2.84 | 9.2 | 0.92 | 4.5 | 12.7 |

Per the symbol table, `k₁⁻² = md²/Iₓ` and `k₂⁻² = md²/I_y`.

**Worked conversion for the M107:**

```
m   = 95.2 lb = 43.18 kg
d   = 0.155 m,  d² = 0.024025 m²
md² = 1.0374 kg·m²

Ix = md² / 7.1  = 0.1461 kg·m²
Iy = md² / 0.81 = 1.2807 kg·m²
```

> **⚠ Check this against what you used.** `Ix = 0.146` agrees well with the
> commonly cited M107 value (~0.1444). But `Iy = 1.281` is well below the
> figure usually quoted for a 155 mm HE shell (~1.79). Either the report's
> `k₂⁻²` differs from modern published values, or the scanned `0.81` is
> actually `0.61` (which would give `Iy = 1.70`). The digit is legible but the
> scan is poor.
>
> This matters directly for your drift problem. Yaw of repose scales roughly
> as `Ix·p / C_Mα`, and drift as `∫ C_Nα · δ_R`. Mixing inertias from one
> source with coefficients from another is exactly the kind of quiet
> inconsistency that produces an 11% drift error. **Confirm which source your
> `Ix`/`Iy` came from, and make them consistent with the coefficient source.**

---

## 3. Table III — 155-mm M107, full scale

The only full-scale M107 data in the report. Three rounds, all near Mach 0.79.

| Rd | M | δ̄² | C_D | C_Mα | C_Nα | C_Mq+C_Mα̇ | C_Mpα |
|---|---|---|---|---|---|---|---|
| 4816 | 0.784 | 1.5 | 0.1575 | 3.84 | 1.61 | — | −0.36 |
| 4818 | 0.786 | 6.4 | 0.1477 | 3.51 | 1.62 | −9.7 | −0.36 |
| 4819 | 0.791 | 2.3 | 0.1413 | 3.74 | 1.57 | −9.9 | −0.38 |

Average statistical errors: ε_CD 0.0010, ε_CMα 0.05, ε_CNα 0.08,
ε_(CMq+CMα̇) 2.0, ε_CMpα 0.18.

**Everything else in the report is M101 or scaled models.** Say this plainly
in your writeup.

---

## 4. Table II — 155-mm M101 prototype, full scale

`C_D` here is total drag and **includes yaw drag**. To recover `C_D0`,
subtract `C_Dδ² · δ̄²`. Rounds with small `δ̄²` are closest to zero-yaw.

| Rd | M | δ̄² | C_D | C_Mα | C_Nα | C_Mq+C_Mα̇ | C_Mpα |
|---|---|---|---|---|---|---|---|
| 1795 | 0.570 | 5.5 | 0.1393 | 3.28 | 1.81 | −8.9 | −0.15 |
| 1686 | 0.615 | | 0.1329 | | | | |
| 1123 | 0.622 | | 0.1206 | | | | |
| 1793 | 0.646 | 4.2 | 0.1362 | 3.36 | | −8.7 | −0.18 |
| 1794 | 0.646 | 0.5 | 0.1299 | | | | |
| 1684 | 0.649 | | 0.1314 | | | | |
| 1685 | 0.653 | | 0.1337 | | | | |
| 1121 | 0.654 | 2.5 | 0.1372 | 3.41 | | −9.9 | 0.02 |
| 1682 | 0.685 | 3.2 | 0.1355 | 3.51 | | −13.5 | 0.02 |
| 1683 | 0.687 | | 0.1311 | | | | |
| 1116 | 0.709 | | 0.1294 | | | | |
| 1681 | 0.729 | | 0.1304 | | | | |
| 1115 | 0.732 | | 0.1294 | | | | |
| 1680 | 0.736 | | 0.1370 | | | | |
| 1113 | 0.747 | | 0.1266 | | | | |
| 1678 | 0.765 | 7.9 | 0.1451 | 3.58 | 1.71 | −9.2 | |
| 1679 | 0.767 | 1.6 | 0.1314 | 3.41 | | | |
| 1125 | 0.778 | | 0.1245 | | | | |
| 1114 | 0.798 | | 0.1230 | | | | |
| 4820 | 0.809 | 10.1 | 0.1408 | 3.59 | 1.68 | −5.4 | 0.05 |
| 4821 | 0.811 | 6.6 | 0.1542 | 3.60 | 1.65 | −7.6 | 0.15 |
| 1112 | 0.815 | 4.4 | 0.1267 | 3.62 | | −9.9 | 0.31 |
| 1074 | 0.817 | 7.0 | 0.1555 | 3.61 | 1.66 | −5.6 | 0 |
| 1791 | 0.867 | 6.9 | 0.1456 | 3.76 | 1.78 | −7.9 | 0.05 |
| 1792 | 0.869 | 3.8 | 0.1383 | 3.84 | | −13.2 | 0 |
| 4822 | 0.879 | 11.4 | 0.1571 | 3.81 | 1.45 | −7.6 | 0.48 |
| 1126 | 0.885 | | 0.1570 | | | | |
| 1111 | 0.886 | 5.5 | 0.1400 | 3.86 | | −9.2 | −0.15 |
| 1110 | 0.928 | 1.6 | 0.1698 | 4.33 | | | |
| 1075 | 0.934 | 2.9 | 0.1816 | 4.26 | | −14.5 | −0.08 |
| 1797 | 0.947 | 3.7 | 0.1986 | 4.10 | | −21.9 | 0.25 |
| 1109 | 0.950 | 1.5 | 0.2065 | 4.39 | | | |
| 1796 | 0.950 | 6.7 | 0.2068 | 4.06 | | −10.7 | 0.51 |
| 1072 | 0.961 | 3.0 | 0.2091 | 4.19 | | −14.8 | 0.33 |
| 866 | 0.969 | 1.4 | 0.2170 | 4.54 | | | |
| 867 | 0.973 | 0.5 | 0.2256 | | | | |
| 1078 | 0.976 | 6.0 | 0.2072 | 3.97 | | −10.9 | 0.41 |
| 1079 | 0.998 | 3.4 | 0.3484 | 3.67 | | −13.7 | 0.46 |
| 992 | 1.014 | 53.1 | 0.4507 | 3.56 | 2.44 | −4.1 | −0.02 |
| 864 | 1.056 | | 0.3845 | | | | |
| 865 | 1.057 | | 0.3899 | | | | |
| 1799 | 1.099 | 2.5 | 0.3914 | | | −7.4 | 0.05 |
| 863 | 1.159 | | 0.3868 | | | | |
| 862 | 1.162 | 2.4 | 0.3846 | 3.49 | | | |
| 1562 | 1.182 | 7.4 | 0.4064 | 3.41 | 2.50 | −5.1 | −0.05 |
| 1561 | 1.185 | 16.3 | 0.4270 | 3.48 | 2.52 | −7.1 | 0.08 |
| 1560 | 1.192 | 1.5 | 0.3855 | 3.75 | 2.70 | | |
| 1106 | 1.249 | 0.2 | 0.3672 | | | | |
| 860 | 1.274 | 0.2 | 0.3687 | | | | |
| 861 | 1.279 | 0.8 | 0.3664 | | | | |
| 1800 | 1.303 | 5.5 | 0.3754 | 3.57 | | −8.1 | 0.58 |
| 1801 | 1.307 | 6.6 | 0.3704 | | | −7.1 | 0.10 |
| 858 | 1.433 | 1.4 | 0.3540 | 3.45 | | | |
| 1802 | 1.396 | 9.0 | 0.3598 | 3.30 | 2.72 | −7.9 | 0.15 |
| 990 | 1.399 | 5.1 | 0.3551 | 3.40 | 2.55 | −12.2 | |
| 1803 | 1.606 | 3.4 | 0.3400 | 3.35 | 2.67 | −7.6 | 0.28 |
| 991 | 1.613 | 4.7 | 0.3405 | 3.34 | 2.52 | −9.9 | 0.41 |
| 1127 | 1.770 | 4.7 | 0.3239 | 3.45 | 2.62 | −4.8 | 0.08 |
| 989 | 1.934 | | 0.2926 | | | | |
| 1102 | 2.164 | | 0.2748 | | | | |
| 1101 | 2.183 | | 0.2806 | | | | |
| 1556 | 2.190 | 21.0 | 0.3229 | 2.99 | 2.88 | −6.9 | 0.08 |
| 1557 | 2.196 | 10.1 | 0.3050 | 3.00 | 2.98 | −5.4 | −0.05 |
| 1555 | 2.411 | 12.2 | 0.2911 | 2.91 | 3.00 | −6.4 | 0.03 |

Average statistical errors: ε_CD 0.0015, ε_CMα 0.10, ε_CNα 0.10,
ε_(CMq+CMα̇) 2.5, ε_CMpα 0.10.

Rounds 858 and 1802 appear out of Mach order in the scan — likely an OCR
artefact rather than a real ordering.

---

## 5. The centre-of-pressure consistency test

This is the non-fitting way to resolve your drift error. `CP_N` is plotted
independently in Figure 9, and it constrains the *ratio* of the two
coefficients you are unsure about:

```
CP_from_nose  =  x_cg  −  C_Mα / C_Nα        (calibers, x_cg = 2.96)
```

Computed from Table II rows that carry both coefficients:

| M | C_Mα | C_Nα | ratio | CP (cal from nose) |
|---|---|---|---|---|
| 0.570 | 3.28 | 1.81 | 1.81 | 1.15 |
| 0.765 | 3.58 | 1.71 | 2.09 | 0.87 |
| 0.809 | 3.59 | 1.68 | 2.14 | 0.82 |
| 0.811 | 3.60 | 1.65 | 2.18 | 0.78 |
| 0.817 | 3.61 | 1.66 | 2.17 | 0.79 |
| 0.867 | 3.76 | 1.78 | 2.11 | 0.85 |
| 0.879 | 3.81 | 1.45 | 2.63 | 0.33 ⚠ δ̄²=11.4 |
| 1.182 | 3.41 | 2.50 | 1.36 | 1.60 |
| 1.185 | 3.48 | 2.52 | 1.38 | 1.58 |
| 1.192 | 3.75 | 2.70 | 1.39 | 1.57 |
| 1.396 | 3.30 | 2.72 | 1.21 | 1.75 |
| 1.399 | 3.40 | 2.55 | 1.33 | 1.63 |
| 1.606 | 3.35 | 2.67 | 1.25 | 1.71 |
| 1.613 | 3.34 | 2.52 | 1.33 | 1.63 |
| 1.770 | 3.45 | 2.62 | 1.32 | 1.64 |
| 2.190 | 2.99 | 2.88 | 1.04 | 1.92 |
| 2.196 | 3.00 | 2.98 | 1.01 | 1.95 |
| 2.411 | 2.91 | 3.00 | 0.97 | 1.99 |

M107 (Table III): M 0.784 → CP 0.58; M 0.786 → 0.79; M 0.791 → 0.58.

These reproduce the shape of Figure 9 — CP around 0.8–1.1 cal subsonic,
climbing through ~1.6 in the low supersonic and reaching ~1.95 by Mach 2.4.

**How to use it.** Run the same computation on your `aerodata.py` table. If
your CP curve sits above BRL's, your `C_Nα` is too high relative to `C_Mα`;
below, too low. The residual is a function of Mach, so it tells you *where*
the table is wrong — which is precisely the discrimination a uniform rescale
cannot provide, and it is why your instinct that a flat 11% correction was
unjustified was right.

Nothing here is fitted to drift. It is a third independently measured quantity.

---

## 6. What the tabulated C_Nα actually does with Mach

| Mach band | C_Nα | Source |
|---|---|---|
| 0.57 | 1.81 | M101 |
| 0.76–0.88 | 1.65–1.78 | M101 |
| **0.78–0.79** | **1.57–1.62** | **M107** |
| 1.18–1.19 | 2.50–2.70 | M101 |
| 1.40 | 2.55–2.72 | M101 |
| 1.61 | 2.52–2.67 | M101 |
| 1.77 | 2.62 | M101 |
| 2.19–2.20 | 2.88–2.98 | M101 |
| 2.41 | 3.00 | M101 |

Roughly 1.6–1.8 subsonic, a sharp transonic rise, then 2.5–3.0 supersonic.
That is a ~75% variation across the flight envelope — strongly Mach-dependent,
which is why "BRL's other Mach points contradict a uniform rescale" was the
correct read.

Note the M107 runs ~5% below the M101 at the same Mach. Small, but if you are
chasing an 11% discrepancy it is not negligible, and it is the only direct
M107-vs-M101 comparison the report offers.

The M 1.014 row (`C_Nα` 2.44) has `δ̄²` = 53.1 — a very large yaw. Weight it
lightly or drop it.

---

## 7. Magnus moment sign change

Tabulated `C_Mpα` for the M101 runs from about −0.18 at M 0.57 through zero
near M 0.65–0.82, then positive and scattered (0.05 to 0.58) transonic, and
small positive (−0.05 to 0.08) above M 2.

The M107 rows are all **−0.36 to −0.38** at M ≈ 0.79, where the M101 sits near
zero to slightly positive. That is a real difference between the two shells at
the same Mach, and it is larger than the stated 0.18 error bar.

Figures 11 and 12 show the sign change clearly. Two points:

- Confirm your table reproduces a sign change near Mach 1 rather than a
  monotonic curve.
- **Figure 11 plots two curves, for CG at 2.8 and 3.2 cal from the nose.**
  Magnus moment *is* CG-dependent, unlike `C_Mα` whose station happened to
  coincide with your CG. If you took `C_Mpα` from a source with a different
  reference station, that one does need transferring.

---

## 8. Approximate figure readings

Only for shape where the tables are sparse. The scans are rotated and
degraded; treat these as ±10% and digitise properly if you rely on them.

**Fig 7 — C_D0 vs Mach:** ~0.13 at M 0.6, ~0.15 at 0.85, rising steeply from
M 0.95, peak ~0.39–0.40 around M 1.15–1.20, then 0.37 at 1.3, 0.35 at 1.45,
0.34 at 1.6, 0.32 at 1.8, 0.29 at 2.0, 0.28 at 2.2, 0.27 at 2.4.

**Fig 8 — C_Nα vs Mach:** ~1.8 at 0.6, ~1.85 at 0.9, ~2.25 at 1.0, ~2.5 at
1.2, ~2.6 at 1.4, ~2.7 at 1.8, ~2.8 at 2.0, ~2.9 at 2.4.

**Fig 9 — CP_N vs Mach:** ~0.6–0.9 subsonic with a dip near M 0.85, ~1.2–1.5
at M 1.0, ~1.6 at 1.2, ~1.65 at 1.6, ~1.8 at 2.0, ~1.9 at 2.4.

**Fig 10 — C_Mα vs Mach:** ~3.3 at 0.6, rising to a peak ~4.3–4.5 near M
0.93–0.97, then falling: 3.7 at 1.05, 3.5 at 1.2, 3.4 at 1.5, 3.4 at 1.8,
3.0 at 2.2, 2.9 at 2.4.

---

## 9. Caveats for the writeup

1. This is **M101/M107 aerodynamics, 1964**. Full-scale M107 data is three
   rounds near Mach 0.79. Everything supersonic is M101 or scaled models.
   Your problem statement targets a modern shell with different ogive and
   boat-tail geometry. State this before a judge finds it.
2. The report's abstract notes that model and full-scale data agree at
   supersonic speeds but **differ at transonic and subsonic** — so prefer
   full-scale rows below Mach ~1.2.
3. The `Iy` question in §2 is unresolved and touches your drift directly.
4. Original 1964 printing carries a distribution restriction. The report is
   now in DTIC's public collection as AD0454925 and is widely cited in the
   open literature; check the current DTIC distribution statement before
   formally citing it.
