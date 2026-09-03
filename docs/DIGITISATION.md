# Digitisation of BRL MR-1582 Figures 8 and 10

**Outcome: ATTEMPTED AND REJECTED.** No figure trace feeds the flight model.
The measured C_Nα correction that was adopted is built from the **tabulated
rows** of BRL Tables II and III, which are the same data at higher precision
and with the series labelled. This document records the method, the evidence
that the trace cannot be defended, and why the tables are the better source
anyway.

---

## 1. Page map — corrected

The review's figure page numbers are off by one throughout. Verified by
rendering each page at 300 dpi and reading its printed title:

| PDF page | Actual figure | Title on the page |
|---|---|---|
| 41 | Figure 7 | zero-yaw drag coefficient vs Mach |
| **42** | **Figure 8** | **NORMAL FORCE COEFFICIENT vs MACH NUMBER** |
| 43 | Figure 9 | CENTER OF PRESSURE OF THE NORMAL FORCE vs MACH NUMBER |
| **44** | **Figure 10** | **OVERTURNING MOMENT COEFFICIENT vs MACH NUMBER** |
| 45 | Figure 11 | Magnus moment, semi-scaled model |
| 46 | Figure 12 | Magnus moment, 155-mm M101 |

(The review listed Fig 7 at page 42, Fig 8 at 43, Fig 9 at 44, Fig 10 at 45.)
All figure pages are rotated 90° in the PDF.

## 2. Method

- Rendered with PyMuPDF at **600 dpi**, greyscale, with a 90° pre-rotation
  applied in the render matrix (de-skew before any tracing).
  Figure 8 → 6464 × 5120 px; Figure 10 → 6462 × 5120 px.
- Binarised at grey < 150.
- Gridlines located by row/column ink profiles taken in bands clear of the
  title and legend.
- Marker candidates found by connected-component labelling, filtered on
  bounding-box size and aspect ratio, and classified open-vs-filled by the
  ratio of component area to filled-hole area.

### Axis calibration

Both figures are on the same graph paper: fine rules every **126 px**, heavy
rules every four fine cells = **504 px**. Eleven heavy vertical rules bound
ten cells, and the ten x-axis labels (0.6 … 2.4 in steps of 0.2) are centred
**in the cells, not on the rules** — confirmed by cropping and reading the
label strip directly.

| | Figure 8 | Figure 10 |
|---|---|---|
| heavy x rules (px) | 1011, 1532, 2047, 2560, 3072, 3585, 4105, 4631, 5160, 5694, 6228 | 950, 1466, 1980, 2490, 3005, 3523, 4040, 4562, 5092, 5622, 6155 |
| px per Mach | 2605 | 2600 |
| y calibration | heavy rules at 817, 1310, 1805, 2308, 2808, 3321, 3833, 4341 px = values 6, 5, 4, 3, 2, 1, 0, −1 → **C_Nα = (3837.4 − y)/503.4** | labels 4.8, 4.0, 3.2, 2.4; 504 px per 0.8 units → **630 px per unit** |

### Legends, read from the pages

Figure 8: open circle = 155-mm M101 full scale; filled circle = 155-mm M107
full scale; open triangle = semi-scaled model; filled triangle = exact-scaled
model; plus an unlabelled solid faired curve.

Figure 10 — **the centre-of-gravity trap is real and worse than warned**:

| Marker | Series | CG station |
|---|---|---|
| open circle | 155-mm M101 | **2.96 cal** |
| filled circle | 155-mm M107 | **2.96 cal** |
| open triangle | semi-scaled model | 2.80 cal |
| filled triangle | exact-scaled model | 2.84 cal |
| dashed line | MODEL | 2.96 cal |
| × | semi-scaled model | 3.20 cal |

C_Mα is station-dependent, so only the two circle series are comparable with
Tables II/III and with this project's table. Note in particular that the
**dashed faired curve is model data**, not full-scale — fairing to it would be
exactly the pooling error to avoid. (Figure 9's dashed curve is likewise
labelled "C_Mα vs CM values of semi-scaled model".)

---

## 3. Why the extraction was rejected

### Quantitative evidence

Connected-component analysis inside the plot frame:

| | Figure 8 | Figure 10 |
|---|---|---|
| total components | 13 245 | 7 762 |
| components exceeding 140 px in a dimension | 166 | 76 |
| **share of plot-area ink in those oversized components** | **65.6 %** | **71.8 %** |
| largest single component | 1786 × 5207 px | 1232 × 5163 px |

Two thirds of the ink in each plot belongs to components spanning thousands of
pixels. The graph-paper rules, the faired curves and the markers are fused
into a handful of giant masses. The largest component in Figure 8 is
essentially the whole plot area.

The open/filled classifier confirmed it independently: of 31 marker candidates
in Figure 8, **0** were classified as open (fill fraction < 0.75) — the open
circles' outlines are broken and merge with the rules, so hole-filling closes
nothing and every glyph reads as solid.

### Visual confirmation

At 600 dpi, magnified:

- **Mach 0.74–0.92 in Figure 8** — three open circles are individually
  readable, but the filled glyphs form a single black mass that also merges
  with the faired curve and a heavy rule. **The three full-scale M107 rounds
  live in this blob.** The points most wanted are precisely the ones the scan
  cannot deliver.
- **Mach 2.0–2.5 in Figure 8** — the fine rules are dense enough that markers,
  rules and the faired curve are continuous ink.
- **Mach 0.75–1.05 in Figure 10** — six series overlap through the transonic
  peak.

### Self-validation checks: both unrunnable as specified

**(a) Value check** — cannot be run per row. It requires assigning each
extracted marker to a series before comparing with the tabulated value, and
series assignment is exactly what fails. Any comparison would be a guess
dressed as a residual.

**(b) Count check** — cannot discriminate. Tables II and III together carry
**22** full-scale rows with C_Nα (19 M101 + 3 M107) and **41** with C_Mα
(38 + 3). Figure 8 yielded 31 candidates and Figure 10 forty, but those counts
mix all four series *and* undercount the merged blobs, which the size filter
silently discards. The counts are neither close enough nor clean enough to
demonstrate separation.

The task's own instruction applies: *"If the scan does not support separating
the glyphs at any Mach band, say which bands and stop trying to extract
those."* The bands are listed above; the extraction stops.

---

## 4. Stated read precision

For an **isolated** marker, the centre can be located to about ±15 px:

| | precision |
|---|---|
| C_Nα | **±0.03** |
| C_Mα | **±0.024** |

Glyphs are 55–80 px across (0.11–0.16 C_Nα units), so any overlap is
unresolvable.

**That precision is beside the point.** The rows these figures plot carry
BRL's own stated per-round standard errors:

| | Table II | Table III |
|---|---|---|
| ε(C_Nα) | 0.10 | 0.08 |
| ε(C_Mα) | 0.10 | 0.05 |

The *measurement* uncertainty is two to three times the *reading* uncertainty.
A figure trace can therefore never improve on the table — it can only add
reading error and lose the series label that the table states explicitly.

---

## 5. What was used instead

The measured C_Nα correction in `sim/aerodata.py` is built from the tabulated
rows in `analysis/brl_reference.py` — 19 M101 rows over Mach 0.57–2.41 plus
3 M107 rows at Mach 0.784–0.791, yaw-filtered at 25 deg², grouped into the six
Mach clusters the firing programme produced. See `docs/COEFFICIENTS.md` §8.

This is the same data the figures plot, with three advantages the figures
cannot offer: three significant figures instead of ±0.03, the series named
rather than inferred from a glyph, and a stated per-round error.

**C_Mα was not corrected.** The same 21 rows give measured/ASAT = 1.0186 with
sd 0.0386 and no Mach trend — measurement *confirms* the computed overturning
moment, and changing it would be fitting noise.

---

## 6. One thing the figure work did settle

Reading the pages resolved two things the tables alone could not:

1. **The Figure 10 CG-station legend.** Four of the six series are at 2.80,
   2.84 or 3.20 cal, and the faired curve is model data. Anyone digitising
   this figure without reading that legend would silently mix stations.
2. **The page map**, which was off by one in the review.

Both are recorded in `analysis/brl_figures.py` so the next person does not
repeat the attempt.
