"""
BRL MR-1582 figure pages: page map, axis calibration, extraction attempt,
and the evidence that the extraction cannot be defended.

CONCLUSION UP FRONT
-------------------
The figure digitisation was ATTEMPTED AND REJECTED. Nothing in this module
feeds the flight model. The measured C_Nalpha correction actually used by
sim/aerodata.py is built from the TABULATED rows in analysis/brl_reference.py,
which are the same data at higher precision and with the series labelled.

Full method and both self-validation checks: docs/DIGITISATION.md.

WHY IT WAS REJECTED, in one line: 66 % (Fig 8) and 72 % (Fig 10) of the ink
inside the plot frame belongs to connected components spanning thousands of
pixels -- the graph-paper rules, the faired curves and the markers are fused
into a handful of giant blobs -- so the four marker series cannot be
separated, and separating them is the whole problem.

PAGE MAP -- CORRECTED
---------------------
The external review gave figure page numbers that are off by one. Verified by
reading each page title at 300 dpi:

    PDF page 41   Figure 7    zero-yaw drag coefficient vs Mach
    PDF page 42   Figure 8    NORMAL FORCE COEFFICIENT vs Mach
    PDF page 43   Figure 9    CENTER OF PRESSURE OF THE NORMAL FORCE vs Mach
    PDF page 44   Figure 10   OVERTURNING MOMENT COEFFICIENT vs Mach
    PDF page 45   Figure 11   Magnus moment, semi-scaled model
    PDF page 46   Figure 12   Magnus moment, 155-mm M101

(The review listed Fig 7 at page 42, Fig 8 at 43, and so on.) All figure pages
are rotated 90 degrees in the PDF and must be de-skewed before tracing.

LEGENDS, READ FROM THE PAGES
----------------------------
Figure 8 (C_Nalpha):
    open circle      155-mm M101      full scale
    filled circle    155-mm M107      full scale
    open triangle    semi-scaled model
    filled triangle  exact-scaled model
    solid line       unlabelled fairing

Figure 10 (C_Malpha) -- NOTE THE CENTRE-OF-GRAVITY TRAP:
    open circle      155-mm M101       CG = 2.96 cal from nose
    filled circle    155-mm M107       CG = 2.96 cal from nose
    open triangle    semi-scaled model CG = 2.80 cal from nose
    filled triangle  exact-scaled model CG = 2.84 cal from nose
    dashed line      MODEL             CG = 2.96 cal from nose
    x                semi-scaled model CG = 3.20 cal from nose

C_Malpha is station-dependent, so only the two circle series are directly
comparable with Tables II and III and with this project's table. The dashed
curve is the MODEL data transferred to 2.96 cal -- it is not full-scale data,
and fairing to it would be exactly the pooling error to avoid.

AXIS CALIBRATION (600 dpi renders, de-skewed 90 degrees)
--------------------------------------------------------
Both figures are on the same graph paper: fine rules every 126 px, heavy rules
every 4 fine cells = 504 px. Eleven heavy vertical rules bound ten cells, and
the ten x-axis labels (.6 through 2.4 in steps of 0.2) are centred IN the
cells, not on the rules -- confirmed by cropping the label strip.

    Figure 8   heavy x rules at px 1011 1532 2047 2560 3072 3585 4105 4631
                                  5160 5694 6228     -> 2605 px per Mach
               heavy y rules at px 817 1310 1805 2308 2808 3321 3833 4341
                                  = values 6 5 4 3 2 1 0 -1
               C_Nalpha = (3837.4 - y) / 503.4

    Figure 10  heavy x rules at px 950 1466 1980 2490 3005 3523 4040 4562
                                  5092 5622 6155     -> 2600 px per Mach
               y labels 4.8 4.0 3.2 2.4, heavy rules 504 px apart
               = 630 px per unit of C_Malpha

READ PRECISION, HONESTLY
------------------------
Where a marker is isolated, its centre can be located to about +-15 px, which
is +-0.03 in C_Nalpha and +-0.024 in C_Malpha. The glyphs are about 55-80 px
across, i.e. 0.11-0.16 C_Nalpha units, so overlapping markers cannot be
resolved at all.

That precision is IRRELEVANT, because the tabulated rows this figure plots
carry BRL's own stated per-round standard errors of 0.10 (Table II C_Nalpha),
0.08 (Table III C_Nalpha) and 0.10 / 0.05 (C_Malpha). The measurement
uncertainty is larger than the reading uncertainty, so a figure trace can
never improve on the table -- it can only lose the series labels.

MACH BANDS THAT CANNOT BE RESOLVED AT ALL
-----------------------------------------
Figure 8:  Mach 0.74-0.92. The filled glyphs fuse into a single black mass
           that also merges with the faired curve and a heavy rule. This band
           contains the three full-scale M107 rounds -- the only full-scale
           M107 C_Nalpha data in the report -- so the points most wanted are
           precisely the ones the scan cannot deliver.
Figure 8:  Mach 2.0-2.5. Dense fine rules merge with the markers and the
           faired curve.
Figure 10: Mach 0.75-1.05. Six series overlap through the transonic peak.

The remaining bands are legible but add nothing the tables do not already
give with the series named.
"""

from __future__ import annotations

__all__ = ["FIGURE_PAGES", "AXIS_CALIBRATION", "READ_PRECISION", "UNRESOLVABLE_BANDS",
           "DAMPING_FORCE_SEMISCALED"]

#: PDF page index (0-based) of each figure, verified by reading page titles.
FIGURE_PAGES = {
    "fig7_CD0": 41,
    "fig8_CNalpha": 42,
    "fig9_CP": 43,
    "fig10_CMalpha": 44,
    "fig11_CMpalpha_model": 45,
    "fig12_CMpalpha_M101": 46,
}

#: Axis calibration at 600 dpi with a 90-degree de-skew.
AXIS_CALIBRATION = {
    "fig8_CNalpha": dict(
        x_heavy_rules_px=[1011, 1532, 2047, 2560, 3072, 3585, 4105, 4631, 5160, 5694, 6228],
        mach_of_first_cell_centre=0.6,
        mach_per_cell=0.2,
        y_value_at_px=lambda y: (3837.4 - y) / 503.4,
        px_per_unit=503.4,
    ),
    "fig10_CMalpha": dict(
        x_heavy_rules_px=[950, 1466, 1980, 2490, 3005, 3523, 4040, 4562, 5092, 5622, 6155],
        mach_of_first_cell_centre=0.6,
        mach_per_cell=0.2,
        px_per_unit=630.0,
    ),
}

#: Stated read precision, coefficient units, for an ISOLATED marker only.
READ_PRECISION = {"C_Nalpha": 0.03, "C_Malpha": 0.024}

#: Mach bands where the marker series cannot be separated on this scan.
UNRESOLVABLE_BANDS = {
    "fig8_CNalpha": [(0.74, 0.92), (2.00, 2.50)],
    "fig10_CMalpha": [(0.75, 1.05)],
}

# ---------------------------------------------------------------------------
# Damping FORCE coefficient -- recorded here because it is the one quantity
# the flight model omits entirely, and because it is a figure-adjacent table
# rather than one of the main coefficient tables.
#
# BRL MR-1582, section 4: "The damping force coefficient was obtained only for
# the semiscaled model by using the damping moment coefficients versus
# center-of-mass relations. The results are listed below for the
# center-of-mass position of the M101."
#
# So (C_Nq + C_Nalphadot) IS tabulated -- but for the SEMI-SCALED MODELS only,
# transferred to the M101 CG station. There is NO full-scale measurement of it
# anywhere in the report, and the report separately warns that the semi-scaled
# model damping data disagree badly with full scale (the damping MOMENT is
# about +1, destabilising, for the models against about -9, stabilising, for
# the full-scale M101 and M107).
#
# Values, (C_Nq + C_Nalphadot) at 2.96 cal from the nose, per radian:
DAMPING_FORCE_SEMISCALED = (
    (0.70, 0.3),
    (0.90, 19.9),
    (1.00, 16.8),
    (1.05, 13.2),
    (1.20, 3.6),
    (2.00, 3.3),
    (2.40, 3.3),
)
