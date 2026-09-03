"""
BRL MR-1582 free-flight measurements, transcribed from the scanned report.

Source: B. G. Karpov & L. E. Schmidt (rev. K. Krial, L. C. MacAllister),
"The Aerodynamic Properties of the 155-mm Shell M101 from Free Flight Range
Tests of Full Scale and 1/12 Scale Models", BRL Memorandum Report No. 1582,
Aberdeen Proving Ground, June 1964 (DTIC AD0454925).

TRANSCRIPTION PROVENANCE
------------------------
Every value below was read from the PDF page images rendered at 900-2400 dpi,
not from the OCR text layer. The OCR of this scan confuses 3/5 and 2/8
routinely and cannot be trusted for digits.

Three values in this file differ from the external review document
BRL1582EXTRACT.md, and in each case the high-resolution image was decisive:

  * Table III C_Mpalpha reads -.56, -.56, -.58, NOT -0.36/-0.36/-0.38.
    The glyph is the same "5" that appears in ".1575" and "3.51" on the same
    rows; a "3" in this typeface has an open left side and no closed counter.
    This matters: the review used -0.36 to claim 2% agreement with ASAT's
    -0.355. The true figure is about 58% apart.

  * Table I M107 mass reads 95.8 lb, NOT 95.2. At 2400 dpi the final glyph
    has two stacked closed counters (an 8); a "2" has no closed counter at
    all. Both the M101 and M107 rows carry the same value.

  * Table II rounds 1802 and 990 are at Mach 1.596 and 1.599, NOT 1.396 and
    1.399. Confirmed both by the glyph and by the report's own statement that
    rounds are "numbered in order of increasing Mach number" -- 1.396 would
    place them before round 858 at 1.435, which is what produced the review's
    "appears out of Mach order" note.

One value in this file was corrected in the other direction: round 1678 has
C_Malpha = 3.58, which the review had right and an earlier low-resolution
reading of this project had wrong (3.30).

CONVENTIONS (BRL Appendix I, quoted verbatim in docs/COEFFICIENTS.md)
    reference length d, reference area pi d^2/4, (1/2) rho V^2 S normalisation
    positive C_Nalpha  -> normal force along the total angle of attack
    positive C_Malpha  -> moment which INCREASES the total angle of attack
    reduced spin and transverse rate are pd/V and qd/V (NOT pd/2V)
"""

from __future__ import annotations

__all__ = [
    "M107_TABLE_III",
    "M101_TABLE_II_PAIRS",
    "CG_CALIBERS",
    "cp_from_nose",
    "brl_cnalpha_pairs",
]

#: Centre of gravity, calibers from the nose. BRL Table I, both M101 and M107.
CG_CALIBERS = 2.96

# --- Table III: the only full-scale M107 data in the report ---------------
# (Mach, delta_bar^2 [deg^2], C_D, C_Malpha, C_Nalpha, C_Mq+C_Malphadot, C_Mpalpha)
M107_TABLE_III = [
    (0.784, 1.5, 0.1575, 3.84, 1.61, None, -0.56),
    (0.786, 6.4, 0.1477, 3.51, 1.62, -9.7, -0.56),
    (0.791, 2.3, 0.1413, 3.74, 1.57, -9.9, -0.58),
]

# --- Table II: full-scale M101 rows that carry BOTH C_Malpha and C_Nalpha --
# (Mach, delta_bar^2 [deg^2], C_Malpha, C_Nalpha)
M101_TABLE_II_PAIRS = [
    (0.570, 5.5, 3.28, 1.81),
    (0.763, 7.9, 3.58, 1.71),
    (0.809, 10.1, 3.59, 1.68),
    (0.811, 6.6, 3.60, 1.65),
    (0.817, 7.0, 3.61, 1.66),
    (0.867, 6.9, 3.76, 1.78),
    (0.879, 11.4, 3.81, 1.45),   # high yaw, weight lightly
    (1.014, 53.1, 3.56, 2.44),   # delta^2 = 53 deg^2; excluded by default
    (1.182, 7.4, 3.41, 2.50),
    (1.185, 16.3, 3.48, 2.52),
    (1.192, 1.5, 3.73, 2.70),
    (1.596, 9.0, 3.50, 2.72),
    (1.599, 5.1, 3.40, 2.55),
    (1.606, 3.4, 3.35, 2.67),
    (1.615, 4.7, 3.34, 2.52),
    (1.770, 4.7, 3.45, 2.62),
    (2.190, 21.0, 2.99, 2.88),
    (2.195, 10.1, 3.00, 2.98),
    (2.411, 12.2, 2.91, 3.00),
]

#: Rows with a mean squared yaw above this (deg^2) are excluded from fits.
#: BRL itself warns that the large-yaw rows carry nonlinear contamination.
MAX_YAW_SQ_FOR_FIT = 25.0


def cp_from_nose(C_Malpha: float, C_Nalpha: float, cg_cal: float = CG_CALIBERS) -> float:
    """
    Centre of pressure of the normal force, calibers aft of the nose.

        C_Malpha|CG = C_Nalpha * (x_cg - x_cp)/d      [positive = destabilising]
    =>  x_cp/d      = x_cg/d - C_Malpha/C_Nalpha

    BRL plots this quantity independently in its Figure 9, so it is a third
    measured quantity that constrains the RATIO of the two coefficients.
    """
    return cg_cal - C_Malpha / C_Nalpha


def brl_cnalpha_pairs(include_m101: bool = True, max_yaw_sq: float = MAX_YAW_SQ_FOR_FIT):
    """
    All (Mach, C_Nalpha, C_Malpha, source) rows usable for a fit, M107 first.
    """
    out = []
    for mach, yaw2, _cd, cma, cna, _cmq, _cmpa in M107_TABLE_III:
        out.append((mach, cna, cma, "M107"))
    if include_m101:
        for mach, yaw2, cma, cna in M101_TABLE_II_PAIRS:
            if yaw2 <= max_yaw_sq:
                out.append((mach, cna, cma, "M101"))
    out.sort()
    return out
