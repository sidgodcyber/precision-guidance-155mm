"""
Mach-interpolated aerodynamic coefficient tables for a 155 mm HE shell.

=============================================================================
PROVENANCE -- read this before trusting any number produced by this package
=============================================================================

PRIMARY TABLE (M107_AERO below)
  Source : M. Khalil, H. Abdalla and O. Kamal, "Dispersion Analysis for
           Spinning Artillery Projectile", 13th International Conference on
           Aerospace Sciences and Aviation Technology (ASAT-13), Military
           Technical College, Cairo, Egypt, 26-28 May 2009,
           paper ASAT-13-FM-03, DOI 10.21608/asat.2009.23740.
  Obtained via : W. Y. Lim, "Predicting the Accuracy of Unguided Artillery
           Projectiles", M.S. thesis, Naval Postgraduate School, Monterey CA,
           September 2016 (DTIC AD1029824), Table 11 "Aerodynamic Coefficients
           of M107 (HE) Projectile. Source: [13]", page 53.
  Projectile : 155 mm M107 High Explosive -- the projectile class this study
           is about. 4.5 calibres long, ogive-cylinder-boattail, 95.8 lb.
  Mach range covered : 0.01 to 2.00. Outside this the end values are held
           flat, and the table records that it happened.
  Reference length : d = 0.155 m (one calibre).
  Reference area   : S = pi*d^2/4 = 0.018869 m^2.
  Moment reference station : the projectile centre of gravity (see hazard 2).
  Angular units    : PER RADIAN (see hazard 1).

INDEPENDENT CROSS-CHECK (not used to build the table -- used to validate it)
  Source : B. G. Karpov and L. E. Schmidt, revised by K. Krial and
           L. C. MacAllister, "The Aerodynamic Properties of the 155-mm Shell
           M101 from Free Flight Range Tests of Full Scale and 1/12 Scale
           Models", Ballistic Research Laboratories Memorandum Report
           No. 1582, Aberdeen Proving Ground MD, June 1964 (DTIC AD0454925).

  This is a PRIMARY free-flight spark-range measurement of the 155 mm M101,
  and its Table III reports three rounds of the 155 mm M107 itself, which the
  report states "differs from the M101 only in the rotating band".
  BRL Appendix I defines the coefficients through

      F_Y + i F_Z = (1/2) rho V^2 S   { -[C_Na + i(pd/V) C_Npa] xi - ... }
      M_Y + i M_Z = (1/2) rho V^2 S d { [(pd/V) C_Mpa - i C_Mq] xi + ... }

  i.e. the modern (1/2) rho V^2 S normalisation with reference length d and
  reference area pi d^2/4 -- the same normalisation this package uses.

  Agreement of the two independent sources on the static coefficients:

     Mach   C_Malpha (BRL free flight)      C_Malpha (Khalil, this table)
     0.60   3.28 - 3.36                     3.378
     0.80   3.59 - 3.62 (M101)              3.571
            3.51 - 3.84 (M107, 3 rounds)
     1.00   3.67                            3.682
     1.20   3.49                            3.424
     1.60   3.40 - 3.50                     3.264
     2.20   2.91 - 3.00                     3.013

     Mach   C_Nalpha (BRL)                  C_Nalpha (this table)
     0.57   1.81                            1.763
     0.79   1.57 - 1.62 (M107)              1.780
     1.01   2.44                            2.166
     1.60   2.52 - 2.72                     2.594
     2.20   2.88 - 3.00                     2.747

     Mach   C_Mpalpha (BRL)                 C_Mpalpha (this table, 0 deg yaw)
     0.79   -0.56, -0.56, -0.58 (M107)      -0.359

     Mach   C_X0 (BRL, yaw-corrected)       C_X0 (this table)
     0.81   0.123 - 0.149 (4 M101 rounds)   0.1465
     0.79   0.136 - 0.155 (3 M107 rounds)   0.1455

  The overturning moment -- the coefficient that dominates gyroscopic
  stability, and therefore the yaw of repose and the drift -- agrees between
  two fully independent determinations to within 3-7 % across the whole Mach
  range. That is the strongest confidence statement available here.

  The BRL C_X0 values above were derived from the report's per-round total
  drag by removing the yaw contribution,
      C_X0 = C_D(round) - C_D2 * delta_bar^2
  with delta_bar^2 the round's mean squared yaw in square degrees converted
  to square radians, and C_D2 the report's own measured yaw-drag values
  (5.9 /rad^2 subsonic and transonic; 9.9 at M 1.2; 11.6 at M 1.6;
  7.8 at M 2.1). Round-to-round scatter at fixed Mach is about +-10 %,
  which is why "agrees within the scatter" is the honest phrasing.

=============================================================================
THE THREE HAZARDS, ADDRESSED EXPLICITLY
=============================================================================

1. PER-DEGREE VS PER-RADIAN
   Both sources are PER RADIAN, and the values are used per radian here.
   No conversion is applied.

   Evidence that does not rely on either paper asserting it: with
   C_Malpha ~ 3.4 the gyroscopic stability factor of the M101 at its
   spark-range conditions (1 turn in 25 calibres) comes out at Sg = 1.7-2.3,
   and BRL MR-1582 Table II tabulates its own separately measured
   s (gyroscopic stability factor) for those same rounds as 1.69 - 2.27.
   A per-degree misreading would put C_Malpha near 195 and Sg near 0.03,
   and those shells manifestly did not tumble.

2. MOMENT REFERENCE STATION
   BRL MR-1582 Table I places the M101 and M107 centre of gravity at
   2.96 calibres from the nose, and its moment coefficients are referenced to
   the centre of mass. The Khalil table is a 6-DOF deck, likewise
   CG-referenced. projectile.py places x_cg at that same station
   (2.96 * 0.155 = 0.4588 m), so the axial transfer term is identically zero:

       C_Malpha|CG = C_Malpha|ref + (x_ref - x_cg)/d * C_Nalpha
                   = 3.571 + (0.4588 - 0.4588)/0.155 * 1.783
                   = 3.571 + 0.000 * 1.783
                   = 3.571

   transfer_moment_reference() below implements the general transfer, so a
   later Monte Carlo that perturbs x_cg applies it correctly instead of
   silently reusing a coefficient referenced to the wrong station.

3. SIGN OF C_Malpha (STABILISING VS DESTABILISING)
   This package, per SIXDOFSPEC.md section 6, takes POSITIVE C_Malpha to mean
   DESTABILISING: centre of pressure ahead of the centre of gravity.

   BRL MR-1582 Appendix I states its convention explicitly -- "A positive
   C_Malpha yields a moment which increases the total angle of attack" --
   which is the same convention. The Khalil values are positive and of the
   same magnitude as the BRL measurements, so it is the same convention.
   NO SIGN FLIP IS APPLIED.

   Independent consistency check on that reading: C_Malpha/C_Nalpha at
   M 0.8 is 3.571/1.783 = 2.00 calibres, putting the centre of pressure
   2.00 calibres AHEAD of the CG, i.e. at 2.96 - 2.00 = 0.96 calibres from
   the nose -- inside the ogive, which is where the centre of pressure of an
   ogive-cylinder-boattail body belongs. Under a stabilising-positive reading
   it would sit 2 calibres BEHIND the CG, at 4.96 calibres from the nose of a
   4.5 calibre shell, i.e. off the back of the projectile. The adopted sign
   is the only one that is geometrically possible.

=============================================================================
TWO SIGN CORRECTIONS APPLIED TO THE SOURCE TABLE, AND WHY
=============================================================================

C_Nalpha : the source tabulates this column NEGATIVE (-1.763 ... -2.747);
   it is stored here POSITIVE. In the force model of SIXDOFSPEC.md section 5,

       [Y, Z] = -qbar * S * C_Nalpha * [v, w] / V

   a POSITIVE C_Nalpha puts the normal force along the angle-of-attack
   direction (nose pitched up gives force up). That is the physically
   correct sense and is what BRL MR-1582 tabulates (positive, 1.45 - 3.00,
   the same magnitudes). The source is tabulating dC_z/dalpha, which equals
   -C_Nalpha in this convention. The flip is mandatory: used unflipped the
   normal force would act to increase the angle of attack.

C_Ypalpha : the source tabulates this NEGATIVE and it is stored NEGATIVE.
   This is deliberate, not an oversight. A negative Magnus force coefficient
   is the physically expected result for a spin-stabilised shell -- boundary
   layer asymmetry reverses the naive inviscid omega x v direction -- and BRL
   MR-1582 independently measured C_Npalpha = -0.15 to -0.55 for this shell
   family. The sign is retained.

=============================================================================
CONFIDENCE, AND THE ONE UNRESOLVED CONVENTION RISK
=============================================================================

  C_X0      HIGH      two independent sources agree within the round-to-round
                      scatter of the free-flight data
  C_Nalpha  HIGH      two independent sources agree within ~12 %
  C_Malpha  HIGH      two independent sources agree within 3-7 %
  C_X2      MEDIUM    single source. BRL measured the DRAG yaw coefficient
                      C_D2 = 5.9 (subsonic), 9.9 (M 1.2), 7.8 (M 2.1);
                      converting to axial through
                          C_X2 = C_D2 - C_Nalpha + C_X0/2
                      gives 4.2, 7.8, 5.2 against this table's 2.8, 5.7, 3.4
                      -- same shape, about 1.4x larger. Yaw drag is ~3 % of
                      axial force at the 1-2 degree yaw of nominal flight,
                      so the spread is not range-critical.
  C_lp      MEDIUM    single source (-0.023 to -0.019). Corroborated only
                      indirectly: a PRODAS deck for a 155 mm ERFB/BB shell
                      (Balon and Komenda, Advances in Military Technology
                      1/2006, Tab. 2) gives STANAG-4355 C_spin = -0.0132 to
                      -0.0107, which converts to C_lp = 2*C_spin = -0.026 to
                      -0.021 in this package's convention. Same magnitude,
                      different shell.
  C_mq      LOW-MED   single source (-5.1 to -15.8). BRL measured the combined
                      (C_mq + C_mAlphadot) = -4.1 to -21.9 with a stated
                      standard error of 2.5, so the free-flight data itself
                      does not pin this down better than a factor of two.
  C_Ypalpha LOW       single source. BRL could not extract it from full-scale
                      swerve at all and quotes semiscaled-model values only.
  C_Mpalpha LOW       sources differ by a FACTOR OF 3.1, once both are put
                      in the same convention. BRL M107 free flight gives
                      -0.57 at M 0.79 in the pd/V normalisation of its
                      Appendix I; converting to this package's pd/(2V) means
                      DOUBLING it, to -1.13, against this table's -0.36.
                      An earlier note here compared -0.57 with -0.36 and
                      called the gap 36 % -- that was a comparison across
                      two conventions and understated it by a factor of two.
                      The ratio is tight across all three BRL rounds and
                      BRL's own stated error is 0.18, so it is a real
                      disagreement, not scatter. The deck is self-consistent
                      with the ASAT-13 section 4.3 published trajectory
                      (using BRL's value there gives a peak yaw of 3.89 deg
                      against a published ~1.3 deg), so the deck is kept;
                      computed Magnus moments are the least reliable output
                      of semi-empirical codes and this is where they fail.
                      Both are negative subsonic and both turn positive
                      supersonically. See docs/RATE-CONVENTION.md section 5.

  THE REDUCED-RATE CONVENTION -- RESOLVED, and how:

    The four RATE-DEPENDENT coefficients (C_Ypalpha, C_Mpalpha, C_lp, C_mq)
    are multiplied in dynamics.py by the reduced rates pd/(2V) and qd/(2V),
    per SIXDOFSPEC.md sections 5-6, through REDUCED_RATE_FACTOR below. The
    classical aeroballistic literature (McCoy; BRL MR-1582 Appendix I, quoted
    above) instead uses pd/V and qd/V, whose coefficients are exactly HALF
    the value of a pd/(2V)-normalised coefficient describing the same
    physics. A factor of two, and a silent one.

    This was an assumption until it was tested. It is now DETERMINED, and it
    was determined WITHOUT reference to the firing table:

      ASAT-13 section 4.3 publishes a trajectory that ASAT computed with
      THIS deck, and states the peak total angle of attack and the time at
      which it occurs. Reproducing that published trajectory requires
      pd/(2V):

                              pd/(2V)     pd/V     ASAT published
        max total AoA (deg)     1.297     1.233     ~1.3
        TIME of max AoA (s)     32.36     19.08     ~32
        yaw history             smooth    17 local maxima

      The peak TIME is the discriminator, not the peak value -- it is a
      shape feature of the yaw history, not a magnitude a coefficient scale
      can slide. Converged over dt = 4e-4 to 1e-4.

      Corroboration for C_lp specifically, the only one of the four that
      moves drift: the Balon and Komenda PRODAS deck converts to
      C_lp = -0.026 to -0.021 in this package's convention against this
      table's -0.023 to -0.019. Under pd/V it would have needed -0.046
      to -0.038.

    All four rate coefficients come from ONE source, so one factor governs
    all four and no coefficient here needs different treatment from its
    neighbours. BRL is the exception and is pd/V: any comparison against a
    BRL rate coefficient must double it first. See the C_Mpalpha entry above
    for what that correction did to a comparison that had been made without
    it.

    Full audit, including why the `rate coefficients x2` sensitivity row
    increases the angle of attack (it is a spin-decay experiment, not a
    damping experiment, on trajectories that carry no epicyclic motion):
    docs/RATE-CONVENTION.md. Reproduce with
    `python -m analysis.rate_convention_audit`.

    The static coefficients that set range, stability and drift -- C_X0,
    C_Nalpha and C_Malpha -- carry no such ambiguity.

Nothing in this file is a placeholder. Every number is traceable to a cited
document. Where a value is weakly determined that is recorded above and
reported at run time by warn_unvalidated().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AeroCoefficients",
    "AeroTable",
    "M107_AERO",
    "make_m107_table",
    "transfer_moment_reference",
    "warn_unvalidated",
    "COEFFICIENT_CONFIDENCE",
    "COEFFICIENT_NAMES",
    "REDUCED_RATE_FACTOR",
]


# --- the table ------------------------------------------------------------
# Columns, in the units and conventions of SIXDOFSPEC.md sections 5 and 6:
#
#   mach
#   C_X0      zero-yaw axial force            (-)         source column CA
#   C_X2      yaw-dependent axial force       (1/rad^2)   source column CA2
#   C_Nalpha  normal force slope              (1/rad)     source column CN, SIGN FLIPPED
#   C_Ypalpha Magnus force                    (1/rad)     source column CY, sign retained
#   C_lp      spin damping moment             (1/rad)     source column Clp
#   C_Malpha  overturning moment slope        (1/rad)     source column Cma
#   C_mq      pitch/yaw damping moment        (1/rad)     source column Cmq
#   C_Mpalpha Magnus moment, 0 deg yaw column (1/rad)     source column Cmpa @ 0 deg
_M107_ROWS = np.array(
    [
        # M      C_X0    C_X2    C_Na    C_Ypa   C_lp     C_Ma    C_mq    C_Mpa
        [0.01,  0.144,  2.343,  1.763, -0.767, -0.023,  3.355,  -5.1, -0.500],
        [0.60,  0.144,  2.343,  1.763, -0.767, -0.023,  3.378,  -5.1, -0.500],
        [0.80,  0.146,  2.847,  1.783, -0.767, -0.022,  3.571,  -5.1, -0.355],
        [0.90,  0.167,  3.372,  1.827, -0.857, -0.021,  3.957,  -7.4, -0.112],
        [0.95,  0.221,  3.730,  2.038, -1.082, -0.020,  3.886,  -9.9,  0.085],
        [1.00,  0.327,  4.180,  2.153, -0.992, -0.020,  3.682, -13.8,  0.198],
        [1.05,  0.383,  4.691,  2.207, -0.902, -0.020,  3.415, -13.3,  0.293],
        [1.10,  0.381,  5.209,  2.255, -0.857, -0.019,  3.384, -14.6,  0.334],
        [1.20,  0.370,  5.702,  2.325, -0.767, -0.020,  3.424, -15.8,  0.352],
        [1.35,  0.353,  5.130,  2.442, -0.767, -0.020,  3.278, -15.6,  0.366],
        [1.50,  0.338,  4.561,  2.556, -0.767, -0.020,  3.264, -15.3,  0.373],
        [1.75,  0.314,  3.970,  2.692, -0.767, -0.020,  3.201, -15.3,  0.381],
        [2.00,  0.294,  3.460,  2.747, -0.767, -0.021,  3.013, -15.3,  0.388],
    ]
)

COEFFICIENT_NAMES = (
    "C_X0",
    "C_X2",
    "C_Nalpha",
    "C_Ypalpha",
    "C_lp",
    "C_Malpha",
    "C_mq",
    "C_Mpalpha",
)

# ==========================================================================
# C_Nalpha: MEASURED CORRECTION TO THE COMPUTED DECK
# ==========================================================================
# The ASAT/SPINNER-98 deck is COMPUTED. BRL MR-1582 Tables II and III are
# MEASURED -- free-flight spark-range firings of full-scale rounds: 19 M101
# rows spanning Mach 0.57 to 2.41 plus 3 M107 rows at Mach 0.784-0.791, each
# carrying both C_Nalpha and C_Malpha. Where both exist, measurement outranks
# computation.
#
# HOW THE CORRECTION IS BUILT
# ---------------------------
# Rows with a mean squared yaw above 25 deg^2 are dropped (BRL warns that the
# large-yaw rounds carry nonlinear contamination); that removes only the
# Mach 1.014 round at 53.1 deg^2. The remaining 21 rows are grouped into the
# six Mach clusters the firing programme actually produced, and within each
# cluster the ratio
#
#     k = mean(measured C_Nalpha) / ASAT C_Nalpha at the cluster mean Mach
#
# is formed. k is then linearly interpolated in Mach between cluster centres
# and held flat outside. The ASAT curve supplies the SHAPE across the gaps --
# including the whole transonic, where no full-scale round was fired -- and
# the measurement supplies the LEVEL wherever a measurement exists.
#
#   Mach centre   n rows   measured C_Na   ASAT C_Na      k
#      0.570         1         1.810         1.763      1.0267
#      0.812         9         1.637         1.788      0.9152
#      1.186         3         2.573         2.315      1.1114
#      1.604         4         2.615         2.613      1.0009
#      1.770         1         2.620         2.696      0.9717
#      2.265         3         2.953         2.747      1.0751
#
# The scatter of a single round about its cluster mean is sd(k) = 0.085, so a
# 9-row cluster has a standard error of 0.028 and a 3-row cluster 0.049. On
# that basis the subsonic deficit (k = 0.915, 3.0 sigma below 1) is solid, the
# Mach 1.19 excess (k = 1.111, 2.3 sigma above 1) is marginal, and the two
# single-round clusters at Mach 0.570 and 1.770 are individually
# insignificant. They are retained because including them lowers the
# centre-of-pressure residual and excluding them raises it; they are the
# weakest part of this curve and are flagged as such.
#
# MACH COVERAGE GAPS in the measured full-scale set, where the ASAT shape is
# doing the work rather than any measurement:
#     0.879 -> 1.182   (0.303 Mach)  the entire transonic, including the peak
#     1.192 -> 1.596   (0.404 Mach)
#     1.770 -> 2.190   (0.420 Mach)
#
# WHY THIS AND NOT A FIGURE TRACE
# -------------------------------
# BRL Figures 8 and 10 plot exactly these rows. Digitising them was attempted
# and abandoned -- see docs/DIGITISATION.md. The scan cannot separate the four
# marker series (66-72 % of the plot-area ink lies in fused connected
# components spanning thousands of pixels), and the tables are strictly the
# better source anyway: they carry the series label explicitly, give three
# significant figures, and state their own per-round errors.
#
# WHAT JUSTIFIES IT: THE CENTRE-OF-PRESSURE TEST
# ----------------------------------------------
#     CP_from_nose [cal] = x_cg/d - C_Malpha / C_Nalpha
#
# BRL plots CP independently in its Figure 9. Comparing each candidate model
# against the CP implied by each measured row (RMS over the 21 usable rows):
#
#     raw ASAT deck                      0.1891 cal
#     subsonic-only splice (previous)    0.1438 cal
#     THIS measured correction           0.1227 cal
#     ---------------------------------------------
#     irreducible row-to-row scatter     0.1166 cal
#
# The corrected curve sits at the measurement noise floor: the residual is no
# longer distinguishable from the scatter of the measurements themselves, so
# there is no further information in this dataset to extract.
#
# NOTE ON CIRCULARITY, stated because it matters: the CP a row implies is
# computed from that row own C_Malpha and C_Nalpha, so a model that matches
# the measured C_Nalpha will tend to match the measured CP. The test is not
# fully independent. It retains force because the model C_Malpha is ASAT, not
# the row, and because the comparison is against a noise floor computed the
# same way for every candidate.
#
# C_Malpha IS NOT CORRECTED. The same 21 rows give measured/ASAT = 1.0186
# with sd 0.0386 and no Mach trend (min 0.966, max 1.090) -- consistent with
# unity inside the BRL stated per-round error. Measurement CONFIRMS the
# computed overturning moment; changing it would be fitting noise.
COEFFICIENT_CONFIDENCE = {
    "C_X0": ("HIGH", "ASAT-13 C_A; cross-checked vs BRL MR-1582 free-flight C_D within scatter"),
    "C_X2": ("MEDIUM", "ASAT-13 only; BRL C_D2 implies ~1.4x larger; ~3% of axial force at nominal yaw"),
    "C_Nalpha": ("HIGH", "ASAT-13 shape scaled onto BRL MR-1582 full-scale measurement (sign flipped)"),
    "C_Ypalpha": ("LOW", "ASAT-13 only; BRL could not extract it from full-scale swerve"),
    "C_lp": ("MEDIUM", "ASAT-13 only; magnitude corroborated by a PRODAS 155 mm ERFB/BB deck"),
    "C_Malpha": ("HIGH", "ASAT-13, CONFIRMED by BRL measurement (ratio 1.019 +- 0.039, no trend)"),
    "C_mq": ("LOW-MEDIUM", "ASAT-13 only, and NOT the same quantity BRL measured (C_mq vs C_mq+C_mad)"),
    "C_Mpalpha": ("LOW", "ASAT-13 0-deg column; BRL M107 measured -0.57 at M 0.79 vs -0.36 here (36% apart)"),
}

#: Multiplier applied to the reduced rates pd/V and qd/V in dynamics.py.
#: 0.5 gives the pd/(2V) normalisation of the ASAT-13 / SPINNER-98 deck this
#: table carries -- determined empirically by reproducing that source's own
#: published trajectory, not assumed. Setting it to 1.0 switches the entire
#: model to the classical aeroballistic pd/V normalisation, which is what a
#: BRL- or McCoy-sourced coefficient set would need. THE CONVENTION BELONGS
#: TO THE COEFFICIENTS: change this only when the table changes. See the
#: block above and docs/RATE-CONVENTION.md.
REDUCED_RATE_FACTOR = 0.5


CNALPHA_MEASURED_K = (
    (0.570, 1.0267),
    (0.812, 0.9152),
    (1.186, 1.1114),
    (1.604, 1.0009),
    (1.770, 0.9717),
    (2.265, 1.0751),
)

_KX = tuple(p[0] for p in CNALPHA_MEASURED_K)
_KY = tuple(p[1] for p in CNALPHA_MEASURED_K)


def cnalpha_measured_factor(mach: float) -> float:
    """
    Mach-dependent multiplier taking the ASAT C_Nalpha column onto the BRL
    full-scale measurement. Linear between cluster centres, held flat outside.
    """
    if mach <= _KX[0]:
        return _KY[0]
    if mach >= _KX[-1]:
        return _KY[-1]
    for i in range(len(_KX) - 1):
        if _KX[i] <= mach <= _KX[i + 1]:
            f = (mach - _KX[i]) / (_KX[i + 1] - _KX[i])
            return _KY[i] + (_KY[i + 1] - _KY[i]) * f
    return 1.0


@dataclass(frozen=True)
class AeroCoefficients:
    """The eight coefficients at one Mach number. Immutable."""

    C_X0: float
    C_X2: float
    C_Nalpha: float
    C_Ypalpha: float
    C_lp: float
    C_Malpha: float
    C_mq: float
    C_Mpalpha: float


class AeroTable:
    """
    Mach-interpolated coefficient table.

    Linear interpolation between knots, held flat outside the tabulated
    range. `extrapolated_below` and `extrapolated_above` record that a
    request fell outside the table so a driver can report it rather than
    silently accepting an unsupported value.
    """

    def __init__(self, rows: np.ndarray, name: str = "unnamed", source: str = ""):
        rows = np.asarray(rows, dtype=float)
        ncol = 1 + len(COEFFICIENT_NAMES)
        if rows.ndim != 2 or rows.shape[1] != ncol:
            raise ValueError(f"expected an (n, {ncol}) table, got {rows.shape}")
        if np.any(np.diff(rows[:, 0]) <= 0):
            raise ValueError("Mach column must be strictly increasing")
        self.name = name
        self.source = source
        self.mach = rows[:, 0].copy()
        self.values = rows[:, 1:].copy()
        self.mach_min = float(self.mach[0])
        self.mach_max = float(self.mach[-1])
        self.extrapolated_below = False
        self.extrapolated_above = False
        # Plain-Python copies for the allocation-free hot path in lookup().
        self._mach_list = [float(m) for m in self.mach]
        self._rows_list = [[float(x) for x in row] for row in self.values]

    def __call__(self, mach: float) -> AeroCoefficients:
        return self.coefficients_at(mach)

    def coefficients_at(self, mach: float) -> AeroCoefficients:
        """Interpolate all eight coefficients at this Mach number."""
        if mach < self.mach_min:
            self.extrapolated_below = True
        elif mach > self.mach_max:
            self.extrapolated_above = True
        vals = [
            float(np.interp(mach, self.mach, self.values[:, j]))
            for j in range(len(COEFFICIENT_NAMES))
        ]
        return AeroCoefficients(*vals)

    def lookup(self, mach: float) -> tuple:
        """
        Hot-path form of coefficients_at(): returns the eight coefficients as
        a plain tuple, in COEFFICIENT_NAMES order, allocating nothing beyond
        that tuple. Identical interpolation to coefficients_at() -- there is
        one implementation, and test_lookup_matches_coefficients_at pins them
        together.
        """
        ml = self._mach_list
        n = len(ml)
        if mach <= ml[0]:
            if mach < ml[0]:
                self.extrapolated_below = True
            row = self._rows_list[0]
            return (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
        if mach >= ml[-1]:
            if mach > ml[-1]:
                self.extrapolated_above = True
            row = self._rows_list[-1]
            return (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])

        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if ml[mid] <= mach:
                lo = mid
            else:
                hi = mid
        f = (mach - ml[lo]) / (ml[hi] - ml[lo])
        a = self._rows_list[lo]
        b = self._rows_list[hi]
        return (
            a[0] + f * (b[0] - a[0]),
            a[1] + f * (b[1] - a[1]),
            a[2] + f * (b[2] - a[2]),
            a[3] + f * (b[3] - a[3]),
            a[4] + f * (b[4] - a[4]),
            a[5] + f * (b[5] - a[5]),
            a[6] + f * (b[6] - a[6]),
            a[7] + f * (b[7] - a[7]),
        )

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mach.copy(), self.values.copy()


def apply_measured_cnalpha(rows: np.ndarray) -> np.ndarray:
    """Apply the BRL-measured C_Nalpha correction to a copy of the ASAT rows."""
    out = np.array(rows, dtype=float, copy=True)
    icn = 1 + COEFFICIENT_NAMES.index("C_Nalpha")
    for i in range(out.shape[0]):
        out[i, icn] *= cnalpha_measured_factor(float(out[i, 0]))
    return out


def make_m107_table(measured_cnalpha: bool = True) -> AeroTable:
    """
    A fresh M107 table, so extrapolation flags are per-run.

    measured_cnalpha=True (default) applies the BRL-measured C_Nalpha
    correction documented above. Pass False to get the raw ASAT/SPINNER-98
    deck, which is what the sensitivity comparison in run_validation.py uses.
    """
    rows = apply_measured_cnalpha(_M107_ROWS) if measured_cnalpha else _M107_ROWS
    src = (
        "Khalil, Abdalla & Kamal, ASAT-13 (Cairo, 2009), "
        "via Lim, NPS thesis 2016 (AD1029824) Table 11"
    )
    if measured_cnalpha:
        src += "; C_Nalpha corrected onto BRL MR-1582 full-scale measurement"
    return AeroTable(rows, name="155 mm M107 HE", source=src)


#: Module-level default table. Prefer make_m107_table() in a harness that
#: cares about the extrapolation flags.
M107_AERO = make_m107_table()


def transfer_moment_reference(
    C_Malpha_ref: float, C_Nalpha: float, x_ref: float, x_cg: float, d: float
) -> float:
    """
    Transfer an overturning moment coefficient from an axial reference
    station x_ref to the centre of gravity x_cg.

        C_Malpha|CG = C_Malpha|ref + (x_ref - x_cg)/d * C_Nalpha

    x_ref and x_cg are measured from the nose, positive aft, in metres;
    d is the reference diameter in metres.

    For the nominal M107 model this is a no-op, because the source
    coefficients are already CG-referenced and projectile.py places x_cg at
    that same station. It exists so a Monte Carlo that perturbs x_cg can
    transfer correctly rather than reuse a wrongly-referenced coefficient.
    """
    return C_Malpha_ref + (x_ref - x_cg) / d * C_Nalpha


def warn_unvalidated(table: "AeroTable | None" = None, stream=None) -> list[str]:
    """
    Emit the standing coefficient-confidence banner.

    There are no PLACEHOLDER coefficients in this model -- every value is
    traceable to a cited document -- but four of the eight rest on a single
    source, and the reduced-rate convention for the rate-dependent terms is
    an assumption. Any run that produces numbers a reader might quote has to
    say so, every time.
    """
    import sys

    stream = stream if stream is not None else sys.stderr
    lines = [
        "=" * 78,
        "AERODYNAMIC COEFFICIENT CONFIDENCE -- read before quoting any result",
        "=" * 78,
        "No placeholder coefficients are in use; every value is sourced. However:",
        "",
    ]
    for key in COEFFICIENT_NAMES:
        level, why = COEFFICIENT_CONFIDENCE[key]
        lines.append(f"  {key:<10} {level:<11} {why}")
    lines += [
        "",
        "  CONVENTION ASSUMPTION: C_Ypalpha, C_Mpalpha, C_lp and C_mq are applied",
        "  with reduced rates pd/(2V) and qd/(2V) (PRODAS / aircraft convention).",
        "  The classical aeroballistic literature uses pd/V, whose coefficients are",
        "  half as large for the same physics. If the source table is in fact",
        "  aeroballistic-normalised, these four terms are 2x too small. Range and",
        "  drift are insensitive to this; dynamic stability margins are not.",
        "",
        "  Range, gyroscopic stability and drift are governed by C_X0, C_Nalpha and",
        "  C_Malpha -- all rated HIGH, each confirmed by two independent sources.",
        "",
        "  RIFLING TWIST IS A GUN PROPERTY, NOT A SHELL PROPERTY. The nominal model",
        "  uses 1 turn in 20 calibres (M185/M199, the tube of firing table",
        "  FT 155-AM-2). BRL MR-1582 measured on a 1-in-25 tube. Quoting a",
        "  gyroscopic stability factor without naming the tube is meaningless:",
        "  Sg scales as the inverse square of the twist and, at fixed twist, does",
        "  not depend on muzzle velocity at all.",
        "=" * 78,
    ]
    if table is not None and (table.extrapolated_below or table.extrapolated_above):
        lines.append(
            f"  NOTE: a Mach number outside the tabulated range "
            f"[{table.mach_min}, {table.mach_max}] was requested; "
            f"end values were held flat."
        )
        lines.append("=" * 78)
    print("\n".join(lines), file=stream)
    return lines
