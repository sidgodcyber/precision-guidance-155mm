"""
Projectile physical properties and launch conditions.

Everything here is a configurable input, never a hard constant, because the
Monte Carlo in step 6 will perturb every one of these.

PROVENANCE OF THE NOMINAL 155 mm M107 VALUES
--------------------------------------------
Mass, centre of gravity, length and the radii of gyration come from
B. G. Karpov and L. E. Schmidt (rev. K. Krial, L. C. MacAllister),
"The Aerodynamic Properties of the 155-mm Shell M101 from Free Flight Range
Tests of Full Scale and 1/12 Scale Models", BRL Memorandum Report No. 1582,
June 1964 (DTIC AD0454925), Table I "Physical Properties", which lists the
155-mm M101 and the 155-mm M107 with identical entries:

    mass                     95.8 lb
    CG from nose             2.96 calibres
    k_axial^-2               7.10      (so k_axial^2 = 1/7.10 = 0.140845)
    k_transverse^-2          0.81      (so k_trans^2  = 1/0.81 = 1.234568)
    length                   4.5 calibres
    diameter                 155 mm

The radii of gyration are non-dimensionalised on the diameter, so

    I_axial      = k_axial^2      * m * d^2 = 0.140845 * 43.454 * 0.024025
                 = 0.14705 kg m^2
    I_transverse = k_transverse^2 * m * d^2 = 1.234568 * 43.454 * 0.024025
                 = 1.28888 kg m^2

Independent confirmation of the axial inertia: W. Y. Lim, NPS thesis 2016
(DTIC AD1029824) Table 12 lists Ixx = 0.1461 kg m^2 and mass = 43.091 kg for
the M107, from an unrelated source. The two determinations of I_axial agree
to 0.7 %, which is also a check that the k^-2 columns were read the right way
round -- swapping them would give I_axial = 1.29 kg m^2, nine times too big,
and the shell would not spin-stabilise at all.

The nominal projectile below uses the BRL mass of 43.454 kg (95.8 lb). The
M107 nominal service weight is 95 lb / 43.09 kg; the difference is fuze and
lot variation and is exactly the sort of thing the Monte Carlo will sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "Projectile",
    "LaunchConditions",
    "Environment",
    "M107",
    "CHARGE_TABLE",
    "LB_TO_KG",
    "MIL_TO_RAD",
]

LB_TO_KG = 0.45359237
#: NATO mil: 6400 mils to the full circle.
MIL_TO_RAD = 2.0 * math.pi / 6400.0


@dataclass(frozen=True)
class Projectile:
    """
    Rigid-body physical properties of an axisymmetric spin-stabilised shell.

    Frozen so nothing can mutate it inside an integration loop. Use
    dataclasses.replace() (or .perturbed()) to make a varied copy.
    """

    name: str
    mass: float            # kg
    diameter: float        # m   reference diameter d
    I_axial: float         # kg m^2   Ix, about the spin axis
    I_transverse: float    # kg m^2   It, about any transverse axis through the CG
    x_cg: float            # m   centre of gravity aft of the nose, positive aft
    length: float          # m   overall length
    twist_calibers: float  # calibres of travel per revolution of rifling
                           # positive = RIGHT-HAND rifling
                           #
                           # NOTE: twist is a property of the GUN TUBE, not of
                           # the projectile. The same M107 shell is fired from
                           # tubes of different twist and therefore leaves with
                           # different spin. It lives on this dataclass only
                           # because the muzzle spin is a projectile initial
                           # condition. See TUBES below and always set it to
                           # match the weapon whose data you are comparing to.

    @property
    def reference_area(self) -> float:
        """S = pi d^2 / 4."""
        return 0.25 * math.pi * self.diameter * self.diameter

    @property
    def x_cg_calibers(self) -> float:
        return self.x_cg / self.diameter

    def muzzle_spin(self, muzzle_velocity: float) -> float:
        """
        Axial spin rate at the muzzle, rad/s.

            p0 = 2 pi V / (twist_calibers * d)

        Right-hand rifling gives a POSITIVE p about the body x axis (out of
        the nose), which is what makes a spin-stabilised shell drift RIGHT.
        """
        return 2.0 * math.pi * muzzle_velocity / (self.twist_calibers * self.diameter)

    def perturbed(self, **kwargs) -> "Projectile":
        """A copy with fields replaced. For the step-6 Monte Carlo."""
        return replace(self, **kwargs)


def _m107() -> Projectile:
    d = 0.155
    mass = 95.8 * LB_TO_KG            # 43.4542 kg
    md2 = mass * d * d
    return Projectile(
        name="155 mm M107 HE",
        mass=mass,
        diameter=d,
        I_axial=md2 / 7.10,           # k_axial^-2      = 7.10  (BRL MR-1582 Table I)
        I_transverse=md2 / 0.81,      # k_transverse^-2 = 0.81  (BRL MR-1582 Table I)
        x_cg=2.96 * d,                # 2.96 calibres from the nose
        length=4.5 * d,               # 4.5 calibres
        twist_calibers=20.0,          # 1 turn in 20 cal, M185/M199 39-calibre tube
    )


#: Nominal 155 mm M107 high-explosive projectile, as fired from an M185/M199
#: 39-calibre tube (the weapon of firing table FT 155-AM-2).
M107 = _m107()


# ==========================================================================
# RIFLING TWIST BY TUBE -- this is a weapon property, not a shell property
# ==========================================================================
# The twist question was raised in review, checked against primary sources,
# and resolved as follows.
#
#   M185 / M199, 39 calibre (M109A1/A2/A3, M198)   1 turn in 20 calibres
#     * R. L. McCoy, "Modern Exterior Ballistics", 2nd ed., ch. 13, states
#       verbatim: the M549 shell "fired at Charge 4 from the M109A1 Howitzer
#       with a rifling twist rate of 1 turn in 20 calibers of travel". The
#       M109A1 mounts the M185.
#     * W. Y. Lim, NPS thesis 2016 (DTIC AD1029824), Table 12, "Twist Rate
#       20 Calibers/rev" -- and that is the study whose FT 155-AM-2
#       comparison data this model is validated against, so 20 is the value
#       consistent with the reference data.
#
#   M1 / M1A1, 23 calibre (M114 towed howitzer)     1 turn in 25 calibres
#     * BRL MR-1582 states its own M101 firings used "standard 155-mm
#       artillery pieces with a twist of one turn in 25 calibers", and later
#       refers to "the muzzle spin of 1/25". This is the older tube, and it
#       is why BRL's own measured gyroscopic stability factors (1.69-2.27)
#       are lower than this model's for the M185.
#
# A muzzle velocity of 684 m/s is a 39-calibre-tube figure, so the nominal
# M107 model uses 1/20. ASAT-13 states V0 = 684.3 m/s together with
# p0 = 175.48 rps, which implies 684.3/175.48/0.155 = 25.16 calibres per turn
# -- an M114-era twist paired with an M185-era muzzle velocity. That pairing
# is not physically consistent with any single US tube; ASAT's configuration
# is reproduced faithfully in validation rung 5b, but it is not adopted for
# the nominal model.
TUBES = {
    "M185": 20.0,   # also M199; 39 cal; FT 155-AM-2
    "M1": 25.0,     # M114 towed howitzer, 23 cal; BRL MR-1582 firings
    "ASAT": 684.3 / 175.48 / 0.155,  # 25.159, implied by ASAT-13 section 4.1
}


def _m107_asat() -> Projectile:
    """
    The M107 exactly as specified in ASAT-13 section 4.3, for validation
    rung 5b. Mass properties are ASAT's own (Inventor + PRODAS), which agree
    with BRL Table I to 2.1 % on Ix and 6.0 % on Iy.
    """
    d = 0.155
    return Projectile(
        name="155 mm M107 HE (ASAT-13 section 4.3 specification)",
        mass=43.0,
        diameter=d,
        I_axial=0.144,
        I_transverse=1.216,
        x_cg=0.459,
        length=0.698,
        twist_calibers=TUBES["ASAT"],
    )


#: The ASAT-13 section 4.3 configuration, used only by validation rung 5b.
M107_ASAT = _m107_asat()


@dataclass(frozen=True)
class LaunchConditions:
    """Muzzle state. QE and azimuth in radians."""

    muzzle_velocity: float          # m/s
    quadrant_elevation: float       # rad, positive nose-up
    azimuth: float = 0.0            # rad, 0 = along the frame X axis
    initial_roll: float = 0.0       # rad
    #: Optional initial transverse rates (rad/s). Left at zero for step 1;
    #: a Monte Carlo will use them to inject muzzle tip-off.
    initial_q: float = 0.0
    initial_r: float = 0.0

    @classmethod
    def from_degrees(cls, muzzle_velocity: float, qe_deg: float, **kw) -> "LaunchConditions":
        return cls(muzzle_velocity, math.radians(qe_deg), **kw)

    @classmethod
    def from_mils(cls, muzzle_velocity: float, qe_mils: float, **kw) -> "LaunchConditions":
        """NATO mils, 6400 to the circle -- the unit firing tables use."""
        return cls(muzzle_velocity, qe_mils * MIL_TO_RAD, **kw)

    @property
    def qe_degrees(self) -> float:
        return math.degrees(self.quadrant_elevation)

    @property
    def qe_mils(self) -> float:
        return self.quadrant_elevation / MIL_TO_RAD


@dataclass(frozen=True)
class Environment:
    """Firing-site environment."""

    latitude: float = math.radians(45.0)   # rad
    include_coriolis: bool = True
    include_inverse_square_gravity: bool = True
    #: Height of the muzzle above mean sea level, m. The trajectory origin is
    #: AT THE MUZZLE (r0 = 0, per SIXDOFSPEC.md section 9), so this offsets
    #: only the altitude fed to the atmosphere and gravity models:
    #:     altitude = site_altitude - z
    site_altitude: float = 0.0

    @classmethod
    def from_degrees(cls, latitude_deg: float, **kw) -> "Environment":
        return cls(math.radians(latitude_deg), **kw)


#: Muzzle velocities for the M107 propelling charges, as used in the
#: FT 155-AM-2 comparison tables of Lim (NPS 2016, AD1029824) Tables 15-19.
#: Charge 8 (684 m/s) is the maximum-range charge for a 39-calibre tube and
#: is the "M107-class, MV ~684 m/s, max range ~18.1 km" anchor in the spec.
CHARGE_TABLE = {
    4: 337.0,
    5: 397.0,
    6: 474.0,
    7: 568.0,
    8: 684.0,
}
