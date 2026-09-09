"""
sim -- shared flight-dynamics library.

Step 1 of the software roadmap: a 6-DOF rigid-body simulator for a
spin-stabilised 155 mm projectile in unguided ballistic flight. Step 2.5 added
the despun-nose degree of freedom and the canard model on top of it, taking
the state from 13 elements to 15; `canards` is the only module that is not
step-1 work, and it is the only one whose aerodynamics are estimated rather
than measured -- see its docstring before using any number it produces.

Frame conventions (see SIXDOFSPEC.md section 1) -- these hold everywhere
in this package and must never be deviated from:

  Earth frame : NED, origin at the muzzle.
                X = downrange (azimuth of fire), Y = right, Z = DOWN.
                Gravity is +Z. Altitude is -z. Impact is z >= 0 descending.
  Body frame  : x = forward out of the nose, y = right, z = down.
  Attitude    : quaternion q = [w, x, y, z] mapping BODY -> EARTH.
                v_earth = R(q) @ v_body
  Wind        : the velocity OF THE AIR. A wind *from* the north has a
                negative X component.
"""

from . import frames, atmosphere, aerodata, projectile, dynamics, integrate, diagnostics
from . import canards

__all__ = [
    "frames",
    "atmosphere",
    "aerodata",
    "projectile",
    "dynamics",
    "integrate",
    "diagnostics",
    "canards",
]
