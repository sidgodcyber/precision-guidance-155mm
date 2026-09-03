"""
sim -- shared flight-dynamics library.

Step 1 of the software roadmap: a 6-DOF rigid-body simulator for a
spin-stabilised 155 mm projectile in unguided ballistic flight.

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

__all__ = [
    "frames",
    "atmosphere",
    "aerodata",
    "projectile",
    "dynamics",
    "integrate",
    "diagnostics",
]
