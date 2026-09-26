"""
Read-only access to `docs/roll_servo.json` and `docs/authority_results.json`
-- the precomputed artifacts Physics Lab's cards read instead of calling
`analysis.roll_servo`/`analysis.authority` live.

`analysis.roll_servo.baseline()` (the function that produced the "baseline"
entry below) flies two full 6-DOF trajectories to impact at dt=5e-4;
measured directly at ~33 s on this machine. That is far too slow for a
slider-driven card, so `adopted_deployment_condition()` reads its ALREADY
STORED output instead of recomputing it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache

from scipy.optimize import brentq

from sim.atmosphere import GAMMA, isa_scalars

from setter.config import AUTHORITY_RESULTS_PATH, ROLL_SERVO_PATH

__all__ = [
    "DeploymentCondition", "adopted_deployment_condition",
    "authority_phi_sweep", "authority_deploy_sweep",
]


@lru_cache(maxsize=1)
def _roll_servo() -> dict:
    with open(ROLL_SERVO_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _authority() -> dict:
    with open(AUTHORITY_RESULTS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass(frozen=True)
class DeploymentCondition:
    """One flight condition at nose deployment, read from the stored
    `analysis.roll_servo` baseline -- the adopted configuration's own
    deployment point, not a live simulation."""

    time_s: float
    qbar_pa: float
    mach: float
    spin_rad_s: float
    airspeed_ms: float
    altitude_m: float
    source: str


def _altitude_and_airspeed_from_mach_qbar(mach: float, qbar_pa: float) -> tuple:
    """Closed-form: qbar = 0.5 * mach^2 * GAMMA * p(h) (since rho*a^2 =
    GAMMA*p), solved for the altitude h whose ISA pressure matches, then
    airspeed = mach * a(h). No simulation, no iteration beyond the
    bisection -- `docs/roll_servo.json` does not store altitude or airspeed
    directly, only (mach, qbar)."""
    p_target = 2.0 * qbar_pa / (GAMMA * mach * mach)
    h = brentq(lambda alt: isa_scalars(alt)[1] - p_target, -1000.0, 30000.0)
    _, _, _, a = isa_scalars(h)
    return h, mach * a


def adopted_deployment_condition() -> DeploymentCondition:
    """The 'long' engagement's adopted-configuration deployment point (charge
    8, QE 525.3 mils, canards at 25 mm/3 deg -- `analysis.roll_servo`'s
    adopted configuration): t=5.55 s, M1.52, q-bar 137 kPa, spin 1308 rad/s.
    Read from `docs/roll_servo.json["baseline"]`, the same numbers
    `docs/CONTROL-CHARACTERISATION.md` documents -- not recomputed."""
    b = _roll_servo()["baseline"]
    mach = float(b["mach_at_deploy"])
    qbar = float(b["qbar_at_deploy"])
    altitude, airspeed = _altitude_and_airspeed_from_mach_qbar(mach, qbar)
    return DeploymentCondition(
        time_s=float(b["deploy_time"]), qbar_pa=qbar, mach=mach,
        spin_rad_s=float(b["spin_at_deploy"]), airspeed_ms=airspeed,
        altitude_m=altitude,
        source="docs/roll_servo.json['baseline'] (adopted configuration)")


def authority_phi_sweep(tag: str = "c8_qe525.3") -> list:
    """The 12 stored `(phi_deg, d_range, d_deflection)` rows for one
    `docs/authority_results.json['main']` tag -- Card 3's precomputed
    ground-truth overlay. Generated with `sim.canards.NOMINAL_GEOMETRY`
    (station 90 mm, steering 5 deg -- confirmed against the stored rows'
    own `station_from_nose_m`/`steering_deflection_deg` fields), NOT the
    25 mm/3 deg configuration `analysis.roll_servo` adopts elsewhere on
    this page; see `setter/pages/physics_lab.py` for why that mismatch is
    deliberate here.
    """
    entry = _authority()["main"].get(tag)
    if entry is None:
        return []
    return [(r["phi_deg"], r["d_range"], r["d_deflection"]) for r in entry["rows"]]


def authority_deploy_sweep() -> dict:
    """`docs/authority_results.json['deploy']` verbatim: 3 tags
    (`c4_qe97.2`, `c6_qe258.4`, `c8_qe525.3`) x 6 `deploy_fraction` points
    each. These 3 tags are NOT the setter's 5 named engagements --
    `c6_qe258.4` (charge 6, QE 258.4) is a different, intermediate case
    from the named 'middle' engagement (charge 6, QE 378.6 mils,
    `analysis.mpmm_compare.FIRING_TABLE[7]`) -- presented as itself, not
    silently relabelled."""
    return _authority()["deploy"]
