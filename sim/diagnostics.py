"""
Stability diagnostics: gyroscopic and dynamic stability factors, yaw of
repose, and the validation-ladder helpers.

CONVENTION NOTE THAT MATTERS HERE
---------------------------------
The classical stability factors were derived in the aeroballistic
normalisation, where the reduced spin is pd/V. This package stores the
rate-dependent coefficients in the pd/(2V) convention (SIXDOFSPEC.md
sections 5-6, aerodata.REDUCED_RATE_FACTOR = 0.5). The dynamic stability
factor therefore converts C_Mpalpha and C_mq back to the pd/V normalisation
before applying McCoy's formula. The gyroscopic stability factor involves
only C_Malpha, which carries no such ambiguity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import atmosphere as atm
from .aerodata import REDUCED_RATE_FACTOR, AeroTable
from .dynamics import FlightModel, aero_state, unpack
from .integrate import Trajectory
from .projectile import Projectile

__all__ = [
    "gyroscopic_stability_factor",
    "dynamic_stability_factor",
    "dynamic_stability_limit",
    "yaw_of_repose_estimate",
    "StabilityPoint",
    "stability_history",
    "muzzle_stability",
]


def gyroscopic_stability_factor(
    projectile: Projectile, density: float, airspeed: float, spin: float, C_Malpha: float
) -> float:
    """
    Sg = Ix^2 p^2 / (2 rho S d It V^2 C_Malpha)      SIXDOFSPEC.md section 10

    Sg > 1 is required for a spin-stabilised shell. Typical values are
    1.3 - 2.5 at the muzzle, rising through flight because spin decays more
    slowly than velocity.

    This formula was validated against BRL MR-1582, which tabulates its own
    independently measured gyroscopic stability factor s for the 155 mm M101
    at 1.69 - 2.27 over M 0.57 - 2.41. Evaluating this function with the same
    projectile, twist (1 turn in 25 calibres) and conditions reproduces that
    range -- see tests/test_sim.py::test_gyroscopic_factor_matches_brl.
    """
    Ix = projectile.I_axial
    It = projectile.I_transverse
    S = projectile.reference_area
    d = projectile.diameter
    denom = 2.0 * density * S * d * It * airspeed * airspeed * C_Malpha
    if denom == 0.0:
        return math.inf
    return (Ix * Ix * spin * spin) / denom


def dynamic_stability_factor(
    projectile: Projectile,
    C_Nalpha: float,
    C_X0: float,
    C_Mpalpha: float,
    C_mq: float,
) -> float:
    """
    McCoy's dynamic stability factor

        Sd = 2 (C_Lalpha + kx^-2 C_Mpalpha) /
             (C_Lalpha - C_D - ky^-2 (C_mq + C_mAlphadot))

    with kx^-2 = m d^2 / Ix and ky^-2 = m d^2 / It.

    C_Lalpha (lift slope) is recovered from the body-axis normal force slope
    as C_Lalpha = C_Nalpha - C_D0, and C_D is taken as C_X0.

    C_Mpalpha and C_mq are divided by 2 to convert this package's pd/(2V)
    normalisation into the pd/V normalisation McCoy's derivation assumes.
    Free-flight measurement of (C_mq + C_mAlphadot) for this shell family
    scatters by a factor of two, so treat Sd as indicative, not decisive.
    The convention-free statement about dynamic stability is the direct
    observable from the 6-DOF run: whether the total angle of attack stays
    bounded (validation rung 6).
    """
    m, d = projectile.mass, projectile.diameter
    kx_inv2 = m * d * d / projectile.I_axial
    ky_inv2 = m * d * d / projectile.I_transverse

    # pd/(2V) -> pd/V is exactly a factor REDUCED_RATE_FACTOR (= 0.5).
    conv = REDUCED_RATE_FACTOR
    C_Mpa = C_Mpalpha * conv
    C_mq_c = C_mq * conv

    C_Lalpha = C_Nalpha - C_X0
    num = 2.0 * (C_Lalpha + kx_inv2 * C_Mpa)
    den = C_Lalpha - C_X0 - ky_inv2 * C_mq_c
    if den == 0.0:
        return math.inf
    return num / den


def dynamic_stability_limit(Sd: float) -> float:
    """
    The dynamic stability requirement is Sg > 1 / (Sd (2 - Sd)).

    Returns that lower bound on Sg. If Sd is outside (0, 2) no amount of spin
    stabilises the shell and this returns +inf.
    """
    prod = Sd * (2.0 - Sd)
    if prod <= 0.0:
        return math.inf
    return 1.0 / prod


def yaw_of_repose_estimate(
    projectile: Projectile,
    density: float,
    airspeed: float,
    spin: float,
    C_Malpha: float,
    pitch_angle: float,
) -> float:
    """
    Closed-form magnitude of the yaw of repose, for cross-checking the 6-DOF.

    The velocity vector pitches down at gravity's doing, at a rate
    thetadot = -g cos(theta) / V. For the shell axis to precess with it, the
    overturning moment must supply the required gyroscopic torque:

        (1/2) rho V^2 S d C_Malpha delta_R = Ix p |thetadot|

    so

        delta_R = 2 Ix p g cos(theta) / (rho V^3 S d C_Malpha)

    For RIGHT-HAND rifling (p > 0) the equilibrium yaw is to the RIGHT, which
    is what makes the shell drift right. Returns the magnitude in radians.
    """
    Ix = projectile.I_axial
    S = projectile.reference_area
    d = projectile.diameter
    g = atm.G0
    denom = density * airspeed**3 * S * d * C_Malpha
    if denom == 0.0:
        return 0.0
    return 2.0 * Ix * spin * g * math.cos(pitch_angle) / denom


@dataclass(frozen=True)
class StabilityPoint:
    t: float
    mach: float
    airspeed: float
    altitude: float
    density: float
    spin: float
    Sg: float
    Sd: float
    Sg_required: float
    gyro_stable: bool
    dynamically_stable: bool
    yaw_of_repose_rad: float


def stability_history(traj: Trajectory, model: FlightModel) -> list[StabilityPoint]:
    """Evaluate the stability factors at every logged trajectory sample."""
    proj = model.projectile
    out: list[StabilityPoint] = []
    for i in range(traj.t.size):
        V = float(traj.airspeed[i])
        if V <= 0.0:
            continue
        rho = float(traj.density[i])
        mach = float(traj.mach[i])
        p = float(traj.omega[i, 0])
        c = model.aero.coefficients_at(mach)
        Sg = gyroscopic_stability_factor(proj, rho, V, p, c.C_Malpha)
        Sd = dynamic_stability_factor(proj, c.C_Nalpha, c.C_X0, c.C_Mpalpha, c.C_mq)
        need = dynamic_stability_limit(Sd)
        # Flight path angle from the earth-frame velocity.
        v = traj.velocity[i]
        theta = math.atan2(-v[2], math.hypot(v[0], v[1]))
        out.append(
            StabilityPoint(
                t=float(traj.t[i]),
                mach=mach,
                airspeed=V,
                altitude=float(-traj.position[i, 2]),
                density=rho,
                spin=p,
                Sg=Sg,
                Sd=Sd,
                Sg_required=need,
                gyro_stable=Sg > 1.0,
                dynamically_stable=Sg > need,
                yaw_of_repose_rad=yaw_of_repose_estimate(
                    proj, rho, V, p, c.C_Malpha, theta
                ),
            )
        )
    return out


def muzzle_stability(y0: np.ndarray, model: FlightModel) -> StabilityPoint:
    """Stability factors evaluated at the muzzle state."""
    st = aero_state(0.0, y0, model)
    _, v, _, omega = unpack(y0)
    proj = model.projectile
    c = model.aero.coefficients_at(st.mach)
    Sg = gyroscopic_stability_factor(proj, st.density, st.airspeed, float(omega[0]), c.C_Malpha)
    Sd = dynamic_stability_factor(proj, c.C_Nalpha, c.C_X0, c.C_Mpalpha, c.C_mq)
    need = dynamic_stability_limit(Sd)
    theta = math.atan2(-v[2], math.hypot(v[0], v[1]))
    return StabilityPoint(
        t=0.0,
        mach=st.mach,
        airspeed=st.airspeed,
        altitude=st.altitude,
        density=st.density,
        spin=float(omega[0]),
        Sg=Sg,
        Sd=Sd,
        Sg_required=need,
        gyro_stable=Sg > 1.0,
        dynamically_stable=Sg > need,
        yaw_of_repose_rad=yaw_of_repose_estimate(
            proj, st.density, st.airspeed, float(omega[0]), c.C_Malpha, theta
        ),
    )
