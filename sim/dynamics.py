"""
Rigid-body derivative function: state in, derivative out.

This module is PURE. Given the same arguments it returns the same result, it
performs no I/O, it holds no module-level mutable state, and it never mutates
its inputs. That is what lets the same function be dropped into a Monte Carlo
harness (step 6) or a hardware-in-the-loop rig without being rewritten.

(One apparent exception: AeroTable.lookup latches a flag when it is asked for
a Mach number outside its tabulated range. That latch is a monotone
diagnostic, is never read back here, and cannot influence the derivative.)

STATE VECTOR -- 13 elements, SIXDOFSPEC.md section 2
    y[0:3]   r = [x, y, z]        earth NED position       m
    y[3:6]   v = [vx, vy, vz]     earth NED velocity       m/s
    y[6:10]  q = [qw,qx,qy,qz]    attitude, body -> earth  -
    y[10:13] w = [p, q, r]        body angular rates       rad/s

Index it only through unpack()/pack(), never by hand, so that appending the
14th despun-nose state in a later step touches one place.

EQUATIONS OF MOTION -- SIXDOFSPEC.md section 3
    rdot      = v
    vdot      = (1/m) R(q) F_body + g_ned + a_coriolis
    qdot      = 0.5 q (x) [0, omega]
    pdot      = L / Ix
    qdot_rate = [M + (It - Ix) r p] / It
    rdot_rate = [N - (It - Ix) p q] / It

The (It - Ix) cross terms are the gyroscopic coupling. They are the entire
reason a spinning shell behaves differently from a missile.

A NOTE ON THE SHAPE OF THIS FILE
--------------------------------
The force and moment model exists exactly once, in _aero_core(), written in
scalars. Both the hot-path derivative() and the diagnostic aero_state()
wrapper call it. That is deliberate: a "fast" copy and a "readable" copy of
projectile aerodynamics would eventually disagree, and a silent sign error in
one of them would still produce a plausible-looking trajectory. The cost of
the scalar style is that _aero_core is dense; the benefit is that there is
only one place a sign can be wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import atmosphere as atm
from . import frames
from .aerodata import REDUCED_RATE_FACTOR, AeroCoefficients, AeroTable
from .projectile import Environment, LaunchConditions, Projectile

__all__ = [
    "STATE_SIZE",
    "FlightModel",
    "AeroState",
    "unpack",
    "pack",
    "initial_state",
    "aero_state",
    "forces_moments",
    "derivative",
    "ControlCallback",
]

STATE_SIZE = 13

_IR = slice(0, 3)
_IV = slice(3, 6)
_IQ = slice(6, 10)
_IW = slice(10, 13)

#: Below this relative airspeed the aerodynamic model is switched off rather
#: than dividing by V. A shell is never this slow in flight; the guard exists
#: so a degenerate initial condition produces zeros instead of NaNs.
V_EPS = 1e-6

#: Signature of the step-4 canard model. It receives the time, the state and
#: the already-computed aerodynamic state, and returns an ADDITIONAL body-frame
#: force and moment about the CG. Step 1 passes None.
ControlCallback = Callable[[float, np.ndarray, "AeroState"], tuple]


@dataclass(frozen=True)
class FlightModel:
    """
    Everything the derivative needs that is not the state.

    Frozen, so an integration loop cannot accidentally reconfigure the
    physics halfway through a trajectory.
    """

    projectile: Projectile
    aero: AeroTable
    environment: Environment = field(default_factory=Environment)
    #: wind(altitude_m) -> [north, east, 0], the VELOCITY OF THE AIR, m/s.
    wind: Callable[[float], np.ndarray] = atm.no_wind
    #: Step-4 hook. None in step 1 (unguided ballistic flight).
    control: Optional[ControlCallback] = None
    #: Set False to zero every aerodynamic force and moment (validation rung 1).
    aero_enabled: bool = True
    #: Set False to zero all aerodynamic MOMENTS and the normal and Magnus
    #: forces, leaving axial drag only (validation rung 2).
    moments_enabled: bool = True
    #: Multiplies the four rate-dependent coefficients. 1.0 is nominal; the
    #: convention sensitivity case in run_ballistic.py uses 2.0.
    rate_coefficient_scale: float = 1.0
    #: Multiplies C_X0. 1.0 is the raw sourced table. Used only to run the
    #: documented form-factor sensitivity, never to tune a result into range.
    drag_scale: float = 1.0
    #: VALIDATION RUNG 2 ONLY. Forces the total angle of attack to zero by
    #: applying the axial force along the RELATIVE VELOCITY rather than along
    #: the body axis, and zeroing every moment. The 13-state integration still
    #: runs in full, so this compares the atmosphere, the Mach-indexed drag
    #: lookup, gravity, Coriolis and the RK4 driver against an independent
    #: 3-DOF point-mass script. Never set this for a physics run.
    alpha_zero_drag_only: bool = False

    def __post_init__(self):
        # Cache everything constant over a trajectory. These are read four
        # times per RK4 step over hundreds of thousands of steps, so an
        # attribute chain or a recomputed property is not free.
        w = atm.earth_rate_ned(self.environment.latitude, 0.0)
        object.__setattr__(self, "_omega_ned", (float(w[0]), float(w[1]), float(w[2])))
        p = self.projectile
        object.__setattr__(self, "_d", float(p.diameter))
        object.__setattr__(self, "_S", float(p.reference_area))
        object.__setattr__(self, "_inv_mass", 1.0 / float(p.mass))
        object.__setattr__(self, "_Ix", float(p.I_axial))
        object.__setattr__(self, "_It", float(p.I_transverse))
        object.__setattr__(self, "_dI", float(p.I_transverse - p.I_axial))
        object.__setattr__(self, "_site_alt", float(self.environment.site_altitude))
        object.__setattr__(self, "_wind_is_zero", self.wind is atm.no_wind)


@dataclass(frozen=True)
class AeroState:
    """
    Aerodynamic state at an instant, returned alongside the derivative for
    truth logging and for the step-4 control law, so neither has to recompute
    it.
    """

    altitude: float          # m above mean sea level
    density: float           # kg/m^3
    sound_speed: float       # m/s
    mach: float
    airspeed: float          # m/s, magnitude of velocity relative to the air
    dynamic_pressure: float  # Pa
    total_aoa: float         # rad, total angle of attack delta
    alpha: float             # rad, pitch-plane angle of attack
    beta: float              # rad, sideslip angle
    v_rel_body: np.ndarray   # m/s, [u, v, w]
    coefficients: Optional[AeroCoefficients]
    force_body: np.ndarray   # N,   aerodynamic force only (no gravity)
    moment_body: np.ndarray  # N m, about the CG


def unpack(y: np.ndarray):
    """Split the flat state into (r, v, q, omega). Views, not copies."""
    return y[_IR], y[_IV], y[_IQ], y[_IW]


def pack(r, v, q, omega) -> np.ndarray:
    """Assemble a flat state vector."""
    y = np.empty(STATE_SIZE)
    y[_IR] = r
    y[_IV] = v
    y[_IQ] = q
    y[_IW] = omega
    return y


def initial_state(projectile: Projectile, launch: LaunchConditions) -> np.ndarray:
    """
    Muzzle state, SIXDOFSPEC.md section 9.

        r0 = [0, 0, 0]                       origin AT the muzzle
        q0 = q_from_euler(azimuth, QE, roll)
        v0 = R(q0) @ [V_muzzle, 0, 0]
        p0 = 2 pi V / (twist_calibers * d)   right-hand rifling -> positive
    """
    q0 = frames.quat_from_euler(
        launch.azimuth, launch.quadrant_elevation, launch.initial_roll
    )
    v0 = frames.dcm_from_quat(q0) @ np.array([launch.muzzle_velocity, 0.0, 0.0])
    r0 = np.zeros(3)
    omega0 = np.array(
        [projectile.muzzle_spin(launch.muzzle_velocity), launch.initial_q, launch.initial_r]
    )
    return pack(r0, v0, q0, omega0)


def _aero_core(y: np.ndarray, model: FlightModel) -> tuple:
    """
    THE single implementation of the force and moment model.

    Scalars in, scalars out, no allocation beyond the returned tuple.

    Returns
        (fx, fy, fz, mx, my, mz,          body-frame aero force N, moment N m
         V, mach, delta, qbar, rho, a_snd, altitude,
         u, v, w,                          relative velocity, body axes, m/s
         r00, r01, r02, r10, r11, r12, r20, r21, r22)   body->earth DCM
    """
    d = model._d
    S = model._S

    # float() at extraction is essential, not cosmetic: indexing a numpy
    # array yields np.float64, whose arithmetic is roughly ten times slower
    # than a Python float. Without these casts every operation below becomes
    # a numpy scalar op and the derivative costs 40 us instead of 4 us.
    # When y is already a list of Python floats (the integrator hot path)
    # these casts are near-free.
    qw = float(y[6])
    qx = float(y[7])
    qy = float(y[8])
    qz = float(y[9])

    # NORMALISE BEFORE BUILDING THE DCM. This is not defensive tidying, it is
    # required for correctness, and getting it wrong is silent.
    #
    # dcm_from_quat assumes a unit quaternion; fed a quaternion of norm n it
    # returns a rotation scaled by n^2. Inside an RK4 stage the quaternion is
    # NOT unit: the stage state q + (dt/2) qdot leaves the unit sphere, and
    # because qdot is orthogonal to q the norm grows as
    #     |q| = sqrt(1 + (dt |omega| / 4)^2).
    # At the muzzle spin of a 155 mm shell (1386 rad/s) that is |q| = 1.06 at
    # dt = 1e-3. The aerodynamic force here scales as |q|^4 -- once through
    # the earth-to-body transform and once through body-to-earth -- so the
    # three interior RK4 stages were being evaluated with forces up to 25 %
    # too large. The resulting trajectory still looked entirely plausible and
    # still landed within about 1 % of the firing table; validation rung 2,
    # comparing against an independent 3-DOF integration, is what exposed it.
    n2 = qw * qw + qx * qx + qy * qy + qz * qz
    if n2 != 1.0:
        inv_n = 1.0 / math.sqrt(n2)
        qw *= inv_n
        qx *= inv_n
        qy *= inv_n
        qz *= inv_n

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    # The origin is at the muzzle, so height above sea level offsets by the
    # site altitude.
    altitude = model._site_alt - float(y[2])
    _T, _p, rho, a_snd = atm.isa_scalars(altitude)

    vx = float(y[3])
    vy = float(y[4])
    vz = float(y[5])
    if not model._wind_is_zero:
        wind = model.wind(altitude)
        vx -= float(wind[0])
        vy -= float(wind[1])
        vz -= float(wind[2])

    # earth -> body is the transpose of R
    u = r00 * vx + r10 * vy + r20 * vz
    v = r01 * vx + r11 * vy + r21 * vz
    w = r02 * vx + r12 * vy + r22 * vz
    V = math.sqrt(u * u + v * v + w * w)

    if (not model.aero_enabled) or V < V_EPS:
        return (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            V, V / a_snd, 0.0, 0.5 * rho * V * V, rho, a_snd, altitude,
            u, v, w,
            r00, r01, r02, r10, r11, r12, r20, r21, r22,
        )

    if model.alpha_zero_drag_only:
        # Rung 2: pure drag along the relative velocity, no moments.
        mach = V / a_snd
        qbar = 0.5 * rho * V * V
        cx0 = model.aero.lookup(mach)[0]
        k = -qbar * S * model.drag_scale * cx0 / V
        return (
            k * u, k * v, k * w, 0.0, 0.0, 0.0,
            V, mach, 0.0, qbar, rho, a_snd, altitude,
            u, v, w,
            r00, r01, r02, r10, r11, r12, r20, r21, r22,
        )

    v_t = math.sqrt(v * v + w * w)
    sin_delta = v_t / V
    if sin_delta > 1.0:
        sin_delta = 1.0
    delta = math.asin(sin_delta)
    mach = V / a_snd
    qbar = 0.5 * rho * V * V

    (C_X0, C_X2, C_Nalpha, C_Ypalpha,
     C_lp, C_Malpha, C_mq, C_Mpalpha) = model.aero.lookup(mach)

    ks = model.rate_coefficient_scale
    p = float(y[10])
    qr = float(y[11])
    rr = float(y[12])
    inv_V = 1.0 / V
    # Reduced rates. REDUCED_RATE_FACTOR = 0.5 gives pd/(2V), per the spec.
    p_hat = REDUCED_RATE_FACTOR * p * d * inv_V
    q_hat = REDUCED_RATE_FACTOR * qr * d * inv_V
    r_hat = REDUCED_RATE_FACTOR * rr * d * inv_V

    qS = qbar * S

    # --- forces, body frame (SIXDOFSPEC.md section 5) ---------------------
    # Axial force, opposing the body x axis.
    fx = -qS * (model.drag_scale * C_X0 + C_X2 * delta * delta)
    fy = 0.0
    fz = 0.0

    if model.moments_enabled:
        # Normal force: opposes the transverse velocity, i.e. it acts along
        # the angle-of-attack direction. Nose pitched up (w > 0) gives a
        # force up (fz < 0). The destabilising behaviour lives entirely in
        # the moment, not the force.
        cn = -qS * C_Nalpha * inv_V
        fy += cn * v
        fz += cn * w
        # Magnus force: 90 degrees to the normal force, scaled by reduced
        # spin. Small; its moment is not.
        magnus = qS * (ks * C_Ypalpha) * p_hat * inv_V
        fy += magnus * (-w)
        fz += magnus * v

    # --- moments about the CG, body frame (SIXDOFSPEC.md section 6) -------
    if model.moments_enabled:
        qSd = qS * d
        # Overturning moment. POSITIVE C_Malpha is DESTABILISING: nose up
        # (w > 0) gives my > 0, which is further nose up. That is correct for
        # a spin-stabilised shell, whose centre of pressure is ahead of the
        # CG; gyroscopic stiffness, not aerodynamics, keeps it nose-forward.
        ovt = qSd * C_Malpha * inv_V
        mx = 0.0
        my = ovt * w
        mz = -ovt * v
        # Magnus moment.
        magnus_m = qSd * (ks * C_Mpalpha) * p_hat * inv_V
        my += magnus_m * (-v)
        mz += magnus_m * (-w)
        # Spin damping. C_lp is negative, so this always opposes p.
        mx += qSd * (ks * C_lp) * p_hat
        # Pitch and yaw damping. C_mq is negative, so these oppose q and r.
        my += qSd * (ks * C_mq) * q_hat
        mz += qSd * (ks * C_mq) * r_hat
    else:
        mx = 0.0
        my = 0.0
        mz = 0.0

    return (
        fx, fy, fz, mx, my, mz,
        V, mach, delta, qbar, rho, a_snd, altitude,
        u, v, w,
        r00, r01, r02, r10, r11, r12, r20, r21, r22,
    )


def aero_state(t: float, y: np.ndarray, model: FlightModel) -> AeroState:
    """
    Aerodynamic angles, coefficients and the body-frame aerodynamic force and
    moment about the CG. Thin wrapper over _aero_core(); the physics is there.
    """
    c = _aero_core(y, model)
    fx, fy, fz, mx, my, mz = c[0], c[1], c[2], c[3], c[4], c[5]
    V, mach, delta, qbar, rho, a_snd, altitude = c[6], c[7], c[8], c[9], c[10], c[11], c[12]
    u, v, w = c[13], c[14], c[15]

    coeffs = None
    alpha = 0.0
    beta = 0.0
    if model.aero_enabled and V >= V_EPS:
        coeffs = AeroCoefficients(*model.aero.lookup(mach))
        alpha = math.atan2(w, u)
        beta = math.atan2(v, math.hypot(u, w))

    return AeroState(
        altitude=altitude,
        density=rho,
        sound_speed=a_snd,
        mach=mach,
        airspeed=V,
        dynamic_pressure=qbar,
        total_aoa=delta,
        alpha=alpha,
        beta=beta,
        v_rel_body=np.array([u, v, w]),
        coefficients=coeffs,
        force_body=np.array([fx, fy, fz]),
        moment_body=np.array([mx, my, mz]),
    )


def forces_moments(t: float, y: np.ndarray, model: FlightModel):
    """
    Total body-frame force and moment about the CG, including any control
    contribution. Returns (F_body, M_body, AeroState).
    """
    st = aero_state(t, y, model)
    F = st.force_body
    M = st.moment_body
    if model.control is not None:
        dF, dM = model.control(t, y, st)
        F = F + np.asarray(dF, dtype=float)
        M = M + np.asarray(dM, dtype=float)
    return F, M, st


def _deriv_scalars(t: float, y, model: FlightModel) -> list:
    """
    The 13-element state derivative as a plain list of Python floats.

    This is what the integrator calls. It contains no numpy at all: over
    hundreds of thousands of steps, boxing every intermediate into np.float64
    and allocating a small array per call dominates the runtime.

    derivative() is the numpy-facing public wrapper around this.
    """
    env = model.environment

    c = _aero_core(y, model)
    fx = c[0]; fy = c[1]; fz = c[2]
    mx = c[3]; my = c[4]; mz = c[5]
    altitude = c[12]
    r00 = c[16]; r01 = c[17]; r02 = c[18]
    r10 = c[19]; r11 = c[20]; r12 = c[21]
    r20 = c[22]; r21 = c[23]; r22 = c[24]

    if model.control is not None:
        st = aero_state(t, np.asarray(y, dtype=float), model)
        dF, dM = model.control(t, y, st)
        fx += float(dF[0]); fy += float(dF[1]); fz += float(dF[2])
        mx += float(dM[0]); my += float(dM[1]); mz += float(dM[2])

    # --- translation: body force to earth, plus gravity and Coriolis ------
    inv_m = model._inv_mass
    ax = (r00 * fx + r01 * fy + r02 * fz) * inv_m
    ay = (r10 * fx + r11 * fy + r12 * fz) * inv_m
    az = (r20 * fx + r21 * fy + r22 * fz) * inv_m

    if env.include_inverse_square_gravity:
        ratio = atm.R_EARTH / (atm.R_EARTH + altitude)
        az += atm.G0 * ratio * ratio
    else:
        az += atm.G0

    vx = float(y[3])
    vy = float(y[4])
    vz = float(y[5])

    if env.include_coriolis:
        ox, oy, oz = model._omega_ned
        # a_cor = -2 Omega x v
        ax += -2.0 * (oy * vz - oz * vy)
        ay += -2.0 * (oz * vx - ox * vz)
        az += -2.0 * (ox * vy - oy * vx)

    # --- attitude: qdot = 0.5 q (x) [0, omega] ----------------------------
    qw = float(y[6])
    qx = float(y[7])
    qy = float(y[8])
    qz = float(y[9])
    p = float(y[10])
    qr = float(y[11])
    rr = float(y[12])

    # --- rotation: Euler equations for an axisymmetric body ---------------
    dI = model._dI
    return [
        vx,
        vy,
        vz,
        ax,
        ay,
        az,
        0.5 * (-qx * p - qy * qr - qz * rr),
        0.5 * (qw * p + qy * rr - qz * qr),
        0.5 * (qw * qr - qx * rr + qz * p),
        0.5 * (qw * rr + qx * qr - qy * p),
        mx / model._Ix,
        (my + dI * rr * p) / model._It,
        (mz - dI * p * qr) / model._It,
    ]


def derivative(t: float, y: np.ndarray, model: FlightModel) -> np.ndarray:
    """
    The 13-element state derivative. Pure.

    t     time, s. Unused by the ballistic model; present for the control hook.
    y     13-element state.
    model FlightModel bundling projectile, aero, environment, wind, control.

    This is the numpy-facing form. The integrator uses _deriv_scalars()
    directly for speed; both call the same physics.
    """
    return np.array(_deriv_scalars(t, y, model))
