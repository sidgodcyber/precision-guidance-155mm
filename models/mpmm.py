"""
Modified point-mass model (MPMM), STANAG 4355 formulation.

This is the reduced-order model intended to run on the projectile flight
computer. Step 3 calls it in a tight loop to predict the impact point from the
current state; step 7 ports it to C.

IT SHARES THE 6-DOF AERODYNAMICS. The coefficient table, the atmosphere, the
gravity and Coriolis models and the projectile definition are imported from
sim/, not copied. Nothing in this file redefines a coefficient. That is what
makes the model-to-model comparison in docs/MPMM-VALIDATION.md meaningful:
any difference between the two models is structural, not a difference of
inputs.

=============================================================================
STATE VECTOR -- 7 elements
=============================================================================
    y[0:3]  r = [x, y, z]      earth NED position, m   (same frame as sim/)
    y[3:6]  v = [vx, vy, vz]   earth NED velocity, m/s
    y[6]    p                  axial spin rate, rad/s

There is NO attitude state. That is the entire point of the reduction: the
6-DOF carries a quaternion and three body rates and must resolve a 221 rev/s
spin, which forces dt ~ 2e-4 s. The MPMM replaces integrated attitude with an
ALGEBRAIC yaw of repose and can take steps two orders of magnitude larger.

=============================================================================
THE YAW OF REPOSE -- the structural difference between the two models
=============================================================================
STANAG 4355 computes the yaw of repose as the steady-state balance between the
aerodynamic overturning moment and the gyroscopic moment of the spinning
shell, evaluated algebraically from the current state:

    alpha_e = -( 8 Ix p (v x vdot) ) / ( pi rho d^3 C_Malpha |v|^4 )

Source: STANAG 4355 (the NATO standard for the modified point mass trajectory
model), as reproduced in R. Balon and J. Komenda, "Analysis of the 155 mm
ERFB/BB Projectile Trajectory", Advances in Military Technology 1/2006,
equation (15), which cites the standard as its reference [5]. That paper
writes the denominator with a cubic term, (C_Malpha + C_Malpha3 * alpha_e^2);
the cubic coefficient is not available for the M107 in either source used by
this project, so the linear form is used and the omission is recorded in
docs/MPMM-VALIDATION.md.

TWO INDEPENDENT CHECKS THAT THIS FORM IS RIGHT, neither taken on faith:

1. DIRECTION. For velocity along +X and vdot dominated by gravity (+Z, down),
       v x vdot = (V,0,0) x (0,0,g) = (0, -Vg, 0)
   so alpha_e points along +Y -- to the RIGHT -- for positive (right-hand)
   spin. The lift then acts along alpha_e, so the shell drifts right. That is
   the correct and well-known behaviour, and it is asserted in the tests.

2. MAGNITUDE. |v x vdot| = V g cos(theta) for gravity alone, so

       |alpha_e| = 8 Ix p g cos(theta) / (pi rho d^3 C_Malpha V^3)

   Substituting S = pi d^2 / 4 turns this into

       |alpha_e| = 2 Ix p g cos(theta) / (rho V^3 S d C_Malpha)

   which is EXACTLY the closed-form yaw of repose that sim/diagnostics.py
   computes and that step 1 showed the 6-DOF epicyclic motion settles onto
   (see the diagnostics figure in docs/figures). The two models therefore
   agree on the steady state by construction; where they differ is that the
   6-DOF superimposes epicyclic motion on it and the MPMM does not.

WHAT vdot IS. alpha_e depends on the acceleration, which depends on the lift,
which depends on alpha_e. This implementation breaks that loop explicitly:
vdot is evaluated from drag + gravity + Coriolis only -- the terms that do not
depend on the yaw of repose -- so the derivative stays a pure function of the
state with no iteration and no history. The neglected feedback is not merely
assumed small: lift acts along alpha_e, which is perpendicular to v, so its
contribution to (v x vdot) is perpendicular to alpha_e itself and rotates the
yaw of repose rather than amplifying it. `yaw_of_repose(..., iterate=True)`
performs one fixed-point pass including lift and Magnus so the size of the
effect can be measured rather than argued; docs/MPMM-COMPUTE.md reports it.

=============================================================================
COEFFICIENT CORRESPONDENCE
=============================================================================
The shared table is in the 6-DOF body-axis convention (axial force C_X0/C_X2,
normal force C_Nalpha) with reduced spin pd/(2V). STANAG is written in the
wind-axis convention (drag C_D, lift C_Lalpha). The forces below are therefore
expressed directly in the SHARED table's coefficients, using the exact
small-angle relations, rather than by importing a second, renamed table:

    C_D      = C_X0 + (C_X2 + C_Nalpha - C_X0/2) * alpha_e^2
               [drag along the velocity vector; the alpha^2 group is the
                body-axis-to-wind-axis conversion derived in step 1]
    C_Lalpha = C_Nalpha - C_X0
               [lift slope from normal-force slope; the axial force has a
                component perpendicular to the velocity at non-zero yaw]
    Magnus   : written directly from the 6-DOF Magnus force with the same
               C_Ypalpha and the same pd/(2V) reduced spin (see _forces)
    spin     : Ix pdot = qbar S d C_lp (p d / 2V), identical to the 6-DOF

Every one of these is exercised against the 6-DOF in tests/test_mpmm.py.

=============================================================================
FITTING FACTORS -- DELIBERATELY UNUSED
=============================================================================
Operational STANAG 4355 implementations carry fitting factors determined from
firing trials of one specific projectile lot: a form factor i on drag, a lift
factor fL, a Magnus factor QM, a yaw-drag factor QD. They exist to make the
model reproduce a particular firing table.

THEY ARE ALL UNITY HERE AND MUST STAY THAT WAY. Fitting them would make the
MPMM match the firing table trivially, would make the comparison against the
6-DOF meaningless, and would turn the model-error term in docs/MODEL-ERROR.md
-- the number this whole step exists to produce -- into something manufactured
rather than measured. A fitted MPMM says nothing about how the onboard model
behaves on a lot that has not been fired.

They are present as named constants only so that a future reader can see that
they exist, that they were considered, and that they are not in use.
`tests/test_mpmm.py::test_all_fitting_factors_are_unity` asserts it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from sim import atmosphere as atm
from sim.aerodata import REDUCED_RATE_FACTOR, AeroTable
from sim.projectile import Environment, LaunchConditions, Projectile

__all__ = [
    "MPMM_STATE_SIZE",
    "FittingFactors",
    "MpmmModel",
    "unpack",
    "pack",
    "initial_state",
    "state_from_sixdof",
    "yaw_of_repose",
    "derivative",
    "propagate_to_impact",
    "MpmmResult",
]

MPMM_STATE_SIZE = 7

#: Below this airspeed the aerodynamic model is switched off rather than
#: dividing by V. Mirrors sim.dynamics.V_EPS.
V_EPS = 1e-6


@dataclass(frozen=True)
class FittingFactors:
    """
    STANAG 4355 ballistic fitting factors. ALL UNITY, DELIBERATELY.

    See the module docstring. These are recorded, not used. Changing any of
    them away from 1.0 invalidates every number in docs/MODEL-ERROR.md and
    docs/MPMM-VALIDATION.md, because those exist to measure how the model
    behaves WITHOUT trial-fitted corrections.
    """

    form_factor_i: float = 1.0      # drag multiplier
    lift_factor_fL: float = 1.0     # lift multiplier
    magnus_factor_QM: float = 1.0   # Magnus force multiplier
    yaw_drag_factor_QD: float = 1.0  # yaw-drag multiplier

    def all_unity(self) -> bool:
        return (
            self.form_factor_i == 1.0
            and self.lift_factor_fL == 1.0
            and self.magnus_factor_QM == 1.0
            and self.yaw_drag_factor_QD == 1.0
        )


@dataclass(frozen=True)
class MpmmModel:
    """Everything the MPMM derivative needs that is not the state."""

    projectile: Projectile
    aero: AeroTable
    environment: Environment = field(default_factory=Environment)
    wind: Callable[[float], object] = atm.no_wind
    factors: FittingFactors = field(default_factory=FittingFactors)
    #: Include the Magnus force. On by default; used by the term-ablation
    #: study in docs/MPMM-COMPUTE.md.
    include_magnus: bool = True
    #: Include lift (the yaw-of-repose force). Switching this off reduces the
    #: model to a 3-DOF point mass and removes all drift.
    include_lift: bool = True
    #: Include the alpha^2 yaw-drag contribution to C_D.
    include_yaw_drag: bool = True
    #: Perform one fixed-point pass on the yaw of repose, so that the dv/dt
    #: driving alpha_e includes the lift and Magnus accelerations that alpha_e
    #: itself produces. OFF by default: with it off the derivative is a
    #: closed-form function of the state with no inner iteration, which is
    #: what the C port in step 7 wants. What the pass is worth is measured in
    #: docs/MPMM-COMPUTE.md -- it is not assumed to be negligible.
    iterate_yaw: bool = False

    def __post_init__(self):
        p = self.projectile
        object.__setattr__(self, "_d", float(p.diameter))
        object.__setattr__(self, "_S", float(p.reference_area))
        object.__setattr__(self, "_mass", float(p.mass))
        object.__setattr__(self, "_inv_mass", 1.0 / float(p.mass))
        object.__setattr__(self, "_Ix", float(p.I_axial))
        object.__setattr__(self, "_site_alt", float(self.environment.site_altitude))
        object.__setattr__(self, "_wind_is_zero", self.wind is atm.no_wind)
        w = atm.earth_rate_ned(self.environment.latitude, 0.0)
        object.__setattr__(self, "_omega_ned", (float(w[0]), float(w[1]), float(w[2])))


def unpack(y):
    """Split the flat MPMM state into (r, v, p)."""
    return y[0:3], y[3:6], y[6]


def pack(r, v, p) -> np.ndarray:
    y = np.empty(MPMM_STATE_SIZE)
    y[0:3] = r
    y[3:6] = v
    y[6] = p
    return y


def initial_state(projectile: Projectile, launch: LaunchConditions) -> np.ndarray:
    """Muzzle state, matching sim.dynamics.initial_state for the shared states."""
    ce = math.cos(launch.quadrant_elevation)
    se = math.sin(launch.quadrant_elevation)
    ca = math.cos(launch.azimuth)
    sa = math.sin(launch.azimuth)
    v0 = launch.muzzle_velocity
    return pack(
        (0.0, 0.0, 0.0),
        (v0 * ce * ca, v0 * ce * sa, -v0 * se),
        projectile.muzzle_spin(launch.muzzle_velocity),
    )


def state_from_sixdof(y6: np.ndarray) -> np.ndarray:
    """
    Build an MPMM state from a 6-DOF state vector.

    Position, velocity and axial spin transfer directly. The 6-DOF attitude
    quaternion and transverse body rates are DISCARDED -- there is nowhere in
    the reduced model to put them, and that discarded information is exactly
    what docs/MODEL-ERROR.md measures the cost of.
    """
    return pack(y6[0:3], y6[3:6], float(y6[10]))


def _base_acceleration(model: MpmmModel, r, v_rel, v_ground, V, altitude, rho,
                       mach):
    """
    Acceleration from the terms that do NOT depend on the yaw of repose:
    drag along the relative velocity, gravity, and Coriolis.

    This is what the yaw-of-repose expression differentiates against, and it
    is what keeps the derivative a pure function of state.

    NOTE the two velocities. Drag acts along the AIR-relative velocity
    ``v_rel``; the Coriolis acceleration -2 Omega x v is a kinematic term of
    the rotating earth frame and takes the GROUND velocity ``v_ground``. They
    differ whenever there is wind, and sim/dynamics.py makes the same
    distinction -- it uses y[3:6] for Coriolis. Passing v_rel here would be a
    silent error of order (wind speed / flight speed) that no wind-free test
    could catch.
    """
    C_X0, C_X2, C_Nalpha, _C_Ypa, _C_lp, _C_Ma, _C_mq, _C_Mpa = model.aero.lookup(mach)
    qbar = 0.5 * rho * V * V
    # Zero-yaw drag only here; the yaw-drag increment is second order in
    # alpha_e and is added in _forces once alpha_e is known.
    k = -qbar * model._S * model.factors.form_factor_i * C_X0 / (V * model._mass)
    ax = k * v_rel[0]
    ay = k * v_rel[1]
    az = k * v_rel[2]

    if model.environment.include_inverse_square_gravity:
        ratio = atm.R_EARTH / (atm.R_EARTH + altitude)
        az += atm.G0 * ratio * ratio
    else:
        az += atm.G0

    if model.environment.include_coriolis:
        ox, oy, oz = model._omega_ned
        # a_cor = -2 Omega x v, on the ground velocity. Identical to
        # sim/dynamics.py::_aero_core.
        gx, gy, gz = v_ground
        ax += -2.0 * (oy * gz - oz * gy)
        ay += -2.0 * (oz * gx - ox * gz)
        az += -2.0 * (ox * gy - oy * gx)
    return ax, ay, az


def yaw_of_repose(t: float, y: np.ndarray, model: MpmmModel, iterate: bool = False):
    """
    The STANAG 4355 algebraic yaw of repose vector, in earth NED components.

        alpha_e = -( 8 Ix p (v x vdot) ) / ( pi rho d^3 C_Malpha |v|^4 )

    Returns (alpha_x, alpha_y, alpha_z, magnitude). See the module docstring
    for the derivation, the source, and the two checks on it.

    iterate=True performs one fixed-point pass with lift and Magnus included
    in vdot. It is off by default: the derivative is then a pure function of
    the state with no iteration, and docs/MPMM-COMPUTE.md reports how much
    the extra pass is worth.
    """
    v = (float(y[3]), float(y[4]), float(y[5]))
    p = float(y[6])
    altitude = model._site_alt - float(y[2])
    _T, _pr, rho, a_snd = atm.isa_scalars(altitude)

    if model._wind_is_zero:
        v_rel = v
    else:
        w = model.wind(altitude)
        v_rel = (v[0] - float(w[0]), v[1] - float(w[1]), v[2] - float(w[2]))
    V = math.sqrt(v_rel[0] ** 2 + v_rel[1] ** 2 + v_rel[2] ** 2)
    if V < V_EPS:
        return 0.0, 0.0, 0.0, 0.0
    mach = V / a_snd

    ax, ay, az = _base_acceleration(model, y[0:3], v_rel, v, V, altitude, rho, mach)

    if iterate:
        fx, fy, fz, _ = _forces(t, y, model, (ax, ay, az))
        ax += fx * model._inv_mass
        ay += fy * model._inv_mass
        az += fz * model._inv_mass

    C_Malpha = model.aero.lookup(mach)[5]
    # v x vdot
    cx = v_rel[1] * az - v_rel[2] * ay
    cy = v_rel[2] * ax - v_rel[0] * az
    cz = v_rel[0] * ay - v_rel[1] * ax

    denom = math.pi * rho * model._d ** 3 * C_Malpha * V ** 4
    if denom == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    scale = -8.0 * model._Ix * p / denom
    axr, ayr, azr = scale * cx, scale * cy, scale * cz
    return axr, ayr, azr, math.sqrt(axr * axr + ayr * ayr + azr * azr)


def _forces(t: float, y: np.ndarray, model: MpmmModel, base_acc=None):
    """
    Yaw-dependent forces: lift, the yaw-drag increment, and Magnus.
    Returns (fx, fy, fz, alpha_e_magnitude) in earth NED, newtons.

    Separated from the drag/gravity/Coriolis part so that yaw_of_repose can
    optionally iterate on it without recursion.
    """
    v = (float(y[3]), float(y[4]), float(y[5]))
    altitude = model._site_alt - float(y[2])
    _T, _pr, rho, a_snd = atm.isa_scalars(altitude)
    if model._wind_is_zero:
        v_rel = v
    else:
        w = model.wind(altitude)
        v_rel = (v[0] - float(w[0]), v[1] - float(w[1]), v[2] - float(w[2]))
    V = math.sqrt(v_rel[0] ** 2 + v_rel[1] ** 2 + v_rel[2] ** 2)
    if V < V_EPS:
        return 0.0, 0.0, 0.0, 0.0
    mach = V / a_snd
    C_X0, C_X2, C_Nalpha, C_Ypalpha, _C_lp, C_Malpha, _C_mq, _C_Mpa = model.aero.lookup(mach)

    if base_acc is None:
        base_acc = _base_acceleration(model, y[0:3], v_rel, v, V, altitude, rho, mach)
    ax, ay, az = base_acc

    cx = v_rel[1] * az - v_rel[2] * ay
    cy = v_rel[2] * ax - v_rel[0] * az
    cz = v_rel[0] * ay - v_rel[1] * ax
    denom = math.pi * rho * model._d ** 3 * C_Malpha * V ** 4
    if denom == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    scale = -8.0 * model._Ix * float(y[6]) / denom
    aex, aey, aez = scale * cx, scale * cy, scale * cz
    alpha = math.sqrt(aex * aex + aey * aey + aez * aez)

    qbar = 0.5 * rho * V * V
    qS = qbar * model._S
    fx = fy = fz = 0.0

    if model.include_lift:
        # Lift acts along alpha_e. C_Lalpha = C_Nalpha - C_X0 converts the
        # body-axis normal-force slope to the wind-axis lift slope.
        cl = qS * (C_Nalpha - C_X0) * model.factors.lift_factor_fL
        fx += cl * aex
        fy += cl * aey
        fz += cl * aez

    if model.include_yaw_drag:
        # Yaw drag: the alpha^2 increment to C_D, along -vhat.
        cd2 = C_X2 + C_Nalpha - 0.5 * C_X0
        kd = -qS * cd2 * alpha * alpha * model.factors.yaw_drag_factor_QD / V
        fx += kd * v_rel[0]
        fy += kd * v_rel[1]
        fz += kd * v_rel[2]

    if model.include_magnus:
        # Magnus force, written directly from the 6-DOF form:
        #   F = -qbar S C_Ypalpha (p d / 2V) (vhat x alpha_e)
        # with the SAME C_Ypalpha and the SAME pd/(2V) reduced spin.
        p_hat = REDUCED_RATE_FACTOR * float(y[6]) * model._d / V
        km = -qS * C_Ypalpha * p_hat * model.factors.magnus_factor_QM / V
        # vhat x alpha_e  (the 1/V is folded into km)
        mx = v_rel[1] * aez - v_rel[2] * aey
        my = v_rel[2] * aex - v_rel[0] * aez
        mz = v_rel[0] * aey - v_rel[1] * aex
        fx += km * mx
        fy += km * my
        fz += km * mz

    return fx, fy, fz, alpha


def derivative(t: float, y, model: MpmmModel) -> list:
    """
    The 7-element MPMM state derivative. PURE: state in, derivative out, no
    I/O, no globals, no hidden state, no mutation of the input.

    Returns a plain list of Python floats, for the same reason
    sim.dynamics._deriv_scalars does: this is called in a tight loop and
    ported to C in step 7.
    """
    v = (float(y[3]), float(y[4]), float(y[5]))
    altitude = model._site_alt - float(y[2])
    _T, _pr, rho, a_snd = atm.isa_scalars(altitude)

    if model._wind_is_zero:
        v_rel = v
    else:
        w = model.wind(altitude)
        v_rel = (v[0] - float(w[0]), v[1] - float(w[1]), v[2] - float(w[2]))
    V = math.sqrt(v_rel[0] ** 2 + v_rel[1] ** 2 + v_rel[2] ** 2)

    if V < V_EPS:
        return [v[0], v[1], v[2], 0.0, 0.0, atm.G0, 0.0]

    mach = V / a_snd
    ax, ay, az = _base_acceleration(model, y[0:3], v_rel, v, V, altitude, rho, mach)
    if model.iterate_yaw:
        # One fixed-point pass: form the yaw-dependent forces from the
        # drag/gravity/Coriolis acceleration, fold them into dv/dt, and form
        # them again from the improved dv/dt.
        gx, gy, gz, _a0 = _forces(t, y, model, (ax, ay, az))
        acc = (ax + gx * model._inv_mass,
               ay + gy * model._inv_mass,
               az + gz * model._inv_mass)
    else:
        acc = (ax, ay, az)
    fx, fy, fz, _alpha = _forces(t, y, model, acc)
    ax += fx * model._inv_mass
    ay += fy * model._inv_mass
    az += fz * model._inv_mass

    # Spin damping, identical in form to the 6-DOF:
    #     Ix pdot = qbar S d C_lp (p d / 2V)
    C_lp = model.aero.lookup(mach)[4]
    qbar = 0.5 * rho * V * V
    p_hat = REDUCED_RATE_FACTOR * float(y[6]) * model._d / V
    pdot = qbar * model._S * model._d * C_lp * p_hat / model._Ix

    return [v[0], v[1], v[2], ax, ay, az, pdot]


@dataclass
class MpmmResult:
    """Outcome of one MPMM propagation."""

    impact_state: np.ndarray
    impact_time: float
    range_m: float
    drift_m: float
    impact_velocity: float
    impact_angle_rad: float
    max_ordinate: float
    steps: int
    terminated: str
    dt: float
    t: np.ndarray = field(default_factory=lambda: np.empty(0))
    position: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    velocity: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    spin: np.ndarray = field(default_factory=lambda: np.empty(0))
    alpha_e: np.ndarray = field(default_factory=lambda: np.empty(0))
    mach: np.ndarray = field(default_factory=lambda: np.empty(0))


def propagate_to_impact(
    y0,
    model: MpmmModel,
    dt: float = 0.01,
    t0: float = 0.0,
    t_max: float = 300.0,
    log_every: int = 0,
    ground_z: float = 0.0,
) -> MpmmResult:
    """
    Fixed-step RK4 to ground impact, with the final step interpolated to
    z = ground_z exactly (the same scheme as sim/integrate.py).

    log_every = 0 records nothing but the endpoints, which is the mode step 3
    will use. Any positive value records that many steps apart.
    """
    y = [float(v) for v in y0]
    t = float(t0)
    steps = 0
    max_ord = -float(y[2])

    log_t, log_r, log_v, log_p, log_a, log_m = [], [], [], [], [], []

    def _record(tt, yy):
        log_t.append(tt)
        log_r.append((yy[0], yy[1], yy[2]))
        log_v.append((yy[3], yy[4], yy[5]))
        log_p.append(yy[6])
        _ax, _ay, _az, al = yaw_of_repose(tt, yy, model)
        log_a.append(al)
        alt = model._site_alt - yy[2]
        V = math.sqrt(yy[3] ** 2 + yy[4] ** 2 + yy[5] ** 2)
        log_m.append(V / atm.isa_scalars(alt)[3])

    if log_every:
        _record(t, y)

    terminated = "max_time"
    while True:
        if t >= t_max:
            terminated = "max_time"
            break
        y_prev = list(y)
        t_prev = t

        k1 = derivative(t, y, model)
        y2 = [y[i] + 0.5 * dt * k1[i] for i in range(MPMM_STATE_SIZE)]
        k2 = derivative(t + 0.5 * dt, y2, model)
        y3 = [y[i] + 0.5 * dt * k2[i] for i in range(MPMM_STATE_SIZE)]
        k3 = derivative(t + 0.5 * dt, y3, model)
        y4 = [y[i] + dt * k3[i] for i in range(MPMM_STATE_SIZE)]
        k4 = derivative(t + dt, y4, model)
        y = [
            y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(MPMM_STATE_SIZE)
        ]
        t = t_prev + dt
        steps += 1

        if not all(math.isfinite(c) for c in y):
            terminated = "nan"
            y = y_prev
            break

        if -y[2] > max_ord:
            max_ord = -y[2]

        if y[2] >= ground_z and y[5] > 0.0 and steps > 1:
            z0, z1 = y_prev[2], y[2]
            frac = 0.0 if z1 == z0 else (ground_z - z0) / (z1 - z0)
            frac = min(1.0, max(0.0, frac))
            y = [y_prev[i] + frac * (y[i] - y_prev[i]) for i in range(MPMM_STATE_SIZE)]
            t = t_prev + frac * dt
            terminated = "impact"
            break

        if log_every and steps % log_every == 0:
            _record(t, y)

    if log_every:
        _record(t, y)

    speed = math.sqrt(y[3] ** 2 + y[4] ** 2 + y[5] ** 2)
    horiz = math.hypot(y[3], y[4])
    angle = math.atan2(y[5], horiz) if horiz > 0 else 0.0

    return MpmmResult(
        impact_state=np.array(y),
        impact_time=t,
        range_m=y[0],
        drift_m=y[1],
        impact_velocity=speed,
        impact_angle_rad=angle,
        max_ordinate=max_ord,
        steps=steps,
        terminated=terminated,
        dt=dt,
        t=np.array(log_t),
        position=np.array(log_r) if log_r else np.empty((0, 3)),
        velocity=np.array(log_v) if log_v else np.empty((0, 3)),
        spin=np.array(log_p),
        alpha_e=np.array(log_a),
        mach=np.array(log_m),
    )
