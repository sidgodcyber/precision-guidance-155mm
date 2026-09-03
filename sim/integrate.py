"""
Fixed-step RK4 integration, impact detection and trajectory logging.

WHY FIXED-STEP RK4 AND NOT AN ADAPTIVE SOLVER
---------------------------------------------
The step size here is set by the SPIN RATE, not by the trajectory. Muzzle
spin for a 155 mm shell at 1 turn in 20 calibres and 684 m/s is 1386 rad/s,
i.e. 221 revolutions per second. The Magnus and gyroscopic terms are
functions of body-frame quantities that rotate at that rate, so the step has
to resolve a revolution, not a ballistic arc. An adaptive solver with loose
tolerances will happily step over whole revolutions and alias the spin.

    dt = 1e-4 s  ->  45 steps per revolution at the muzzle

A 45 s flight is then ~450,000 steps.

QUATERNION RENORMALISATION
--------------------------
RK4 does not preserve the unit norm, and over half a million steps the drift
compounds. The quaternion is renormalised after every completed step.

IMPACT DETECTION
----------------
Integration runs until z >= 0 while descending (vz > 0). The final step is
then linearly interpolated back to z = 0 exactly. Taking the last step
instead would quantise the impact point at ~0.03 m; interpolating is free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import frames
from .dynamics import STATE_SIZE, FlightModel, aero_state, derivative, unpack

__all__ = ["Trajectory", "rk4_step", "integrate", "IntegrationResult"]


@dataclass
class Trajectory:
    """
    Logged trajectory. Arrays are (n,) or (n, k) with n samples.

    Logs everything the step-5 navigation filter will later need to be
    compared against: true position, velocity, attitude quaternion, body
    rates, plus the aerodynamic diagnostics.
    """

    t: np.ndarray
    position: np.ndarray        # (n,3) earth NED, m
    velocity: np.ndarray        # (n,3) earth NED, m/s
    quaternion: np.ndarray      # (n,4) body->earth
    omega: np.ndarray           # (n,3) body rates, rad/s
    mach: np.ndarray            # (n,)
    airspeed: np.ndarray        # (n,) m/s
    total_aoa: np.ndarray       # (n,) rad
    alpha: np.ndarray           # (n,) rad
    beta: np.ndarray            # (n,) rad
    density: np.ndarray         # (n,) kg/m^3
    dynamic_pressure: np.ndarray  # (n,) Pa
    quat_norm_error: np.ndarray   # (n,) |norm(q) - 1| BEFORE renormalisation

    @property
    def altitude(self) -> np.ndarray:
        """Height above the muzzle plane, m."""
        return -self.position[:, 2]

    @property
    def downrange(self) -> np.ndarray:
        return self.position[:, 0]

    @property
    def crossrange(self) -> np.ndarray:
        """Positive is to the RIGHT of the line of fire."""
        return self.position[:, 1]

    @property
    def spin(self) -> np.ndarray:
        return self.omega[:, 0]

    def euler(self) -> np.ndarray:
        """(n,3) array of (yaw, pitch, roll) in radians. Diagnostic only."""
        return np.array([frames.euler_from_quat(q) for q in self.quaternion])


@dataclass
class IntegrationResult:
    """Outcome of one trajectory."""

    trajectory: Trajectory
    impact_state: np.ndarray          # 13-element state interpolated to z = 0
    impact_time: float                # s
    range_m: float                    # downrange distance at impact, m
    drift_m: float                    # crossrange at impact, positive RIGHT, m
    impact_velocity: float            # m/s
    impact_angle_rad: float           # below horizontal, positive
    max_ordinate: float               # m, apogee above the muzzle plane
    #: Peak total angle of attack, rad. NOTE: this is a maximum over the
    #: LOGGED samples, not over every integration step, because evaluating the
    #: aerodynamic state at every step would roughly double the run time. With
    #: log_every = 200 at dt = 2e-4 that is a sample every 0.04 s against an
    #: epicyclic period of order 0.5 s, which resolves the envelope. A large
    #: log_every (say 100000) will badly under-report this number -- if you
    #: care about peak angle of attack, keep log_every small.
    max_total_aoa: float              # rad
    max_quat_norm_error: float
    steps: int
    terminated: str                   # "impact", "max_time", "nan", "max_steps"
    dt: float

    @property
    def time_of_flight(self) -> float:
        return self.impact_time


def rk4_step(t: float, y: np.ndarray, dt: float, model: FlightModel) -> np.ndarray:
    """One classical RK4 step. The quaternion is NOT renormalised here."""
    k1 = derivative(t, y, model)
    k2 = derivative(t + 0.5 * dt, y + 0.5 * dt * k1, model)
    k3 = derivative(t + 0.5 * dt, y + 0.5 * dt * k2, model)
    k4 = derivative(t + dt, y + dt * k3, model)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _interp_to_ground(y0: np.ndarray, y1: np.ndarray, t0: float, dt: float):
    """
    Linearly interpolate the final step back to z = 0.

    Returns (y_impact, t_impact). The quaternion is renormalised after the
    interpolation, since a linear blend of two unit quaternions is not unit.
    """
    z0, z1 = y0[2], y1[2]
    if z1 == z0:
        return y1.copy(), t0 + dt
    frac = (0.0 - z0) / (z1 - z0)
    frac = min(1.0, max(0.0, frac))
    y = y0 + frac * (y1 - y0)
    y[6:10] = frames.quat_normalize(y[6:10])
    return y, t0 + frac * dt


def integrate(
    y0: np.ndarray,
    model: FlightModel,
    dt: float = 1e-4,
    t_max: float = 200.0,
    log_every: int = 100,
    max_steps: int = 20_000_000,
    stop_on_impact: bool = True,
    progress: Optional[Callable[[float, np.ndarray], None]] = None,
) -> IntegrationResult:
    """
    Integrate from y0 until ground impact.

    y0         13-element initial state (see dynamics.initial_state)
    dt         fixed step, s. Must resolve the spin: see the module docstring.
    log_every  log one sample every this many steps (plus the first and the
               interpolated impact sample)
    """
    if y0.shape != (STATE_SIZE,):
        raise ValueError(f"expected a {STATE_SIZE}-element state, got {y0.shape}")

    y = y0.astype(float).copy()
    y[6:10] = frames.quat_normalize(y[6:10])
    t = 0.0

    log_t: list[float] = []
    log_r: list[np.ndarray] = []
    log_v: list[np.ndarray] = []
    log_q: list[np.ndarray] = []
    log_w: list[np.ndarray] = []
    log_mach: list[float] = []
    log_V: list[float] = []
    log_aoa: list[float] = []
    log_alpha: list[float] = []
    log_beta: list[float] = []
    log_rho: list[float] = []
    log_qbar: list[float] = []
    log_qerr: list[float] = []

    max_norm_err = 0.0
    max_aoa = 0.0
    max_ord = 0.0

    def _log(tt: float, yy: np.ndarray, norm_err: float) -> None:
        st = aero_state(tt, yy, model)
        log_t.append(tt)
        log_r.append(yy[0:3].copy())
        log_v.append(yy[3:6].copy())
        log_q.append(yy[6:10].copy())
        log_w.append(yy[10:13].copy())
        log_mach.append(st.mach)
        log_V.append(st.airspeed)
        log_aoa.append(st.total_aoa)
        log_alpha.append(st.alpha)
        log_beta.append(st.beta)
        log_rho.append(st.density)
        log_qbar.append(st.dynamic_pressure)
        log_qerr.append(norm_err)

    _log(t, y, 0.0)

    terminated = "max_time"
    step = 0
    while True:
        if step >= max_steps:
            terminated = "max_steps"
            break
        if t >= t_max:
            terminated = "max_time"
            break

        y_prev = y.copy()
        t_prev = t

        y_new = rk4_step(t, y, dt, model)

        if not np.all(np.isfinite(y_new)):
            terminated = "nan"
            break

        # Quaternion renormalisation, and record the drift we just removed.
        qn = np.linalg.norm(y_new[6:10])
        norm_err = abs(qn - 1.0)
        if norm_err > max_norm_err:
            max_norm_err = norm_err
        y_new[6:10] = y_new[6:10] / qn

        y = y_new
        t = t_prev + dt
        step += 1

        alt = -y[2]
        if alt > max_ord:
            max_ord = alt

        # Impact test: at or below the muzzle plane while descending.
        if stop_on_impact and y[2] >= 0.0 and y[5] > 0.0 and step > 1:
            y_imp, t_imp = _interp_to_ground(y_prev, y, t_prev, dt)
            st = aero_state(t_imp, y_imp, model)
            max_aoa = max(max_aoa, st.total_aoa)
            _log(t_imp, y_imp, norm_err)
            y, t = y_imp, t_imp
            terminated = "impact"
            break

        if step % log_every == 0:
            st = aero_state(t, y, model)
            max_aoa = max(max_aoa, st.total_aoa)
            _log(t, y, norm_err)
            if progress is not None:
                progress(t, y)

    traj = Trajectory(
        t=np.array(log_t),
        position=np.array(log_r),
        velocity=np.array(log_v),
        quaternion=np.array(log_q),
        omega=np.array(log_w),
        mach=np.array(log_mach),
        airspeed=np.array(log_V),
        total_aoa=np.array(log_aoa),
        alpha=np.array(log_alpha),
        beta=np.array(log_beta),
        density=np.array(log_rho),
        dynamic_pressure=np.array(log_qbar),
        quat_norm_error=np.array(log_qerr),
    )

    r, v, _, _ = unpack(y)
    speed = float(np.linalg.norm(v))
    horiz = float(np.hypot(v[0], v[1]))
    impact_angle = float(np.arctan2(v[2], horiz)) if horiz > 0 else 0.0

    return IntegrationResult(
        trajectory=traj,
        impact_state=y.copy(),
        impact_time=float(t),
        range_m=float(r[0]),
        drift_m=float(r[1]),
        impact_velocity=speed,
        impact_angle_rad=impact_angle,
        max_ordinate=float(max_ord),
        max_total_aoa=float(max(max_aoa, traj.total_aoa.max() if traj.total_aoa.size else 0.0)),
        max_quat_norm_error=float(max_norm_err),
        steps=step,
        terminated=terminated,
        dt=dt,
    )
