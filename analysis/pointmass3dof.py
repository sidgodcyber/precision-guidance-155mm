"""
An independent 3-DOF point-mass trajectory integrator.

This exists ONLY as the independent reference for validation rung 2. It is
deliberately written as a standalone script-style implementation: its own
state layout, its own RK4, its own impact interpolation, and its own copy of
the ISA constants. It shares nothing with sim/ except the drag table, which
is the thing being compared.

If this file imported sim.integrate or sim.atmosphere it would not be an
independent check -- a sign error or an off-by-one in the shared code would
cancel out on both sides and the rung would pass while the model was wrong.

State: [x, y, z, vx, vy, vz] in the same earth NED frame as the 6-DOF
(X downrange, Y right, Z DOWN, so gravity is +Z and altitude is -z).

Model: drag only, acting along the negative relative-velocity direction, with
C_D = C_X0(Mach). No lift, no moments, no spin -- the projectile is a point.
"""

from __future__ import annotations

import math

__all__ = ["PointMassResult", "integrate_point_mass"]

# --- ISA 1976 constants, written out again on purpose ---------------------
_R = 287.05287
_GAMMA = 1.4
_G0 = 9.80665
_RE = 6356766.0
_P0 = 101325.0
_T0 = 288.15
_LAPSE = -0.0065
_H_TROP = 11000.0
_T_TROP = 216.65
_P_TROP = _P0 * (_T_TROP / _T0) ** (-_G0 / (_R * _LAPSE))
_OMEGA_E = 7.292115e-5


def _atmosphere(h_geometric):
    """(density, speed of sound) at geometric altitude h, ISA 1976."""
    h = _RE * h_geometric / (_RE + h_geometric)
    if h < _H_TROP:
        t = _T0 + _LAPSE * h
        p = _P0 * (t / _T0) ** (-_G0 / (_R * _LAPSE))
    else:
        t = _T_TROP
        p = _P_TROP * math.exp(-_G0 * (h - _H_TROP) / (_R * _T_TROP))
    return p / (_R * t), math.sqrt(_GAMMA * _R * t)


class PointMassResult:
    def __init__(self, range_m, drift_m, tof, impact_speed, max_ordinate, steps):
        self.range_m = range_m
        self.drift_m = drift_m
        self.time_of_flight = tof
        self.impact_speed = impact_speed
        self.max_ordinate = max_ordinate
        self.steps = steps

    def __repr__(self):
        return (
            f"PointMassResult(range={self.range_m:.2f} m, tof={self.time_of_flight:.3f} s, "
            f"drift={self.drift_m:.3f} m, v_impact={self.impact_speed:.2f} m/s)"
        )


def integrate_point_mass(
    muzzle_velocity,
    qe_rad,
    mass,
    diameter,
    cd_of_mach,
    azimuth_rad=0.0,
    dt=0.001,
    latitude_rad=None,
    site_altitude=0.0,
    inverse_square_gravity=True,
    t_max=300.0,
):
    """
    Integrate a drag-only point mass to ground impact.

    cd_of_mach(mach) -> zero-yaw axial force coefficient.
    latitude_rad     -> None disables Coriolis; a value enables it.
    """
    area = math.pi * diameter * diameter / 4.0

    ca = math.cos(azimuth_rad)
    sa = math.sin(azimuth_rad)
    ce = math.cos(qe_rad)
    se = math.sin(qe_rad)
    # Frame X is along the firing azimuth, so the launch velocity has no Y.
    state = [
        0.0,
        0.0,
        0.0,
        muzzle_velocity * ce,
        0.0,
        -muzzle_velocity * se,
    ]

    if latitude_rad is None:
        om = None
    else:
        cl = math.cos(latitude_rad)
        sl = math.sin(latitude_rad)
        om = (_OMEGA_E * cl * ca, -_OMEGA_E * cl * sa, -_OMEGA_E * sl)

    def deriv(s):
        vx, vy, vz = s[3], s[4], s[5]
        alt = site_altitude - s[2]
        rho, a_snd = _atmosphere(alt)
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        if speed > 0.0:
            cd = cd_of_mach(speed / a_snd)
            # -1/2 rho V^2 S Cd * vhat  ==  -1/2 rho S Cd * V * v
            k = -0.5 * rho * area * cd * speed / mass
            ax, ay, az = k * vx, k * vy, k * vz
        else:
            ax = ay = az = 0.0
        if inverse_square_gravity:
            g = _G0 * (_RE / (_RE + alt)) ** 2
        else:
            g = _G0
        az += g
        if om is not None:
            ax += -2.0 * (om[1] * vz - om[2] * vy)
            ay += -2.0 * (om[2] * vx - om[0] * vz)
            az += -2.0 * (om[0] * vy - om[1] * vx)
        return [vx, vy, vz, ax, ay, az]

    t = 0.0
    steps = 0
    max_ord = 0.0
    while t < t_max:
        prev = list(state)
        t_prev = t

        k1 = deriv(state)
        s2 = [state[i] + 0.5 * dt * k1[i] for i in range(6)]
        k2 = deriv(s2)
        s3 = [state[i] + 0.5 * dt * k2[i] for i in range(6)]
        k3 = deriv(s3)
        s4 = [state[i] + dt * k3[i] for i in range(6)]
        k4 = deriv(s4)
        state = [
            state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(6)
        ]
        t += dt
        steps += 1

        if -state[2] > max_ord:
            max_ord = -state[2]

        if state[2] >= 0.0 and state[5] > 0.0 and steps > 1:
            z0, z1 = prev[2], state[2]
            frac = 0.0 if z1 == z0 else (0.0 - z0) / (z1 - z0)
            frac = min(1.0, max(0.0, frac))
            final = [prev[i] + frac * (state[i] - prev[i]) for i in range(6)]
            t_imp = t_prev + frac * dt
            speed = math.sqrt(final[3] ** 2 + final[4] ** 2 + final[5] ** 2)
            return PointMassResult(final[0], final[1], t_imp, speed, max_ord, steps)

    speed = math.sqrt(state[3] ** 2 + state[4] ** 2 + state[5] ** 2)
    return PointMassResult(state[0], state[1], t, speed, max_ord, steps)
