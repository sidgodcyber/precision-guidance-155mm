"""
Adapts the existing reduced-order trajectory engine (`models.mpmm`) into a
plain time-series structure for the event engine and the setter's ground
track view.

This module builds no physics of its own: it constructs `MpmmModel` and
calls `propagate_to_impact` exactly as `analysis.monte_carlo._apogee_time`
does, then reshapes the result. Acceleration is not one of MPMM's logged
channels, so it is derived here by finite-differencing the logged velocity --
a display/adapter step, not a new dynamics model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis import roll_servo as rs
from models import mpmm
from sim import aerodata, projectile as pr

__all__ = ["TrajectorySample", "generate_trajectory"]


@dataclass(frozen=True)
class TrajectorySample:
    """One reduced-order trajectory, launch to impact.

    All arrays share length and index. Position channels are earth NED,
    metres; `altitude` is height above the muzzle plane (-z).
    """

    t: np.ndarray
    downrange: np.ndarray
    crossrange: np.ndarray
    altitude: np.ndarray
    velocity_ms: np.ndarray
    acceleration_ms2: np.ndarray
    spin_rad_s: np.ndarray
    alpha_e_rad: np.ndarray
    mach: np.ndarray

    @property
    def duration_s(self) -> float:
        return float(self.t[-1]) if self.t.size else 0.0

    @property
    def range_m(self) -> float:
        return float(self.downrange[-1]) if self.downrange.size else 0.0

    def as_rows(self) -> list:
        """Plain dicts, one per logged sample -- what `fuze.sensors` and the
        setter's plotting module consume, so neither needs numpy."""
        return [
            {
                "t": float(self.t[i]),
                "downrange_m": float(self.downrange[i]),
                "crossrange_m": float(self.crossrange[i]),
                "altitude_m": float(self.altitude[i]),
                "velocity_ms": float(self.velocity_ms[i]),
                "acceleration_ms2": float(self.acceleration_ms2[i]),
                "spin_rad_s": float(self.spin_rad_s[i]),
                "alpha_e_rad": float(self.alpha_e_rad[i]),
                "mach": float(self.mach[i]),
            }
            for i in range(self.t.size)
        ]


def generate_trajectory(base: dict, met=None, dt: float = 0.05) -> TrajectorySample:
    """One MPMM trajectory for `base` (an engagement baseline dict, as
    returned by `analysis.guidance_cep.engagement_context`), optionally flown
    through a `sim.atmosphere.MetProfile`.

    This is a single lightweight simulation run for visualization, NOT a
    campaign result -- it carries no dispersion draw and is not a stand-in
    for the Monte Carlo statistics in `docs/monte_carlo.json`.
    """
    kw = {}
    if met is not None and not met.is_standard:
        kw = {"wind": met.wind, "atmosphere": met.scalars}
    model = mpmm.MpmmModel(
        projectile=pr.M107,
        aero=aerodata.make_m107_table(),
        environment=rs.base_model().environment,
        iterate_yaw=True,
        **kw,
    )
    launch = pr.LaunchConditions.from_mils(base["muzzle_velocity"], base["qe_mils"])
    y0 = mpmm.initial_state(pr.M107, launch)
    r = mpmm.propagate_to_impact(y0, model, dt=dt, log_every=1)

    t = np.asarray(r.t, dtype=float)
    pos = np.asarray(r.position, dtype=float)
    vel = np.asarray(r.velocity, dtype=float)
    speed = np.linalg.norm(vel, axis=1)
    accel = np.gradient(speed, t) if t.size > 1 else np.zeros_like(speed)

    return TrajectorySample(
        t=t,
        downrange=pos[:, 0],
        crossrange=pos[:, 1],
        altitude=-pos[:, 2],
        velocity_ms=speed,
        acceleration_ms2=accel,
        spin_rad_s=np.asarray(r.spin, dtype=float),
        alpha_e_rad=np.asarray(r.alpha_e, dtype=float),
        mach=np.asarray(r.mach, dtype=float),
    )
