"""
Atmosphere, gravity and wind models.

ISA / U.S. Standard Atmosphere 1976, implemented from the layer table so
that the published sea-level and 11 km values reproduce exactly.

Gravity uses the inverse-square variation with altitude, worth ~0.4 % at a
12 km apogee (metres at the target).

Wind is the VELOCITY OF THE AIR in earth NED components. A wind blowing
*from* the north is a NEGATIVE X component. Getting that backwards puts
every range correction the wrong way round.

Pure functions and small immutable dataclasses only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp as _exp
from math import sqrt as _sqrt
from typing import Callable

import numpy as np

__all__ = [
    "R_AIR",
    "GAMMA",
    "G0",
    "R_EARTH",
    "OMEGA_EARTH",
    "AtmoState",
    "isa_atmosphere",
    "isa_scalars",
    "gravity",
    "gravity_ned",
    "coriolis_acceleration",
    "earth_rate_ned",
    "no_wind",
    "constant_wind",
    "geopotential_altitude",
]

# --- physical constants (U.S. Standard Atmosphere 1976 / WGS-84) ----------
R_AIR = 287.05287        # J/(kg K)  specific gas constant for dry air
GAMMA = 1.4              # ratio of specific heats
G0 = 9.80665             # m/s^2     standard gravity
R_EARTH = 6356766.0      # m         ISA 1976 effective earth radius
OMEGA_EARTH = 7.292115e-5  # rad/s   earth rotation rate
P0 = 101325.0            # Pa        sea level pressure
T0 = 288.15              # K         sea level temperature

# ISA layers: (base geopotential altitude m, base temperature K, lapse rate K/m)
_LAYERS = (
    (0.0, 288.15, -0.0065),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 0.001),
    (32000.0, 228.65, 0.0028),
    (47000.0, 270.65, 0.0),
    (51000.0, 270.65, -0.0028),
    (71000.0, 214.65, -0.002),
)

# base pressures, computed once by integrating hydrostatic balance upward
def _base_pressures() -> tuple[float, ...]:
    p = [P0]
    for i in range(len(_LAYERS) - 1):
        h0, t0, lam = _LAYERS[i]
        h1 = _LAYERS[i + 1][0]
        if lam == 0.0:
            p.append(p[i] * np.exp(-G0 * (h1 - h0) / (R_AIR * t0)))
        else:
            t1 = t0 + lam * (h1 - h0)
            p.append(p[i] * (t1 / t0) ** (-G0 / (R_AIR * lam)))
    return tuple(p)


_BASE_P = _base_pressures()

# Flattened layer columns, for the allocation-free hot path.
_LAYER_H = tuple(l[0] for l in _LAYERS)
_LAYER_T = tuple(l[1] for l in _LAYERS)
_LAYER_L = tuple(l[2] for l in _LAYERS)
_N_LAYERS = len(_LAYERS)


@dataclass(frozen=True)
class AtmoState:
    """Air state at a point. Immutable so it can never be mutated in a loop."""

    temperature: float   # K
    pressure: float      # Pa
    density: float       # kg/m^3
    sound_speed: float   # m/s


def geopotential_altitude(h_geometric: float) -> float:
    """Geometric -> geopotential altitude. ~35 m difference at 15 km."""
    return R_EARTH * h_geometric / (R_EARTH + h_geometric)


def isa_scalars(h_geometric: float) -> tuple:
    """
    U.S. Standard Atmosphere 1976 at a geometric altitude in metres,
    returned as the plain tuple (T, p, rho, a).

    This is the hot-path form: it allocates nothing and uses the math module
    rather than numpy, because it is called four times per RK4 step and a
    long trajectory takes hundreds of thousands of steps. isa_atmosphere()
    wraps it for readable use elsewhere -- there is only one implementation
    of the physics.

    Below sea level the sea-level layer is extrapolated (a projectile can
    legitimately impact below the muzzle plane). Above 71 km the top layer is
    extrapolated; artillery never gets there.
    """
    h = R_EARTH * h_geometric / (R_EARTH + h_geometric)

    idx = 0
    for i in range(_N_LAYERS):
        if h >= _LAYER_H[i]:
            idx = i
        else:
            break

    hb = _LAYER_H[idx]
    tb = _LAYER_T[idx]
    lam = _LAYER_L[idx]
    pb = _BASE_P[idx]
    dh = h - hb

    if lam == 0.0:
        t = tb
        p = pb * _exp(-G0 * dh / (R_AIR * tb))
    else:
        t = tb + lam * dh
        p = pb * (t / tb) ** (-G0 / (R_AIR * lam))

    return t, p, p / (R_AIR * t), _sqrt(GAMMA * R_AIR * t)


def isa_atmosphere(h_geometric: float) -> AtmoState:
    """U.S. Standard Atmosphere 1976 at a geometric altitude in metres."""
    t, p, rho, a = isa_scalars(h_geometric)
    return AtmoState(t, p, rho, a)


def gravity(h_geometric: float) -> float:
    """Inverse-square gravity magnitude at geometric altitude h (m)."""
    ratio = R_EARTH / (R_EARTH + h_geometric)
    return G0 * ratio * ratio


def gravity_ned(z: float) -> np.ndarray:
    """
    Gravity vector in the earth NED frame given the DOWN coordinate z.

    Altitude is -z, and gravity points along +Z (down), so this returns
    [0, 0, +g].
    """
    return np.array([0.0, 0.0, gravity(-z)])


def earth_rate_ned(latitude_rad: float, azimuth_rad: float = 0.0) -> np.ndarray:
    """
    Earth rotation rate resolved in the simulation frame.

    The simulation X axis points along the firing azimuth. With
    azimuth = 0 (firing due north) this reduces exactly to the spec form
      Omega = omega_e * [cos(lat), 0, -sin(lat)]
    and the azimuth term is a strict generalisation for other headings.
    """
    cl, sl = np.cos(latitude_rad), np.sin(latitude_rad)
    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    return OMEGA_EARTH * np.array([cl * ca, -cl * sa, -sl])


def coriolis_acceleration(v_ned: np.ndarray, omega_ned: np.ndarray) -> np.ndarray:
    """a_cor = -2 * Omega x v. Tens of metres at 20+ km, so it is included."""
    return -2.0 * np.cross(omega_ned, v_ned)


_ZERO_WIND = (0.0, 0.0, 0.0)


def no_wind(altitude: float):
    """Still air. Returns a tuple of Python floats, not a numpy array, so
    the hot path in dynamics does not pay numpy-scalar arithmetic costs."""
    return _ZERO_WIND


def constant_wind(north: float, east: float) -> Callable[[float], tuple]:
    """
    Altitude-independent wind, as the VELOCITY OF THE AIR.

    constant_wind(-10, 0) is a 10 m/s wind blowing FROM the north
    (a head wind for a northward shot).
    """
    w = (float(north), float(east), 0.0)

    def _wind(altitude: float):
        return w

    return _wind
