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


# ==========================================================================
# STEP 6 -- A REALISED ATMOSPHERE, AND THE MET MESSAGE THAT DESCRIBES IT
# ==========================================================================
# Everything above this line is the STANDARD atmosphere: one deterministic
# profile, the same for every round. A dispersion study needs two more things
# and they are different things:
#
#   THE ATMOSPHERE THE ROUND FLIES THROUGH  -- a realised wind, density and
#       temperature profile, different every round, which is an ALEATORIC
#       quantity and belongs inside a CEP.
#   THE ATMOSPHERE THE FUZE BELIEVES IN     -- the met message uploaded at
#       fuze setting, which is a measurement of the first one taken some
#       hours earlier and somewhere else. The DIFFERENCE between the two is
#       the atmospheric-knowledge error, and docs/ATMOSPHERIC-ERROR.md is
#       what it costs.
#
# The message structure modelled is the NATO ballistic met message (STANAG
# 4061, METB3/METCM): wind direction and speed, air temperature as a
# percentage of standard, and air density as a percentage of standard, given
# for a set of altitude lines above the met datum plane. That is why the
# perturbation below is expressed as a wind vector plus two RATIOS to the ISA
# value rather than as absolute thermodynamic variables -- it is the shape
# the real message has, so a stale message is modelled by staling exactly the
# quantities a real one carries.
#
# THE NUMBERS ARE ESTIMATES AND ARE GRADED AS SUCH. See
# docs/ATMOSPHERIC-ERROR.md section 2: the STRUCTURE (correlated Gaussian
# profiles, variance growing with altitude, exponential decorrelation in
# time) is standard and defensible; the MAGNITUDES are engineering estimates
# and every result that depends on them is reported per unit sigma as well as
# at the nominal, so a reader who disagrees with a sigma can rescale the
# answer without re-running anything.

#: Altitude grid the profiles are realised on, m above the site. 0 to 16 km
#: covers every trajectory in this project: the long engagement's apogee is
#: about 11.9 km and it is well inside.
MET_GRID = tuple(float(h) for h in range(0, 16001, 500))

#: Wind speed 1 sigma PER COMPONENT against altitude, m/s. Piecewise linear
#: between the anchors. Mid-latitude, all seasons, no terrain.
#:
#: CONFIDENCE: LOW on the magnitude, MEDIUM on the shape. That the wind is
#: weak near the ground and strongest around 10-12 km is not controversial;
#: that the 1 sigma is 15 m/s rather than 10 or 20 is an estimate.
WIND_SIGMA_ANCHORS = ((0.0, 3.0), (12000.0, 15.0), (16000.0, 13.0))

#: Vertical correlation length of the wind field, m. Sets how many
#: independent layers a profile has: at 2 km over a 12 km trajectory the
#: round flies through roughly six, which is why a single constant wind is a
#: poor model of a real one and why the profile is realised rather than
#: reduced to one number.
WIND_CORRELATION_M = 2000.0

#: Air density 1 sigma as a FRACTION of the ISA value, and its vertical
#: correlation length. Density anomalies are deep -- they follow the pressure
#: field rather than the local lapse rate -- so the correlation length is
#: longer than the wind's.
DENSITY_SIGMA_ANCHORS = ((0.0, 0.015), (16000.0, 0.030))
DENSITY_CORRELATION_M = 4000.0

#: Air temperature 1 sigma, K, and its vertical correlation length.
#: Temperature reaches the trajectory only through the sound speed and
#: therefore through the Mach-indexed drag table, which is why it is carried
#: separately from density rather than folded into it.
TEMPERATURE_SIGMA_ANCHORS = ((0.0, 3.0), (16000.0, 4.0))
TEMPERATURE_CORRELATION_M = 4000.0

#: Time constant of the exponential decorrelation of each field, hours. A met
#: message `dt` hours old describes an atmosphere correlated exp(-dt/tau)
#: with the one the round actually flies through.
#:
#: CONFIDENCE: LOW. Six hours for the wind is a synoptic-evolution timescale
#: and it is the right order for the free troposphere; the boundary layer
#: turns over faster than that and this model does not separate the two, so
#: the low-altitude wind error at long staleness is UNDERSTATED.
DECORRELATION_HOURS = {"wind": 6.0, "density": 12.0, "temperature": 8.0}

#: Error of a FRESH message: instrument error plus the representativeness
#: error of describing the air over a 16 km trajectory by a sounding taken at
#: the battery. Not zero, and a staleness sweep that started from zero would
#: flatter a fresh message.
#: CONFIDENCE: LOW.
FRESH_MESSAGE_SIGMA = {"wind": 1.5, "density": 0.005, "temperature": 1.0}


def _sigma_at(anchors, h: float) -> float:
    """Piecewise-linear interpolation of a sigma schedule, held flat outside."""
    if h <= anchors[0][0]:
        return anchors[0][1]
    if h >= anchors[-1][0]:
        return anchors[-1][1]
    for i in range(len(anchors) - 1):
        h0, v0 = anchors[i]
        h1, v1 = anchors[i + 1]
        if h0 <= h <= h1:
            return v0 + (v1 - v0) * (h - h0) / (h1 - h0)
    return anchors[-1][1]


def _correlated_profile(rng, grid, sigma_of, length_m: float) -> np.ndarray:
    """
    One realisation of a zero-mean Gaussian field on `grid` with an
    exponential vertical correlation of length `length_m`.

    Generated by the exact AR(1) recursion for an Ornstein-Uhlenbeck process
    sampled on a grid -- x[i] = a x[i-1] + sqrt(1 - a^2) w, with
    a = exp(-dh/L) -- so the realisation has exactly the stated correlation
    structure and unit variance at every node before the sigma schedule is
    applied. Doing it this way rather than by factorising a covariance matrix
    keeps it cheap enough to call twice per round.
    """
    n = len(grid)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        a = _exp(-(grid[i] - grid[i - 1]) / length_m)
        x[i] = a * x[i - 1] + _sqrt(max(0.0, 1.0 - a * a)) * rng.standard_normal()
    return x * np.array([sigma_of(h) for h in grid])


@dataclass(frozen=True)
class MetProfile:
    """
    One realised atmosphere: wind, density and temperature against altitude.

    `density_ratio` and `temperature_ratio` are multipliers on the ISA value
    at the same altitude, which is the form the NATO message uses. Pressure is
    NOT carried independently: it follows from p = rho R T, so the state is
    always thermodynamically self-consistent even though density and
    temperature are perturbed separately. That separation is deliberate and it
    is what the real message does -- at a point in a real atmosphere the two
    are not tied, because the pressure varies too.

    `MetProfile.standard()` is the ISA in still air, which is what every step
    before 6 flew, and it returns `isa_scalars` bit for bit.
    """

    grid: tuple
    wind_north: tuple
    wind_east: tuple
    density_ratio: tuple
    temperature_ratio: tuple
    label: str = ""

    def __post_init__(self):
        wn = np.asarray(self.wind_north, dtype=float)
        we = np.asarray(self.wind_east, dtype=float)
        dr = np.asarray(self.density_ratio, dtype=float)
        tr = np.asarray(self.temperature_ratio, dtype=float)
        object.__setattr__(self, "_wn", wn)
        object.__setattr__(self, "_we", we)
        object.__setattr__(self, "_dr", dr)
        object.__setattr__(self, "_tr", tr)
        # The lookups below run four times per RK4 step over hundreds of
        # thousands of steps, so they use plain lists and the math module
        # rather than numpy indexing, exactly as isa_scalars does.
        g = [float(v) for v in self.grid]
        object.__setattr__(self, "_gl", g)
        object.__setattr__(self, "_wnl", [float(v) for v in wn])
        object.__setattr__(self, "_wel", [float(v) for v in we])
        object.__setattr__(self, "_drl", [float(v) for v in dr])
        object.__setattr__(self, "_trl", [float(v) for v in tr])
        object.__setattr__(self, "_n", len(g))
        object.__setattr__(self, "_h0", g[0])
        object.__setattr__(self, "_dh", g[1] - g[0])
        object.__setattr__(
            self, "_is_standard",
            bool(not wn.any() and not we.any()
                 and np.all(dr == 1.0) and np.all(tr == 1.0)))

    # -- lookup ------------------------------------------------------------
    def _frac(self, altitude: float):
        """(index, fraction) into the uniform grid, clamped at both ends."""
        u = (altitude - self._h0) / self._dh
        if u <= 0.0:
            return 0, 0.0
        if u >= self._n - 1:
            return self._n - 2, 1.0
        i = int(u)
        return i, u - i

    def wind(self, altitude: float):
        """The VELOCITY OF THE AIR at this altitude, m/s, earth NED."""
        i, f = self._frac(altitude)
        wn = self._wnl
        we = self._wel
        return (wn[i] + f * (wn[i + 1] - wn[i]),
                we[i] + f * (we[i + 1] - we[i]),
                0.0)

    def scalars(self, altitude: float):
        """(T, p, rho, a) with this profile's perturbation applied to the ISA."""
        t, p, rho, a = isa_scalars(altitude)
        if self._is_standard:
            return t, p, rho, a
        i, f = self._frac(altitude)
        dr = self._drl
        tr = self._trl
        rr = dr[i] + f * (dr[i + 1] - dr[i])
        tt = tr[i] + f * (tr[i + 1] - tr[i])
        t2 = t * tt
        rho2 = rho * rr
        return t2, rho2 * R_AIR * t2, rho2, _sqrt(GAMMA * R_AIR * t2)

    @property
    def is_standard(self) -> bool:
        return self._is_standard

    # -- construction ------------------------------------------------------
    @classmethod
    def standard(cls) -> "MetProfile":
        n = len(MET_GRID)
        return cls(MET_GRID, (0.0,) * n, (0.0,) * n, (1.0,) * n, (1.0,) * n,
                   label="ISA, still air")

    @classmethod
    def sample(cls, rng, scale: float = 1.0, grid=MET_GRID,
               label: str = "") -> "MetProfile":
        """
        One realisation of the atmosphere.

        `scale` multiplies every sigma, so a result that depends on the
        assumed variability can be reported per unit sigma rather than only at
        the nominal. `scale = 0` returns the standard atmosphere exactly.
        """
        g = list(grid)
        if scale == 0.0:
            return cls.standard()
        wn = _correlated_profile(
            rng, g, lambda h: scale * _sigma_at(WIND_SIGMA_ANCHORS, h),
            WIND_CORRELATION_M)
        we = _correlated_profile(
            rng, g, lambda h: scale * _sigma_at(WIND_SIGMA_ANCHORS, h),
            WIND_CORRELATION_M)
        dr = _correlated_profile(
            rng, g, lambda h: scale * _sigma_at(DENSITY_SIGMA_ANCHORS, h),
            DENSITY_CORRELATION_M)
        dt = _correlated_profile(
            rng, g, lambda h: scale * _sigma_at(TEMPERATURE_SIGMA_ANCHORS, h),
            TEMPERATURE_CORRELATION_M)
        t_isa = np.array([isa_scalars(h)[0] for h in g])
        return cls(tuple(g), tuple(wn), tuple(we), tuple(1.0 + dr),
                   tuple(1.0 + dt / t_isa), label=label or "realised atmosphere")

    def message(self, rng, age_hours, scale: float = 1.0) -> "MetProfile":
        """
        The met message that would have been uploaded for this round: a
        measurement of THIS atmosphere taken `age_hours` ago.

        Each field is staled by the standard result for a stationary Gaussian
        process -- a sample correlated rho = exp(-dt/tau) with the truth is
        `rho * truth + sqrt(1 - rho^2) * independent` -- and a fresh-message
        measurement error is added on top, because even a message uploaded at
        the moment of firing is a sounding taken somewhere else.

        `age_hours = None` returns the STANDARD atmosphere, which is the "no
        met message at all" case and the bound on all of them.

        Differencing a round flown through `self` with a fuze given this is
        the whole of the atmospheric-knowledge measurement.
        """
        if age_hours is None:
            return MetProfile.standard()
        g = list(self.grid)
        t_isa = np.array([isa_scalars(h)[0] for h in g])

        def stale(truth, key, sigma_of, length, fresh_sigma):
            rho = _exp(-float(age_hours) / DECORRELATION_HOURS[key])
            indep = _correlated_profile(rng, g, sigma_of, length)
            meas = _correlated_profile(rng, g, lambda h: scale * fresh_sigma,
                                       length)
            return (rho * np.asarray(truth)
                    + _sqrt(max(0.0, 1.0 - rho * rho)) * indep + meas)

        wn = stale(self._wn, "wind",
                   lambda h: scale * _sigma_at(WIND_SIGMA_ANCHORS, h),
                   WIND_CORRELATION_M, FRESH_MESSAGE_SIGMA["wind"])
        we = stale(self._we, "wind",
                   lambda h: scale * _sigma_at(WIND_SIGMA_ANCHORS, h),
                   WIND_CORRELATION_M, FRESH_MESSAGE_SIGMA["wind"])
        dr = stale(self._dr - 1.0, "density",
                   lambda h: scale * _sigma_at(DENSITY_SIGMA_ANCHORS, h),
                   DENSITY_CORRELATION_M, FRESH_MESSAGE_SIGMA["density"])
        dt = stale((self._tr - 1.0) * t_isa, "temperature",
                   lambda h: scale * _sigma_at(TEMPERATURE_SIGMA_ANCHORS, h),
                   TEMPERATURE_CORRELATION_M, FRESH_MESSAGE_SIGMA["temperature"])
        return MetProfile(tuple(g), tuple(wn), tuple(we), tuple(1.0 + dr),
                          tuple(1.0 + dt / t_isa),
                          label=f"met message, {float(age_hours):g} h old")

    def ballistic_wind(self) -> tuple:
        """
        The single constant wind that best summarises this profile, m/s.

        Used only where a scalar is unavoidable: `gnc.navigation.GunData`
        carries one wind vector and not a profile, because the warm start
        propagates a reduced-order model that is given one. It is NOT what the
        trajectory flies through and no reported result is derived from it.
        """
        return (float(self._wn.mean()), float(self._we.mean()), 0.0)

    def summary(self) -> dict:
        """Scalars that describe this realisation, for the campaign JSON."""
        return {
            "label": self.label,
            "wind_rms_ms": float(np.sqrt((self._wn ** 2 + self._we ** 2).mean())),
            "wind_max_ms": float(np.hypot(self._wn, self._we).max()),
            "ballistic_wind_ms": list(self.ballistic_wind()[:2]),
            "density_ratio_mean": float(self._dr.mean()),
            "density_ratio_max_dev": float(np.abs(self._dr - 1.0).max()),
            "temperature_ratio_mean": float(self._tr.mean()),
        }


#: The standard atmosphere as a profile, for code that wants one uniformly.
STANDARD_MET = MetProfile.standard()

__all__ = list(__all__) + [
    "MET_GRID",
    "MetProfile",
    "STANDARD_MET",
    "WIND_SIGMA_ANCHORS",
    "WIND_CORRELATION_M",
    "DENSITY_SIGMA_ANCHORS",
    "DENSITY_CORRELATION_M",
    "TEMPERATURE_SIGMA_ANCHORS",
    "TEMPERATURE_CORRELATION_M",
    "DECORRELATION_HOURS",
    "FRESH_MESSAGE_SIGMA",
]
