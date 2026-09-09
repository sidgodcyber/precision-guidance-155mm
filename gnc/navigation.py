"""
Step 5 navigation: a 15-state loosely-coupled error-state extended Kalman
filter, and the roll estimate the servo lives on.

docs/NAV-CONSISTENCY.md, docs/NAV-CEP.md and docs/NAV-DEGRADATION.md are this
module's account. `gnc/sensors.py` is what feeds it.

THE STATE
---------
    dr (3)   position error, earth NED
    dv (3)   velocity error, earth NED
    dpsi (3) attitude error, earth-frame small-angle
    dbg (3)  gyro bias
    dba (3)  accelerometer bias

Error-state rather than total-state, for the usual two reasons: the error is
small so its dynamics are linear where the vehicle's are not, and the
quaternion stays a quaternion because the filter never touches it except
through a reset.

The attitude propagated is the NOSE's, because the IMU is in the despun
section. That is not a compromise. `gnc/sensors.py` shows that the nose
frame's 3-2-1 Euler roll is IDENTICALLY `phi_body + phi_rel` = `phi_nose`,
the variable the servo controls, and that its pitch and yaw are IDENTICALLY
the body's. Navigating the nose gives the servo its controlled variable with
no transformation and gives guidance the body attitude with no transformation.

WHY LOOSELY COUPLED, AND WHAT TIGHT WOULD BUY
---------------------------------------------
Loose: the receiver forms a position/velocity solution and the filter
consumes it. Tight: the filter consumes pseudoranges and deltaranges and
forms the solution itself.

Loose is adequate here and it is adequate for a measurable reason. The
accuracy class is set by a 3 m single-frequency receiver against a 43 s
flight, and over that flight the inertial solution's job is to bridge 0.2 s
between fixes and to supply attitude -- not to carry the position. At that
duty cycle the two couplings differ by centimetres.

TIGHT COUPLING IS THE PRODUCTION UPGRADE PATH, AND THE REASON IS OUTAGE, NOT
ACCURACY. A loosely coupled filter gets NOTHING from three satellites: below
four in view the receiver reports no solution and the filter free-runs on the
IMU. A tightly coupled one uses every pseudorange it has, so three satellites
still constrain the solution, and with a good clock model two can. On a
spinning body whose antenna pattern sweeps -- so that satellites drop in and
out at 208 Hz -- partial visibility is the normal condition rather than the
exception, and docs/NAV-DEGRADATION.md section 2 measures what full outages
cost precisely so that the value of not having them is priced.

THE ROLL DEGREE OF FREEDOM IS DIFFERENT FROM THE OTHER TWO
-----------------------------------------------------------
Pitch and yaw are slow, well observed, and barely matter: the specific force
is 98 % axial, so a pitch error rotates a 2.1 g vector and a roll error
rotates the 0.06 g that is left. Roll is the opposite -- it barely affects
the position solution and it is the servo's entire input.

And roll is the one axis where the gyro is not available when it is first
needed. On the adopted trajectory the despun gyro is PINNED AT FULL SCALE for
the first 2.10 s after deployment while the nose despins through 74 952 deg/s.
The filter therefore does three things this file makes explicit:

  1. it detects saturation from the sensor's own flag and stops trusting the
     gyro on that axis, rather than integrating a clipped constant;
  2. it takes the roll rate from the MAGNETOMETER during that window. At
     500 Hz sampling the roll phase advances at most 2.62 rad per sample at
     the deployment spin of 1308 rad/s, which is inside pi -- the despin is
     unambiguously resolvable by an analog AMR bridge sampled at the IMU rate,
     and that is why `MagSpec.bandwidth` is a HIGH-confidence entry in the
     register;
  3. it never lets a pinned rate look like a settled one. A saturated gyro
     returns a CONSTANT, and a constant is exactly what
     `gnc.roll_control.RollAngleController`'s model-free settling detector is
     looking for. `NavigationSystem.rate_valid` is what stops it.

WHAT IS NOT MODELLED, AND WHY THAT IS DEFENSIBLE HERE
------------------------------------------------------
Earth rotation and transport rate are omitted from the attitude propagation.
Over 43 s the earth turns 0.18 deg and the transport rate over 16 km is
0.14 deg; both are below the 0.9 deg roll floor and far below the pitch and
yaw the magnetometer holds. `sim/dynamics.py` keeps Coriolis as an option in
the TRUTH model and `gnc/sensors.py` subtracts it from the specific force
when it is on, so turning it on does not silently corrupt the sensor model --
it just leaves the filter with an unmodelled term, which is what step 6 would
have to price.

PURITY
------
No module-level mutable state, no I/O, deterministic given the seed handed to
the sensor suite. `NavigationSystem` is a `step_hook`, exactly as `BrakeLaw`
and `GuidanceLaw` are, and it adds no third seam to the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from sim import atmosphere as atm
from sim import frames

from . import sensors as sn

__all__ = [
    "NavConfig", "GunData", "warm_start_state", "NavigationFilter",
    "NavigationSystem", "NavSample", "IDX",
]

#: Index blocks into the 15-element error state. Named once, used everywhere,
#: for the same reason `sim.dynamics` forbids indexing the 6-DOF state by hand.
IDX = {"r": slice(0, 3), "v": slice(3, 6), "psi": slice(6, 9),
       "bg": slice(9, 12), "ba": slice(12, 15)}
N_STATES = 15

#: Nominal gravity, for the one place the F matrix needs a specific force
#: before the accelerometers are usable.
G_NOMINAL = 9.80665


def _cross3(a, b) -> np.ndarray:
    """
    Cross product of two 3-vectors.

    `numpy.cross` is general over shapes and axes and spends most of its time
    proving that; on the 3-vectors this module actually uses it is a third of
    the whole navigation runtime. Measured: 17.4 s of a 52 s profile.
    """
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def _skew(v) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class NavConfig:
    """
    Every number the filter runs on, and where it came from.

    THE TUNING RULE OF THIS STEP. Nothing here was moved to make a CEP look
    better. The process noises are the sensor specs; the measurement noises
    are the receiver's own claims; and the two that are NOT directly a spec --
    `gnss_position_inflation` and `mag_sigma_floor` -- were set by the Task E
    consistency test and by nothing else. docs/NAV-CONSISTENCY.md section 4
    shows the NEES before and after, and docs/NAV-CEP.md reports whatever CEP
    that filter then produced.
    """

    imu_rate: float = 500.0
    #: Covariance propagation rate. The strapdown runs at the IMU rate; the
    #: covariance does not need to, and 100 Hz is 5x the fastest measurement.
    covariance_rate: float = 100.0
    #: Magnetometer attitude update rate in normal flight. Raised to the IMU
    #: rate while the gyro is saturated, because then it is the only thing
    #: holding the roll.
    mag_rate: float = 100.0
    mag_rate_saturated: float = 500.0

    #: GNSS measurement-noise inflation. SET BY TASK E, NOT BY TASK F.
    #:
    #: The receiver's 3 m is right for one fix and wrong for two hundred of
    #: them: 80 % of the error is a Gauss-Markov process at a 100 s
    #: correlation time (`sensors.GnssSpec.error_tau`), so a filter that
    #: averages 5 Hz fixes as if they were independent drives its own
    #: covariance below its actual error and reports a confidence it does not
    #: have. A 15-state filter has nowhere to put a GNSS bias state, so the
    #: correlated part is CONSIDERED rather than estimated, by inflating R.
    #: The value is whatever makes the NEES sit inside its chi-square bounds.
    gnss_position_inflation: float = 1.0
    gnss_velocity_inflation: float = 1.0

    #: CONSIDER-COVARIANCE FLOORS. SET BY TASK E, AND THE REASON R INFLATION
    #: IS NOT ENOUGH ON ITS OWN.
    #:
    #: A Kalman filter driven by a CORRELATED measurement error converges to
    #: that error, and its covariance converges to something smaller --
    #: however large R is. Measured on the tuning sweep: inflating the GNSS
    #: position R by 200x moved the 6-state NEES only from 91 to 38 against an
    #: expected 6, and made the position error WORSE, from 3.5 m to 20.7 m,
    #: because a filter told not to believe its measurements believes its
    #: propagation instead. Inflation trades error for confidence; it does not
    #: buy consistency.
    #:
    #: What does is a floor: the filter declares that it cannot know its
    #: position better than the receiver's correlated error, no matter how
    #: many fixes it averages. That is the Schmidt-Kalman "consider"
    #: treatment of a bias state the 15-state filter has no room for, applied
    #: as a diagonal congruence so the covariance stays symmetric and positive
    #: semi-definite and the correlation structure survives.
    #:
    #: The floors are DERIVED from the sensor specs, not chosen --
    #: `NavigationFilter.__init__` computes them -- and these multipliers are
    #: the only free numbers, set by the NEES and by nothing else.
    #: ADOPTED after the Task E sweep. The position and velocity floors are
    #: taken exactly as derived -- scale 1.0 -- and only the attitude one
    #: needed a factor, because a scalar floor cannot express an
    #: azimuth-dependent roll observability and the measured roll error on the
    #: adopted (worst-case) azimuth is larger than the median geometry implies.
    #:
    #: 1.8 is on a plateau, not at a fitted point: over the swept range 1.0 to
    #: 2.4 the position and velocity errors do not move AT ALL
    #: (2.11/2.62/5.57 m and 0.52/0.36/0.53 m/s at every value), because this
    #: knob changes only what the filter CLAIMS, never what it estimates. A
    #: consider factor that cannot move the estimate cannot be fitting a CEP.
    floor_enabled: bool = True
    floor_position_scale: float = 1.0
    floor_velocity_scale: float = 1.0
    floor_attitude_scale: float = 1.8

    #: Magnetometer measurement noise, as a fraction of field magnitude. The
    #: white part is nanotesla; this is the CALIBRATION RESIDUAL, which is not
    #: noise at all -- it is a fixed distortion the filter cannot estimate --
    #: and declaring it as noise is the honest way to stop the filter
    #: believing the magnetometer more than it deserves.
    mag_sigma_floor: float = 0.015
    #: The velocity-vector attitude reference. 1.78 deg is the MEASURED rms
    #: angle between the body axis and the air-relative velocity on the
    #: adopted trajectory (mean 1.59, worst 4.40), not an assumption -- see
    #: `NavigationFilter.update_velocity_attitude`. Without this reference the
    #: magnetometer leaves 53 % of a roll error unobservable.
    alignment_sigma: float = 1.78 * sn.DEG
    alignment_rate: float = 20.0
    #: The velocity reference is only as good as the velocity. Above this the
    #: update is skipped -- see the guard in `NavigationSystem.sample`.
    alignment_max_sigma_v: float = 5.0

    #: Below this transverse-field fraction the roll is not observable from
    #: the magnetometer and the update is skipped rather than divided by a
    #: small number. sin(15 deg).
    mag_min_transverse: float = 0.259

    #: Initial covariance when the filter starts from a GNSS fix alone.
    p0_position: float = 10.0
    p0_velocity: float = 5.0
    p0_attitude: float = 10.0 * sn.DEG
    p0_roll_cold: float = math.pi
    #: ... and when it starts from gun data. See `warm_start_state`.
    p0_position_warm: float = 30.0
    p0_velocity_warm: float = 4.0
    p0_attitude_warm: float = 3.0 * sn.DEG

    #: A solution is USABLE when its position error covariance is below this
    #: and it has been corrected at least once, or when the warm start's own
    #: a-priori covariance is. Task D measures the time to reach it.
    usable_position_sigma: float = 30.0
    usable_velocity_sigma: float = 3.0

    #: Validity. The filter declares itself INVALID rather than confident and
    #: wrong -- the distinction docs/NAV-DEGRADATION.md section 3 exists to
    #: test. The threshold is on the filter's own claimed sigma, which is the
    #: only thing it has; Task E is what licenses believing that claim.
    invalid_position_sigma: float = 150.0
    #: Free-running on the IMU alone, position error grows as the accelerometer
    #: bias integrates twice. This is a hard time limit on top of the
    #: covariance test, because a covariance can be optimistic and a clock
    #: cannot.
    max_coast_s: float = 30.0


# ===========================================================================
# Task D: the warm start
# ===========================================================================
@dataclass
class GunData:
    """
    What the fuze setter uploads before firing.

    The kit ALREADY has a setter and it already uploads target coordinates.
    Everything below is data the fire-control solution has computed anyway, in
    the message it is already sending. The marginal cost of a warm start is
    therefore a few dozen bytes on a link that exists, which is why it is worth
    checking whether it buys anything -- and Task D checks rather than assumes.

    `muzzle_velocity_sigma` is the gun's round-to-round dispersion, not a
    measurement error: the setter uploads the EXPECTED muzzle velocity from
    the firing table and the charge temperature, and this round will differ
    from it. That dispersion is the dominant term in the a-priori trajectory
    and it is what sets `p0_position_warm`.
    """

    muzzle_velocity: float
    quadrant_elevation: float          # rad
    azimuth: float = 0.0               # rad, relative to the trajectory X axis
    site_position: tuple = (0.0, 0.0, 0.0)   # the origin IS the muzzle
    muzzle_velocity_sigma: float = 2.4       # m/s, 1 sigma round to round
    quadrant_elevation_sigma: float = 0.5e-3  # rad (about 0.9 mil)
    azimuth_sigma: float = 0.5e-3            # rad
    #: The met message. Carried as the wind the a-priori propagation uses; the
    #: atmosphere is ISA, as everywhere else in this project.
    wind: tuple = (0.0, 0.0, 0.0)


def warm_start_state(gun: GunData, projectile, environment, t: float,
                     dt: float = 0.05) -> tuple:
    """
    The a-priori state at time `t`, from gun data alone.

    Propagates the SAME reduced-order model `gnc.guidance.ImpactPredictor`
    already carries -- `models.mpmm` -- from the muzzle. That reuse is the
    point: a warm start costs the flight computer no new model, no new table
    and no new code path, only a propagation it is already able to run.

    Returns (position, velocity, spin, sigma_position, sigma_velocity), with
    the sigmas propagated by differencing the trajectory against a
    one-sigma-perturbed one in each of the three uploaded quantities. That is
    a linearised covariance and it is cheap; over 8 s of flight the
    trajectory is close enough to linear in muzzle velocity that the
    difference from a full sigma-point propagation is below the terms it is
    being compared with.
    """
    from models import mpmm
    from sim import projectile as pr

    def fly(dmv: float, dqe: float, daz: float):
        launch = pr.LaunchConditions(
            muzzle_velocity=gun.muzzle_velocity + dmv,
            quadrant_elevation=gun.quadrant_elevation + dqe,
            azimuth=gun.azimuth + daz)
        y = mpmm.initial_state(projectile, launch)
        model = mpmm.MpmmModel(projectile=projectile,
                               aero=_warm_aero(), environment=environment,
                               iterate_yaw=False)
        # Land exactly on `t`. A fixed-step propagator asked for t_max
        # smaller than its own step overshoots by a whole step, which at
        # 684 m/s is 34 m -- and it overshoots MOST at the small times the
        # warm start is being judged on.
        steps = max(1, int(math.ceil(float(t) / dt)))
        r = mpmm.propagate_to_impact(y, model, dt=float(t) / steps, t0=0.0,
                                     t_max=float(t), ground_z=1.0e9)
        s = np.asarray(r.impact_state, dtype=float)
        return s[0:3].copy(), s[3:6].copy(), float(s[6])

    r0, v0, p0 = fly(0.0, 0.0, 0.0)
    dr = np.zeros(3)
    dv = np.zeros(3)
    for d, s in ((0, gun.muzzle_velocity_sigma), (1, gun.quadrant_elevation_sigma),
                 (2, gun.azimuth_sigma)):
        pert = [0.0, 0.0, 0.0]
        pert[d] = s
        r1, v1, _ = fly(*pert)
        dr += (r1 - r0) ** 2
        dv += (v1 - v0) ** 2
    return r0, v0, p0, np.sqrt(dr), np.sqrt(dv)


def _warm_aero():
    """
    The onboard aerodynamic table for the a-priori propagation.

    The BARE SHELL table, not the kit one: before deployment there is no kit
    in the flow. `gnc.guidance.kit_drag_table` is the deployed-phase
    equivalent and the two must not be confused -- using the kit table here
    would over-predict drag by the panel area for a phase in which the panels
    are stowed.
    """
    from sim import aerodata
    return aerodata.make_m107_table()


# ===========================================================================
# The filter
# ===========================================================================
class NavigationFilter:
    """
    15-state loosely-coupled error-state EKF.

    Sequential scalar measurement updates throughout. Every measurement this
    filter takes has diagonal R -- the receiver reports per-axis sigmas, the
    magnetometer axes are independent after calibration, the C/N0 roll is
    scalar -- so the sequential form is EXACT, not an approximation, and it
    replaces a 3x3 or 6x6 inverse with three or six divisions. On a fuze whose
    compute budget step 7 has to size, that is the difference between a filter
    that fits and one that does not.

    The nominal state is carried outside the covariance:

        r   position, earth NED, m
        v   velocity, earth NED, m/s
        q   attitude quaternion, NOSE -> earth
        bg  gyro bias, nose axes, rad/s
        ba  accelerometer bias, nose axes, m/s^2

    and `P` is the covariance of the 15 errors in `IDX` order.
    """

    def __init__(self, config: NavConfig, gyro: sn.GyroSpec,
                 accel: sn.AccelSpec, field: sn.GeomagneticField,
                 imu_lever=(0.399, 0.025, 0.0), gnss_spec=None):
        self.config = config
        self.gyro = gyro
        self.accel = accel
        self.field = field
        self.imu_lever = np.asarray(imu_lever, dtype=float)

        self.r = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.P = np.eye(N_STATES)
        self.initialised = False
        self.t = 0.0

        # Bookkeeping the consistency test reads back.
        keys = ("gnss_pos", "gnss_vel", "mag", "cn0", "align")
        self.updates = {k: 0 for k in keys}
        self.innovations: dict = {k: [] for k in keys}
        self.nis: dict = {k: [] for k in keys}
        self.nis_t: dict = {k: [] for k in keys}
        self.last_correction_t: Optional[float] = None
        #: The last time a measurement corrected POSITION, which on this kit
        #: means the last GNSS fix. Tracked separately from
        #: `last_correction_t` because the magnetometer and the velocity
        #: alignment reference correct ATTITUDE and nothing else -- see the
        #: note in `NavigationSystem.valid`.
        self.last_position_correction_t: Optional[float] = None
        self._omega = np.zeros(3)
        self._omega_prev = np.zeros(3)
        self._has_omega = False
        self._f_nose = np.zeros(3)
        self._sat_axes = np.zeros(3, dtype=bool)
        self._sat_sigma = 0.0

        # -- the consider floors, derived from the sensor specs -----------
        # Position: the correlated part of the receiver's error. sqrt(0.8) of
        # 3 m horizontal and 5 m vertical.
        g = gnss_spec if gnss_spec is not None else sn.GNSS_L1
        self.gnss_spec = g
        c = math.sqrt(max(g.correlated_fraction, 0.0))
        self._floor_r = np.array([g.sigma_horizontal, g.sigma_horizontal,
                                  g.sigma_vertical]) * c             * config.floor_position_scale
        # Velocity: what the drift of that bias looks like over the interval
        # the filter can average across, sigma_c * sqrt(2 / (T tau)) with T the
        # correlation time itself as the longest useful window.
        self._floor_v = (float(np.mean(self._floor_r))
                         * math.sqrt(2.0) / max(g.error_tau, 1e-9)
                         * config.floor_velocity_scale) * g.error_tau ** 0.5
        # Attitude: the two references cannot do better than their own fixed
        # errors -- the magnetometer's calibration residual and the angle of
        # attack -- neither of which is noise that averages down.
        self._floor_psi = config.alignment_sigma * config.floor_attitude_scale

        # Pre-computed process-noise densities.
        self._q_att = gyro.arw ** 2
        self._q_vel = accel.vrw ** 2
        self._q_bg = 2.0 * gyro.bias_stability ** 2 / max(gyro.bias_tau, 1e-9)
        self._q_ba = 2.0 * accel.bias_stability ** 2 / max(accel.bias_tau, 1e-9)

    # -- initialisation ---------------------------------------------------
    def initialise(self, t: float, r, v, q, sigma_r, sigma_v, sigma_att) -> None:
        self.t = float(t)
        self.r = np.asarray(r, dtype=float).copy()
        self.v = np.asarray(v, dtype=float).copy()
        self.q = frames.quat_normalize(np.asarray(q, dtype=float).copy())
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        p = np.zeros(N_STATES)
        p[IDX["r"]] = np.asarray(sigma_r, dtype=float) ** 2
        p[IDX["v"]] = np.asarray(sigma_v, dtype=float) ** 2
        p[IDX["psi"]] = np.asarray(sigma_att, dtype=float) ** 2
        p[IDX["bg"]] = self.gyro.bias_repeatability ** 2
        p[IDX["ba"]] = self.accel.bias_repeatability ** 2
        self.P = np.diag(p)
        self.initialised = True

    def set_attitude(self, q, sigma) -> None:
        """
        Declare the attitude DETERMINED rather than estimated, and say so in
        the covariance.

        Used only in the MODEL phase, where there is no usable gyro and the
        attitude comes from two direct observations each sample -- the velocity
        vector and the magnetometer. That is a determination, not a
        propagation, and integrating a process noise across it is wrong in a
        way that does not stay local:

          measured, before this existed -- the attitude covariance integrated
          the substituted magnetometer rate's uncertainty at 1308 rad/s for the
          whole pre-deployment phase and reached sigma_roll = 625 deg. The
          first magnetometer update then crushed it to 0.47 deg, but the
          CROSS-covariances with the gyro bias and the velocity survived at
          their inflated values, and the filter spent the rest of the flight
          attributing every attitude innovation to a gyro bias. That estimate
          ran away to 6 731 deg/hr against a physical 37.5, and took the roll
          (49 deg), the velocity (30 m/s) and the position with it.

        So the attitude block is SET, and its cross terms are zeroed, because
        a directly determined attitude really is uncorrelated with the states
        that did not determine it.
        """
        self.q = frames.quat_normalize(np.asarray(q, dtype=float))
        s = np.asarray(sigma, dtype=float)
        i = IDX["psi"]
        self.P[i, :] = 0.0
        self.P[:, i] = 0.0
        self.P[i, i] = np.diag(s ** 2)

    # -- attitude helpers -------------------------------------------------
    @property
    def dcm(self) -> np.ndarray:
        """Nose -> earth."""
        return frames.dcm_from_quat(self.q)

    @property
    def euler(self) -> tuple:
        """(heading, pitch, roll) of the NOSE frame. Roll IS `phi_nose`."""
        return frames.euler_from_quat(self.q)

    # -- propagation ------------------------------------------------------
    def propagate_state(self, dt: float, gyro: np.ndarray, accel: np.ndarray,
                        saturated_axes=None, substitute_rate=None,
                        substitute_sigma: float = 0.0) -> None:
        """
        Strapdown mechanisation, one IMU period. Runs at the full IMU rate.

        `saturated_axes` marks axes whose reading is pinned. On those the
        clipped value is discarded and `substitute_rate` is used instead --
        the roll rate the MAGNETOMETER phase supplies, which is an independent
        measurement and not the filter's own opinion.

        SUBSTITUTE, DO NOT ZERO. Zeroing looks safer and is not. The rate does
        not only propagate the attitude: it is also what compensates the GNSS
        antenna's lever arm, and at 1308 rad/s on a 10 mm offset that lever
        arm is 13 m/s. Zeroing the axis therefore hands the filter an
        uncompensated 13 m/s velocity innovation every fix -- measured, before
        this was fixed: 7.4 m/s of velocity error at deployment against a
        0.1 m/s receiver.

        `substitute_sigma` is what the substituted rate can be wrong by, and
        it is proportional to the rate itself: the magnetometer's calibration
        residual is a FIXED distortion in the nose frame, so differentiating
        the phase turns a 1.5 % angular error into a 1.5 % RATE error --
        20 rad/s at the deployment spin. That is what `propagate_covariance`
        inflates the attitude process noise with, and it is two orders of
        magnitude smaller than inflating by the gyro's full scale.
        """
        w = np.asarray(gyro, dtype=float) - self.bg
        if saturated_axes is not None:
            sat = np.asarray(saturated_axes, dtype=bool)
            if substitute_rate is not None:
                w = np.where(sat, np.asarray(substitute_rate, dtype=float), w)
            else:
                w = np.where(sat, 0.0, w)
            self._sat_axes = sat
            self._sat_sigma = float(substitute_sigma)
        else:
            self._sat_axes = np.zeros(3, dtype=bool)
            self._sat_sigma = 0.0
        self._omega_prev = self._omega if self._has_omega else w
        self._omega = w
        self._has_omega = True

        f = np.asarray(accel, dtype=float) - self.ba
        # -- THE LEVER ARM, AND THE TERM THAT IS NOT SMALL -----------------
        #
        # The IMU sits 0.399 m forward of the CG and 25 mm off the axis, and
        # the filter navigates the CG, so both transport terms come off.
        #
        # The CENTRIPETAL term w x (w x r) is the one the despun mounting
        # exists to kill, and it does: 0.18 m/s^2 mean here, against 42 779
        # on the body.
        #
        # The TANGENTIAL term alpha x r is the surprise, and it is the larger
        # of the two by two orders of magnitude. The nose's moment of inertia
        # is 1.8 g m^2 and the brake applies up to 0.85 N m, so the servo's own
        # actuator swings the nose at up to 294 rad/s^2. At 25 mm that is
        # **66 m/s^2 rms against a 5.6 m/s^2 flight signal** -- the kit's own
        # control action is twelve times louder at the accelerometer than the
        # trajectory is. Measured, not estimated.
        #
        # It is removed by differencing the gyro. That is affordable ONLY
        # because the gyro is an integrating device: consecutive Dtheta/dt
        # samples differenced give the AVERAGE angular acceleration over the
        # interval, which is exactly the alpha that acted. The noise it costs
        # is 0.015 m/s^2 against the 66 it removes.
        alpha = (w - self._omega_prev) / dt if dt > 0.0 else np.zeros(3)
        f = f - _cross3(alpha, self.imu_lever)               - _cross3(w, _cross3(w, self.imu_lever))
        self._f_nose = f

        # Attitude: exact rotation-vector update, so the 2.62 rad per sample
        # of the despin is handled without a small-angle assumption.
        theta = w * dt
        ang = float(np.linalg.norm(theta))
        if ang > 1e-12:
            axis = theta / ang
            s, c = math.sin(0.5 * ang), math.cos(0.5 * ang)
            dq = np.array([c, s * axis[0], s * axis[1], s * axis[2]])
        else:
            dq = np.array([1.0, 0.5 * theta[0], 0.5 * theta[1], 0.5 * theta[2]])
        self.q = frames.quat_normalize(frames.quat_multiply(self.q, dq))

        C = self.dcm
        g = atm.gravity_ned(float(self.r[2]))
        a = C @ f + g
        self.v = self.v + a * dt
        self.r = self.r + self.v * dt - 0.5 * a * dt * dt
        self.t += dt

    def propagate_covariance(self, dt: float) -> None:
        """
        One covariance step. Runs at `covariance_rate`, not the IMU rate.

        F is the standard NED error-state dynamics with earth rotation and
        transport rate dropped -- see the module docstring for why that is
        defensible over 43 s and 16 km, and what would have to change if it
        were not.
        """
        C = self.dcm
        F = np.zeros((N_STATES, N_STATES))
        F[IDX["r"], IDX["v"]] = np.eye(3)
        F[IDX["v"], IDX["psi"]] = -_skew(C @ self._f_nose)
        F[IDX["v"], IDX["ba"]] = -C
        F[IDX["psi"], IDX["bg"]] = -C
        tg, ta = max(self.gyro.bias_tau, 1e-9), max(self.accel.bias_tau, 1e-9)
        F[IDX["bg"], IDX["bg"]] = -np.eye(3) / tg
        F[IDX["ba"], IDX["ba"]] = -np.eye(3) / ta

        Phi = np.eye(N_STATES) + F * dt + 0.5 * (F @ F) * dt * dt
        Q = np.zeros(N_STATES)
        Q[IDX["v"]] = self._q_vel * dt
        Q[IDX["psi"]] = self._q_att * dt
        Q[IDX["bg"]] = self._q_bg * dt
        Q[IDX["ba"]] = self._q_ba * dt
        Qm = np.diag(Q)
        # A saturated gyro axis is not a gyro. Its attitude process noise is
        # raised to the rate the axis could plausibly be turning at, so the
        # filter's covariance says out loud that it does not know the roll and
        # the magnetometer update is believed instead of being rejected as an
        # outlier by a confident prior.
        if self._sat_axes.any():
            big = (self._sat_sigma ** 2) * dt
            axis_cov = C @ np.diag(np.where(self._sat_axes, big, 0.0)) @ C.T
            Qm[IDX["psi"], IDX["psi"]] += axis_cov
        self.P = Phi @ self.P @ Phi.T + Qm
        self.P = 0.5 * (self.P + self.P.T)
        self.apply_floor()

    def apply_floor(self) -> None:
        """
        Raise any floored variance that has fallen below what the filter can
        actually know, by a diagonal congruence P <- D P D.

        A congruence, not an assignment: setting P[i,i] alone breaks the
        correlations and can leave the matrix indefinite, while scaling the
        whole row and column preserves symmetry, preserves positive
        semi-definiteness, and keeps the cross terms in proportion.
        """
        if not self.config.floor_enabled:
            return
        floors = np.concatenate([self._floor_r ** 2,
                                 np.full(3, self._floor_v ** 2),
                                 np.full(3, self._floor_psi ** 2),
                                 np.full(6, 0.0)])
        d = np.ones(N_STATES)
        diag = np.diag(self.P)
        m = (floors > 0.0) & (diag < floors)
        if not m.any():
            return
        d[m] = np.sqrt(floors[m] / np.maximum(diag[m], 1e-30))
        self.P = (self.P * d) * d[:, None]
        self.P = 0.5 * (self.P + self.P.T)

    def apply_roll_floor(self, sin_theta: float) -> None:
        """
        Floor the attitude variance ALONG THE BODY AXIS at what the
        magnetometer can actually resolve there.

        The roll signal is |B| sin(field-to-axis angle), so a transverse field
        error costs 1/sin of it in roll -- and on the adopted engagement, which
        fires along the field, the median sin is 0.424 and the floor is 2.0 deg
        rather than the 0.86 deg the calibration residual gives on its own
        (docs/SENSOR-MODELS.md section 4). A single scalar attitude floor
        cannot express that, because it is a floor in ONE DIRECTION.

        Applied as a rank-1 addition along the body x axis in earth
        coordinates, which inflates only that direction and leaves the
        pitch/yaw block and every cross term alone.
        """
        if not self.config.floor_enabled:
            return
        u = self.dcm[:, 0]
        s = max(float(sin_theta), math.sin(self.config.mag_min_transverse
                                           if self.config.mag_min_transverse < 1.0
                                           else 1.0))
        s = max(s, 1e-3)
        floor = (self.config.mag_sigma_floor / s) * self.config.floor_attitude_scale
        i = IDX["psi"]
        block = self.P[i, i]
        v = float(u @ block @ u)
        if v < floor * floor:
            self.P[i, i] = block + (floor * floor - v) * np.outer(u, u)
            self.P = 0.5 * (self.P + self.P.T)

    # -- measurement updates ---------------------------------------------
    def _scalar_update(self, h: np.ndarray, innovation: float, r: float) -> None:
        """
        One scalar Kalman update, Joseph-free but symmetrised.

        Exact for diagonal R, which every measurement in this filter has, and
        it costs one division where a vector update costs an inverse.
        """
        Ph = self.P @ h
        s = float(h @ Ph) + r
        if s <= 0.0:
            return
        k = Ph / s
        self._dx = self._dx + k * (innovation - float(h @ self._dx))
        self.P = self.P - np.outer(k, Ph)
        self.P = 0.5 * (self.P + self.P.T)

    def _begin(self) -> None:
        self._dx = np.zeros(N_STATES)

    def _commit(self, t: float) -> None:
        dx = self._dx
        self.r = self.r + dx[IDX["r"]]
        self.v = self.v + dx[IDX["v"]]
        self.bg = self.bg + dx[IDX["bg"]]
        self.ba = self.ba + dx[IDX["ba"]]
        dpsi = dx[IDX["psi"]]
        # C_true = (I + [dpsi]x) C_est, so the reset is a LEFT multiplication
        # by an earth-frame small rotation.
        n = float(np.linalg.norm(dpsi))
        if n > 1e-14:
            axis = dpsi / n
            s, c = math.sin(0.5 * n), math.cos(0.5 * n)
            dq = np.array([c, s * axis[0], s * axis[1], s * axis[2]])
        else:
            dq = np.array([1.0, 0.5 * dpsi[0], 0.5 * dpsi[1], 0.5 * dpsi[2]])
        self.q = frames.quat_normalize(frames.quat_multiply(dq, self.q))
        self.last_correction_t = float(t)
        self.apply_floor()

    def _joint_nis(self, H: np.ndarray, nu: np.ndarray, R: np.ndarray) -> float:
        S = H @ self.P @ H.T + R
        try:
            return float(nu @ np.linalg.solve(S, nu))
        except np.linalg.LinAlgError:
            return float("nan")

    def update_gnss(self, fix: "sn.GnssFix", phi_rel: float, p_rel: float,
                    antenna_lever, log_nis: bool = False) -> None:
        """
        Position and velocity from one receiver solution.

        THE LEVER ARM IS THE INTERESTING PART. The antenna is on the BODY, 10 mm
        off the spin axis, and the body turns at up to 1308 rad/s, so the phase
        centre moves at 13 m/s -- one hundred and thirty times the receiver's
        0.1 m/s velocity noise. It is deterministic given the body roll angle,
        so it is subtracted; and the residual after subtraction is set by the
        ROLL ERROR, which is how Task C's roll accuracy reaches the velocity
        solution and then the CEP. That path is measured in
        docs/NAV-CEP.md section 5, not assumed to be small.
        """
        cfg = self.config
        C = self.dcm
        C_body = C @ sn.nose_from_body(phi_rel).T
        lever = np.asarray(antenna_lever, dtype=float)
        lever_e = C_body @ lever

        # -- LATENCY -------------------------------------------------------
        # The receiver reports where the projectile WAS. At 500 m/s the 100 ms
        # of transport and solution latency is 50 m, and a filter that applies
        # the fix at arrival time sees that 50 m as a persistent innovation.
        # It has no position-bias state to put it in, so it puts it in the
        # ATTITUDE and the accelerometer bias -- which is how a 100 ms timing
        # error becomes tens of degrees of roll error and takes the servo with
        # it. Measured before the fix was in: 34 m of position error and 67
        # degrees of roll bias, with the filter claiming 0.5 m and 0.03 deg.
        #
        # The fix is extrapolated forward to now on the filter's own velocity
        # and specific force, and R is inflated by what that extrapolation can
        # be wrong by.
        lag = max(0.0, float(fix.t) - float(fix.t_valid))
        a_e = C @ self._f_nose + atm.gravity_ned(float(self.r[2]))
        z_pos = np.asarray(fix.position, dtype=float) + self.v * lag + 0.5 * a_e * lag * lag
        z_vel = np.asarray(fix.velocity, dtype=float) + a_e * lag
        lag_var_p = (self.sigma_velocity * lag) ** 2
        lag_var_v = (0.5 * float(np.linalg.norm(a_e)) * lag * lag) ** 2

        self._begin()
        # -- position ------------------------------------------------------
        nu = z_pos - (self.r + lever_e)
        Hp = np.zeros((3, N_STATES))
        Hp[:, IDX["r"]] = np.eye(3)
        Hp[:, IDX["psi"]] = -_skew(lever_e)
        Rp = np.diag((np.asarray(fix.sigma_position, dtype=float) ** 2)
                     * cfg.gnss_position_inflation + lag_var_p)
        if log_nis:
            self.nis["gnss_pos"].append(self._joint_nis(Hp, nu, Rp))
            self.nis_t["gnss_pos"].append(fix.t)
            self.innovations["gnss_pos"].append(nu.copy())
        for i in range(3):
            self._scalar_update(Hp[i], float(nu[i]), float(Rp[i, i]))
        self.updates["gnss_pos"] += 1

        # -- velocity ------------------------------------------------------
        # NO LEVER-ARM COMPENSATION HERE, and that is deliberate. The receiver
        # forms its Doppler over ~20 ms, which is four spin revolutions, so
        # the antenna's 13 m/s rotational velocity is averaged away INSIDE the
        # receiver before the filter ever sees it. Subtracting it again would
        # add 13 m/s rather than remove it. What survives is a residual of
        # about 1 m/s, and the receiver reports it in its own sigma --
        # `sensors.Gnss.sample` says how it is sized.
        nu_v = z_vel - self.v
        Hv = np.zeros((3, N_STATES))
        Hv[:, IDX["v"]] = np.eye(3)
        sv = (np.asarray(fix.sigma_velocity, dtype=float) ** 2
              * cfg.gnss_velocity_inflation + lag_var_v)
        Rv = np.diag(sv)
        if log_nis:
            self.nis["gnss_vel"].append(self._joint_nis(Hv, nu_v, Rv))
            self.nis_t["gnss_vel"].append(fix.t)
            self.innovations["gnss_vel"].append(nu_v.copy())
        for i in range(3):
            self._scalar_update(Hv[i], float(nu_v[i]), float(Rv[i, i]))
        self.updates["gnss_vel"] += 1

        # -- roll from carrier-to-noise modulation -------------------------
        if fix.cn0_roll is not None:
            self._cn0_roll(fix.cn0_roll, fix.cn0_roll_sigma, phi_rel,
                           log_nis=log_nis, t=fix.t)
        self._commit(fix.t)
        self.last_position_correction_t = float(fix.t)

    def _cn0_roll(self, phi_body_meas: float, sigma: float, phi_rel: float,
                  log_nis: bool = False, t: float = 0.0) -> None:
        """
        The third roll source: `phi_nose = phi_body + phi_rel`.

        Independent of both the magnetometer and the gyro, which is its whole
        value -- a common-mode magnetic disturbance moves the magnetometer and
        this not at all. It measures the BODY, because the antenna is on the
        body, and the resolver converts.
        """
        _, _, roll = self.euler
        z = _wrap_pi(float(phi_body_meas) + float(phi_rel))
        nu = _wrap_pi(z - roll)
        # Roll about the body x axis; in the earth-frame error parametrisation
        # that direction is the body x axis expressed in earth axes.
        h = np.zeros(N_STATES)
        h[IDX["psi"]] = self.dcm[:, 0]
        r = float(sigma) ** 2
        if log_nis:
            s = float(h @ self.P @ h) + r
            self.nis["cn0"].append(nu * nu / s if s > 0 else float("nan"))
            self.nis_t["cn0"].append(float(t))
            self.innovations["cn0"].append(nu)
        self._scalar_update(h, nu, r)
        self.updates["cn0"] += 1

    def update_magnetometer(self, t: float, mag: np.ndarray,
                            log_nis: bool = False) -> bool:
        """
        Three-axis field update. Returns False if the roll was not observable
        and the update was skipped.

        THE OBSERVABILITY TEST IS NOT A ROBUSTNESS HACK. Only the component of
        the field TRANSVERSE to the body axis is modulated by roll, and its
        magnitude is |B| sin(angle between B and the axis). As that angle goes
        to zero the roll signal vanishes and any measurement error is divided
        by its sine on the way into the roll estimate. Skipping below 15 deg
        is the filter declining to divide by a small number; section 4 of
        docs/SENSOR-MODELS.md sweeps the firing azimuth to show where on the
        compass this actually bites.
        """
        cfg = self.config
        C = self.dcm
        b_e = self.field.field_ned(self.r)
        b_pred = C.T @ b_e
        mag = np.asarray(mag, dtype=float)

        mag_norm = float(np.linalg.norm(b_e))
        transverse = math.hypot(float(b_pred[1]), float(b_pred[2]))
        if mag_norm <= 0.0 or transverse / mag_norm < cfg.mag_min_transverse:
            return False

        H = np.zeros((3, N_STATES))
        H[:, IDX["psi"]] = C.T @ _skew(b_e)
        nu = mag - b_pred
        # R is the CALIBRATION RESIDUAL, not the bridge noise. The residual is
        # 700 nT against a 1 nT/sqrt(Hz) part: declaring the small number would
        # make the filter believe a distortion it cannot estimate.
        sig = cfg.mag_sigma_floor * mag_norm
        R = np.eye(3) * sig * sig
        if log_nis:
            self.nis["mag"].append(self._joint_nis(H, nu, R))
            self.nis_t["mag"].append(float(t))
            self.innovations["mag"].append(nu.copy())
        self._begin()
        for i in range(3):
            self._scalar_update(H[i], float(nu[i]), float(R[i, i]))
        self._commit(t)
        self.updates["mag"] += 1
        return True

    def update_velocity_attitude(self, t: float, wind=(0.0, 0.0, 0.0),
                                 log_nis: bool = False) -> bool:
        """
        The second attitude reference, and without it the magnetometer is not
        enough.

        A MAGNETOMETER IS A TWO-DEGREE-OF-FREEDOM ATTITUDE SENSOR. One vector
        observation leaves the rotation ABOUT that vector free, and on the
        adopted engagement the body axis has a 0.53 component along the field,
        so 53 % of a roll error lives in the direction the magnetometer cannot
        see. Measured, before this update existed: a 5 deg roll error settled
        at 2.46 deg and stopped, with the filter's own sigma stuck at 5.34 deg
        against an initial 10 deg -- exactly cos(57.8 deg) of it.

        The second vector is the shell's own velocity. A spin-stabilised shell
        flies with its axis along the air-relative velocity to within the total
        angle of attack, and on this trajectory that is **1.59 deg mean, 1.78
        deg rms, 4.40 deg worst**, measured on the 6-DOF rather than assumed --
        which is what sets R here. Two non-parallel references determine the
        full attitude; this is the classic two-vector solution, run as a Kalman
        update so its uncertainty is carried rather than discarded.

        WHAT IT DOES NOT REMOVE. The angle of attack is not zero-mean: the
        steady yaw of repose puts the nose a fraction of a degree off the
        velocity in a FIXED direction, and this update has no model of it, so
        that part becomes an attitude bias. `models.mpmm.yaw_of_repose` --
        which the kit already carries for the impact prediction -- would
        remove it, and docs/NAV-CONSISTENCY.md section 5 measures what
        leaving it in costs before deciding whether it is worth the code.

        WIND ENTERS HERE. The reference is the AIR-relative velocity, and the
        filter only knows the wind the met message gave it. A 10 m/s error in
        crosswind at 450 m/s is 1.3 deg -- comparable with the angle of attack
        itself, and the reason the met message is part of the Task D upload
        rather than an afterthought.
        """
        v_rel = self.v - np.asarray(wind, dtype=float)
        speed = float(np.linalg.norm(v_rel))
        if speed < 1.0:
            return False
        u_e = v_rel / speed
        C = self.dcm
        pred = C.T @ u_e                      # the axis direction, nose axes
        nu = np.array([1.0, 0.0, 0.0]) - pred
        H = np.zeros((3, N_STATES))
        H[:, IDX["psi"]] = C.T @ _skew(u_e)
        s = self.config.alignment_sigma
        R = np.eye(3) * s * s
        if log_nis:
            self.nis["align"].append(self._joint_nis(H, nu, R))
            self.nis_t["align"].append(float(t))
            self.innovations["align"].append(nu.copy())
        self._begin()
        # The x component carries no attitude information -- it is 1 - cos of
        # a small angle, second order -- so only the two transverse ones are
        # applied. Applying all three would feed a second-order residual into
        # a first-order filter.
        for i in (1, 2):
            self._scalar_update(H[i], float(nu[i]), float(R[i, i]))
        self._commit(t)
        self.updates["align"] += 1
        return True

    # -- what the consumers ask ------------------------------------------
    @property
    def sigma_position(self) -> float:
        return float(math.sqrt(max(0.0, np.trace(self.P[0:3, 0:3]))))

    @property
    def sigma_velocity(self) -> float:
        return float(math.sqrt(max(0.0, np.trace(self.P[3:6, 3:6]))))

    @property
    def sigma_roll(self) -> float:
        """1 sigma on `phi_nose`, rad, from the attitude covariance block."""
        x = self.dcm[:, 0]
        return float(math.sqrt(max(0.0, x @ self.P[6:9, 6:9] @ x)))


# ===========================================================================
# The system: sensors + filter, shaped as a step_hook
# ===========================================================================
@dataclass
class NavSample:
    """One logged navigation epoch. What the consistency analysis differences."""

    t: float
    r: np.ndarray
    v: np.ndarray
    phi_nose: float
    p_nose: float
    sigma_position: float
    sigma_velocity: float
    sigma_roll: float
    mode: str
    valid: bool
    gyro_saturated: bool
    #: The BODY axial spin, `p_nose - p_rel`. Logged because it is the third
    #: and last thing `models.mpmm.state_from_sixdof` reads, so it is the
    #: third and last channel by which an estimate error can reach the impact
    #: prediction -- see docs/NAV-ERROR-DECOMPOSITION.md. Defaulted so that
    #: nothing constructing a NavSample positionally has to change.
    p_body: float = 0.0


class NavigationSystem:
    """
    The whole navigation function: sensors, filter, and the two answers the
    rest of the kit asks for.

    USAGE mirrors `BrakeLaw` and `GuidanceLaw`, and it adds no third seam:

        nav = NavigationSystem(cfg, suite_cfg, seed, gun=gun_data, ...)
        law_b = rc.BrakeLaw(controller, guidance.command, ..., nav=nav)
        integrate(..., step_hook=nav.chain(guidance.chain(law_b.sample)))

    `nav` runs FIRST in the chain, so the servo and the guidance law both see
    an estimate that already includes this step's measurements.

    TWO PHASES, AND THE REASON IS MEASURED
    --------------------------------------
    Before the nose despins, the kit's IMU is inside a body turning at
    1308 rad/s. At the 25 mm transverse offset a fuze-well kit can hold, the
    accelerometers see 4 361 g against a +-40 g range, and the roll gyro is
    pinned at full scale. NEITHER IS RECOVERABLE BY MOUNTING CARE: getting the
    centripetal term under 40 g at that spin needs the proof mass on the axis
    to within 0.23 mm.

    So there is no inertial solution at all until the nose has despun, and the
    filter says so rather than integrating clipped readings:

      MODEL phase   position and velocity propagate on the SAME reduced-order
                    model `gnc.guidance.ImpactPredictor` already carries,
                    seeded from gun data (Task D) and corrected by GNSS.
                    Attitude is determined directly -- pitch and heading from
                    the velocity vector, roll from the magnetometer, which is
                    the one sensor that works throughout.
      INERTIAL      full strapdown, once the gyro and the accelerometers are
      phase         both out of saturation. On the adopted trajectory that is
                    t_dep + 2.10 s.

    The reduced-order model is not a fallback of convenience. docs/MODEL-ERROR.md
    measures it at 0.65 m RMS in range over a WHOLE flight, so over the 8 s
    before the IMU wakes up its contribution is negligible beside the gun's own
    muzzle-velocity dispersion. What the warm start is worth is therefore a
    question about the GUN's dispersion, not about the model, and Task D
    measures it that way.
    """

    MODEL, INERTIAL = "model", "inertial"

    def __init__(self, config: NavConfig, suite: "sn.SuiteConfig", seed: int,
                 projectile, environment, gun: Optional[GunData] = None,
                 muzzle_time: float = 0.0, log: bool = True,
                 log_nis: bool = False, warm_start: bool = True,
                 sensor_suite: Optional["sn.SuiteConfig"] = None):
        self.config = config
        #: WHAT THE FILTER BELIEVES. The process noise, the measurement noise
        #: and the Schmidt consider floors are all derived from these part
        #: specs, so this object is the filter's TUNING and not a description
        #: of the world.
        self.suite_config = suite
        #: WHAT THE WORLD IS. Normally the same object. They separate for one
        #: purpose -- the ablation of docs/NAV-ERROR-DECOMPOSITION.md -- and
        #: separating them is not a refinement there, it is the difference
        #: between a decomposition and a nonsense:
        #:
        #:   `_q_att = gyro.arw**2`, `_q_vel = accel.vrw**2`, `_floor_r` from
        #:   the GNSS sigmas. Zeroing a sensor's error term to ask what it
        #:   contributes ALSO zeroes the filter's process noise for that
        #:   channel, and a filter with no process noise stops believing its
        #:   measurements. Measured, when this was one object: every sensor
        #:   error set to zero at once gave 425 m of range contribution
        #:   against a 32 m baseline -- a filter diverging on perfect data.
        #:
        #: So an error-term ablation moves this one and leaves the tuning
        #: alone, and a GEOMETRY change -- the antenna offset, the fix rate --
        #: moves both, because the filter is entitled to know its own layout.
        self.sensor_config = sensor_suite if sensor_suite is not None else suite
        self.projectile = projectile
        self.environment = environment
        self.gun = gun
        self.warm_start = bool(warm_start and gun is not None)
        self.log_enabled = bool(log)
        self.log_nis = bool(log_nis)

        self.sensors = sn.SensorSuite(self.sensor_config, seed,
                                      muzzle_time=muzzle_time)
        self.filter = NavigationFilter(config, suite.gyro, suite.accel,
                                       suite.field, imu_lever=suite.imu_lever,
                                       gnss_spec=suite.gnss)
        self.mode = self.MODEL
        self.muzzle_time = float(muzzle_time)

        self.period = 1.0 / config.imu_rate
        self._next = -math.inf
        self._cov_accum = 0.0
        self._mag_accum = 0.0
        self._align_accum = 0.0
        self._model_accum = 0.0
        self.samples = 0
        #: The met message's wind, as the air-relative velocity reference
        #: needs it. Zero when no gun data was uploaded, which is itself a
        #: modelled degradation rather than an omission.
        self.wind = np.asarray(gun.wind if gun is not None else (0.0, 0.0, 0.0),
                               dtype=float)

        self._psi_mag_prev: Optional[float] = None
        self._mpmm = None
        self.first_usable_t: Optional[float] = None
        self.first_fix_t: Optional[float] = None
        self.inertial_start_t: Optional[float] = None
        self.log: list = []
        #: Latched for the consumers, so `roll()` is a function of nothing but
        #: this object's state -- the same restriction `BrakeLaw.brake_command`
        #: lives under.
        self._phi_nose = 0.0
        self._p_nose = 0.0
        self._p_body = 0.0
        self._rate_valid = False
        self._state = np.zeros(15)

    # -- the seam ---------------------------------------------------------
    def chain(self, *hooks) -> Callable:
        """A step_hook that runs navigation first, then `hooks`, in order."""
        def hook(t, y, model):
            self.sample(t, y, model)
            for h in hooks:
                h(t, y, model)
        return hook

    # -- what the consumers read -----------------------------------------
    def roll(self) -> tuple:
        """(phi_nose, p_nose, p_body) as the servo needs them. Held between samples."""
        return self._phi_nose, self._p_nose, self._p_body

    def state(self) -> np.ndarray:
        """
        A 15-element 6-DOF-shaped ESTIMATE, so `gnc.guidance` can consume it
        through exactly the interface it already consumes truth through.

        The attitude slot carries the estimated NOSE quaternion and the nose
        states carry the resolver's relative angle and rate, which is what the
        kit actually knows. `models.mpmm.state_from_sixdof` reads position,
        velocity and the axial rate and discards the rest, so what guidance
        depends on is exactly the three quantities this filter estimates well.
        """
        return self._state

    @property
    def rate_valid(self) -> bool:
        """
        False while the roll rate is not measurable.

        THIS EXISTS TO STOP A SPECIFIC FAILURE. A saturated gyro returns a
        CONSTANT, and a constant is precisely what
        `gnc.roll_control.RollAngleController`'s model-free settling detector
        treats as evidence that the despin transient is over. Without this
        flag the servo engages its angle loop on a pinned reading, at a nose
        rate two orders of magnitude outside the actuator's reachable set.
        """
        return self._rate_valid

    @property
    def valid(self) -> bool:
        """
        Whether the solution may be steered on.

        A filter that reports itself invalid is worth more than one that
        reports a confident wrong answer, and docs/NAV-DEGRADATION.md section 3
        is the test of that claim rather than the assertion of it.
        """
        f = self.filter
        if not f.initialised:
            return False
        if f.sigma_position > self.config.invalid_position_sigma:
            return False
        # THE COAST TIMER MUST COUNT FROM THE LAST *POSITION* CORRECTION.
        #
        # Measured, when it counted from any correction: under COMPLETE GNSS
        # DENIAL the filter reported itself VALID on 20 of 24 rounds all the
        # way to impact, and guidance saw a mean of 1.2 invalid cycles out of
        # forty. The magnetometer and the velocity alignment reference keep
        # updating -- they always do, they need no satellites -- so the timer
        # kept being reset by measurements that correct ATTITUDE and touch
        # position not at all.
        #
        # That is precisely the failure docs/NAV-DEGRADATION.md exists to test
        # for: a filter producing a confident wrong answer instead of
        # admitting it is lost. The fix is to time the right thing.
        last = f.last_position_correction_t
        if last is None:
            # No position measurement has EVER arrived. A warm start is an
            # a-priori trajectory, not a fix, and it decays; `max_coast_s`
            # from launch is as long as it may be steered on.
            if (f.t - self.muzzle_time) > self.config.max_coast_s:
                return False
        elif (f.t - last) > self.config.max_coast_s:
            return False
        return True

    @property
    def usable(self) -> bool:
        """The Task D threshold: accurate enough to seed an impact prediction."""
        f = self.filter
        return (f.initialised
                and f.sigma_position <= self.config.usable_position_sigma
                and f.sigma_velocity <= self.config.usable_velocity_sigma)

    # -- the cycle --------------------------------------------------------
    def sample(self, t: float, y: np.ndarray, model) -> None:
        """One navigation epoch. `step_hook`-shaped, runs on its own grid."""
        if t < self._next:
            return
        dt = self.period if self._next != -math.inf else self.period
        if self._next == -math.inf:
            self._next = t
        self._next += self.period
        if self._next <= t:
            self._next = t + self.period
        self.samples += 1

        m = self.sensors.measure(t, y, model, dt)
        imu, mag = m["imu"], m["mag"]
        phi_rel, p_rel = m["phi_rel"], m["p_rel"]
        f = self.filter

        gyro_sat = np.abs(imu.gyro) >= self.suite_config.gyro.full_scale * 0.999
        accel_sat = np.abs(imu.accel) >= self.suite_config.accel.full_scale * 0.999
        inertial_ok = not (gyro_sat.any() or accel_sat.any())

        # -- roll rate from the magnetometer, valid through the despin ----
        # The transverse field phase advances by p_nose*dt per sample; at the
        # 500 Hz IMU rate and the measured 1308 rad/s deployment spin that is
        # 2.62 rad, inside pi, so the despin is UNAMBIGUOUS. This is the whole
        # practical content of the HMC1053's DC-to-5 MHz bandwidth line.
        psi_m = math.atan2(float(mag[2]), float(mag[1]))
        p_mag = None
        if self._psi_mag_prev is not None and dt > 0.0:
            p_mag = -_wrap_pi(psi_m - self._psi_mag_prev) / dt
        self._psi_mag_prev = psi_m

        # -- initialise ----------------------------------------------------
        if not f.initialised:
            self._initialise(t, m, p_mag)

        if f.initialised:
            if inertial_ok and self.mode == self.MODEL:
                self.mode = self.INERTIAL
                self.inertial_start_t = float(t)
            sub = np.array([p_mag if p_mag is not None else 0.0, 0.0, 0.0])
            sub_sigma = self.config.mag_sigma_floor * abs(float(sub[0])) + 0.3
            if self.mode == self.INERTIAL:
                f.propagate_state(dt, imu.gyro, imu.accel,
                                  saturated_axes=gyro_sat,
                                  substitute_rate=sub,
                                  substitute_sigma=sub_sigma)
            else:
                # The MODEL phase does not propagate on the gyro, but the
                # filter still needs the ANGULAR RATE: it is what compensates
                # the GNSS antenna's lever arm, and before deployment that
                # lever arm is 13 m/s. Leaving `_omega` at zero here cost
                # 7.4 m/s of velocity error at deployment, measured.
                f._omega = np.where(gyro_sat, sub,
                                    np.asarray(imu.gyro) - f.bg)
                # No saturation inflation here: the attitude is DETERMINED
                # each sample by `set_attitude`, so there is no propagated
                # attitude error for a process noise to describe.
                f._sat_axes = np.zeros(3, dtype=bool)
                f._sat_sigma = 0.0
                # TWO RATES, AND THE SPLIT IS NOT AN OPTIMISATION DETAIL.
                #
                # POSITION AND VELOCITY propagate on the COVARIANCE grid.
                # There is no inertial data to integrate in this phase -- the
                # IMU is saturated -- so a 500 Hz RK4 of a smooth ballistic
                # model buys nothing and costs four reduced-order derivative
                # evaluations per sample.
                #
                # ATTITUDE is determined EVERY sample, because the servo reads
                # it every sample. Determining it at 100 Hz while the servo
                # consumes it at 500 Hz hands the loop an angle up to 10 ms
                # stale, and during the despin the nose turns 9 rad in 10 ms.
                # Measured, when this was one rate: 25.3 deg rms of roll error
                # over the first 3 s after deployment on the `middle`
                # engagement against 2.2 deg thereafter, and 18.1 m of
                # navigation contribution to CEP against 4.3 m at short range.
                self._model_accum += dt
                cp = 1.0 / self.config.covariance_rate
                if self._model_accum >= cp - 1e-12:
                    self._propagate_model(self._model_accum, m, p_mag)
                    self._model_accum = 0.0
                self._determine_attitude(m)

            self._cov_accum += dt
            cov_period = 1.0 / self.config.covariance_rate
            if self._cov_accum >= cov_period - 1e-12:
                f.propagate_covariance(self._cov_accum)
                self._cov_accum = 0.0

            # -- magnetometer ---------------------------------------------
            self._mag_accum += dt
            rate = (self.config.mag_rate_saturated if gyro_sat[0]
                    else self.config.mag_rate)
            if self._mag_accum >= 1.0 / rate - 1e-12:
                self._mag_accum = 0.0
                if self.mode == self.INERTIAL:
                    f.update_magnetometer(t, mag, log_nis=self.log_nis)
                    # The roll floor is a function of the field geometry, so
                    # it is applied where that geometry is known.
                    b_e = self.suite_config.field.field_ned(f.r)
                    x_e = f.dcm[:, 0]
                    nb = float(np.linalg.norm(b_e))
                    if nb > 0.0:
                        c = float(np.dot(b_e, x_e)) / nb
                        f.apply_roll_floor(math.sqrt(max(0.0, 1.0 - c * c)))

            # -- the second attitude reference -----------------------------
            # Not optional. One vector observation fixes two degrees of
            # freedom; the velocity vector supplies the third. See
            # `NavigationFilter.update_velocity_attitude`.
            self._align_accum += dt
            if self._align_accum >= 1.0 / self.config.alignment_rate - 1e-12:
                self._align_accum = 0.0
                if (self.mode == self.INERTIAL
                        and f.sigma_velocity <= self.config.alignment_max_sigma_v):
                    # The reference is the filter's OWN velocity, so running
                    # this while the velocity is poorly known would let an
                    # attitude error and a velocity error confirm each other.
                    # The guard is the one place a loop could close, and it is
                    # closed here rather than discovered in a CEP.
                    f.update_velocity_attitude(t, wind=self.wind,
                                               log_nis=self.log_nis)

            # -- GNSS ------------------------------------------------------
            fix = m["gnss"]
            if fix is not None:
                if self.first_fix_t is None:
                    self.first_fix_t = float(fix.t)
                f.update_gnss(fix, phi_rel, p_rel,
                              self.suite_config.antenna_lever,
                              log_nis=self.log_nis)

        self._latch(t, m, imu, gyro_sat, p_mag, phi_rel, p_rel)

    # -- initialisation ---------------------------------------------------
    def _initialise(self, t: float, m: dict, p_mag) -> None:
        """
        Task D, and the whole of it: gun data or nothing.

        WARM: the a-priori trajectory from the fuze setter's own upload, at
        this instant, with the covariance the gun's dispersion implies. The
        filter has a solution from the first sample and GNSS arrives as a
        correction to it.

        COLD: no solution until the receiver produces one. The IMU cannot
        supply the missing one -- it is saturated -- so this is not "coast on
        inertial until GNSS": it is nothing at all.
        """
        cfg = self.config
        f = self.filter
        if self.warm_start:
            r, v, p, sr, sv = warm_start_state(
                self.gun, self.projectile, self.environment,
                max(t - self.muzzle_time, 1e-3))
            q = self._attitude_from(v, m["mag"], p_mag)
            f.initialise(t, r, v, q,
                         np.maximum(sr, 1.0),
                         np.maximum(sv, 0.5),
                         np.full(3, cfg.p0_attitude_warm))
            return
        fix = m["gnss"]
        if fix is None:
            return
        q = self._attitude_from(fix.velocity, m["mag"], p_mag)
        f.initialise(t, fix.position, fix.velocity, q,
                     np.asarray(fix.sigma_position) * 1.5,
                     np.full(3, cfg.p0_velocity),
                     np.array([cfg.p0_roll_cold, cfg.p0_attitude,
                               cfg.p0_attitude]))

    def _attitude_from(self, v, mag, p_mag) -> np.ndarray:
        """
        Direct attitude determination: pitch and heading from the velocity
        vector, roll from the magnetometer.

        Legitimate on THIS vehicle and not in general. A spin-stabilised shell
        flies at a yaw of repose of a fraction of a degree, so its axis and its
        velocity differ by that and by the angle of attack -- both small, both
        bounded, and both already characterised by step 1. Section 5 of
        docs/NAV-CONSISTENCY.md measures the residual instead of asserting it.
        """
        v = np.asarray(v, dtype=float)
        sp = float(np.hypot(v[0], v[1]))
        heading = math.atan2(float(v[1]), float(v[0])) if sp > 1e-6 else 0.0
        pitch = math.atan2(-float(v[2]), sp) if sp > 1e-6 else 0.0
        roll = self._roll_from_mag(mag, heading, pitch)
        return frames.quat_from_euler(heading, pitch, roll)

    def _roll_from_mag(self, mag, heading: float, pitch: float) -> float:
        """
        Roll from the transverse field, given pitch and heading.

        With `u` the field in the roll-free intermediate frame,
        B_nose = Rx(-phi) u, so atan2(B_z, B_y) = atan2(u_z, u_y) - phi, and
        the roll follows in one atan2 pair. The error is the transverse
        measurement error divided by |u| -- which is |B| sin(field-to-axis
        angle), the observability factor `GeomagneticField.axis_angle`
        returns and section 4 of docs/SENSOR-MODELS.md sweeps.
        """
        b_e = self.suite_config.field.field_ned(self.filter.r)
        cy, sy = math.cos(heading), math.sin(heading)
        cp, sp = math.cos(pitch), math.sin(pitch)
        # Rz(heading)^T then Ry(pitch)^T
        u1 = np.array([cy * b_e[0] + sy * b_e[1],
                       -sy * b_e[0] + cy * b_e[1], b_e[2]])
        u = np.array([cp * u1[0] - sp * u1[2], u1[1],
                      sp * u1[0] + cp * u1[2]])
        if math.hypot(float(u[1]), float(u[2])) < 1e-15:
            return 0.0
        return _wrap_pi(math.atan2(float(u[2]), float(u[1]))
                        - math.atan2(float(mag[2]), float(mag[1])))

    # -- the model phase --------------------------------------------------
    def _propagate_model(self, dt: float, m: dict, p_mag) -> None:
        """
        Propagate on the reduced-order model, because the IMU is saturated.

        One RK4 step of `models.mpmm.derivative` on the filter's own position,
        velocity and spin. The attitude is re-determined each sample rather
        than propagated: with no usable gyro there is nothing to propagate it
        WITH, and the magnetometer supplies the roll directly at 500 Hz.
        """
        from models import mpmm
        f = self.filter
        if self._mpmm is None:
            from sim import aerodata
            self._mpmm = mpmm.MpmmModel(
                projectile=self.projectile, aero=aerodata.make_m107_table(),
                environment=self.environment, iterate_yaw=False)
        y = mpmm.pack(f.r, f.v, self._p_body if self._p_body else 0.0)
        t = f.t
        k1 = np.array(mpmm.derivative(t, y, self._mpmm))
        k2 = np.array(mpmm.derivative(t + 0.5 * dt, y + 0.5 * dt * k1, self._mpmm))
        k3 = np.array(mpmm.derivative(t + 0.5 * dt, y + 0.5 * dt * k2, self._mpmm))
        k4 = np.array(mpmm.derivative(t + dt, y + dt * k3, self._mpmm))
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        f.r, f.v = y[0:3].copy(), y[3:6].copy()
        f.t += dt
        # A crude axial specific force, for the F matrix alone. The MODEL
        # phase does not integrate it -- the reduced-order model supplies the
        # acceleration -- so it only shapes the attitude-to-velocity coupling.
        f._f_nose = np.array([-2.0 * G_NOMINAL, 0.0, 0.0])

    def _determine_attitude(self, m: dict) -> None:
        """
        Determine the attitude directly, at the FULL IMU rate.

        Pitch and heading from the velocity vector, roll from the
        magnetometer. Runs every sample, not on the covariance grid, because
        `gnc.roll_control.BrakeLaw` reads `phi_nose` every sample and the nose
        is turning at up to 1308 rad/s -- see the note in `sample`.
        """
        f = self.filter
        heading = math.atan2(float(f.v[1]), float(f.v[0]))
        pitch = math.atan2(-float(f.v[2]), float(np.hypot(f.v[0], f.v[1])))
        roll = self._roll_from_mag(m["mag"], heading, pitch)
        # The roll's accuracy is the calibration residual DIVIDED BY THE
        # OBSERVABILITY: only |B| sin(field-to-axis angle) of the field is
        # modulated by roll, so an error transverse to the field costs
        # 1/sin of it in roll. Pitch and heading come from the velocity and
        # carry the measured angle of attack.
        cfg = self.config
        b_e = self.suite_config.field.field_ned(f.r)
        x_e = frames.dcm_from_quat(frames.quat_from_euler(heading, pitch, roll))[:, 0]
        sin_a = max(math.sin(self.suite_config.field.axis_angle(f.r, x_e)), 1e-3)
        s_roll = min(cfg.mag_sigma_floor / sin_a, math.pi)
        f.set_attitude(frames.quat_from_euler(heading, pitch, roll),
                       np.array([s_roll, cfg.alignment_sigma, cfg.alignment_sigma]))

    # -- latch what the consumers read ------------------------------------
    def _latch(self, t: float, m: dict, imu, gyro_sat, p_mag,
               phi_rel: float, p_rel: float) -> None:
        f = self.filter
        _, _, roll = f.euler if f.initialised else (0.0, 0.0, 0.0)
        self._phi_nose = float(roll)

        # The roll RATE. From the gyro when it has one, from the magnetometer
        # phase when it does not. Both are measurements; neither is the
        # filter's own opinion, which is what makes the substitution safe.
        if not gyro_sat[0]:
            self._p_nose = float(imu.gyro[0] - f.bg[0])
            self._rate_valid = True
        elif p_mag is not None:
            self._p_nose = float(p_mag)
            self._rate_valid = True
        else:
            self._p_nose = float(math.copysign(
                self.suite_config.gyro.full_scale, imu.gyro[0]))
            self._rate_valid = False
        self._p_body = self._p_nose - float(p_rel)

        s = self._state
        s[0:3] = f.r
        s[3:6] = f.v
        s[6:10] = f.q
        s[10] = self._p_body
        # Transverse body rates: the gyro measures them in NOSE axes and the
        # resolver's angle rotates them to body axes. They are 22 deg/s at
        # most and guidance discards them, but the servo's flight condition
        # does not, so they are carried rather than zeroed.
        w_body = sn.nose_from_body(phi_rel) @ (np.asarray(imu.gyro) - f.bg)
        s[11] = w_body[1]
        s[12] = w_body[2]
        s[13] = float(phi_rel)
        s[14] = float(p_rel)

        if self.usable and self.first_usable_t is None:
            self.first_usable_t = float(t)

        if self.log_enabled:
            self.log.append(NavSample(
                t=float(t), r=f.r.copy(), v=f.v.copy(),
                phi_nose=self._phi_nose, p_nose=self._p_nose,
                sigma_position=f.sigma_position,
                sigma_velocity=f.sigma_velocity,
                sigma_roll=f.sigma_roll, mode=self.mode, valid=self.valid,
                gyro_saturated=bool(gyro_sat.any()),
                p_body=float(self._p_body)))

    # -- reduction --------------------------------------------------------
    def summary(self) -> dict:
        f = self.filter
        return {
            "samples": self.samples,
            "mode": self.mode,
            "first_fix_s": self.first_fix_t,
            "first_usable_s": self.first_usable_t,
            "inertial_start_s": self.inertial_start_t,
            "gnss_fixes": self.sensors.gnss.fixes,
            "gnss_reacquire_s": self.sensors.gnss.reacquire_time,
            "gyro_saturated_samples": self.sensors.gyro_saturated_samples,
            "updates": dict(f.updates),
            "sigma_position": f.sigma_position,
            "sigma_velocity": f.sigma_velocity,
            "sigma_roll_deg": f.sigma_roll / sn.DEG,
            "valid": self.valid,
        }

    def errors_against(self, truth_t, truth_r, truth_v, truth_phi,
                       truth_p_body=None) -> dict:
        """
        Difference the log against truth. The analysis calls this; the filter
        never sees any of it.

        `truth_p_body` is optional because only the error decomposition of
        docs/NAV-ERROR-DECOMPOSITION.md needs the spin channel; every earlier
        caller passes four arguments and gets what it always got.
        """
        if not self.log:
            return {}
        t = np.array([s.t for s in self.log])
        r = np.array([s.r for s in self.log])
        v = np.array([s.v for s in self.log])
        phi = np.array([s.phi_nose for s in self.log])
        tr = np.column_stack([np.interp(t, truth_t, np.asarray(truth_r)[:, k])
                              for k in range(3)])
        tv = np.column_stack([np.interp(t, truth_t, np.asarray(truth_v)[:, k])
                              for k in range(3)])
        tp = np.interp(t, truth_t, np.unwrap(np.asarray(truth_phi)))
        dphi = np.array([_wrap_pi(a - b) for a, b in zip(phi, tp)])
        out = {"t": t, "position_error": r - tr, "velocity_error": v - tv,
               "roll_error": dphi}
        if truth_p_body is not None:
            pb = np.array([s.p_body for s in self.log])
            out["spin_error"] = pb - np.interp(t, truth_t,
                                               np.asarray(truth_p_body))
        return out
