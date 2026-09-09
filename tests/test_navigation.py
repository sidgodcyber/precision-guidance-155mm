"""
Step 5 tests: the sensor models, the filter, and the properties the whole
navigation architecture rests on.

The load-bearing ones, in order of how much would break if they failed:

  * the despun IMU's own Euler roll IS the servo's controlled variable;
  * an integrating gyro and accelerometer make the lever-arm compensation
    exact, which is what stops the strapdown drifting;
  * saturation is modelled, latched, and never silently integrated;
  * the filter is a pure, seed-deterministic object step 6 can build
    thousands of;
  * one attitude vector observation is not enough, and the test proves the
    velocity reference is what makes the roll observable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim import dynamics as dyn, frames, projectile as pr
from gnc import navigation as nv, sensors as sn


# ===========================================================================
# The identity the architecture rests on
# ===========================================================================
def test_despun_euler_roll_is_the_controlled_variable():
    """
    C_N^E = C_B^E Rx(phi_rel) = Rz(psi) Ry(theta) Rx(phi_body + phi_rel), so
    the nose frame's 3-2-1 Euler roll IS `phi_nose` and its pitch and yaw ARE
    the body's.

    This is why the IMU goes in the despun section and why the servo needs no
    transformation. If it ever fails, `gnc/sensors.py`'s whole account is
    wrong.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        psi, th, ph = rng.uniform(-math.pi, math.pi), rng.uniform(-1.4, 1.4), \
            rng.uniform(-math.pi, math.pi)
        phi_rel = rng.uniform(-50.0, 50.0)
        q = frames.quat_from_euler(psi, th, ph)
        C = frames.dcm_from_quat(q) @ sn.nose_from_body(phi_rel)
        theta_n = -math.asin(max(-1.0, min(1.0, C[2, 0])))
        roll_n = math.atan2(C[2, 1], C[2, 2])
        yaw_n = math.atan2(C[1, 0], C[0, 0])
        d = roll_n - (ph + phi_rel)
        assert abs(math.atan2(math.sin(d), math.cos(d))) < 1e-9
        assert abs(theta_n - th) < 1e-12
        dy = yaw_n - psi
        assert abs(math.atan2(math.sin(dy), math.cos(dy))) < 1e-12


def test_nose_from_body_is_a_rotation_about_x():
    for a in (-3.1, 0.0, 0.7, 41.0):
        C = sn.nose_from_body(a)
        assert np.allclose(C @ C.T, np.eye(3), atol=1e-14)
        assert abs(np.linalg.det(C) - 1.0) < 1e-14
        assert np.allclose(C[:, 0], [1.0, 0.0, 0.0])


# ===========================================================================
# Saturation is modelled, not assumed away
# ===========================================================================
def test_gyro_saturation_clips_and_latches():
    imu = sn.Imu(sn.GYRO_DESPUN, sn.ACCEL_DESPUN, np.random.default_rng(1))
    fs = sn.GYRO_DESPUN.full_scale
    s = imu.sample(0.0, 0.002, np.array([50.0 * fs, 0.0, 0.0]), np.zeros(3))
    assert s.gyro_saturated
    assert abs(s.gyro[0]) <= fs * 1.0000001
    # ... and it is a CLIPPED VALUE, not a NaN and not a flag in the data.
    assert np.isfinite(s.gyro).all()
    assert imu.saturated_samples == 1


def test_body_mounted_gyro_is_pinned_for_the_whole_flight():
    """
    The measurement that decides the layout. 1308 rad/s at deployment against
    a 2000 deg/s part is a factor of 37.
    """
    spin_deploy = 1308.0
    assert spin_deploy > 20.0 * sn.GYRO_DESPUN.full_scale
    # and a body-mounted accelerometer at any offset a fuze-well kit can hold
    a = sn.body_mounted_centripetal(spin_deploy, 0.025)
    assert a > 100.0 * sn.ACCEL_DESPUN.full_scale
    # the offset that would keep it inside range is sub-millimetre
    r_ok = sn.ACCEL_DESPUN.full_scale / spin_deploy ** 2
    assert r_ok < 3.0e-4


def test_despun_centripetal_term_is_negligible():
    """The same expression, at the rate the despun section actually turns."""
    assert sn.body_mounted_centripetal(3.0, 0.025) < 0.3


# ===========================================================================
# Determinism and purity -- what step 6 needs
# ===========================================================================
def test_sensor_suite_is_seed_deterministic():
    cfg = sn.SuiteConfig()
    model, y = _toy_model_state()
    a = sn.SensorSuite(cfg, seed=11)
    b = sn.SensorSuite(cfg, seed=11)
    c = sn.SensorSuite(cfg, seed=12)
    ra = [a.measure(0.002 * k, y, model, 0.002) for k in range(5)]
    rb = [b.measure(0.002 * k, y, model, 0.002) for k in range(5)]
    rc = [c.measure(0.002 * k, y, model, 0.002) for k in range(5)]
    for x, z in zip(ra, rb):
        assert np.allclose(x["imu"].gyro, z["imu"].gyro)
        assert np.allclose(x["mag"], z["mag"])
    assert not np.allclose(ra[-1]["mag"], rc[-1]["mag"])


def test_measuring_does_not_mutate_the_state_vector():
    """`sim/dynamics.py`'s purity rule extends to everything that reads y."""
    cfg = sn.SuiteConfig()
    model, y = _toy_model_state()
    before = y.copy()
    sn.SensorSuite(cfg, seed=3).measure(0.0, y, model, 0.002)
    assert np.array_equal(y, before)


def test_confidence_register_covers_every_low_entry_and_prints_them():
    import io
    keys = sn.low_confidence_keys()
    assert keys, "the register must expose its own weakest entries"
    buf = io.StringIO()
    sn.warn_low_confidence(stream=buf)
    text = buf.getvalue()
    for k in keys:
        assert k in text
    # the two the whole step turns on
    assert "mag_calibration_residual" in keys
    assert "gnss_reacquire_s" in keys


# ===========================================================================
# The filter
# ===========================================================================
def _filter(seed=0):
    return nv.NavigationFilter(nv.NavConfig(), sn.GYRO_DESPUN, sn.ACCEL_DESPUN,
                               sn.NAGPUR)


def _toy_model_state():
    from analysis import roll_servo as rs
    model = rs.base_model()
    launch = pr.LaunchConditions.from_mils(684.0, 525.3)
    y = dyn.initial_state(pr.M107, launch)
    y = y.astype(float)
    y[0:3] = np.array([2900.0, 5.0, -1500.0])
    y[3:6] = np.array([462.0, 1.6, -213.0])
    return model, y


def test_magnetometer_alone_cannot_determine_roll():
    """
    One vector observation fixes two attitude degrees of freedom and leaves the
    rotation about that vector free. This is the measurement that forced the
    velocity reference into the design, so it is pinned as a test rather than
    left as a remark.
    """
    f = _filter()
    psi, th, phi = 0.05, 0.43, 0.9
    q_true = frames.quat_from_euler(psi, th, phi)
    b_e = sn.NAGPUR.field_ned(np.zeros(3))
    mag = frames.dcm_from_quat(q_true).T @ b_e
    v = 450.0 * frames.dcm_from_quat(q_true)[:, 0]
    q0 = frames.quat_from_euler(psi, th, phi + math.radians(20.0))
    f.initialise(0.0, np.zeros(3), v, q0, np.full(3, 5.0), np.full(3, 1.0),
                 np.full(3, math.radians(10.0)))
    f._f_nose = np.array([-20.0, 0.0, 0.0])
    for k in range(60):
        f.propagate_covariance(0.01)
        f.update_magnetometer(0.01 * k, mag)
    _, _, roll = f.euler
    left = abs(math.degrees(nv._wrap_pi(roll - phi)))
    assert left > 5.0, "the magnetometer alone should NOT fix the roll"


def test_velocity_reference_makes_the_roll_observable():
    """The same case, with the second attitude vector. 20 deg -> under 0.1."""
    f = _filter()
    psi, th, phi = 0.05, 0.43, 0.9
    q_true = frames.quat_from_euler(psi, th, phi)
    b_e = sn.NAGPUR.field_ned(np.zeros(3))
    mag = frames.dcm_from_quat(q_true).T @ b_e
    v = 450.0 * frames.dcm_from_quat(q_true)[:, 0]
    q0 = frames.quat_from_euler(psi, th, phi + math.radians(20.0))
    f.initialise(0.0, np.zeros(3), v, q0, np.full(3, 5.0), np.full(3, 1.0),
                 np.full(3, math.radians(10.0)))
    f._f_nose = np.array([-20.0, 0.0, 0.0])
    for k in range(60):
        f.propagate_covariance(0.01)
        f.update_magnetometer(0.01 * k, mag)
        f.update_velocity_attitude(0.01 * k)
    _, _, roll = f.euler
    assert abs(math.degrees(nv._wrap_pi(roll - phi))) < 0.1


def test_covariance_stays_symmetric_and_positive():
    f = _filter()
    f.initialise(0.0, np.zeros(3), np.array([450.0, 0.0, -100.0]),
                 np.array([1.0, 0.0, 0.0, 0.0]), np.full(3, 5.0),
                 np.full(3, 1.0), np.full(3, 0.1))
    rng = np.random.default_rng(2)
    b_e = sn.NAGPUR.field_ned(np.zeros(3))
    for k in range(200):
        f.propagate_state(0.002, rng.normal(0, 0.1, 3),
                          np.array([-20.0, 0.0, 0.0]) + rng.normal(0, 0.01, 3))
        if k % 5 == 0:
            f.propagate_covariance(0.01)
            f.update_magnetometer(0.002 * k,
                                  f.dcm.T @ b_e + rng.normal(0, 5e-7, 3))
            f.update_velocity_attitude(0.002 * k)
    assert np.allclose(f.P, f.P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(f.P) > -1e-12)
    assert abs(np.linalg.norm(f.q) - 1.0) < 1e-12


def test_saturated_axis_uses_the_substitute_rate_not_zero():
    """
    Zeroing a saturated axis costs 13 m/s of GNSS lever-arm velocity. The
    substituted magnetometer rate has to reach `_omega`, because that is what
    the lever-arm compensation reads.
    """
    f = _filter()
    f.initialise(0.0, np.zeros(3), np.array([450.0, 0.0, 0.0]),
                 np.array([1.0, 0.0, 0.0, 0.0]), np.full(3, 5.0),
                 np.full(3, 1.0), np.full(3, 0.1))
    sat = np.array([True, False, False])
    f.propagate_state(0.002, np.array([34.9, 0.1, -0.2]), np.zeros(3),
                      saturated_axes=sat,
                      substitute_rate=np.array([1308.0, 0.0, 0.0]),
                      substitute_sigma=20.0)
    assert abs(f._omega[0] - 1308.0) < 1e-9
    assert f._sat_axes[0] and not f._sat_axes[1]


def test_attitude_reset_convention_matches_the_jacobians():
    """
    C_true = (I + [dpsi]x) C_est. If the reset and the Jacobians ever disagree
    in sign the filter diverges quietly, so the convention is pinned.
    """
    f = _filter()
    C0 = frames.dcm_from_quat(frames.quat_from_euler(0.3, -0.2, 1.1))
    f.q = frames.quat_from_euler(0.3, -0.2, 1.1)
    f.P = np.eye(nv.N_STATES)
    dpsi = np.array([0.002, -0.001, 0.0015])
    f._begin()
    f._dx[nv.IDX["psi"]] = dpsi
    f._commit(0.0)
    expect = (np.eye(3) + nv._skew(dpsi)) @ C0
    assert np.allclose(f.dcm, expect, atol=1e-6)


# ===========================================================================
# Warm start
# ===========================================================================
def test_warm_start_lands_on_the_requested_time():
    """
    A fixed-step propagator asked for a t_max smaller than its own step
    overshoots by a whole step -- 34 m at 684 m/s, and worst exactly where the
    warm start is judged. Regression test.
    """
    from analysis import roll_servo as rs
    gun = nv.GunData(muzzle_velocity=684.0,
                     quadrant_elevation=525.3 * pr.MIL_TO_RAD)
    env = rs.base_model().environment
    r_small, _, _, _, _ = nv.warm_start_state(gun, pr.M107, env, 0.001)
    assert np.linalg.norm(r_small) < 1.5


def test_warm_start_matches_the_6dof_at_deployment():
    """
    The a-priori trajectory is the SAME reduced-order model the impact
    predictor already carries, so the warm start costs the flight computer no
    new model. docs/MODEL-ERROR.md prices it at 0.65 m RMS over a whole
    flight; over the first 5.5 s it is well under a metre.
    """
    from analysis import guidance_cep as gc, roll_servo as rs
    md = gc.load_maps()
    base = gc.engagement_context(md, "long")["base"]
    gun = nv.GunData(muzzle_velocity=base["muzzle_velocity"],
                     quadrant_elevation=base["qe_mils"] * pr.MIL_TO_RAD)
    env = rs.base_model().environment
    r, v, p, sr, sv = nv.warm_start_state(gun, pr.M107, env,
                                          base["deploy_time"])
    y = np.asarray(base["y_deploy"], dtype=float)
    assert np.linalg.norm(r - y[0:3]) < 1.0
    assert np.linalg.norm(v - y[3:6]) < 0.2
    # the uncertainty is the GUN's dispersion, and it dominates the model
    assert sr[0] > 5.0


# ===========================================================================
# The seams
# ===========================================================================
def test_navigation_adds_no_third_seam():
    """
    `NavigationSystem.chain` is a step_hook exactly as `BrakeLaw.sample` and
    `GuidanceLaw.sample` are, and `roll()` is a function of the object's own
    latched state -- not of the time it is called at.
    """
    from analysis import guidance_cep as gc, nav_common as ncm, roll_servo as rs
    md = gc.load_maps()
    base = gc.engagement_context(md, "long")["base"]
    nav = ncm.make_nav(0, base, log=False)
    model, y = _toy_model_state()
    nav.sample(0.0, y, model)
    a = nav.roll()
    b = nav.roll()
    assert a == b
    assert np.array_equal(nav.state(), nav.state())
    calls = []
    hook = nav.chain(lambda t, yy, m: calls.append(t))
    hook(0.004, y, model)
    assert calls == [0.004]


def test_filter_reports_itself_invalid_rather_than_confident_and_wrong():
    """
    The degradation ladder's `reversionary` state depends on this: a filter
    with nothing to go on must say so, not produce a number.
    """
    from analysis import guidance_cep as gc, nav_common as ncm
    md = gc.load_maps()
    base = gc.engagement_context(md, "long")["base"]
    suite = sn.SuiteConfig(gnss_denied=True)
    nav = ncm.make_nav(0, base, suite=suite, warm_start=False, log=False)
    model, y = _toy_model_state()
    for k in range(50):
        nav.sample(0.002 * k, y, model)
    assert not nav.filter.initialised
    assert not nav.valid


def test_state_estimate_is_shaped_like_a_6dof_state():
    """
    So `gnc.guidance` consumes it through the interface it already consumes
    truth through, and `models.mpmm.state_from_sixdof` reads it unchanged.
    """
    from models import mpmm
    from analysis import guidance_cep as gc, nav_common as ncm
    md = gc.load_maps()
    base = gc.engagement_context(md, "long")["base"]
    nav = ncm.make_nav(0, base, log=False)
    model, y = _toy_model_state()
    nav.sample(0.0, y, model)
    s = nav.state()
    assert s.shape == (dyn.STATE_SIZE,)
    y7 = mpmm.state_from_sixdof(s)
    assert y7.shape == (mpmm.MPMM_STATE_SIZE,)
    assert np.isfinite(y7).all()


# ===========================================================================
# Geomagnetic geometry
# ===========================================================================
def test_field_magnitude_and_dip_are_what_the_site_says():
    b = sn.NAGPUR.field_ned(np.zeros(3))
    assert abs(np.linalg.norm(b) - sn.NAGPUR.intensity) < 1e-15
    dip = math.asin(float(b[2]) / np.linalg.norm(b))
    assert abs(dip - sn.NAGPUR.inclination) < 1e-12


def test_roll_observability_vanishes_along_the_field():
    """
    The 1/sin amplification is the reason `mag_min_transverse` exists.
    """
    f = sn.NAGPUR
    b = f.field_ned(np.zeros(3))
    along = b / np.linalg.norm(b)
    assert f.axis_angle(np.zeros(3), along) < 1e-6
    perp = np.cross(along, [0.0, 0.0, 1.0])
    perp /= np.linalg.norm(perp)
    assert abs(f.axis_angle(np.zeros(3), perp) - math.pi / 2) < 1e-9


def test_magnetometer_percentages_are_of_full_scale_not_signal():
    """
    0.1 %FS on a +-6 gauss part is 600 nT, which is 1.3 % of a 46 500 nT
    field. Reading these as percentages of signal understates the magnetometer
    error tenfold, and it is the easiest mistake in the sensor model.
    """
    fs_error = sn.MAG_AMR.linearity * sn.MAG_AMR.full_scale
    assert fs_error / sn.NAGPUR.intensity > 0.01


def test_magnetometer_bandwidth_resolves_the_deployment_spin():
    """
    At 500 Hz sampling the roll phase advances 2.62 rad per sample at
    1308 rad/s -- inside pi, so the despin is unambiguous. This is the whole
    practical content of the part's DC-to-5 MHz line.
    """
    advance = 1308.0 / 500.0
    assert advance < math.pi
    assert sn.MAG_AMR.bandwidth > 1000.0


# ===========================================================================
# Consistency machinery
# ===========================================================================
def test_chi2_bounds_bracket_the_dimension():
    from analysis import nav_consistency as ncons
    for dof in (6, 9, 15):
        lo, hi = ncons.chi2_bounds(dof, 64)
        assert lo < dof < hi
        assert hi / dof < 1.5


def test_whiteness_detects_a_correlated_sequence():
    from analysis import nav_consistency as ncons
    rng = np.random.default_rng(4)
    white = rng.normal(size=4000)
    corr = np.convolve(rng.normal(size=4200), np.ones(50) / 50.0)[:4000]
    assert abs(ncons._whiteness(white)["acf"][0]) < 0.1
    assert ncons._whiteness(corr)["acf"][0] > 0.5


def test_truth_fed_run_reproduces_step_3_exactly():
    """
    The paired comparison of Task F is only a measurement of NAVIGATION if its
    truth-fed half is the same round step 3 flew. `nav_common.run_guided_nav`
    with `use_nav=False` must therefore reproduce
    `analysis.guidance_cep.run_guided` to the last digit -- not approximately,
    and not "close enough", because any difference would be silently added to
    the navigation contribution.

    Slow (two 6-DOF trajectories) and worth it: this is the assumption the
    whole of docs/NAV-CEP.md rests on.
    """
    from analysis import guidance_cep as gc, nav_cep as ncep, nav_common as ncm
    md = gc.load_maps()
    ctx = gc.engagement_context(md, "long")
    opts = gc.scheduler_options(md, "long")["proportional"]
    draws = gc.make_draws(ctx, 4)
    case = ncep._cases(ctx, draws, use_nav=False, scheduler_opts=opts)[0]
    a = ncm.run_guided_nav(case)
    b = gc.run_guided({**{k: v for k, v in case.items() if k != "use_nav"},
                       "scheduler": "proportional", "scheduler_opts": opts})
    assert abs(a["range_m"] - b["range_m"]) < 1e-9
    assert abs(a["drift_m"] - b["drift_m"]) < 1e-9
    assert abs(a["miss_m"] - b["miss_m"]) < 1e-9


def test_attitude_updates_do_not_reset_the_position_coast_timer():
    """
    The defect docs/NAV-DEGRADATION.md section 3 found, pinned so it cannot
    return.

    The magnetometer and the velocity alignment reference need no satellites,
    so they update continuously even under total GNSS denial. When they reset
    the coast timer, a filter with ZERO position measurements reported itself
    valid for the whole flight -- measured, before the fix: VALID on 20 of 24
    rounds at impact; after it, 0 of 12, with every round entering the invalid
    branch.
    """
    f = _filter()
    b_e = sn.NAGPUR.field_ned(np.zeros(3))
    f.initialise(0.0, np.zeros(3), np.array([450.0, 0.0, -100.0]),
                 frames.quat_from_euler(0.0, 0.2, 0.5),
                 np.full(3, 5.0), np.full(3, 1.0), np.full(3, 0.05))
    f._f_nose = np.array([-20.0, 0.0, 0.0])
    f.t = 40.0
    f.update_magnetometer(40.0, f.dcm.T @ b_e)
    f.update_velocity_attitude(40.0)

    # An attitude update IS a correction ...
    assert f.last_correction_t == 40.0
    # ... and it is NOT a position correction.
    assert f.last_position_correction_t is None
    assert f.updates["gnss_pos"] == 0

    # A GNSS fix is, and only it moves the position timer.
    fix = sn.GnssFix(t=41.0, t_valid=40.9, position=np.zeros(3),
                     velocity=np.array([450.0, 0.0, -100.0]),
                     sigma_position=np.full(3, 3.0),
                     sigma_velocity=np.full(3, 0.5))
    f.update_gnss(fix, 0.0, 0.0, (0.439, 0.010, 0.0))
    assert f.last_position_correction_t == 41.0


def test_denied_navigation_declares_itself_invalid():
    """
    The end-to-end form of the same property: with no satellites at all, the
    system must stop claiming a solution once it has coasted past
    `max_coast_s` from launch -- because a warm start is an a-priori
    trajectory, not a fix, and it decays.
    """
    from analysis import guidance_cep as gc, nav_common as ncm
    md = gc.load_maps()
    base = gc.engagement_context(md, "long")["base"]
    suite = sn.SuiteConfig(gnss_denied=True)
    nav = ncm.make_nav(0, base, suite=suite, warm_start=True, log=False)
    model, y = _toy_model_state()
    nav.sample(0.0, y, model)
    assert nav.filter.initialised, "the warm start should give it a solution"
    assert nav.valid, "and it is usable at first"
    nav.filter.t = nav.config.max_coast_s + 1.0
    assert nav.filter.last_position_correction_t is None
    assert not nav.valid, "past the coast limit with no fix, it must say so"
