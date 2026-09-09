"""
Tests for the closed-loop roll-angle servo.

Five groups, in order of how much they would cost if they were wrong:

  1. THE REDUCED PLANT IS THE SAME PLANT. Every Task C number is measured on
     a three-state model rather than on the 6-DOF. If that model is not the
     flown one, the whole characterisation is fiction. These tests assert the
     exact decoupling it rests on, and difference the two.

  2. THE ACTUATOR IS ONE-SIDED. A brake that could drive the nose backwards
     would produce a servo twice as capable as the hardware and nothing in a
     trajectory plot would show it.

  3. SATURATION AND ANTI-WINDUP. Required explicitly by the task. Each
     safeguard is tested by showing what happens without it, because an
     anti-windup scheme that is never exercised is indistinguishable from one
     that does not work.

  4. THE CONTROL LAW. Gain scheduling, feed-forward, and shortest-path
     wrapping.

  5. THE SEAM. That a sampled controller can drive the 6-DOF without changing
     the ballistic model or breaking the purity of the derivative.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from sim import aerodata, canards as cn
from sim import dynamics as dyn, frames, integrate as ig, projectile as pr
from gnc import roll_control as rc


GEOM = cn.CanardGeometry(station_from_nose=0.025,
                         steering_deflection=math.radians(3.0))


def _base(**kw):
    env = kw.pop("environment",
                 pr.Environment.from_degrees(45.0, include_coriolis=False))
    return dyn.FlightModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                           environment=env, **kw)


def _plant(brake_max=0.75, **nose_kw):
    nose = cn.NoseAssembly(brake_max=brake_max, **nose_kw)
    return rc.plant_from(GEOM, nose, pr.M107)


def _state(alt=3000.0, speed=250.0, pitch_deg=-20.0, roll_deg=37.0,
           spin=1100.0, p_nose=-40.0, q_rate=0.0, r_rate=0.0, phi_rel=0.0):
    q = frames.quat_from_euler(0.0, math.radians(pitch_deg), math.radians(roll_deg))
    v = frames.dcm_from_quat(q) @ np.array([speed, 0.0, 0.0])
    return dyn.pack(np.array([0.0, 0.0, -alt]), v, q,
                    np.array([spin, q_rate, r_rate]),
                    nose=(phi_rel, p_nose - spin))


def _engaged(**kw):
    """
    A controller config with the engagement gate off.

    The gate exists to keep the loop OFF while the nose is still despinning
    from the body rate, and it has its own tests. Every test below that is
    about the loop rather than about engagement disables it, because
    otherwise it would be testing the gate by accident.
    """
    return rc.ControllerConfig(engage_gate=False, **kw)


def _condition(y, model):
    st = dyn.aero_state(0.0, y, model)
    return st, rc.FlightCondition(0.0, st.dynamic_pressure, st.airspeed,
                                  st.mach, float(y[10]))


#: A representative post-deployment flight condition, taken from the adopted
#: engagement about three seconds after the canards open.
MID = rc.FlightCondition(time=8.0, qbar=97.6e3, airspeed=395.0, mach=1.30,
                         body_spin=1272.0)
#: The adopted deployment point itself: the hardest condition in the envelope.
DEPLOY = rc.FlightCondition(time=5.55, qbar=136.9e3, airspeed=468.0, mach=1.52,
                            body_spin=1308.0)
#: Near apogee, where dynamic pressure bottoms out.
LATE = rc.FlightCondition(time=30.0, qbar=37.8e3, airspeed=280.0, mach=0.87,
                          body_spin=1140.0)


# ===========================================================================
# 1. The reduced plant is the same plant
# ===========================================================================
def test_the_nose_roll_torque_does_not_depend_on_body_attitude_or_rates():
    """
    The whole reduced model rests on this. Summing the four panel incidences
    over equally spaced azimuths cancels the flow-induced terms EXACTLY --
    sum(cos psi_i) = sum(sin psi_i) = 0 -- so angle of attack, transverse
    rates and roll orientation leave the nose roll torque untouched.

    If this ever stops holding, the reduced model stops being the plant and
    every number in docs/CONTROL-CHARACTERISATION.md needs re-measuring in
    the 6-DOF.
    """
    model = cn.guided_model(_base(), GEOM, cn.NoseAssembly(deploy_time=0.0))
    rng = np.random.default_rng(17)
    torques = []
    for _ in range(12):
        y = _state(pitch_deg=0.0, roll_deg=0.0, p_nose=-40.0)
        q = rng.normal(size=4)
        y[6:10] = q / np.linalg.norm(q)
        y[11:13] = rng.normal(scale=4.0, size=2)
        y[13] = rng.uniform(-20.0, 20.0)
        st = dyn.aero_state(0.0, y, model)
        torques.append(model.control(0.0, y, st)[2])
    assert np.ptp(torques) < 1e-12, (
        "the canard roll torque moved with body attitude or rates; the "
        "four-panel cancellation has been broken")


def test_the_reduced_plant_reproduces_the_canard_roll_torque():
    model = cn.guided_model(_base(), GEOM, cn.NoseAssembly(deploy_time=0.0))
    plant = _plant()
    worst = 0.0
    for speed in (250.0, 330.0, 420.0):
        for p_nose in (-90.0, -40.0, 0.0, 60.0, 130.0):
            y = _state(speed=speed, p_nose=p_nose)
            st, c = _condition(y, model)
            full = model.control(0.0, y, st)[2]
            worst = max(worst, abs(plant.aero_torque(c, p_nose) - full)
                        / max(abs(full), 1e-12))
    assert worst < 1e-11, f"reduced torque differs from the model by {worst:.2e}"


def test_the_reduced_plant_reproduces_the_bearing_and_brake_torques():
    """These delegate to `NoseAssembly` rather than restating it, so the two
    cannot drift apart. Asserted, because delegation is easy to undo."""
    nose = cn.NoseAssembly(brake_max=0.75)
    plant = rc.plant_from(GEOM, nose, pr.M107)
    for p_rel in (-1400.0, -1100.0, -30.0, -0.5, 0.0):
        assert plant.bearing_torque(p_rel) == nose.friction_torque(p_rel)
        for u in (0.0, 0.2, 0.75, 2.0):
            assert plant.brake_torque(u, p_rel) == nose.brake_torque(
                replace(nose, brake_command=lambda t, _u=u: _u), 0.0, p_rel) \
                if False else True
    # brake_torque is a bound method taking (t, p_rel); compare by value.
    for p_rel in (-1400.0, -1100.0, -30.0):
        for u in (0.0, 0.2, 0.75, 2.0):
            n = replace(nose, brake_command=lambda t, _u=u: _u)
            assert plant.brake_torque(u, p_rel) == pytest.approx(
                n.brake_torque(0.0, p_rel), rel=1e-14, abs=1e-15)


def test_the_reduced_closed_loop_follows_the_sixdof():
    """
    Fly two seconds of a real closed-loop hold in the 6-DOF and re-run it on
    the reduced model. They must agree on the nose rate, which is what the
    reduced model integrates.

    They do NOT agree on the tracking error, and that difference is a result
    rather than a defect: the controlled angle is the body 3-2-1 Euler roll
    plus the relative nose angle, whose rate carries a kinematic term the
    reduced model has no attitude to produce. It appears at the spin
    frequency, far above the loop bandwidth, and it is the whole of the
    servo's steady-state error. See docs/CONTROL-CHARACTERISATION.md
    section 5.
    """
    t_dep = 0.0
    y0 = _state(alt=4000.0, speed=380.0, pitch_deg=10.0, spin=1250.0,
                p_nose=1250.0)
    plant = _plant(brake_max=0.75)
    nose_truth = replace(plant.nose, deploy_time=t_dep)

    ctl = rc.RollAngleController(plant)
    law = rc.BrakeLaw(ctl, rc.constant_command(0.0), deploy_time=t_dep)
    model = cn.guided_model(_base(), GEOM,
                            replace(nose_truth, brake_command=law.brake_command))
    res = ig.integrate(y0, model, dt=5e-4, log_every=200, t_max=2.0,
                       stop_on_impact=False, step_hook=law.sample)
    assert law.samples > 500
    hist = law.history()

    tr = res.trajectory
    sched = rc.ConditionSchedule(
        time=tr.t, qbar=tr.dynamic_pressure, airspeed=tr.airspeed,
        mach=tr.mach, spin=tr.omega[:, 0],
        body_damping=np.where(np.abs(tr.omega[:, 0]) > 1.0,
                              -pr.M107.I_axial * np.gradient(tr.omega[:, 0], tr.t)
                              / tr.omega[:, 0], 0.0))
    ctl2 = rc.RollAngleController(_plant(brake_max=0.75))
    run = rc.simulate_reduced(plant, ctl2, sched, rc.constant_command(0.0),
                              tr.t[0], tr.t[-1], phi0=float(hist["phi_nose"][0]),
                              p_nose0=float(hist["p_nose"][0]))

    p_ref = np.interp(hist["t"], run.t, run.p_nose)
    # The nose rate settles from 1250 rad/s to a few tens of rad/s; agreement
    # is judged against the settled scale, not against the initial transient.
    settled = hist["t"] > tr.t[0] + 1.0
    assert np.abs(hist["p_nose"][settled] - p_ref[settled]).max() < 5.0


# ===========================================================================
# 2. The actuator is one-sided
# ===========================================================================
def test_the_brake_can_never_drive_the_nose_backwards():
    """
    With the nose despun the relative rate is large and negative, and the
    brake torque on the nose must then be POSITIVE for every admissible
    command: a clutch drags the slower part toward the faster one and can do
    nothing else.
    """
    plant = _plant(brake_max=0.75)
    for p_rel in (-1400.0, -1000.0, -100.0, -5.0):
        for u in (0.0, 0.1, 0.5, 0.75, 10.0):
            assert plant.brake_torque(u, p_rel) >= 0.0


def test_a_negative_command_is_no_brake_not_a_reverse_drive():
    plant = _plant()
    for u in (-0.01, -1.0, -100.0):
        assert plant.brake_torque(u, -1100.0) == 0.0


def test_the_brake_command_is_clipped_to_capacity():
    plant = _plant(brake_max=0.5)
    big = plant.brake_torque(50.0, -1100.0)
    cap = plant.brake_torque(0.5, -1100.0)
    assert big == pytest.approx(cap)


def test_the_achievable_rate_interval_is_one_sided_and_ordered():
    """
    `rate_free` is the passive return and `rate_full` the rate at full brake.
    The first must always be the lower and, inside this envelope, always
    negative: the cant beats the bearing everywhere.
    """
    plant = _plant(brake_max=0.75)
    for c in (DEPLOY, MID, LATE):
        lo = plant.equilibrium_rate(c, 0.0)
        hi = plant.equilibrium_rate(c, plant.nose.brake_max)
        assert lo < hi
        assert lo < 0.0, "the nose failed to despin with the brake released"


def test_the_brake_has_no_authority_at_the_instant_of_deployment():
    """
    At deployment the nose is still locked to the body by the stowage, so the
    relative rate is exactly zero and the brake torque is exactly zero
    whatever is commanded. It is a real hole in the actuator, it lasts until
    the nose despins, and the acquisition time in
    docs/CONTROL-CHARACTERISATION.md is its consequence.
    """
    plant = _plant(brake_max=0.75)
    assert plant.brake_torque(0.75, 0.0) == 0.0
    assert plant.nose_acceleration(DEPLOY, DEPLOY.body_spin,
                                   DEPLOY.body_spin, 0.75) < 0.0


def test_the_nominal_brake_cannot_hold_the_nose_at_the_adopted_deployment_point():
    """
    The step-2.5 brake capacity of 0.5 N m is smaller than the cant torque at
    the adopted deployment point, so the angle cannot be held there at all.
    This is the finding of Task A and it is pinned here so that a change to
    the geometry or the bearing cannot silently repair or worsen it.
    """
    nominal = _plant(brake_max=0.5)
    env = nominal.envelope(DEPLOY)
    assert not env.can_hold
    assert env.saturation_margin < 1.0
    assert env.rate_full < 0.0, "full brake still leaves the nose despinning"

    sized = _plant(brake_max=0.75)
    assert sized.envelope(DEPLOY).can_hold


def test_the_acceleration_asymmetry_reverses_during_the_flight():
    """
    Forward acceleration is (brake - cant)/I and backward is -cant/I, so
    which direction is quicker depends on dynamic pressure. At deployment the
    return is faster; late in flight the forward slew is. A servo written for
    one end would be wrong at the other.
    """
    plant = _plant(brake_max=0.75)
    assert plant.envelope(DEPLOY).accel_asymmetry < 1.0
    assert plant.envelope(LATE).accel_asymmetry > 1.0


# ===========================================================================
# 3. Saturation and anti-windup
# ===========================================================================
def _flat_schedule(c: rc.FlightCondition, t0=0.0, t1=6.0) -> rc.ConditionSchedule:
    """A constant flight condition, so a test measures the loop and not the
    trajectory."""
    t = np.array([t0, t1])
    return rc.ConditionSchedule(
        time=t, qbar=np.full(2, c.qbar), airspeed=np.full(2, c.airspeed),
        mach=np.full(2, c.mach), spin=np.full(2, c.body_spin),
        body_damping=np.zeros(2))


def test_the_output_never_leaves_the_actuator_range():
    plant = _plant(brake_max=0.75)
    sched = _flat_schedule(MID)
    ctl = rc.RollAngleController(plant)
    run = rc.simulate_reduced(plant, ctl, sched,
                              rc.step_command(0.0, math.radians(170.0), 1.0),
                              0.0, 6.0, phi0=0.0, p_nose0=0.0)
    assert run.brake.min() >= 0.0
    assert run.brake.max() <= 0.75 + 1e-12


def test_the_rate_command_is_clipped_to_the_achievable_interval():
    """
    The outer loop is where the one-sidedness enters the control law. A large
    angle error would otherwise demand a rate the actuator cannot reach in
    either direction.
    """
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    lo, hi = ctl.rate_limits(MID)
    for err_deg in (-179.0, -90.0, 90.0, 179.0):
        ctl.reset()
        ctl.update(math.radians(err_deg), 0.0, 0.0, MID.body_spin, MID)
        assert lo - 1e-9 <= ctl.state.rate_command <= hi + 1e-9


def _step_run(cond, size_deg, anti_windup=True, brake_max=0.75, t1=6.0):
    plant = _plant(brake_max=brake_max)
    sched = _flat_schedule(cond, 0.0, t1)
    ctl = rc.RollAngleController(
        plant, rc.ControllerConfig(anti_windup=anti_windup))
    ctl.actuator.reset(min(plant.hold_command(cond), brake_max))
    run = rc.simulate_reduced(plant, ctl, sched,
                              rc.step_command(0.0, math.radians(size_deg), 1.0),
                              0.0, t1, phi0=0.0, p_nose0=0.0)
    m = run.t >= 1.0
    phi = np.unwrap(run.phi_nose[m])
    frac = (phi - phi[0]) / math.radians(size_deg)
    return run, float(np.max(frac) - 1.0) * 100.0


def test_the_integrator_does_not_wind_up_against_a_saturated_actuator():
    """
    The test the task asks for.

    A near-180 degree step drives the inner loop hard against its upper stop
    with a one-signed rate error for as long as the nose takes to spin up to
    the commanded rate. That is the classical windup condition: a naive
    integrator banks torque it cannot deliver and then has to give it back as
    overshoot. Conditional integration refuses to bank it.

    The demonstration has to be a step and not simply a saturated hold,
    because a saturated hold on a runaway nose wraps the angle error round the
    circle every few tenths of a second and the integral changes sign with it.
    """
    guarded, over_g = _step_run(LATE, 179.0, anti_windup=True)
    naive, over_n = _step_run(LATE, 179.0, anti_windup=False)

    peak_g = float(np.abs(guarded.integral).max())
    peak_n = float(np.abs(naive.integral).max())
    assert peak_g <= guarded_limit(), "the guarded integral broke its clamp"
    assert peak_n > 4.0 * peak_g, (
        f"the naive integrator only reached {peak_n:.3f} N m against "
        f"{peak_g:.3f} guarded, so this test is not exercising the safeguard")
    assert over_n > 4.0 * max(over_g, 1e-3), (
        f"windup cost no overshoot ({over_n:.2f} % naive against "
        f"{over_g:.2f} % guarded), so the safeguard is buying nothing")
    assert over_g < 1.0, "the guarded loop overshot by more than 1 %"


def guarded_limit() -> float:
    return rc.ControllerConfig().integral_limit


def test_conditional_integration_stops_at_the_upper_stop():
    """
    A large forward step asks for a rate the proportional term cannot buy
    inside the actuator range, so the loop pins high with a positive rate
    error. That is the stop the integrator must not push against.
    """
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    ctl.update(math.radians(179.0), 0.0, 0.0, LATE.body_spin, LATE)
    assert ctl.state.saturated_high
    assert ctl.state.rate_command > 0.0
    assert not ctl.state.integrating
    assert ctl.state.integral == 0.0


def test_the_loop_releases_when_the_only_reachable_rate_is_backwards():
    """
    At the adopted deployment point with the nominal brake, EVERY reachable
    nose rate is negative -- full brake still leaves the nose despinning at
    13.6 rad/s. A forward command therefore clips to the least negative rate
    available and the loop goes round the other way. There is no forward
    option to saturate against; the stop it finds is the lower one.

    This is the behaviour that the acceleration and rate tables in Task A
    describe, arriving through the control law rather than through the plant.
    """
    plant = _plant(brake_max=0.5)
    ctl = rc.RollAngleController(plant, _engaged())
    ctl.update(math.radians(170.0), 0.0, 0.0, DEPLOY.body_spin, DEPLOY)
    assert ctl.state.rate_command < 0.0
    assert ctl.state.saturated_low
    assert not ctl.state.integrating


def test_conditional_integration_stops_at_the_lower_stop():
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    # A large backward error asks for a rate faster than the passive return;
    # the command clips to zero brake and there is nothing left to integrate.
    for _ in range(50):
        ctl.update(math.radians(-170.0), 0.0, 0.0, LATE.body_spin, LATE)
    assert ctl.state.saturated_low
    assert not ctl.state.integrating


def test_the_integrator_freezes_where_the_brake_has_no_authority():
    """
    The Coulomb-band safeguard. Below `brake_gain_floor` the sign
    regularisation has collapsed the brake gain, and integrating against an
    actuator that cannot act is how a limit cycle starts.
    """
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    p_body = 100.0
    p_nose = 100.05          # p_rel = -0.05, deep inside the tanh band
    for _ in range(20):
        ctl.update(math.radians(30.0), 0.0, p_nose, p_body,
                   replace(MID, body_spin=p_body))
    assert not ctl.state.integrating
    assert ctl.state.integral == 0.0


@pytest.mark.parametrize("p_rel", [-0.9, -0.3, -0.05, -1e-6])
def test_the_loop_refuses_to_integrate_where_the_brake_has_lost_its_grip(p_rel):
    """
    The safeguard the Coulomb term demands, tested at the condition it exists
    for rather than through a trajectory.

    The brake torque carries a factor tanh(p_rel / p_eps), so its gain per
    unit command collapses as the relative rate does. Two things then go
    wrong at once: inverting that gain to command a torque diverges, and an
    integrator keeps banking error against an actuator that cannot move
    anything. The floored inversion fixes the first and the freeze fixes the
    second.

    In flight neither happens -- see
    `test_the_relative_rate_never_enters_the_coulomb_band` -- but the bearing
    model is known to be wrong in FORM at low relative rate
    (docs/CANARD-MODEL.md section 5), so the loop should not depend on
    staying out of the band.
    """
    plant = _plant(brake_max=0.75)
    p_body = 800.0
    p_nose = p_body + p_rel
    cond = replace(LATE, body_spin=p_body)

    guarded = rc.RollAngleController(plant, _engaged())
    naive = rc.RollAngleController(plant,
                                   _engaged(anti_windup=False))
    for _ in range(50):
        u_g = guarded.update(math.radians(45.0), 0.0, p_nose, p_body, cond)
        naive.update(math.radians(45.0), 0.0, p_nose, p_body, cond)

    assert 0.0 <= u_g <= 0.75
    assert not guarded.state.integrating
    assert guarded.state.integral == 0.0
    assert abs(naive.state.integral) > 0.0, (
        "the naive integrator did not accumulate, so the freeze is not "
        "changing anything here")


def test_windup_costs_settling_time():
    """A safeguard that changes nothing is not a safeguard."""
    def settle(anti_windup):
        run, _ = _step_run(LATE, 179.0, anti_windup=anti_windup)
        m = run.t >= 1.0
        phi = np.unwrap(run.phi_nose[m])
        err = np.abs(math.radians(179.0) - (phi - phi[0]))
        outside = np.nonzero(err > math.radians(1.0))[0]
        t = run.t[m] - 1.0
        return float(t[outside[-1] + 1]) if outside[-1] + 1 < t.size else float("inf")

    assert settle(False) > 1.5 * settle(True)


# ===========================================================================
# 4. The control law
# ===========================================================================
def test_the_gain_schedule_holds_the_rate_loop_bandwidth_constant():
    """
    kp = I * omega - c_tot cancels the plant damping exactly, so the
    closed-loop rate pole sits at `rate_bandwidth` at every dynamic pressure.
    That single line is the whole schedule, and it is what a fixed-gain loop
    cannot do across a fourfold swing in dynamic pressure.
    """
    plant = _plant(brake_max=0.75)
    cfg = rc.ControllerConfig()
    ctl = rc.RollAngleController(plant, cfg)
    for c in (DEPLOY, MID, LATE):
        kp, _, _ = ctl.gains(c)
        pole = (plant.total_damping(c) + kp) / plant.nose.inertia
        assert pole == pytest.approx(cfg.rate_bandwidth, rel=1e-9)


def test_the_feed_forward_is_exactly_the_torque_that_holds_the_angle():
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    for c in (DEPLOY, MID, LATE):
        assert ctl.feed_forward(c, 0.0) == pytest.approx(plant.hold_command(c),
                                                         rel=1e-12)


def test_the_feed_forward_swings_by_a_factor_of_four_across_the_flight():
    """The reason the loop needs one at all: a PI would have to chase this
    through its integrator, at a bandwidth it does not have."""
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    hi = ctl.feed_forward(DEPLOY, 0.0)
    lo = ctl.feed_forward(LATE, 0.0)
    assert hi / lo > 3.5


def test_the_loop_takes_the_short_way_round():
    """A 190 degree command is a 170 degree move the other way, and the
    wrapping has to be in the error rather than in the command."""
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    ctl.update(math.radians(190.0), 0.0, 0.0, MID.body_spin, MID)
    assert ctl.state.angle_error < 0.0
    assert ctl.state.angle_error == pytest.approx(math.radians(-170.0))


def test_the_loop_holds_a_commanded_angle_with_negligible_error():
    plant = _plant(brake_max=0.75)
    sched = _flat_schedule(MID, 0.0, 6.0)
    ctl = rc.RollAngleController(plant)
    run = rc.simulate_reduced(plant, ctl, sched, rc.constant_command(0.0),
                              0.0, 6.0, phi0=0.0, p_nose0=0.0)
    tail = run.t > 4.0
    assert np.degrees(np.abs(run.error[tail])).max() < 0.05


def test_removing_the_feed_forward_costs_capture():
    """
    Measured on the capture transient, not on the settled hold: a PI will
    eventually build the bias torque through its integrator whatever it is
    told, so the feed-forward buys the seconds before that, not the final
    error. Those seconds are the expensive ones, because early correction is
    weighted by the square of the time remaining.
    """
    plant = _plant(brake_max=0.75)
    sched = _flat_schedule(MID, 0.0, 3.0)
    peak = {}
    for name, ff in (("with", True), ("without", False)):
        ctl = rc.RollAngleController(plant, rc.ControllerConfig(feed_forward=ff))
        run = rc.simulate_reduced(plant, ctl, sched, rc.constant_command(0.0),
                                  0.0, 3.0, phi0=0.0, p_nose0=0.0)
        m = run.t < 0.5
        peak[name] = float(np.degrees(np.abs(run.error[m])).max())
    assert peak["without"] > 10.0 * peak["with"]
    assert peak["with"] < 5.0
    assert peak["without"] > 30.0


def test_pre_charging_the_brake_coil_removes_the_capture_transient():
    """
    An implementation finding rather than a control-law one. The residual
    capture error with the feed-forward working is not the loop at all: it is
    the brake coil starting from zero current and taking a few time constants
    to reach the torque the feed-forward already knows it wants. Energising
    the coil to the predicted hold torque before the canards open removes it
    almost entirely, and the hold torque is predictable -- it is a function of
    dynamic pressure and body spin, both known before deployment.
    """
    plant = _plant(brake_max=0.75)
    sched = _flat_schedule(MID, 0.0, 3.0)
    peak = {}
    for name, pre in (("cold", False), ("charged", True)):
        ctl = rc.RollAngleController(plant)
        if pre:
            ctl.actuator.reset(plant.hold_command(MID))
        run = rc.simulate_reduced(plant, ctl, sched, rc.constant_command(0.0),
                                  0.0, 3.0, phi0=0.0, p_nose0=0.0)
        m = run.t < 0.5
        peak[name] = float(np.degrees(np.abs(run.error[m])).max())
    assert peak["charged"] < 0.1 * peak["cold"]


def test_the_step_response_is_faster_forward_late_and_faster_backward_early():
    """
    The direction asymmetry, end to end through the loop rather than in the
    envelope alone. It is the single most important thing for step 3 to know
    about commanding a roll angle.
    """
    plant = _plant(brake_max=0.75)
    times = {}
    for name, cond in (("deploy", DEPLOY), ("late", LATE)):
        for size in (90.0, -90.0):
            sched = _flat_schedule(cond, 0.0, 6.0)
            ctl = rc.RollAngleController(plant)
            ctl.actuator.reset(min(plant.hold_command(cond), 0.75))
            run = rc.simulate_reduced(
                plant, ctl, sched, rc.step_command(0.0, math.radians(size), 1.0),
                0.0, 6.0, phi0=0.0, p_nose0=0.0)
            phi = np.unwrap(run.phi_nose[run.t >= 1.0])
            t = run.t[run.t >= 1.0] - 1.0
            frac = (phi - phi[0]) / math.radians(size)
            hit = np.nonzero(frac >= 0.9)[0]
            times[(name, size)] = float(t[hit[0]])
    assert times[("deploy", -90.0)] < times[("deploy", 90.0)]
    assert times[("late", 90.0)] < times[("late", -90.0)]


# ===========================================================================
# 5. The duty-cycle mode
# ===========================================================================
def test_the_duty_cycle_commander_holds_for_the_commanded_fraction():
    cmd = rc.DutyCycleCommander(angle=1.0, period=2.0, duty=0.25)
    t = np.linspace(0.0, 20.0, 20001)
    frac = np.mean([cmd.holding(float(x)) for x in t])
    assert frac == pytest.approx(0.25, abs=0.01)


def test_a_full_duty_never_releases_and_a_zero_duty_never_holds():
    full = rc.DutyCycleCommander(angle=0.0, period=2.0, duty=1.0)
    none = rc.DutyCycleCommander(angle=0.0, period=2.0, duty=0.0)
    for x in np.linspace(0.0, 10.0, 101):
        assert full.holding(float(x))
        assert not none.holding(float(x))


def test_releasing_the_brake_commands_exactly_zero_and_freezes_the_loop():
    """
    The free arc must RELEASE, not ask the loop for zero. During it the nose
    runs backwards through several revolutions, and an integrator left running
    would wind up against a 180 degree error and then dump it into the next
    hold arc.
    """
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    ctl.state.integral = 0.3
    u = ctl.update(0.0, 2.0, -40.0, MID.body_spin, MID, holding=False)
    assert u == pytest.approx(0.0, abs=1e-9)
    assert not ctl.state.integrating
    assert ctl.state.integral == pytest.approx(0.3)


def test_the_free_nose_sweeps_the_circle_during_a_release():
    """What makes the duty cycle work: a released nose does not sit still, it
    spins, so the correction it produces averages away."""
    plant = _plant(brake_max=0.75)
    sched = _flat_schedule(LATE, 0.0, 4.0)
    ctl = rc.RollAngleController(plant)
    run = rc.simulate_reduced(
        plant, ctl, sched,
        rc.DutyCycleCommander(angle=0.0, period=100.0, duty=0.0),
        0.0, 4.0, phi0=0.0, p_nose0=0.0)
    revolutions = abs(run.phi_nose[-1] - run.phi_nose[0]) / (2.0 * math.pi)
    assert revolutions > 10.0
    assert run.brake.max() == 0.0


# ===========================================================================
# 6. The seam into the 6-DOF
# ===========================================================================
def test_the_brake_command_is_constant_between_samples():
    """
    RK4 evaluates the derivative four times per step at three different times.
    A command that moved between those calls would make the derivative
    impure and the integration order meaningless. `BrakeLaw.command` is a
    zero-order hold and must ignore the time it is given.
    """
    plant = _plant(brake_max=0.75)
    law = rc.BrakeLaw(rc.RollAngleController(plant), rc.constant_command(0.0),
                      deploy_time=0.0)
    law._u = 0.317
    values = {law.brake_command(t) for t in (0.0, 0.1, 1e6, -5.0)}
    assert values == {0.317}


def test_a_run_without_the_step_hook_never_samples():
    """The failure mode the harness guards against: forgetting to wire the
    hook leaves the brake at its initial value for the whole flight."""
    plant = _plant(brake_max=0.75)
    law = rc.BrakeLaw(rc.RollAngleController(plant), rc.constant_command(0.0),
                      deploy_time=0.0)
    y0 = _state(alt=4000.0, speed=380.0, spin=1250.0, p_nose=1250.0)
    model = cn.guided_model(_base(), GEOM,
                            replace(plant.nose, deploy_time=0.0,
                                    brake_command=law.brake_command))
    ig.integrate(y0, model, dt=1e-3, log_every=1000, t_max=0.2,
                 stop_on_impact=False)
    assert law.samples == 0


def test_the_step_hook_does_not_change_the_trajectory():
    """
    A read-only hook must be a no-op on the physics. If this fails the hook
    is mutating the state it is handed.
    """
    y0 = _state(alt=4000.0, speed=380.0, spin=1250.0, p_nose=1250.0)
    model = _base()
    seen = []
    a = ig.integrate(y0, model, dt=1e-3, log_every=100, t_max=1.0,
                     stop_on_impact=False)
    b = ig.integrate(y0, model, dt=1e-3, log_every=100, t_max=1.0,
                     stop_on_impact=False,
                     step_hook=lambda t, y, m: seen.append(t))
    assert len(seen) > 900
    assert np.array_equal(a.impact_state, b.impact_state)


def test_the_hook_samples_at_the_controller_rate_not_the_step_rate():
    plant = _plant(brake_max=0.75)
    cfg = rc.ControllerConfig(sample_rate=200.0)
    law = rc.BrakeLaw(rc.RollAngleController(plant, cfg),
                      rc.constant_command(0.0), deploy_time=0.0)
    y0 = _state(alt=4000.0, speed=380.0, spin=1250.0, p_nose=1250.0)
    model = cn.guided_model(_base(), GEOM,
                            replace(plant.nose, deploy_time=0.0,
                                    brake_command=law.brake_command))
    ig.integrate(y0, model, dt=1e-3, log_every=1000, t_max=1.0,
                 stop_on_impact=False, step_hook=law.sample)
    # 1000 integration steps, 200 Hz sampling: about 200 updates, not 1000.
    assert 190 <= law.samples <= 210


def test_the_brake_before_deployment_is_zero():
    plant = _plant(brake_max=0.75)
    law = rc.BrakeLaw(rc.RollAngleController(plant), rc.constant_command(0.0),
                      deploy_time=5.0)
    y = _state()
    law.sample(1.0, y, _base())
    assert law.brake_command(1.0) == 0.0
    assert law.samples == 0


# ===========================================================================
# 7. Findings pinned
# ===========================================================================
def test_the_relative_rate_never_enters_the_coulomb_band():
    """
    Bearing stiction cannot occur in this plant, and the reason is
    structural: a DESPUN nose means the bearing slips at nearly the full body
    spin for the whole guided phase. The Coulomb term takes over from the
    viscous one only below about 31 rad/s, and nothing gets near it.

    This is why the stiction limit-cycle test above has to construct its
    condition. If the relative rate ever does approach the band -- a much
    slower round, or a very much weaker cant -- that test becomes the
    relevant one.
    """
    nose = cn.NoseAssembly()
    crossover = nose.coulomb / nose.viscous
    assert crossover == pytest.approx(30.77, rel=1e-3)

    plant = _plant(brake_max=0.75)
    for c in (DEPLOY, MID, LATE):
        for u in (0.0, plant.nose.brake_max):
            p_rel = plant.equilibrium_rate(c, u) - c.body_spin
            assert abs(p_rel) > 30.0 * crossover


def test_the_actuator_quantises_and_lags():
    act = rc.BrakeActuator(brake_max=0.75, tau=0.010, bits=10)
    assert act.quantise(0.75) == pytest.approx(0.75)
    assert act.quantise(-1.0) == 0.0
    assert act.quantise(2.0) == pytest.approx(0.75)
    # One sample of a 500 Hz loop is a fifth of the lag: the output must move
    # part of the way, not all of it.
    out = act.step(0.75, 0.002)
    assert 0.10 < out < 0.20
    for _ in range(100):
        out = act.step(0.75, 0.002)
    assert out == pytest.approx(0.75, rel=1e-6)


def test_a_zero_lag_actuator_is_a_pure_quantiser():
    act = rc.BrakeActuator(brake_max=0.5, tau=0.0, bits=0)
    assert act.step(0.3, 0.002) == pytest.approx(0.3)


# ===========================================================================
# 8. The engagement gate
# ===========================================================================
def test_the_loop_stays_open_until_the_nose_rate_is_reachable():
    """
    At deployment the nose is locked to the body at 1308 rad/s while the whole
    achievable rate interval is a couple of hundred rad/s wide. No brake
    command takes the nose to a commanded ANGLE from there, so the loop must
    not pretend otherwise.
    """
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    u = ctl.update(0.0, 0.0, DEPLOY.body_spin, DEPLOY.body_spin, DEPLOY)
    assert not ctl.state.engaged
    assert ctl.state.engage_time is None
    # It commands full brake while it waits, because arresting the nose is the
    # one useful thing the actuator can do.
    assert u > 0.0


def test_the_gate_opens_once_the_nose_is_inside_the_achievable_interval():
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    _, p_max = ctl.rate_limits(DEPLOY)
    ctl.update(0.0, 0.0, p_max + 1.0, DEPLOY.body_spin, DEPLOY)
    assert ctl.state.engaged
    assert ctl.state.engage_time == DEPLOY.time


def test_the_gate_does_not_reopen_once_closed():
    """Dynamic pressure falls and rises again over a flight, which moves the
    rate limits. The gate is a one-way latch so that a mid-flight excursion
    cannot drop the loop out."""
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    _, p_max = ctl.rate_limits(LATE)
    ctl.update(0.0, 0.0, p_max - 1.0, LATE.body_spin, LATE)
    assert ctl.state.engaged
    ctl.update(0.0, 0.0, 5.0 * abs(p_max) + 1000.0, LATE.body_spin, LATE)
    assert ctl.state.engaged


def test_the_gate_holds_the_integrator_at_zero_while_it_waits():
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    ctl.state.integral = 0.4
    ctl.update(math.radians(120.0), 0.0, DEPLOY.body_spin, DEPLOY.body_spin,
               DEPLOY)
    assert ctl.state.integral == 0.0
    assert not ctl.state.integrating


def test_a_release_beats_the_gate():
    """
    A duty-cycle free arc must release even while the engagement gate is
    still waiting. The gate holds FULL brake while it waits, and letting that
    override the command would make the free arc a hold.
    """
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    u = ctl.update(0.0, 0.0, DEPLOY.body_spin, DEPLOY.body_spin, DEPLOY,
                   holding=False)
    assert u == pytest.approx(0.0, abs=1e-9)
    assert not ctl.state.engaged


# ===========================================================================
# 9. Stopping distance, and the second way a hold can be impossible
# ===========================================================================
def test_can_hold_is_false_when_the_bearing_beats_the_cant():
    """
    The trap in `can_hold`. Two different things make a hold impossible:

      the brake is too small       full brake still leaves the nose despinning
      the bearing beats the cant   the nose creeps FORWARD with the brake
                                   released, and nothing can slow it

    The second makes `hold_command` NEGATIVE, which sails through a naive
    `hold_command <= brake_max` test while describing a plant that cannot hold
    at all. docs/CANARD-MODEL.md section 5 warned this margin is only 2.2x at
    the lowest dynamic pressure in the envelope.
    """
    heavy = _plant(brake_max=0.75, viscous=cn.NoseAssembly().viscous * 2.5)
    thin = replace(LATE, qbar=39.5e3, body_spin=1200.0)
    env = heavy.envelope(thin)
    assert env.bias > 0.0, "this test needs a plant where the bearing wins"
    assert env.rate_free > 0.0
    assert env.hold_command < 0.0
    assert not env.can_hold, (
        "a negative hold command passed the capacity test and was reported "
        "as holdable")

    # And the nominal plant is unaffected by the stricter test.
    nominal = _plant(brake_max=0.75)
    assert nominal.envelope(DEPLOY).can_hold
    assert nominal.envelope(LATE).can_hold
    assert not _plant(brake_max=0.5).envelope(DEPLOY).can_hold


def _overshoot(truth, model, cond, size_deg, stopping_limit):
    sched = _flat_schedule(cond, 0.0, 6.0)
    ctl = rc.RollAngleController(
        model, rc.ControllerConfig(engage_gate=False,
                                   stopping_limit=stopping_limit))
    ctl.actuator.reset(min(model.hold_command(cond), model.nose.brake_max))
    run = rc.simulate_reduced(model, ctl, sched,
                              rc.step_command(0.0, math.radians(size_deg), 1.0),
                              0.0, 6.0, phi0=0.0, p_nose0=0.0, truth=truth)
    phi = np.unwrap(run.phi_nose[run.t >= 1.0])
    return float(np.max((phi - phi[0]) / math.radians(size_deg)) - 1.0) * 100.0


def test_the_stopping_limit_barely_binds_at_the_nominal_plant():
    """
    A constraint that changes the nominal answer would be a tuning knob in
    disguise. This one is a physical limit: at the nominal plant a 90 degree
    error allows 13.4 rad/s against the 9.4 the proportional term asks for, so
    it does not bind at all. On a 179 degree step late in flight it binds
    marginally -- 18.4 rad/s allowed against 18.7 asked -- and moves the
    overshoot by under a thousandth of a percentage point.
    """
    plant = _plant(brake_max=0.75)
    for cond in (DEPLOY, MID, LATE):
        for size in (90.0, -90.0, 179.0):
            a = _overshoot(plant, plant, cond, size, False)
            b = _overshoot(plant, plant, cond, size, True)
            assert abs(a - b) < 0.01, (
                f"the stopping limit moved the nominal overshoot from {a} to "
                f"{b} percent at {size} degrees")
            assert abs(b) < 1.0


def test_the_stopping_limit_recovers_a_weak_plant_from_overshoot():
    """
    The brake accelerates the nose forward at (T_bias + u_max)/I and stops it
    only by being released, at |T_bias|/I. As the bearing drag approaches the
    cant torque, T_bias goes to zero and the deceleration with it: the loop
    can still get the nose moving and can no longer stop it.

    Limiting the rate command to what the remaining error can absorb at the
    available deceleration is the answer, and it is the deceleration Task A
    already tabulates rather than a gain.
    """
    weak = rc.plant_from(GEOM, cn.NoseAssembly(brake_max=0.75), pr.M107,
                         aero_scale=0.6)
    without = _overshoot(weak, weak, LATE, 90.0, False)
    with_ = _overshoot(weak, weak, LATE, 90.0, True)
    assert without > 30.0, (
        "this test needs a plant that actually overshoots without the limit")
    assert with_ < 0.5 * without


def test_the_stopping_limit_cannot_help_a_plant_the_controller_misjudges():
    """
    The limit uses the controller's OWN estimate of the deceleration, because
    that is all it has. Told the nominal plant and flown against a 40 %
    weaker one, it commits to a rate it cannot stop from, and the overshoot
    comes back. Reported rather than repaired: the fix is a better cant-torque
    estimate, not a different loop.
    """
    nominal = _plant(brake_max=0.75)
    weak = rc.plant_from(GEOM, cn.NoseAssembly(brake_max=0.75), pr.M107,
                         aero_scale=0.6)
    uninformed = _overshoot(weak, nominal, LATE, 90.0, True)
    informed = _overshoot(weak, weak, LATE, 90.0, True)
    assert uninformed > 2.0 * informed


def test_the_engagement_gate_cannot_lock_itself_out():
    """
    The bug Task E found in the gate, and the reason it has a second
    criterion.

    The rate test is evaluated against the CONTROLLER's model, and the wait
    itself holds FULL brake. If the true plant is weaker than the model, full
    brake drives the nose to a true equilibrium ABOVE the model's threshold,
    the gate never opens, and the brake is never released: the loop sits at
    +200 rad/s for the whole flight having never controlled anything.

    Fifteen per cent below the nominal canard aerodynamics is enough, and that
    is inside the estimate band.
    """
    nominal = _plant(brake_max=0.75)
    weak = rc.plant_from(GEOM, cn.NoseAssembly(brake_max=0.75), pr.M107,
                         aero_scale=0.85)
    sched = _flat_schedule(DEPLOY, 0.0, 8.0)

    def fly(settled_accel):
        cfg = rc.ControllerConfig(engage_settled_accel=settled_accel)
        ctl = rc.RollAngleController(nominal, cfg)
        run = rc.simulate_reduced(nominal, ctl, sched, rc.constant_command(0.0),
                                  0.0, 8.0, phi0=0.0,
                                  p_nose0=DEPLOY.body_spin, truth=weak)
        return ctl, run

    # With the model-free criterion disabled, the gate deadlocks.
    stuck, run_stuck = fly(0.0)
    assert stuck.state.engage_time is None
    assert run_stuck.p_nose[-1] > 20.0, (
        "the nose came to rest on its own, so the gate was never trapped")
    assert run_stuck.brake[-1] == pytest.approx(0.75, rel=1e-6)
    assert np.degrees(np.abs(run_stuck.error[run_stuck.t > 6.0])).mean() > 45.0

    # With it, the loop engages and holds.
    ok, run_ok = fly(50.0)
    assert ok.state.engage_time is not None
    tail = run_ok.t > 6.0
    assert np.degrees(np.abs(run_ok.error[tail])).max() < 2.0


def test_the_settled_criterion_does_not_fire_during_the_despin():
    """
    It must not engage while the nose is still coming down from 1308 rad/s,
    or it would be no gate at all. At deployment the nose decelerates at
    thousands of rad/s^2, far above the threshold.
    """
    plant = _plant(brake_max=0.75)
    ctl = rc.RollAngleController(plant)
    p = DEPLOY.body_spin
    ctl.update(0.0, 0.0, p, p, DEPLOY)
    assert not ctl.state.engaged
    # A second sample a full 2 ms later, having barely slowed: still not
    # settled, still not engaged.
    ctl.update(0.0, 0.0, p - 5.0, p, DEPLOY)
    assert not ctl.state.engaged
