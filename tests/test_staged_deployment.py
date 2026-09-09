"""
Tests for the two-stage canard deployment. Step 4.5.

Four groups, in order of how much they would cost if they were wrong:

  1. STAGE 1 IS TWO PANELS, NOT A SCALED FOUR. A stowed panel carries no load
     of any kind. If stage 1 were implemented by scaling a coefficient it
     would get the despin torque right and the transverse force wrong, and
     nothing in a trajectory plot would show it.

  2. THE SINGLE-STAGE PATH IS UNTOUCHED. Every number in steps 2.5 and 4 was
     measured with both pairs released together, and that has to stay bit
     identical or the comparison this step exists to make is not a comparison.

  3. THE STAGE-2 TRIGGER CANNOT DEADLOCK. docs/CONTROL-ROBUSTNESS.md section 6
     records a safeguard that trapped itself because its threshold came from
     the controller's model while its action moved the plant's operating
     point. The stage-2 release has exactly that shape. These tests are the
     ones that would have caught it.

  4. THE SEAM. The release reaches the 6-DOF as a function of time alone, for
     the same reason the brake command does.
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


def _base():
    env = pr.Environment.from_degrees(45.0, include_coriolis=False)
    return dyn.FlightModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                           environment=env)


def _state(alt=3000.0, speed=450.0, pitch_deg=-10.0, roll_deg=0.0,
           spin=1300.0, p_nose=-40.0, q_rate=0.0, r_rate=0.0, phi_rel=0.0,
           aoa_deg=0.0):
    """
    A state with a chosen total angle of attack.

    `aoa_deg` tilts the velocity away from the body x axis in the body z
    direction, which is what puts a real flow incidence on the panels. With it
    zero and the body rates zero, every panel sees only its own deflection.
    """
    q = frames.quat_from_euler(0.0, math.radians(pitch_deg), math.radians(roll_deg))
    a = math.radians(aoa_deg)
    v_body = np.array([speed * math.cos(a), 0.0, speed * math.sin(a)])
    v = frames.dcm_from_quat(q) @ v_body
    return dyn.pack(np.array([0.0, 0.0, -alt]), v, q,
                    np.array([spin, q_rate, r_rate]),
                    nose=(phi_rel, p_nose - spin))


def _model(**nose_kw):
    nose_kw.setdefault("deploy_time", 0.0)
    return cn.guided_model(_base(), GEOM, cn.NoseAssembly(**nose_kw))


def _loads(model, y, t=1.0):
    st = dyn.aero_state(t, y, model)
    return model.control(t, y, st)


#: The adopted deployment point.
DEPLOY = rc.FlightCondition(time=5.55, qbar=136.9e3, airspeed=468.0, mach=1.52,
                            body_spin=1308.0)


def _flat_schedule(c: rc.FlightCondition, t0: float, t1: float, n: int = 400):
    t = np.linspace(t0, t1, n)
    one = np.ones_like(t)
    return rc.ConditionSchedule(time=t, qbar=c.qbar * one, airspeed=c.airspeed * one,
                                mach=c.mach * one, spin=c.body_spin * one,
                                body_damping=np.zeros_like(t))


# ===========================================================================
# 1. Stage 1 is two panels, not a scaled four
# ===========================================================================
def test_stage_one_deploys_the_cant_pair_and_stows_the_steering_pair():
    m = _model(steering_deploy_time=10.0)
    assert m.control.deployed_panels(5.0) == cn.CANT_PANELS
    assert m.control.deployed_panels(10.0) == cn.ALL_PANELS
    assert m.control.deployed_panels(-1.0) == ()


def test_stage_one_leaves_the_despin_torque_alone_and_halves_the_damping():
    """
    The asymmetry the whole staging question turns on.

    `T_cant` sums over the CANT pair, so releasing two panels instead of four
    leaves it untouched. `c_aero` sums over ALL FOUR, so it is halved. Both
    are read off the flown model by differencing the nose roll torque at two
    nose rates, not from the reduced plant's formulae, so this is a statement
    about the physics and not about the algebra.
    """
    staged = _model(steering_deploy_time=10.0)
    single = _model()

    def cant_and_damping(model, t):
        # T_nose = -(T_cant + c_aero p_nose): two rates determine both.
        torques = []
        for p_nose in (0.0, -100.0):
            y = _state(p_nose=p_nose)
            torques.append(_loads(model, y, t)[2])
        c_aero = -(torques[1] - torques[0]) / (-100.0 - 0.0)
        return -torques[0], c_aero

    cant_1, damp_1 = cant_and_damping(staged, 5.0)     # stage 1
    cant_2, damp_2 = cant_and_damping(staged, 11.0)    # stage 2
    cant_s, damp_s = cant_and_damping(single, 5.0)

    assert cant_1 == pytest.approx(cant_s, rel=1e-12), (
        "releasing the cant pair alone changed the despin torque")
    assert cant_2 == pytest.approx(cant_s, rel=1e-12)
    assert damp_1 == pytest.approx(0.5 * damp_s, rel=1e-12), (
        "the roll damping of two panels is not half that of four")
    assert damp_2 == pytest.approx(damp_s, rel=1e-12)


def test_the_reduced_plant_reproduces_the_two_panel_roll_torque():
    """
    The reduced model's `deployed_panels` has to BE the flown stage-1 plant,
    not an approximation of it, or every stage-1 number computed from it is
    fiction.
    """
    model = _model(steering_deploy_time=10.0)
    plant = replace(rc.plant_from(GEOM, cn.NoseAssembly(), pr.M107),
                    deployed_panels=2)
    worst = 0.0
    for speed in (300.0, 468.0):
        for p_nose in (-160.0, -40.0, 0.0, 70.0):
            y = _state(speed=speed, p_nose=p_nose)
            st = dyn.aero_state(5.0, y, model)
            c = rc.FlightCondition(5.0, st.dynamic_pressure, st.airspeed,
                                   st.mach, float(y[10]))
            worst = max(worst, abs(plant.aero_torque(c, p_nose)
                                   - _loads(model, y, 5.0)[2]))
    assert worst < 1e-12, f"reduced two-panel torque differs by {worst} N m"


def test_the_two_panel_roll_torque_is_still_independent_of_attitude_and_rates():
    """
    The four-panel cancellation `sum cos psi = sum sin psi = 0` is what
    licenses the reduced model. A DIAMETRICALLY OPPOSED PAIR cancels the same
    sums, so the property survives stage 1 -- which is not obvious, and if it
    did not hold the stage-1 reduced model would have to be thrown away.
    """
    model = _model(steering_deploy_time=10.0)
    rng = np.random.default_rng(23)
    torques = []
    for _ in range(12):
        y = _state(p_nose=-40.0)
        q = rng.normal(size=4)
        y[6:10] = q / np.linalg.norm(q)
        y[11:13] = rng.normal(scale=4.0, size=2)
        y[13] = rng.uniform(-20.0, 20.0)
        st = dyn.aero_state(5.0, y, model)
        torques.append(model.control(5.0, y, st)[2])
    assert np.ptp(torques) < 1e-12


def test_stage_one_produces_no_commandable_steering_force():
    """
    The point of the staging. At zero angle of attack and zero transverse
    rates the four-panel set produces the steering force `2 qS C_La ds cos
    phi_rel`, which is what sweeps through the flow while the nose despins.
    With the steering pair stowed the cant pair's own contributions cancel
    identically and the transverse force is exactly zero, at every nose angle.
    """
    staged = _model(steering_deploy_time=10.0)
    single = _model()
    for phi_rel in (0.0, 0.7, 1.9, -2.4, math.pi):
        y = _state(p_nose=-40.0, phi_rel=phi_rel, q_rate=0.0, r_rate=0.0,
                   aoa_deg=0.0)
        f1 = _loads(staged, y, 5.0)[0]
        f4 = _loads(single, y, 5.0)[0]
        assert abs(f1[1]) < 1e-9 and abs(f1[2]) < 1e-9, (
            f"the cant pair alone produced a transverse force at phi_rel={phi_rel}")
        assert math.hypot(f4[1], f4[2]) > 10.0, (
            "the four-panel set produced no steering force to compare against")


def test_stage_one_still_carries_an_angle_of_attack_force_and_it_is_modulated():
    """
    Stage 1 is not aerodynamically inert. A diametrically opposed pair
    responds to cross-flow, and unlike the four-panel set its response depends
    on the nose roll orientation: the `sum cos^2` and `sum sin cos` terms no
    longer add to constants. The residue is a transverse force proportional to
    angle of attack and modulated at twice the nose angle.

    This is recorded rather than fixed. It is real, it is half the magnitude
    of the four-panel angle-of-attack force at worst, and it carries no
    commanded deflection, so it cannot be steered with.
    """
    staged = _model(steering_deploy_time=10.0)
    mags = []
    for phi_rel in np.linspace(0.0, math.pi, 9):
        y = _state(p_nose=-40.0, phi_rel=float(phi_rel), aoa_deg=3.0)
        f = _loads(staged, y, 5.0)[0]
        mags.append(math.hypot(f[1], f[2]))
    mags = np.array(mags)
    assert mags.max() > 1.0, "no angle-of-attack force at all from the cant pair"
    assert np.ptp(mags) > 0.2 * mags.max(), (
        "the cant pair's angle-of-attack force did not vary with nose angle, "
        "so the two-panel asymmetry has been lost")


def test_a_stowed_panel_carries_no_load_at_all():
    """
    Force, moment and roll torque. A stowed panel that still contributed axial
    drag would be a different and quieter error than one that still steered.
    """
    y = _state(p_nose=-40.0, aoa_deg=2.0, q_rate=1.0, r_rate=-0.5)
    stowed = _loads(_model(steering_deploy_time=10.0), y, 5.0)
    none_out = _loads(_model(deploy_time=20.0), y, 5.0)
    both = _loads(_model(), y, 5.0)
    # Two panels out is strictly between none and four on the axial force.
    assert abs(none_out[0][0]) == 0.0
    assert abs(stowed[0][0]) < abs(both[0][0])
    assert abs(stowed[0][0]) > 0.0


# ===========================================================================
# 2. The single-stage path is untouched
# ===========================================================================
def test_single_stage_is_the_default_and_is_bit_identical():
    """
    `steering_deploy_time=None` and no gate must reproduce the step-2.5 model
    exactly, not to a tolerance. The comparison this step exists to make is
    against numbers measured on that path.
    """
    old = cn.CanardModel(geometry=GEOM, nose=cn.NoseAssembly(deploy_time=0.0),
                         projectile=pr.M107)
    new = cn.CanardModel(geometry=GEOM,
                         nose=cn.NoseAssembly(deploy_time=0.0,
                                              steering_deploy_time=None,
                                              steering_gate=None),
                         projectile=pr.M107)
    model = _model()
    rng = np.random.default_rng(5)
    for _ in range(8):
        y = _state(p_nose=float(rng.uniform(-160.0, 80.0)),
                   phi_rel=float(rng.uniform(-3.0, 3.0)),
                   aoa_deg=float(rng.uniform(-4.0, 4.0)),
                   q_rate=float(rng.normal(scale=2.0)),
                   r_rate=float(rng.normal(scale=2.0)))
        st = dyn.aero_state(3.0, y, model)
        a, b = old(3.0, y, st), new(3.0, y, st)
        assert (a[0] == b[0]).all() and (a[1] == b[1]).all() and a[2] == b[2]


def test_a_staged_deployment_that_is_not_enabled_is_open_from_the_start():
    dep = rc.StagedDeployment(rc.StagedDeploymentConfig(enabled=False), 5.55)
    assert dep.gate(5.55) is True
    assert dep.release_reason == "single_stage"
    assert dep.delay == 0.0


# ===========================================================================
# 3. The stage-2 trigger cannot deadlock
# ===========================================================================
def test_the_stage_two_trigger_cannot_deadlock():
    """
    THE TEST THIS STEP EXISTS TO WRITE.

    docs/CONTROL-ROBUSTNESS.md section 6: a safeguard whose trigger is
    computed from the onboard model and whose action changes the plant's
    operating point can trap itself. The stage-2 release has exactly that
    shape -- it waits for the servo to capture, and releasing two more panels
    changes the plant the servo is capturing.

    So it is driven here with a stream that defeats every condition it has:
    the loop never engages, the angle error is pinned at 180 degrees, and the
    nose rate is thrown around so that no dwell can ever complete. The
    backstop consults none of that and fires anyway.
    """
    cfg = rc.StagedDeploymentConfig(enabled=True, max_delay=2.0)
    dep = rc.StagedDeployment(cfg, 0.0)
    rng = np.random.default_rng(11)
    t = 0.0
    dt = 1.0 / 500.0
    while t < 3.0 and not dep.released:
        dep.update(t, math.pi, float(rng.uniform(-500.0, 500.0)), engaged=False)
        t += dt
    assert dep.released, "the stage-2 gate never opened"
    assert dep.release_reason == "backstop"
    assert dep.delay <= 2.0 + dt


def test_the_backstop_is_not_what_normally_fires():
    """
    A backstop that fires every time is not a backstop, it is the schedule.
    At a nominal plant the release must come from a condition, well inside the
    backstop, or the layering is decorative.
    """
    plant = rc.plant_from(GEOM, cn.NoseAssembly(brake_max=0.85), pr.M107)
    ctl = rc.RollAngleController(plant)
    cfg = rc.StagedDeploymentConfig(enabled=True, max_delay=4.0)
    dep = rc.StagedDeployment(cfg, 0.0, controller=ctl, plant=plant)
    sched = _flat_schedule(DEPLOY, 0.0, 8.0)
    rc.simulate_reduced(ctl.plant, ctl, sched, rc.constant_command(math.pi),
                        0.0, 8.0, phi0=0.0, p_nose0=DEPLOY.body_spin,
                        truth=plant, deployment=dep)
    assert dep.released
    assert dep.release_reason in ("capture", "quiescent")
    assert dep.delay < 3.0, f"released at {dep.delay} s, too close to the backstop"


def test_the_steering_pair_is_released_even_when_the_servo_never_engages():
    """
    The end-to-end version of the deadlock test, on the plant that trapped the
    engagement gate: the controller is told the nominal canard aerodynamics
    and flown against a truth 15 % weaker, with the gate's own model-free
    criterion disabled so that it locks out exactly as
    `test_the_engagement_gate_cannot_lock_itself_out` shows.

    That round has no servo. It must still end up with a steering surface: a
    round that cannot be steered is worse than one that is steered badly, and
    the backstop is the difference.
    """
    nominal = rc.plant_from(GEOM, cn.NoseAssembly(brake_max=0.85), pr.M107)
    weak = rc.plant_from(GEOM, cn.NoseAssembly(brake_max=0.85), pr.M107,
                         aero_scale=0.85)
    sched = _flat_schedule(DEPLOY, 0.0, 8.0)
    ctl = rc.RollAngleController(nominal,
                                 rc.ControllerConfig(engage_settled_accel=0.0))
    cfg = rc.StagedDeploymentConfig(enabled=True, max_delay=4.0)
    dep = rc.StagedDeployment(cfg, 0.0, controller=ctl, plant=nominal)
    rc.simulate_reduced(ctl.plant, ctl, sched, rc.constant_command(math.pi),
                        0.0, 8.0, phi0=0.0, p_nose0=DEPLOY.body_spin,
                        truth=weak, deployment=dep)
    assert ctl.state.engage_time is None, (
        "the engagement gate did not deadlock, so this is not the case it "
        "was written for")
    assert dep.released and dep.release_reason == "backstop"
    assert dep.delay == pytest.approx(4.0, abs=0.01)


def test_the_quiescent_criterion_does_not_fire_during_the_despin():
    """
    The secondary criterion must not simply defeat the gate. At deployment the
    nose is at 1308 rad/s and nowhere near quiescent, and it must stay shut
    until the nose is actually nearly stationary.
    """
    cfg = rc.StagedDeploymentConfig(enabled=True, max_delay=10.0)
    dep = rc.StagedDeployment(cfg, 0.0)
    # Nothing but the nose rate is keeping the gate shut here: the loop is
    # reported unengaged and the error is at the antipode, so neither capture
    # nor the backstop can fire inside the window.
    for k in range(500):
        dep.update(k / 500.0, math.pi, 1308.0 - 0.5 * k, engaged=False)
        assert not dep.released, "released while the nose was still despinning"


def test_the_capture_criterion_needs_both_engagement_and_a_small_error():
    """
    A loop that is engaged but 90 degrees away has not captured, and neither
    has one sitting at zero error with the gate still shut.
    """
    cfg = rc.StagedDeploymentConfig(enabled=True, max_delay=10.0)
    # The nose is held well outside the quiescent band throughout, so capture
    # is the only criterion in play.
    for engaged, err in ((True, math.radians(90.0)), (False, 0.0)):
        dep = rc.StagedDeployment(cfg, 0.0)
        for k in range(500):
            dep.update(k / 500.0, err, 400.0, engaged=engaged)
        assert not dep.released, f"released on engaged={engaged}, err={err}"
    dep = rc.StagedDeployment(cfg, 0.0)
    for k in range(500):
        dep.update(k / 500.0, math.radians(1.0), 400.0, engaged=True)
    assert dep.released and dep.release_reason == "capture"


def test_a_fixed_delay_releases_exactly_when_asked_and_ignores_the_plant():
    """
    The delay sweep's mode. It is model-free by construction -- it is a clock
    -- so it needs no backstop, and it must not be pre-empted by a condition.
    """
    cfg = rc.StagedDeploymentConfig(enabled=True, fixed_delay=1.5, max_delay=0.5)
    dep = rc.StagedDeployment(cfg, 10.0)
    t = 10.0
    while t < 12.0 and not dep.released:
        dep.update(t, 0.0, 0.0, engaged=True)   # every condition satisfied
        t += 1.0 / 500.0
    assert dep.release_reason == "scheduled"
    assert dep.delay == pytest.approx(1.5, abs=0.005)


def test_the_release_is_latched_and_never_reverts():
    cfg = rc.StagedDeploymentConfig(enabled=True, max_delay=0.1)
    dep = rc.StagedDeployment(cfg, 0.0)
    for k in range(200):
        dep.update(k / 500.0, math.pi, 900.0, engaged=False)
    assert dep.released
    first = dep.release_time
    for k in range(200, 600):
        dep.update(k / 500.0, math.pi, 900.0, engaged=False)
        assert dep.gate(k / 500.0) is True
    assert dep.release_time == first


# ===========================================================================
# 4. The controller's belief follows the stage
# ===========================================================================
def test_the_controller_is_told_how_many_panels_are_out():
    """
    A controller scheduled against the four-panel plant during stage 1 would
    over-cancel the damping and, worse, compute an engagement threshold the
    true plant sits ABOVE -- which is the section 6 deadlock arriving by a
    second route. The release law owns that belief.
    """
    plant = rc.plant_from(GEOM, cn.NoseAssembly(brake_max=0.85), pr.M107)
    ctl = rc.RollAngleController(plant)
    assert ctl.plant.deployed_panels == 4
    dep = rc.StagedDeployment(rc.StagedDeploymentConfig(enabled=True), 0.0,
                              controller=ctl, plant=plant)
    assert ctl.plant.deployed_panels == 2
    dep._fire(1.0, "test")
    assert ctl.plant.deployed_panels == 4
    assert ctl.plant is plant


def test_the_two_panel_model_doubles_the_reachable_rate_interval():
    """
    Task B, in one assertion. Halving the damping leaves `u_hold` alone --
    it contains no damping term -- and roughly doubles both ends of the
    achievable rate interval and the time constant with it. The brake sizing
    rule of docs/CONTROL-ROBUSTNESS.md section 4.3 is therefore unchanged by
    the staging, and that is the reason staging is possible at all.
    """
    four = rc.plant_from(GEOM, cn.NoseAssembly(brake_max=0.85), pr.M107)
    two = replace(four, deployed_panels=2)
    assert two.hold_command(DEPLOY) == pytest.approx(four.hold_command(DEPLOY),
                                                     rel=1e-12)
    assert two.cant_torque(DEPLOY) == pytest.approx(four.cant_torque(DEPLOY),
                                                    rel=1e-12)
    assert two.envelope(DEPLOY).can_hold and four.envelope(DEPLOY).can_hold
    for u in (0.0, 0.85):
        r4 = four.equilibrium_rate(DEPLOY, u)
        r2 = two.equilibrium_rate(DEPLOY, u)
        assert 1.9 < r2 / r4 < 2.01, f"rate ratio {r2 / r4} at u={u}"
    assert 1.9 < two.time_constant(DEPLOY) / four.time_constant(DEPLOY) < 2.01


# ===========================================================================
# 5. The seam
# ===========================================================================
def test_the_steering_gate_is_constant_between_samples():
    """
    Same restriction as `brake_command`, for the same reason: RK4 evaluates
    the derivative four times per step at three different times, and a release
    read from the state inside it would put two panels into the flow for part
    of a step.
    """
    plant = rc.plant_from(GEOM, cn.NoseAssembly(brake_max=0.85), pr.M107)
    ctl = rc.RollAngleController(plant)
    dep = rc.StagedDeployment(rc.StagedDeploymentConfig(enabled=True), 0.0,
                              controller=ctl, plant=plant)
    assert dep.gate(0.0) is False
    assert dep.gate(1e9) is False, "the gate read the time it was given"
    dep._fire(0.5, "test")
    assert dep.gate(0.0) is True, "the gate read the time it was given"


def test_the_staged_run_reaches_the_sixdof_and_changes_the_trajectory():
    """
    End to end and short: a staged deployment flown in the 6-DOF must sample,
    release, and land somewhere different from the single-stage kit at the
    same phase. Three and a half seconds of flight, which covers the
    release.
    """
    t_dep = 5.55
    y = _state(alt=4200.0, speed=468.0, spin=1308.0, p_nose=1308.0)
    out = {}
    for staged in (False, True):
        plant = rc.plant_from(GEOM, cn.NoseAssembly(brake_max=0.85), pr.M107)
        ctl = rc.RollAngleController(plant)
        cfg = rc.StagedDeploymentConfig(enabled=staged, max_delay=4.0)
        dep = rc.StagedDeployment(cfg, t_dep, controller=ctl, plant=plant)
        law = rc.BrakeLaw(ctl, rc.constant_command(math.pi), deploy_time=t_dep,
                          deployment=dep if staged else None)
        nose = replace(plant.nose, deploy_time=t_dep,
                       brake_command=law.brake_command,
                       steering_gate=dep.gate if staged else None)
        model = cn.guided_model(_base(), GEOM, nose)
        res = ig.integrate(y, model, dt=5e-4, log_every=20, t_max=t_dep + 3.5,
                           t_start=t_dep, step_hook=law.sample,
                           stop_on_impact=False)
        assert law.samples > 0
        out[staged] = res.trajectory
        if staged:
            assert dep.samples == law.samples
            assert dep.released, "the steering pair never came out"

    a, b = out[False], out[True]
    assert abs(a.total_aoa.max() - b.total_aoa.max()) > 1e-4, (
        "staging changed nothing in the 6-DOF, so the gate never reached it")
