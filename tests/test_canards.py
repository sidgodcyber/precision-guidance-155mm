"""
Tests for the despun nose degree of freedom and the canard model.

Three groups, in order of how much they would cost if they were wrong:

  1. REGRESSION. Adding states to a validated model is where validated models
     quietly stop being validated. Two tests fly the step-1 ballistic case and
     demand the step-1 answer back.

  2. CONVENTIONS. Which way the canard force points, which rate the
     aerodynamics sees, which torque acts on which body. Every one of these
     produces a plausible-looking trajectory when it is wrong.

  3. BOOKKEEPING. That the register of estimated quantities is maintained,
     and that the geometry fits the shell.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim import aerodata, atmosphere as atm
from sim import canards as cn
from sim import dynamics as dyn, frames, integrate as ig, projectile as pr


def _base(**kw):
    env = kw.pop("environment", pr.Environment.from_degrees(45.0, include_coriolis=False))
    return dyn.FlightModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), environment=env, **kw
    )


def _state(
    alt=3000.0, speed=250.0, pitch_deg=-20.0, roll_deg=37.0, spin=1100.0,
    p_rel=-1100.0, q_rate=0.0, r_rate=0.0, phi_rel=0.0,
):
    """A representative post-apogee state with the nose despun."""
    q = frames.quat_from_euler(0.0, math.radians(pitch_deg), math.radians(roll_deg))
    v = frames.dcm_from_quat(q) @ np.array([speed, 0.0, 0.0])
    return dyn.pack(
        np.array([0.0, 0.0, -alt]), v, q,
        np.array([spin, q_rate, r_rate]), nose=(phi_rel, p_rel),
    )


# ===========================================================================
# 1. Regression -- the step-1 model must survive the new states
# ===========================================================================
#: Charge 4, QE 97.2 mils, dt = 2e-4, latitude 45 N, no Coriolis, ISA, no
#: wind. Captured from the step-1 code BEFORE the nose states existed, to six
#: decimals, and re-captured to full precision afterwards; the two agree at
#: every digit the first capture recorded, which is what makes the extension
#: a no-op rather than merely a small perturbation. If these move, the
#: ballistic model has changed and step 1's validation no longer describes it.
#:
#: pre-extension capture:  2011.662118  3.994791  6.375436  302.089057  657.008235
STEP1_C4_QE97 = {
    "range": 2011.6621181150413,
    "drift": 3.994790934893053,
    "tof": 6.3754361153087595,
    "impact_velocity": 302.0890574989908,
    "impact_spin": 657.0082352839914,
}


def _fly_c4(model, dt=2.0e-4):
    launch = pr.LaunchConditions.from_mils(337.0, 97.2)
    y0 = dyn.initial_state(pr.M107, launch)
    return ig.integrate(y0, model, dt=dt, log_every=2000, t_max=60.0)


def test_appending_the_nose_states_does_not_disturb_the_ballistic_model():
    """
    Task D.2, first half. With no nose assembly the two new states are inert
    and the trajectory must be the step-1 trajectory to integration tolerance
    -- which here means to the last digit, because the arithmetic on the
    first thirteen states is untouched.
    """
    res = _fly_c4(_base())
    assert res.range_m == pytest.approx(STEP1_C4_QE97["range"], abs=1e-9)
    assert res.drift_m == pytest.approx(STEP1_C4_QE97["drift"], abs=1e-9)
    assert res.impact_time == pytest.approx(STEP1_C4_QE97["tof"], abs=1e-12)
    assert res.impact_velocity == pytest.approx(STEP1_C4_QE97["impact_velocity"], abs=1e-9)
    assert float(res.impact_state[10]) == pytest.approx(STEP1_C4_QE97["impact_spin"], abs=1e-9)
    # And the nose states never moved off zero.
    assert np.all(res.trajectory.nose_angle == 0.0)
    assert np.all(res.trajectory.nose_rate == 0.0)


def test_the_nose_degree_of_freedom_alone_barely_moves_the_impact_point():
    """
    Task D.2, second half. Fit the nose assembly -- so the axial inertia is
    split, both new states integrate, the body roll equation goes through the
    new branch and the gyroscopic term picks up the nose's angular momentum --
    but attach no canards.

    The answer must be the step-1 answer to well inside a millimetre. It is
    NOT exactly the step-1 answer, and the reason is physical rather than
    numerical: with no despin torque the nose over-runs the decelerating body,
    reaching p_rel = +17 rad/s with the nominal bearing and +26 rad/s with a
    frictionless one, and the total axial angular momentum that sets the
    gyroscopic stiffness shifts by that much times I_nose. Demanding an exactly
    unchanged trajectory would be demanding the wrong thing.
    """
    for viscous, coulomb in ((0.0, 0.0), (cn.NOMINAL_NOSE.viscous, cn.NOMINAL_NOSE.coulomb)):
        nose = cn.NoseAssembly(viscous=viscous, coulomb=coulomb, deploy_time=0.0)
        bare = dyn.FlightModel(
            projectile=pr.M107, aero=aerodata.make_m107_table(),
            environment=pr.Environment.from_degrees(45.0, include_coriolis=False),
            nose=nose, control=None,
        )
        res = _fly_c4(bare)
        assert res.range_m == pytest.approx(STEP1_C4_QE97["range"], abs=1e-3)
        assert res.drift_m == pytest.approx(STEP1_C4_QE97["drift"], abs=1e-4)
        assert res.impact_time == pytest.approx(STEP1_C4_QE97["tof"], abs=1e-5)
        # The nose leads the decelerating body, and does so more when the
        # bearing cannot drag it along.
        assert float(res.impact_state[14]) > 0.0
    # Explicitly: a real bearing couples them more tightly than none at all.
    def _lead(v, c):
        nose = cn.NoseAssembly(viscous=v, coulomb=c, deploy_time=0.0)
        m = dyn.FlightModel(
            projectile=pr.M107, aero=aerodata.make_m107_table(),
            environment=pr.Environment.from_degrees(45.0, include_coriolis=False),
            nose=nose,
        )
        return float(_fly_c4(m).impact_state[14])

    assert _lead(cn.NOMINAL_NOSE.viscous, cn.NOMINAL_NOSE.coulomb) < _lead(0.0, 0.0)


def test_an_undeflected_kit_costs_range_but_cannot_steer():
    """
    Deploy the canards with no steering deflection and no cant. The kit is
    then rotationally symmetric, so its effect must be COMPLETELY independent
    of where the nose is pointed -- and the whole effect must be drag.

    This is the sharper form of "zero deflection changes nothing": profile
    drag genuinely does shorten the range, so demanding an unchanged impact
    point would be wrong. What must hold is that no part of the change is
    steerable.
    """
    geom = cn.CanardGeometry(steering_deflection=0.0, cant_angle=0.0)
    seen = []
    for phi_deg in (0.0, 90.0, 180.0, 270.0):
        nose = cn.NoseAssembly(viscous=0.0, coulomb=0.0, deploy_time=0.0,
                               hold_angle=math.radians(phi_deg))
        res = _fly_c4(cn.guided_model(_base(), geom, nose))
        seen.append((res.range_m, res.drift_m))
    for r, d in seen[1:]:
        assert r == pytest.approx(seen[0][0], abs=1e-6)
        assert d == pytest.approx(seen[0][1], abs=1e-6)
    # And the effect is a range decrement, i.e. drag.
    assert seen[0][0] < STEP1_C4_QE97["range"]


def test_state_size_grew_by_exactly_two_and_pack_round_trips_the_nose():
    assert dyn.STATE_SIZE == dyn.BODY_STATE_SIZE + 2
    y = dyn.pack(
        np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]),
        np.array([0.5, 0.5, 0.5, 0.5]), np.array([7.0, 8.0, 9.0]),
        nose=(0.25, -1234.0),
    )
    assert y.shape == (dyn.STATE_SIZE,)
    phi, prate = dyn.unpack_nose(y)
    assert phi == 0.25 and prate == -1234.0
    # Omitting the nose gives zeros, not uninitialised memory.
    y2 = dyn.pack(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3))
    assert dyn.unpack_nose(y2) == (0.0, 0.0)


def test_restarting_from_a_logged_state_reproduces_the_trajectory():
    """
    analysis/authority.py restarts the corrected runs from a logged sample of
    the uncorrected one instead of re-flying the shared leg. That is only
    legitimate if a logged sample is an exact integrator state.
    """
    model = _base()
    launch = pr.LaunchConditions.from_mils(337.0, 97.2)
    y0 = dyn.initial_state(pr.M107, launch)
    full = ig.integrate(y0, model, dt=5e-4, log_every=100, t_max=60.0)
    k = full.trajectory.t.size // 2
    t_k = float(full.trajectory.t[k])
    y_k = np.zeros(dyn.STATE_SIZE)
    y_k[0:3] = full.trajectory.position[k]
    y_k[3:6] = full.trajectory.velocity[k]
    y_k[6:10] = full.trajectory.quaternion[k]
    y_k[10:13] = full.trajectory.omega[k]
    rest = ig.integrate(y_k, model, dt=5e-4, log_every=100, t_max=60.0, t_start=t_k)
    assert rest.range_m == pytest.approx(full.range_m, abs=1e-9)
    assert rest.drift_m == pytest.approx(full.drift_m, abs=1e-9)
    assert rest.impact_time == pytest.approx(full.impact_time, abs=1e-12)


# ===========================================================================
# 2. Conventions
# ===========================================================================
def _loads(geom, hold=None, y=None, model_kw=None):
    nose = cn.NoseAssembly(hold_angle=hold, deploy_time=0.0)
    m = cn.guided_model(_base(**(model_kw or {})), geom, nose)
    yy = _state() if y is None else y
    st = dyn.aero_state(0.0, yy, m)
    return m.control(0.0, yy, st), st, m, yy


def test_steering_pair_makes_a_force_and_no_roll_moment():
    (F, M, T), st, m, y = _loads(cn.CanardGeometry(cant_angle=0.0), hold=0.0)
    assert math.hypot(float(F[1]), float(F[2])) > 1.0
    assert abs(T) < 1e-12
    assert float(M[0]) == 0.0


def test_cant_pair_makes_a_roll_moment_and_no_transverse_force():
    (F, M, T), st, m, y = _loads(cn.CanardGeometry(steering_deflection=0.0), hold=0.0)
    assert abs(float(F[1])) < 1e-9 and abs(float(F[2])) < 1e-9
    # Positive cant despins against right-hand rifling, so the torque is
    # negative.
    assert T < -1e-3


def test_cant_torque_exceeds_bearing_drag_at_the_operating_condition():
    """
    The design intent of the cant angle: overcome the bearing with margin.
    If it did not, the nose would be dragged round with the body and there
    would be no despun reference at all.
    """
    (F, M, T), st, m, y = _loads(cn.CanardGeometry(steering_deflection=0.0), hold=0.0)
    drag = cn.NOMINAL_NOSE.friction_torque(-1100.0)
    assert drag > 0.0                    # bearing drags the nose forward
    assert abs(T) > 1.5 * abs(drag)


@pytest.mark.parametrize(
    "phi_deg, want",
    [(0.0, np.array([0.0, 0.0, -1.0])),     # up
     (90.0, np.array([0.0, 1.0, 0.0])),     # right
     (180.0, np.array([0.0, 0.0, 1.0])),    # down
     (270.0, np.array([0.0, -1.0, 0.0]))],  # left
)
def test_phi_nose_walks_the_canard_force_round_the_compass(phi_deg, want):
    """
    phi_nose = 0 up, 90 right, 180 down, 270 left, in EARTH terms, whatever
    the body's own roll angle happens to be. This is the definition the
    steering command depends on and it is the one thing in this module a
    reader is most likely to assume rather than check.
    """
    geom = cn.CanardGeometry(cant_angle=0.0)
    y = _state(roll_deg=37.0, pitch_deg=0.0)   # level, so up is exactly -z
    (F, M, T), st, m, yy = _loads(geom, hold=math.radians(phi_deg), y=y)
    R = frames.dcm_from_quat(y[6:10])
    F_earth = R @ np.array([0.0, float(F[1]), float(F[2])])
    unit = F_earth / np.linalg.norm(F_earth)
    assert np.allclose(unit, want, atol=1e-9)


def test_the_force_direction_does_not_depend_on_the_bodys_own_roll():
    """The whole point of despinning: the force is inertially steady."""
    geom = cn.CanardGeometry(cant_angle=0.0)
    seen = []
    for roll in (0.0, 71.0, 143.0, 300.0):
        y = _state(roll_deg=roll, pitch_deg=-15.0)
        (F, M, T), st, m, yy = _loads(geom, hold=math.radians(90.0), y=y)
        R = frames.dcm_from_quat(y[6:10])
        Fe = R @ np.array([0.0, float(F[1]), float(F[2])])
        seen.append(Fe / np.linalg.norm(Fe))
    for s in seen[1:]:
        assert np.allclose(s, seen[0], atol=1e-9)


def test_reversing_the_deflection_reverses_the_force():
    """Task D.3, at the force level."""
    geom_p = cn.CanardGeometry(cant_angle=0.0, steering_deflection=math.radians(5.0))
    geom_n = cn.CanardGeometry(cant_angle=0.0, steering_deflection=math.radians(-5.0))
    (Fp, _, _), _, _, _ = _loads(geom_p, hold=0.3)
    (Fn, _, _), _, _, _ = _loads(geom_n, hold=0.3)
    assert np.allclose(Fp[1:], -Fn[1:], atol=1e-12)


def test_a_180_degree_roll_command_reverses_the_force():
    geom = cn.CanardGeometry(cant_angle=0.0)
    (Fa, _, _), _, _, _ = _loads(geom, hold=0.3)
    (Fb, _, _), _, _, _ = _loads(geom, hold=0.3 + math.pi)
    assert np.allclose(Fa[1:], -Fb[1:], atol=1e-12)


def test_four_panel_array_response_to_body_aoa_is_isotropic_in_roll():
    """
    With no deflection and no cant, the four-panel "+" array must respond to
    a body angle of attack exactly like a body-fixed surface: the force must
    not depend on where the nose happens to be pointed. If it did, a despun
    kit would inject a roll-orientation-dependent disturbance into the
    airframe that step 4 would then have to fight.
    """
    geom = cn.CanardGeometry(steering_deflection=0.0, cant_angle=0.0)
    y = _state(pitch_deg=-10.0)
    # Give it a real angle of attack by tilting the velocity off the axis.
    y[3:6] = frames.dcm_from_quat(y[6:10]) @ np.array([250.0, 6.0, 9.0])
    ref = None
    for phi in (0.0, 0.7, 1.9, 4.4):
        (F, M, T), st, m, yy = _loads(geom, hold=phi, y=y)
        if ref is None:
            ref = np.array(F, dtype=float)
        else:
            assert np.allclose(F, ref, atol=1e-9)
        assert abs(T) < 1e-12          # and it makes no roll moment either


def test_canard_roll_damping_always_opposes_the_nose_inertial_rate():
    """
    Damping acts on p_nose, the INERTIAL rate, not on p_rel. Getting this
    wrong makes the nose despin to the body's rate instead of to the ground.
    """
    geom = cn.CanardGeometry(steering_deflection=0.0, cant_angle=0.0)
    for p_nose in (-400.0, -50.0, 50.0, 400.0):
        y = _state(spin=1100.0, p_rel=p_nose - 1100.0)
        (F, M, T), st, m, yy = _loads(geom, hold=None, y=y)
        assert T * p_nose < 0.0, f"roll torque did not oppose p_nose={p_nose}"


def test_bearing_friction_always_opposes_relative_motion():
    n = cn.NOMINAL_NOSE
    for p_rel in (-1400.0, -1.0, 1.0, 1400.0):
        assert n.friction_torque(p_rel) * p_rel < 0.0
    assert n.friction_torque(0.0) == 0.0


def test_the_viscous_term_dominates_at_the_operating_relative_rate():
    """
    Stated in the module docstring and in docs/CANARD-MODEL.md; asserted here
    so the claim cannot drift away from the constants.
    """
    n = cn.NOMINAL_NOSE
    p_rel = -1100.0
    viscous = abs(n.viscous * p_rel)
    coulomb = abs(n.coulomb * math.tanh(p_rel / n.p_eps))
    assert viscous / coulomb > 20.0
    # ... and the crossover is at a rate the nose never sees in flight.
    crossover = n.coulomb / n.viscous
    assert crossover < 100.0


def test_the_brake_can_only_oppose_never_drive():
    n = cn.NoseAssembly(brake_command=lambda t: 0.3)
    for p_rel in (-1200.0, -5.0, 5.0, 1200.0):
        assert n.brake_torque(0.0, p_rel) * p_rel <= 0.0
    # A negative command is not a reverse drive; it is nothing.
    n2 = cn.NoseAssembly(brake_command=lambda t: -0.3)
    assert n2.brake_torque(0.0, -1100.0) == 0.0
    # And it saturates at capacity.
    n3 = cn.NoseAssembly(brake_command=lambda t: 99.0, brake_max=0.5)
    assert abs(n3.brake_torque(0.0, -1100.0)) == pytest.approx(0.5, rel=1e-9)


def test_internal_torques_are_equal_and_opposite():
    """
    Newton's third law across the bearing. With no aerodynamics at all, the
    total axial angular momentum of body plus nose must be conserved no
    matter what the bearing and brake do.
    """
    nose = cn.NoseAssembly(deploy_time=0.0, brake_command=lambda t: 0.2)
    model = dyn.FlightModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(),
        environment=pr.Environment.from_degrees(45.0, include_coriolis=False),
        aero_enabled=False, nose=nose, control=None,
    )
    y = _state(p_rel=-900.0)
    d = dyn.derivative(0.0, y, model)
    pdot_body = float(d[10])
    pdot_nose = pdot_body + float(d[14])
    Ix_body = pr.M107.I_axial - nose.inertia
    total = Ix_body * pdot_body + nose.inertia * pdot_nose
    assert abs(total) < 1e-12


def test_the_nose_reaction_torque_reaches_the_body():
    """
    Bearing drag does not vanish into the airframe: it must show up as a body
    spin-down torque. Without this the shell would spin down at exactly its
    ballistic rate no matter how hard the brake was applied, which is the
    quiet version of the error.
    """
    y = _state(p_rel=-1100.0)
    free = dyn.FlightModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(),
        environment=pr.Environment.from_degrees(45.0, include_coriolis=False),
        nose=None,
    )
    with_nose = dyn.FlightModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(),
        environment=pr.Environment.from_degrees(45.0, include_coriolis=False),
        nose=cn.NoseAssembly(deploy_time=0.0),
    )
    d0 = dyn.derivative(0.0, y, free)
    d1 = dyn.derivative(0.0, y, with_nose)
    # The nose is being dragged forward by the bearing, so the body is being
    # dragged back: it spins down faster than the ballistic shell.
    assert float(d1[10]) < float(d0[10])


def test_hold_mode_freezes_the_nose_states_and_still_torques_the_body():
    """
    In hold mode the two nose states carry no meaning, but the despin torque
    still has to reach the shell through the slipping brake.
    """
    geom = cn.CanardGeometry(steering_deflection=0.0)
    nose_held = cn.NoseAssembly(deploy_time=0.0, hold_angle=0.0)
    held = cn.guided_model(_base(), geom, nose_held)
    y = _state()
    d = dyn.derivative(0.0, y, held)
    assert float(d[13]) == 0.0 and float(d[14]) == 0.0

    plain = _base()
    d0 = dyn.derivative(0.0, y, plain)
    # Canted canards despin the nose; the reaction spins the body down too.
    assert float(d[10]) < float(d0[10])


def test_nothing_happens_before_deployment():
    geom = cn.NOMINAL_GEOMETRY
    nose = cn.NoseAssembly(deploy_time=10.0, hold_angle=0.0)
    m = cn.guided_model(_base(), geom, nose)
    y = _state()
    st = dyn.aero_state(0.0, y, m)
    F, M, T = m.control(9.999, y, st)
    assert np.allclose(F, 0.0) and np.allclose(M, 0.0) and T == 0.0
    F, M, T = m.control(10.0, y, st)
    assert np.linalg.norm(F) > 0.0


def test_the_two_tuple_control_seam_still_works():
    """Step 1's callback contract is unchanged: a 2-tuple is still valid."""
    def control(t, y, st):
        return np.array([0.0, 100.0, 0.0]), np.array([0.0, 0.0, 25.0])

    y = _state()
    base = dyn.derivative(0.0, y, _base())
    with_ctl = dyn.derivative(0.0, y, _base(control=control))
    assert with_ctl[12] - base[12] == pytest.approx(25.0 / pr.M107.I_transverse)
    # ... and with no nose assembly the roll equation is untouched.
    assert with_ctl[10] == pytest.approx(base[10])


# ===========================================================================
# The two flight-level sign checks -- task D.3
# ===========================================================================
_SHIFT_CACHE = {}


def _short_shift(phi_deg, deflection_deg=8.0):
    """
    Impact shift for charge 5 / QE 420.6 mils (8 km, 28 s), canards deployed
    at 14 s. dt = 1e-3, coarse but identical between the two runs being
    differenced, which is what a shift needs.
    """
    key = (phi_deg, deflection_deg)
    if key in _SHIFT_CACHE:
        return _SHIFT_CACHE[key]
    launch = pr.LaunchConditions.from_mils(397.0, 420.6)
    base = _base()
    y0 = dyn.initial_state(pr.M107, launch)
    if "unc" not in _SHIFT_CACHE:
        _SHIFT_CACHE["unc"] = ig.integrate(y0, base, dt=1e-3, log_every=500, t_max=60.0)
    unc = _SHIFT_CACHE["unc"]
    geom = cn.CanardGeometry(steering_deflection=math.radians(deflection_deg))
    nose = cn.NoseAssembly(deploy_time=14.0, hold_angle=math.radians(phi_deg))
    cor = ig.integrate(y0, cn.guided_model(base, geom, nose),
                       dt=1e-3, log_every=500, t_max=60.0)
    out = (cor.range_m - unc.range_m, cor.drift_m - unc.drift_m)
    _SHIFT_CACHE[key] = out
    return out


def test_reversing_the_deflection_is_the_same_as_rotating_the_nose_180_deg():
    """
    Task D.3, in its exact form. Negating the steering deflection swaps the
    two steering panels' incidences, and the two roll panels carry equal
    incidence so swapping them is a no-op -- so the set of (azimuth,
    incidence) pairs is identical to the one produced by rotating the nose
    180 degrees. The two trajectories must therefore agree to integration
    precision, not merely in sign.

    Asserting the identity rather than a sign catches an error that a sign
    test would miss: a deflection that reverses the force by the wrong amount
    still reverses it.
    """
    a = _short_shift(90.0, deflection_deg=-8.0)
    b = _short_shift(270.0, deflection_deg=+8.0)
    assert a[0] == pytest.approx(b[0], abs=1e-9)
    assert a[1] == pytest.approx(b[1], abs=1e-9)


def test_reversing_the_canard_deflection_reverses_the_steerable_correction():
    """
    Task D.3 at the trajectory level.

    The comparison has to be made against the right reference. The
    roll-orientation-INDEPENDENT part of the shift -- canard profile drag and
    the yaw drag of the induced trim angle of attack -- survives a sign flip
    unchanged, and it is not the same for a deflected kit as for an
    undeflected one. So the zero-deflection run is NOT the symmetry centre;
    the mean over the four cardinal roll angles at the same deflection is.
    """
    quad = {p: _short_shift(p, deflection_deg=8.0) for p in (0.0, 90.0, 180.0, 270.0)}
    meanR = sum(v[0] for v in quad.values()) / 4.0
    meanD = sum(v[1] for v in quad.values()) / 4.0

    for p, q in ((0.0, 180.0), (90.0, 270.0)):
        dR_a, dD_a = quad[p][0] - meanR, quad[p][1] - meanD
        dR_b, dD_b = quad[q][0] - meanR, quad[q][1] - meanD
        assert dD_a * dD_b < 0.0, f"deflection did not reverse between {p} and {q}"
        assert dR_a * dR_b < 0.0, f"range shift did not reverse between {p} and {q}"
        # Antisymmetric to within the second-harmonic content of the plant.
        assert dD_a == pytest.approx(-dD_b, rel=0.10)
        assert dR_a == pytest.approx(-dR_b, rel=0.10)

    # And the steerable part is real, not numerical noise.
    assert math.hypot(quad[90.0][0] - meanR, quad[90.0][1] - meanD) > 1.0


def test_the_free_nose_despins_to_near_zero_inertial_roll_rate():
    """
    Task D.2's other half: with zero brake torque, the canted pair must drive
    the nose's INERTIAL roll rate toward zero, not toward the body's rate.
    """
    launch = pr.LaunchConditions.from_mils(397.0, 420.6)
    y0 = dyn.initial_state(pr.M107, launch)
    nose = cn.NoseAssembly(deploy_time=14.0)
    m = cn.guided_model(_base(), cn.NOMINAL_GEOMETRY, nose)
    res = ig.integrate(y0, m, dt=1e-3, log_every=200, t_max=60.0)
    tr = res.trajectory
    settled = tr.nose_spin[tr.t > 18.0]
    body = tr.spin[tr.t > 18.0]
    assert settled.size > 5
    # Nose inertial rate is small compared with the body's, and small in
    # absolute terms too.
    assert np.max(np.abs(settled)) < 0.10 * np.min(np.abs(body))
    assert np.max(np.abs(settled)) < 60.0
    # It despins the correct way: the relative rate goes strongly negative.
    assert float(res.impact_state[14]) < -0.8 * float(res.impact_state[10])


# ===========================================================================
# 3. Bookkeeping
# ===========================================================================
def test_the_nominal_geometry_stows_within_the_shell_profile():
    g = cn.NOMINAL_GEOMETRY
    assert g.fits_within_envelope(pr.M107)
    d = g.describe(pr.M107)
    assert d["tip_clearance_mm"] > 10.0
    # Folded aft, each panel's circumferential footprint must fit its quarter
    # of the guidance section.
    available = math.pi * 2.0 * g.body_radius_local / g.n_panels
    assert g.span_exposed < available


def test_the_canards_sit_ahead_of_the_centre_of_gravity():
    assert cn.NOMINAL_GEOMETRY.moment_arm(pr.M107) > 0.0
    assert cn.NOMINAL_GEOMETRY.station_from_nose < pr.M107.x_cg


def test_the_canards_sit_ahead_of_the_subsonic_centre_of_pressure():
    """
    The finding the whole authority result turns on. It is asserted, not
    merely written down, because if a future coefficient revision moved the
    centre of pressure aft of the canard station the sign of the correction
    would flip and every number in docs/AUTHORITY-ENVELOPE.md would be wrong.
    """
    tab = aerodata.make_m107_table()
    for mach in (0.7, 0.8, 0.9, 1.0, 1.2):
        c = tab.coefficients_at(mach)
        x_cp = cn.centre_of_pressure(c.C_Nalpha, c.C_Malpha, pr.M107)
        assert x_cp > cn.NOMINAL_GEOMETRY.station_from_nose, (
            f"at M={mach} the CP has moved forward of the canard station"
        )


def test_the_trim_force_factor_is_negative_and_flips_aft_of_the_cp():
    """
    Closed-form cross-check of the mechanism: a canard AHEAD of the centre of
    pressure produces a net force OPPOSITE to its own, and moving it aft of
    the CP flips the sign.
    """
    tab = aerodata.make_m107_table()
    c = tab.coefficients_at(0.8)
    cla = cn.canard_lift_curve_slope(0.8, cn.NOMINAL_GEOMETRY.aspect_ratio_effective)

    fwd = cn.trim_force_factor(cn.NOMINAL_GEOMETRY, pr.M107, c.C_Nalpha, c.C_Malpha, cla)
    assert fwd["trim_force_factor"] < 0.0
    assert abs(fwd["induced_over_direct"]) > 1.0

    from dataclasses import replace
    aft = cn.trim_force_factor(
        replace(cn.NOMINAL_GEOMETRY, station_from_nose=0.30),
        pr.M107, c.C_Nalpha, c.C_Malpha, cla,
    )
    assert aft["trim_force_factor"] > 0.0


def test_authority_saturates_rather_than_growing_with_panel_area():
    """
    Bigger panels move the combined centre of pressure forward toward the
    canard station, which is where the trim force vanishes. The closed form
    says the net force therefore saturates. If it did not, "make the canards
    bigger" would be a way out of the authority problem, and it is not.
    """
    from dataclasses import replace
    tab = aerodata.make_m107_table()
    c = tab.coefficients_at(0.8)
    g = cn.NOMINAL_GEOMETRY

    def net_per_qbar(scale):
        gg = replace(g, span_exposed=g.span_exposed * math.sqrt(scale),
                     chord=g.chord * math.sqrt(scale))
        cla = cn.canard_lift_curve_slope(0.8, gg.aspect_ratio_effective)
        d = cn.trim_force_factor(gg, pr.M107, c.C_Nalpha, c.C_Malpha, cla)
        direct = 2.0 * gg.panel_area * cla * gg.carryover * g.steering_deflection
        return abs(direct * d["trim_force_factor"])

    one = net_per_qbar(1.0)
    four = net_per_qbar(4.0)
    sixteen = net_per_qbar(16.0)
    assert four > one
    assert sixteen / four < 4.0 * (four / one)      # sub-linear
    assert sixteen < 4.0 * one                      # far short of proportional


def test_canard_lift_curve_slope_is_continuous_and_physically_bounded():
    prev = None
    for i in range(0, 261):
        m = i * 0.01
        cl = cn.canard_lift_curve_slope(m, cn.NOMINAL_GEOMETRY.aspect_ratio_effective)
        assert 0.5 < cl < 5.0
        if prev is not None:
            assert abs(cl - prev) < 0.05, f"jump at M={m}"
        prev = cl
    # Slender-wing limit as the aspect ratio goes to zero.
    tiny = cn.canard_lift_curve_slope(0.0, 0.02)
    assert tiny == pytest.approx(0.5 * math.pi * 0.02, rel=0.02)


def test_carryover_is_the_slender_body_result():
    assert cn.body_wing_carryover(0.030, 0.060) == pytest.approx(2.25)
    assert cn.body_wing_carryover(0.0, 0.060) == pytest.approx(1.0)
    g = cn.CanardGeometry(include_carryover=False)
    assert g.carryover == 1.0


#: Every estimated quantity must be findable in the register by a keyword.
#: Adding a constant without adding an entry fails the test below.
_REGISTER_KEYWORDS = {
    "span_exposed": "semi-span",
    "chord": "chord",
    "station_from_nose": "station",
    "body_radius_local": "local body radius",
    "steering_deflection": "steering differential incidence",
    "cant_angle": "cant angle",
    "induced_drag_efficiency": "induced-drag efficiency",
    "inertia": "nose assembly axial inertia",
    "viscous": "viscous coefficient",
    "coulomb": "Coulomb torque",
    "brake_max": "brake torque capacity",
    "p_eps": "regularisation",
}


def test_every_estimated_quantity_is_declared():
    """
    sim/aerodata.py can name a source for every number it holds. This module
    cannot, so instead it must name a METHOD and an ACCURACY for every one.
    """
    text = " ".join(f"{e.name} {e.value} {e.method} {e.accuracy}" for e in cn.ESTIMATES)
    for field, keyword in _REGISTER_KEYWORDS.items():
        assert keyword.lower() in text.lower(), (
            f"{field} is a tunable canard/nose constant with no entry in "
            f"canards.ESTIMATES (looked for '{keyword}')"
        )
    # Every entry must actually say something about accuracy or say plainly
    # that it is a choice rather than a measurement.
    for e in cn.ESTIMATES:
        assert e.method.strip(), f"{e.name} has no method"
        acc = e.accuracy.lower()
        assert ("%" in acc or "choice" in acc or "immaterial" in acc), (
            f"{e.name} declares no accuracy and does not admit to being a choice"
        )
    assert len(cn.estimate_report()) > len(cn.ESTIMATES)


def test_the_geometry_and_nose_expose_no_undeclared_tunables():
    """
    The register is keyed by dataclass field name, so a new field with a
    numeric default must be added to _REGISTER_KEYWORDS -- which forces
    whoever adds it to decide whether it is an estimate and to write one.
    """
    import dataclasses

    structural = {
        "n_panels", "include_carryover",          # arrangement, not a value
        "deploy_time", "hold_angle", "brake_command",   # inputs, not estimates
        # Staged deployment: WHEN the steering pair is released. An input in
        # exactly the sense deploy_time is, set by the release law in
        # gnc.roll_control, not an aerodynamic estimate.
        "steering_deploy_time", "steering_gate",
        # Step 6. NOT a new estimate: a multiplier of 1.0 on the lift-curve
        # slope, which the register already declares with its accuracy. It
        # exists so the declared +-30 % can be FLOWN rather than only stated,
        # and the register entry for C_Lalpha_canard names it.
        "cla_scale",
    }
    for cls in (cn.CanardGeometry, cn.NoseAssembly):
        for f in dataclasses.fields(cls):
            if f.name in structural:
                continue
            assert f.name in _REGISTER_KEYWORDS, (
                f"{cls.__name__}.{f.name} is a new tunable constant: add it to "
                "canards.ESTIMATES and to _REGISTER_KEYWORDS in this test"
            )


# ===========================================================================
# The steering response: gyroscopic term, Magnus term, and what they do
#
# These pin the Phase 0 finding of the authority-research session: a
# fuze-well canard on a spin-stabilised shell sits at the subsonic centre of
# pressure, where the static trim force vanishes and what is left is the
# MAGNUS force, perpendicular to it. `trim_force_factor()` carries only the
# static part; `steering_response()` carries all of it.
# ===========================================================================
def _response_at(mach, spin, airspeed, geometry=None):
    g = geometry or cn.NOMINAL_GEOMETRY
    tab = aerodata.make_m107_table()
    c = tab.coefficients_at(mach)
    cla = cn.canard_lift_curve_slope(mach, g.aspect_ratio_effective)
    return cn.steering_response(g, pr.M107, c.C_Nalpha, c.C_Malpha,
                                c.C_Ypalpha, c.C_Mpalpha, cla, spin, airspeed)


def test_the_spinning_shell_trims_rather_than_precessing():
    """
    THE MECHANISM QUESTION. A gyroscopically stabilised body responds to a
    steady applied moment by precessing, and its steady-state offset is
    classically rotated ~90 degrees from that moment. That is NOT what happens
    here, because the equilibrium is not set by gyroscopic stiffness: it is
    set by the trajectory-curvature feedback, whose stiffness is m*a_n. The
    gyroscopic term H_x/V is a few per cent of it, so the trim is rotated by
    only a couple of degrees.

    If this ever fails, the closed-form cancellation argument in
    docs/CANARD-MODEL.md section 7 stops being a description of a trim.
    """
    r = _response_at(0.92, 1190.0, 285.0)
    assert 0.0 < r["gyroscopic_over_static"] < 0.10, (
        "the gyroscopic term should be a few per cent of the "
        "trajectory-curvature term for a 155 mm shell at apogee"
    )
    assert 0.0 < r["gyroscopic_rotation_deg"] < 6.0, (
        f"trim rotated by {r['gyroscopic_rotation_deg']:.1f} deg -- if this "
        "approaches 90 the response is a precession, not a trim"
    )


def test_the_magnus_force_dominates_the_residue_subsonically():
    """
    Subsonically the canards sit at the centre of pressure, the in-line term
    (x_c - x_cp) nearly vanishes, and the Magnus term -- which is
    perpendicular and does NOT depend on the canard station -- is larger.
    Supersonically the centre of pressure has moved aft, the in-line term
    dominates, and the step-2.5 picture is recovered.
    """
    sub = _response_at(0.90, 1190.0, 285.0)
    sup = _response_at(2.00, 1350.0, 620.0)
    assert sub["magnus_over_in_line"] > 1.0, (
        "subsonically the Magnus term should exceed the cancellation residue"
    )
    assert sup["magnus_over_in_line"] < 0.2, (
        "supersonically the residue should dominate and step 2.5's closed "
        "form should be adequate"
    )
    # and therefore the net force is well above the static-trim prediction
    # subsonically and equal to it supersonically
    assert sub["magnitude_over_static"] > 1.5
    assert sup["magnitude_over_static"] < 1.1


def test_the_correction_is_not_anti_parallel_to_the_canard_force():
    """
    Step 2.5 reported that "the projectile accelerates opposite to the canard
    force". That is true only supersonically. Subsonically the phase is about
    -120 degrees, and a guidance law that assumed 180 would steer into a
    60-degree error.
    """
    sub = _response_at(0.90, 1190.0, 285.0)
    sup = _response_at(2.00, 1350.0, 620.0)
    assert -160.0 < sub["net_over_direct_phase_deg"] < -90.0, (
        f"subsonic phase {sub['net_over_direct_phase_deg']:.0f} deg"
    )
    assert sup["net_over_direct_phase_deg"] < -170.0


def test_the_magnus_term_does_not_move_with_the_canard_station():
    """
    The in-line term is proportional to (x_c - x_cp) and moves strongly when
    the canards move. The Magnus term moves only through the canards' own
    contribution to the combined centre of pressure, which is a few per cent
    across the whole admissible station band. This is why the step-2.5 station
    sweep changed authority by 13 per cent where the closed form it was
    checked against changed by 2.5x.
    """
    from dataclasses import replace
    g = cn.NOMINAL_GEOMETRY
    fwd = _response_at(0.90, 1190.0, 285.0,
                       replace(g, station_from_nose=0.030))
    nom = _response_at(0.90, 1190.0, 285.0, g)
    magnus_change = abs(fwd["magnus_term_m"] / nom["magnus_term_m"] - 1.0)
    in_line_change = abs(fwd["in_line_term_m"] / nom["in_line_term_m"])
    assert magnus_change < 0.10, (
        f"Magnus term moved {100 * magnus_change:.0f} % across the station band"
    )
    assert in_line_change > 2.0, (
        "the in-line term must move by far more than the Magnus term"
    )


def test_steering_response_reduces_to_the_static_trim_factor():
    """
    With the Magnus coefficients and the spin set to zero, the complete form
    must return exactly what trim_force_factor() returns. If these ever
    diverge, one of them has been changed without the other.
    """
    g = cn.NOMINAL_GEOMETRY
    tab = aerodata.make_m107_table()
    c = tab.coefficients_at(0.85)
    cla = cn.canard_lift_curve_slope(0.85, g.aspect_ratio_effective)
    r = cn.steering_response(g, pr.M107, c.C_Nalpha, c.C_Malpha,
                             0.0, 0.0, cla, 0.0, 285.0,
                             axial_angular_momentum=0.0)
    t = cn.trim_force_factor(g, pr.M107, c.C_Nalpha, c.C_Malpha, cla)
    assert r["net_over_direct"].real == pytest.approx(t["trim_force_factor"],
                                                      rel=1e-12)
    assert abs(r["net_over_direct"].imag) < 1e-12
