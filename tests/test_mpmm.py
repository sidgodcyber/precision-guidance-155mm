"""
Unit tests for the reduced-order modified point-mass model.

The tests that matter most here are the ones checking that the MPMM and the
6-DOF are driven by IDENTICAL aerodynamics, because the whole value of the
model-error measurement rests on that. If a coefficient ever gets duplicated
or converted wrongly, these fail.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from models import mpmm as M
from sim import aerodata, atmosphere as atm
from sim import dynamics as dyn, integrate as ig, projectile as pr


def _models(coriolis=False, **mpmm_kw):
    env = pr.Environment.from_degrees(45.0, include_coriolis=coriolis)
    six = dyn.FlightModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), environment=env
    )
    red = M.MpmmModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), environment=env, **mpmm_kw
    )
    return six, red


# ===========================================================================
# THE CONSTRAINT: no fitting factors
# ===========================================================================
def test_all_fitting_factors_are_unity():
    """
    STANAG 4355 fitting factors are trial-fitted to one projectile lot. Using
    them would make the MPMM match the firing table trivially and would turn
    the measured model-error term in docs/MODEL-ERROR.md into a manufactured
    one. They must stay at 1.0.
    """
    f = M.FittingFactors()
    assert f.form_factor_i == 1.0
    assert f.lift_factor_fL == 1.0
    assert f.magnus_factor_QM == 1.0
    assert f.yaw_drag_factor_QD == 1.0
    assert f.all_unity()

    # and the default model actually uses that default
    _six, red = _models()
    assert red.factors.all_unity()

    # every field of the dataclass is accounted for above
    import dataclasses

    names = {fld.name for fld in dataclasses.fields(M.FittingFactors)}
    assert names == {
        "form_factor_i",
        "lift_factor_fL",
        "magnus_factor_QM",
        "yaw_drag_factor_QD",
    }, "a new fitting factor was added without being asserted unity"


# ===========================================================================
# Shared aerodynamics: the MPMM must not duplicate a coefficient
# ===========================================================================
def test_mpmm_uses_the_same_table_object_type_and_values():
    _six, red = _models()
    shared = aerodata.make_m107_table()
    for m in (0.3, 0.8, 1.2, 2.0):
        assert red.aero.lookup(m) == pytest.approx(shared.lookup(m))


def test_no_hardcoded_coefficients_in_the_module():
    """
    Guard against a coefficient being pasted in.

    Every aerodynamic number must arrive through `model.aero.lookup(mach)`.
    A pasted coefficient would show up as a high-precision float literal in
    the executable code, so the test asserts there are none: the module's own
    constants are all simple (0.5, 1.0, 2.0, 8.0, 1e-6 ...).
    """
    import ast
    import inspect
    import re

    src = inspect.getsource(M)
    tree = ast.parse(src)
    # remove all docstrings, where coefficients are legitimately discussed
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    body = ast.unparse(tree)
    body = re.sub(r"#.*", "", body)

    offenders = [m for m in re.findall(r"\d+\.\d{3,}", body)]
    assert not offenders, f"high-precision literals in mpmm.py: {offenders}"

    # and the aero table really is consulted
    assert "lookup" in body


# ===========================================================================
# The yaw of repose
# ===========================================================================
def test_yaw_of_repose_points_right_for_right_hand_spin():
    """
    The single most important sign check in this module. Positive (right-hand)
    spin must put the yaw of repose to the RIGHT (+Y), which is what makes the
    shell drift right.
    """
    _six, red = _models()
    y = M.pack((0.0, 0.0, -3000.0), (400.0, 0.0, 0.0), 1200.0)
    ax, ay, az, mag = M.yaw_of_repose(0.0, y, red)
    assert ay > 0.0
    assert mag > 0.0
    assert abs(ax) < 1e-9 * mag or abs(ax) < 1e-12

    # left-hand spin reverses it
    y_left = M.pack((0.0, 0.0, -3000.0), (400.0, 0.0, 0.0), -1200.0)
    _bx, by, _bz, _bm = M.yaw_of_repose(0.0, y_left, red)
    assert by < 0.0


def test_yaw_of_repose_matches_the_sixdof_closed_form():
    """
    The STANAG algebraic form must reduce to the closed-form yaw of repose in
    sim/diagnostics.py, which the 6-DOF epicyclic motion settles onto. This is
    the guarantee that the two models share a steady state.

        alpha_e = 8 Ix p g cos(theta) / (pi rho d^3 C_Malpha V^3)
                = 2 Ix p g cos(theta) / (rho V^3 S d C_Malpha)
    """
    from sim import diagnostics as dg

    env = pr.Environment.from_degrees(45.0, include_coriolis=False,
                                      include_inverse_square_gravity=False)
    red = M.MpmmModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                      environment=env)
    table = aerodata.make_m107_table()

    for V, spin, alt in ((400.0, 1200.0, 3000.0), (300.0, 1000.0, 500.0),
                         (600.0, 1350.0, 1000.0)):
        y = M.pack((0.0, 0.0, -alt), (V, 0.0, 0.0), spin)   # level flight
        _ax, _ay, _az, mag = M.yaw_of_repose(0.0, y, red)
        rho = atm.isa_scalars(alt)[2]
        mach = V / atm.isa_scalars(alt)[3]
        c = table.coefficients_at(mach)
        closed = dg.yaw_of_repose_estimate(pr.M107, rho, V, spin, c.C_Malpha, 0.0)
        assert mag == pytest.approx(closed, rel=1e-9)


def test_yaw_of_repose_scales_as_expected():
    _six, red = _models()
    base = M.pack((0.0, 0.0, -3000.0), (400.0, 0.0, 0.0), 1200.0)
    _a, _b, _c, m0 = M.yaw_of_repose(0.0, base, red)
    # linear in spin
    _a, _b, _c, m1 = M.yaw_of_repose(
        0.0, M.pack((0.0, 0.0, -3000.0), (400.0, 0.0, 0.0), 2400.0), red)
    assert m1 == pytest.approx(2.0 * m0, rel=1e-6)
    # zero spin gives zero yaw of repose, hence zero drift
    _a, _b, _c, m2 = M.yaw_of_repose(
        0.0, M.pack((0.0, 0.0, -3000.0), (400.0, 0.0, 0.0), 0.0), red)
    assert m2 == 0.0


# ===========================================================================
# Force-model agreement with the 6-DOF
# ===========================================================================
def test_zero_yaw_drag_matches_the_sixdof_axial_force():
    """
    With spin zero the yaw of repose vanishes, so the MPMM reduces to a
    drag-only point mass. Its acceleration must then equal the 6-DOF's when
    the 6-DOF is put in its alpha-zero drag-only mode -- the same mode rung 2
    of step 1 validated against an independent 3-DOF integration.
    """
    env = pr.Environment.from_degrees(45.0, include_coriolis=False)
    six = dyn.FlightModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), environment=env,
        alpha_zero_drag_only=True,
    )
    red = M.MpmmModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                      environment=env)

    launch = pr.LaunchConditions.from_degrees(684.0, 45.0)
    y6 = dyn.initial_state(pr.M107, launch)
    y6[10] = 0.0                      # kill spin -> no yaw of repose
    ym = M.state_from_sixdof(y6)

    d6 = dyn.derivative(0.0, y6, six)
    dm = M.derivative(0.0, ym, red)
    assert dm[3] == pytest.approx(d6[3], rel=1e-12)
    assert dm[4] == pytest.approx(d6[4], rel=1e-12)
    assert dm[5] == pytest.approx(d6[5], rel=1e-12)


def test_coriolis_matches_the_sixdof_exactly():
    """
    REGRESSION. The Coriolis branch of _base_acceleration referred to an
    undefined name and had never executed, because every run in docs/ was made
    with Coriolis off to match step 1's firing-table convention. It raised
    NameError the first time it was switched on, in the Task E ablation study.

    With the yaw-dependent terms disabled the MPMM is drag + gravity +
    Coriolis, so switching Coriolis on must change its acceleration by exactly
    the vector it changes the 6-DOF's by: -2 Omega x v.
    """
    kw = dict(include_lift=False, include_magnus=False, include_yaw_drag=False)
    six_off, red_off = _models(coriolis=False, **kw)
    six_on, red_on = _models(coriolis=True, **kw)

    launch = pr.LaunchConditions.from_degrees(684.0, 40.0)
    y6 = dyn.initial_state(pr.M107, launch)
    ym = M.state_from_sixdof(y6)

    six_delta = (np.asarray(dyn.derivative(0.0, y6, six_on)[3:6])
                 - np.asarray(dyn.derivative(0.0, y6, six_off)[3:6]))
    mpmm_delta = (np.asarray(M.derivative(0.0, ym, red_on), dtype=float)[3:6]
                  - np.asarray(M.derivative(0.0, ym, red_off), dtype=float)[3:6])

    expected = atm.coriolis_acceleration(np.asarray(y6[3:6]), np.asarray(six_on._omega_ned))
    assert np.linalg.norm(expected) > 1e-3          # the term is not trivial
    assert np.allclose(six_delta, expected, rtol=1e-9, atol=1e-15)
    assert np.allclose(mpmm_delta, expected, rtol=1e-9, atol=1e-15)


def test_coriolis_uses_ground_velocity_not_air_relative_velocity():
    """
    Coriolis is a kinematic term of the rotating earth frame and takes the
    GROUND velocity; drag takes the AIR-RELATIVE velocity. With no wind the
    two are equal and the distinction is invisible -- which is precisely why
    it needs a test of its own with a large wind in it.
    """
    kw = dict(include_lift=False, include_magnus=False, include_yaw_drag=False)
    wind = atm.constant_wind(-40.0, 25.0)
    mk = lambda cor: M.MpmmModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), wind=wind,
        environment=pr.Environment.from_degrees(45.0, include_coriolis=cor), **kw)
    red_on, red_off = mk(True), mk(False)

    launch = pr.LaunchConditions.from_degrees(684.0, 40.0)
    ym = M.state_from_sixdof(dyn.initial_state(pr.M107, launch))

    delta = (np.asarray(M.derivative(0.0, ym, red_on), dtype=float)[3:6]
             - np.asarray(M.derivative(0.0, ym, red_off), dtype=float)[3:6])
    omega = np.asarray(red_on._omega_ned)
    right = atm.coriolis_acceleration(np.asarray(ym[3:6]), omega)
    wrong = atm.coriolis_acceleration(np.asarray(ym[3:6]) - np.array([-40.0, 25.0, 0.0]),
                                      omega)

    assert np.allclose(delta, right, rtol=1e-9, atol=1e-15)
    # the wind is large enough that the two candidates are distinguishable,
    # so this test really does discriminate between them
    assert np.linalg.norm(right - wrong) > 1e-6
    assert not np.allclose(delta, wrong, rtol=1e-6, atol=1e-12)


def test_coriolis_perturbs_the_yaw_of_repose_slightly():
    """
    A structural property of the MPMM that the 6-DOF does not share, pinned
    here so it is not mistaken for a bug later.

    STANAG's yaw of repose is driven by dv/dt, and with Coriolis enabled that
    dv/dt includes the Coriolis acceleration. So switching Coriolis on in the
    FULL model changes the acceleration by slightly more than -2 Omega x v.
    The 6-DOF has no such coupling: its aerodynamic force depends on attitude
    and velocity only, never on acceleration.

    The extra part is a few percent of an already small term. It is recorded,
    not corrected -- v x dv/dt is what the standard specifies.
    """
    six_off, red_off = _models(coriolis=False)
    six_on, red_on = _models(coriolis=True)

    launch = pr.LaunchConditions.from_degrees(684.0, 40.0)
    y6 = dyn.initial_state(pr.M107, launch)
    ym = M.state_from_sixdof(y6)

    six_delta = (np.asarray(dyn.derivative(0.0, y6, six_on)[3:6])
                 - np.asarray(dyn.derivative(0.0, y6, six_off)[3:6]))
    mpmm_delta = (np.asarray(M.derivative(0.0, ym, red_on), dtype=float)[3:6]
                  - np.asarray(M.derivative(0.0, ym, red_off), dtype=float)[3:6])
    cor = atm.coriolis_acceleration(np.asarray(y6[3:6]), np.asarray(six_on._omega_ned))

    # the 6-DOF sees the Coriolis term and nothing else
    assert np.allclose(six_delta, cor, rtol=1e-9, atol=1e-15)
    # the MPMM sees a little more, and the excess is small
    excess = np.linalg.norm(mpmm_delta - cor) / np.linalg.norm(cor)
    assert 0.0 < excess < 0.10


def test_spin_damping_matches_the_sixdof():
    """Ix pdot = qbar S d C_lp (p d / 2V) in both models."""
    six, red = _models()
    launch = pr.LaunchConditions.from_degrees(684.0, 30.0)
    y6 = dyn.initial_state(pr.M107, launch)
    ym = M.state_from_sixdof(y6)
    d6 = dyn.derivative(0.0, y6, six)
    dm = M.derivative(0.0, ym, red)
    assert dm[6] == pytest.approx(d6[10], rel=1e-9)


def test_lift_slope_is_normal_force_slope_minus_axial():
    """
    C_Lalpha = C_Nalpha - C_X0. Checked by driving the MPMM at a known yaw of
    repose and comparing the transverse force it produces with the value that
    relation implies.
    """
    _six, red = _models()
    table = aerodata.make_m107_table()
    y = M.pack((0.0, 0.0, -3000.0), (400.0, 0.0, 0.0), 1200.0)
    fx, fy, fz, alpha = M._forces(0.0, y, red)

    alt, V = 3000.0, 400.0
    rho = atm.isa_scalars(alt)[2]
    mach = V / atm.isa_scalars(alt)[3]
    c = table.coefficients_at(mach)
    qS = 0.5 * rho * V * V * pr.M107.reference_area
    expected_lift = qS * (c.C_Nalpha - c.C_X0) * alpha
    # the transverse (Y) force is lift plus a small Magnus contribution
    assert fy > 0.0
    assert fy == pytest.approx(expected_lift, rel=0.05)


def test_magnus_can_be_switched_off_and_is_small():
    _six, red_on = _models()
    _six2, red_off = _models(include_magnus=False)
    y = M.pack((0.0, 0.0, -3000.0), (400.0, 0.0, 0.0), 1200.0)
    on = M._forces(0.0, y, red_on)
    off = M._forces(0.0, y, red_off)
    tot_on = math.sqrt(sum(v * v for v in on[:3]))
    diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(on[:3], off[:3])))
    assert diff < 0.25 * tot_on


def test_lift_off_gives_no_drift():
    """Without the yaw-of-repose lift there is no mechanism for drift."""
    _six, red = _models(include_lift=False, include_magnus=False)
    y0 = M.initial_state(pr.M107, pr.LaunchConditions.from_mils(684.0, 525.3))
    res = M.propagate_to_impact(y0, red, dt=0.02)
    assert abs(res.drift_m) < 1e-6


# ===========================================================================
# Purity and state handling
# ===========================================================================
def test_yaw_iteration_is_off_by_default_and_changes_only_the_evaluation():
    """
    The fixed-point pass on the yaw of repose is a more faithful evaluation of
    the STANAG formula, not a tuning knob. It must:
      - be OFF by default, so the documented default derivative stays
        closed-form with no inner pass;
      - introduce NO fitting factor -- all four stay unity;
      - change alpha_e by a small amount in the direction of the lift and
        Magnus accelerations it now includes;
      - remain a pure, deterministic function of state.
    """
    _six, plain = _models()
    _six, iterated = _models(iterate_yaw=True)
    assert plain.iterate_yaw is False
    assert iterated.factors.all_unity()

    launch = pr.LaunchConditions.from_mils(684.0, 525.3)
    y = M.initial_state(pr.M107, launch)
    y[2] = -3000.0                      # a mid-flight altitude
    y[3:6] = (250.0, 4.0, 120.0)        # descending, mostly downrange

    d_plain = M.derivative(0.0, y, plain)
    d_iter = M.derivative(0.0, y, iterated)

    # it changes the answer, but only slightly
    rel = max(abs(d_iter[i] - d_plain[i]) / max(abs(d_plain[i]), 1e-9)
              for i in (3, 4, 5))
    assert 0.0 < rel < 0.2

    # still pure and deterministic
    y_before = y.copy()
    again = M.derivative(0.0, y, iterated)
    assert np.array_equal(y, y_before)
    assert again == d_iter


def test_yaw_iteration_is_a_small_correction():
    """
    Pins the size of the correction: a couple of percent of alpha_e at most
    (0.70 % at apogee on the reference engagement, 1.01 % at the state used
    here). If a future change makes the fixed-point pass a large correction
    rather than a small one, the assumption that a SINGLE pass suffices
    (docs/MODEL-ERROR.md limitation 6) no longer holds, and this fails.
    """
    _six, red = _models()
    launch = pr.LaunchConditions.from_mils(684.0, 525.3)
    y = M.initial_state(pr.M107, launch)
    y[2] = -3000.0
    y[3:6] = (250.0, 4.0, 120.0)

    a0 = M.yaw_of_repose(0.0, y, red, iterate=False)[3]
    a1 = M.yaw_of_repose(0.0, y, red, iterate=True)[3]
    assert a0 > 0.0
    assert 0.0 < abs(a1 - a0) / a0 < 0.03


def test_derivative_is_pure_and_deterministic():
    _six, red = _models()
    y = list(M.initial_state(pr.M107, pr.LaunchConditions.from_degrees(684.0, 30.0)))
    before = list(y)
    a = M.derivative(0.0, y, red)
    b = M.derivative(0.0, y, red)
    assert y == before, "derivative mutated its input"
    assert a == b, "derivative is not deterministic"
    assert len(a) == M.MPMM_STATE_SIZE


def test_pack_unpack_roundtrip():
    y = M.pack((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), 7.0)
    r, v, p = M.unpack(y)
    assert list(r) == [1.0, 2.0, 3.0]
    assert list(v) == [4.0, 5.0, 6.0]
    assert p == 7.0


def test_state_from_sixdof_transfers_position_velocity_spin():
    launch = pr.LaunchConditions.from_degrees(684.0, 40.0)
    y6 = dyn.initial_state(pr.M107, launch)
    ym = M.state_from_sixdof(y6)
    assert np.allclose(ym[0:3], y6[0:3])
    assert np.allclose(ym[3:6], y6[3:6])
    assert ym[6] == pytest.approx(y6[10])


def test_initial_state_matches_the_sixdof_muzzle_state():
    for qe in (10.0, 30.0, 60.0):
        launch = pr.LaunchConditions.from_degrees(684.0, qe)
        y6 = dyn.initial_state(pr.M107, launch)
        ym = M.initial_state(pr.M107, launch)
        assert np.allclose(ym[0:3], y6[0:3], atol=1e-12)
        assert np.allclose(ym[3:6], y6[3:6], atol=1e-9)
        assert ym[6] == pytest.approx(y6[10])


# ===========================================================================
# Propagation
# ===========================================================================
def test_impact_is_interpolated_to_ground():
    _six, red = _models()
    y0 = M.initial_state(pr.M107, pr.LaunchConditions.from_mils(684.0, 248.4))
    res = M.propagate_to_impact(y0, red, dt=0.02)
    assert res.terminated == "impact"
    assert abs(res.impact_state[2]) < 1e-9


def test_vacuum_matches_the_analytic_parabola():
    """The MPMM must pass step 1's rung 1 too."""
    env = pr.Environment.from_degrees(
        45.0, include_coriolis=False, include_inverse_square_gravity=False)

    class _NoAero:
        mach_min, mach_max = 0.0, 10.0

        def lookup(self, m):
            return (0.0,) * 8

    red = M.MpmmModel(projectile=pr.M107, aero=_NoAero(), environment=env)
    for qe in (20.0, 45.0, 70.0):
        y0 = M.initial_state(pr.M107, pr.LaunchConditions.from_degrees(500.0, qe))
        res = M.propagate_to_impact(y0, red, dt=1e-3)
        exact = 500.0 ** 2 * math.sin(2 * math.radians(qe)) / atm.G0
        assert res.range_m == pytest.approx(exact, rel=1e-6)
        assert abs(res.drift_m) < 1e-9


def test_drift_is_to_the_right():
    _six, red = _models()
    y0 = M.initial_state(pr.M107, pr.LaunchConditions.from_mils(684.0, 525.3))
    res = M.propagate_to_impact(y0, red, dt=0.02)
    assert res.drift_m > 0.0


def test_close_to_the_sixdof_on_a_reference_engagement():
    """
    Regression guard on the headline agreement: the MPMM must stay within a
    fraction of a percent of the 6-DOF in range and a few percent in drift.
    Loose enough not to be brittle, tight enough that a coefficient or
    conversion error would trip it.

    NOTE ON THE 6-DOF STEP SIZE: this must run the reference at dt = 2e-4.
    Step 1 established that the 6-DOF needs 20-50 samples per revolution of a
    221 rev/s spin; at dt = 1e-3 it is so badly under-resolved that it returns
    21 500 m for an engagement whose converged answer is 15 841 m. A shorter
    charge-4 engagement is used here to keep the test affordable at that step.
    """
    six_model, red = _models()
    launch = pr.LaunchConditions.from_mils(337.0, 211.6)   # 13.5 s flight
    y6 = dyn.initial_state(pr.M107, launch)
    six = ig.integrate(y6, six_model, dt=2e-4, log_every=1000000, t_max=60.0)
    res = M.propagate_to_impact(M.initial_state(pr.M107, launch), red, dt=0.01)
    assert abs(res.range_m - six.range_m) / six.range_m < 0.005
    assert abs(res.drift_m - six.drift_m) / six.drift_m < 0.05
    assert abs(res.impact_time - six.impact_time) < 0.5
