"""
Step 3 tests: the inverse map, the schedulers, the predictor and the ladder.

The expensive things -- 6-DOF closed-loop trajectories -- are not run here.
What is tested is everything that decides what those trajectories will be:
that the inversion is an inversion, that the anisotropy is not silently
averaged away, that a scheduler cannot re-hold after releasing, that the
authority monitor fires on a dead actuator and does not fire on a live one,
and that the predictor's kit-drag increment is the panel aerodynamics
`sim.canards` uses and not a fitted number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim import aerodata, canards as cn, projectile as pr
from gnc import guidance as gd
from gnc import scheduler as sch
from gnc.inverse_map import (AuthorityMap, MapNode, fit_harmonic, ellipse_of,
                             ellipse_contains_fraction, containment_margin)


# ===========================================================================
# Fixtures
# ===========================================================================
def _rot(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s], [s, c]])


def synthetic_map(t_dep: float = 5.55, t_imp: float = 48.24,
                  a: float = 185.5, b: float = 158.8,
                  rotation_deg: float = -75.0, dead: float = 2.74):
    """
    A map with the adopted configuration's measured shape and a plausible
    saturating growth. Used where the test is about the algebra, not about the
    numbers.
    """
    T = t_imp - t_dep
    M = _rot(rotation_deg) @ np.diag([a, b])
    nodes = []
    for off in np.linspace(0.0, T, 25):
        f = 0.0 if off <= dead else 1.0 - ((T - off) / (T - dead)) ** 2
        nodes.append(MapNode(t_go=t_imp - (t_dep + off), G=f * M,
                             c=np.array([-8.0 * f, 0.0]), mach=1.0,
                             residual=0.0))
    return AuthorityMap(nodes, t_imp, t_dep, "synthetic")


# ===========================================================================
# The inversion
# ===========================================================================
@pytest.mark.parametrize("deg", list(range(0, 360, 15)))
def test_inversion_points_where_asked(deg):
    """`invert` returns the angle whose delivered correction is along `u`."""
    A = np.array([[100.0, -160.0], [150.0, 90.0]])
    u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
    phi = AuthorityMap.invert(A, u)
    got = A @ np.array([math.cos(phi), math.sin(phi)])
    # Same direction, and the SAME direction rather than the opposite one.
    assert float(got @ u) > 0.0
    assert abs(math.degrees(math.atan2(got[1], got[0])) - deg) % 360.0 < 1e-6 \
        or abs(abs(math.degrees(math.atan2(got[1], got[0])) - deg) - 360.0) < 1e-6


def test_reach_along_equals_delivered_magnitude():
    """Equation (4) is the magnitude equation (3) actually delivers."""
    A = np.array([[100.0, -160.0], [150.0, 90.0]])
    for deg in range(0, 360, 23):
        u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        phi = AuthorityMap.invert(A, u)
        got = A @ np.array([math.cos(phi), math.sin(phi)])
        assert abs(float(np.hypot(*got)) - AuthorityMap.reach_along(A, u)) < 1e-9


def test_reach_varies_by_the_axis_ratio():
    """
    The best and worst directions differ by exactly the axis ratio.

    This is the anisotropy the schedulers must not average over, and it is a
    property of the map rather than of any law, so it is asserted here.
    """
    A = _rot(-75.0) @ np.diag([185.5, 158.8])
    reach = [AuthorityMap.reach_along(A, [math.cos(t), math.sin(t)])
             for t in np.linspace(0, 2 * math.pi, 721)]
    e = ellipse_of(A)
    assert abs(max(reach) - e["semi_major_m"]) < 1e-6
    assert abs(min(reach) - e["semi_minor_m"]) < 1e-6
    assert abs(max(reach) / min(reach) - e["axis_ratio"]) < 1e-9


def test_singular_map_refuses_to_command():
    """
    Zero authority must come back as `None`, not as an angle.

    A stage-2 mechanical failure leaves an increment with no rank, and a law
    handed a plausible-looking angle for it would keep commanding a dead
    actuator -- which is the class of fault step 4 found in the engagement
    gate.
    """
    assert AuthorityMap.invert(np.zeros((2, 2)), [1.0, 0.0]) is None
    assert AuthorityMap.reach_along(np.zeros((2, 2)), [1.0, 0.0]) == 0.0
    A = np.array([[1.0, 2.0], [2.0, 4.0]])          # rank 1
    assert AuthorityMap.invert(A, [1.0, 0.0]) is None


def test_harmonic_fit_round_trips():
    G = np.array([[100.0, -160.0], [150.0, 90.0]])
    c = np.array([7.0, -3.0])
    angs = np.arange(0.0, 360.0, 45.0)
    xy = [G @ [math.cos(math.radians(a)), math.sin(math.radians(a))] + c
          for a in angs]
    G2, c2, resid = fit_harmonic(angs, xy)
    assert np.allclose(G2, G, atol=1e-9)
    assert np.allclose(c2, c, atol=1e-9)
    assert resid < 1e-12


def test_map_increment_is_a_difference():
    m = synthetic_map()
    A, b = m.increment(30.0, 10.0)
    G1, c1 = m.delivered(10.0)
    G0, c0 = m.delivered(30.0)
    assert np.allclose(A, G1 - G0)
    assert np.allclose(b, c1 - c0)


def test_map_is_flat_outside_its_nodes():
    """No extrapolation of a saturating quantity."""
    m = synthetic_map()
    lo = m.nodes[-1].t_go
    hi = m.nodes[0].t_go
    assert np.allclose(m.delivered(lo - 10.0)[0], m.delivered(lo)[0])
    assert np.allclose(m.delivered(hi + 10.0)[0], m.delivered(hi)[0])


def test_hold_end_is_monotone_and_saturates():
    m = synthetic_map()
    t_go = 40.0
    u = np.array([1.0, 0.0])
    ends = []
    for need in (10.0, 40.0, 90.0, 140.0):
        tge, phi, got, sat = m.hold_end_for(u * need, t_go)
        assert not sat
        assert abs(got - need) < 0.5
        ends.append(tge)
    # More correction required means holding longer, i.e. a smaller t_go_end.
    assert all(ends[i] > ends[i + 1] for i in range(len(ends) - 1))
    # And a demand beyond the set saturates rather than promising it.
    tge, phi, got, sat = m.hold_end_for(u * 100000.0, t_go)
    assert sat and tge == 0.0


def test_map_size_is_six_floats_per_node():
    m = synthetic_map()
    assert m.size_floats == 6 * len(m.nodes)
    assert m.size_bytes(4) == 24 * len(m.nodes)


def test_map_serialisation_round_trips():
    m = synthetic_map()
    m2 = AuthorityMap.from_dict(m.as_dict())
    for t in (0.0, 5.0, 17.3, 42.0):
        assert np.allclose(m.delivered(t)[0], m2.delivered(t)[0])
        assert np.allclose(m.delivered(t)[1], m2.delivered(t)[1])


# ===========================================================================
# Containment
# ===========================================================================
def test_containment_is_one_for_a_huge_set_and_zero_for_a_tiny_one():
    assert ellipse_contains_fraction(1e5, 1e5, 0.0, 200.0, 200.0) > 0.999
    assert ellipse_contains_fraction(1e-3, 1e-3, 0.0, 200.0, 200.0) < 1e-3


def test_containment_margin_inverts_containment():
    a, b, tilt, sr, sd = 185.5, 158.8, -20.0, 244.0, 81.4
    k = containment_margin(a, b, tilt, sr, sd, target=0.5)
    p = ellipse_contains_fraction(a * k, b * k, tilt, sr, sd)
    assert abs(p - 0.5) < 0.01


def test_containment_sees_a_shape_the_scalar_test_cannot():
    """
    Two reachable sets with the SAME semi-major axis contain different
    fractions of the same miss distribution.

    This is the whole reason the scalar criterion is being replaced: a test on
    one semi-axis cannot distinguish them, and the closed loop can.
    """
    sr, sd = 244.0, 81.4         # 3:1 range-dominated, CEP 200 m
    circleish = ellipse_contains_fraction(185.5, 175.0, 0.0, sr, sd)
    elongated = ellipse_contains_fraction(185.5, 60.0, 0.0, sr, sd)
    assert circleish != pytest.approx(elongated, abs=0.02)


# ===========================================================================
# The predictor
# ===========================================================================
def test_kit_drag_is_the_panel_aerodynamics_and_not_a_fitted_number():
    """
    The increment must equal, term for term, what `sim.canards` applies to the
    truth, so that no coefficient in the onboard model has been tuned to the
    6-DOF.
    """
    g = cn.CanardGeometry(station_from_nose=0.025,
                          steering_deflection=math.radians(3.0))
    base = aerodata.make_m107_table()
    tab = gd.kit_drag_table(g, pr.M107, base, n_panels=4)
    ratio = 4 * g.panel_area / pr.M107.reference_area
    pi_ar_e = math.pi * g.aspect_ratio_effective * g.induced_drag_efficiency
    d2 = 0.5 * (g.steering_deflection ** 2 + g.cant_angle ** 2)
    for j, m in enumerate(base.mach):
        want = ratio * (cn.canard_zero_lift_drag(float(m))
                        + cn.canard_lift_curve_slope(
                            float(m), g.aspect_ratio_effective) ** 2
                        * d2 / pi_ar_e)
        assert tab.values[j, 0] - base.values[j, 0] == pytest.approx(want, rel=1e-12)
    # Every other coefficient is untouched.
    assert np.allclose(tab.values[:, 1:], base.values[:, 1:])


def test_kit_drag_two_panels_is_smaller_than_four():
    g = cn.CanardGeometry(station_from_nose=0.025,
                          steering_deflection=math.radians(3.0))
    base = aerodata.make_m107_table()
    four = gd.kit_drag_table(g, pr.M107, base, 4)
    two = gd.kit_drag_table(g, pr.M107, base, 2)
    assert np.all(two.values[:, 0] < four.values[:, 0])
    assert np.all(two.values[:, 0] > base.values[:, 0])


# ===========================================================================
# The schedulers
# ===========================================================================
def _ctx(m, t_go, miss, armed=True, phi=0.0, authority_ok=True):
    return sch.Context(t=m.impact_time - t_go, t_go=t_go,
                       miss=np.asarray(miss, dtype=float), amap=m, armed=armed,
                       holding=True, phi=phi, cycles=1,
                       authority_ok=authority_ok)


@pytest.mark.parametrize("name", sorted(sch.SCHEDULERS))
def test_a_released_scheduler_never_holds_again(name):
    """
    The contiguity constraint. Holds shorter than about 0.5 s deliver nothing
    (docs/CONTROL-CHARACTERISATION.md section 8.3), so a law that releases and
    re-holds is spending authority it will not get back.
    """
    m = synthetic_map()
    s = sch.make_scheduler(name)
    s(_ctx(m, 35.0, [150.0, 0.0]))
    s.released = True
    for t_go in (30.0, 20.0, 10.0):
        d = s(_ctx(m, t_go, [150.0, 0.0]))
        assert d.hold is False


@pytest.mark.parametrize("name", sorted(sch.SCHEDULERS))
def test_no_authority_stops_every_law(name):
    m = synthetic_map()
    s = sch.make_scheduler(name)
    d = s(_ctx(m, 30.0, [150.0, 0.0], authority_ok=False))
    assert d.hold is False and d.note == "no_authority"


@pytest.mark.parametrize("name", sorted(sch.SCHEDULERS))
def test_every_law_holds_when_the_miss_is_large(name):
    m = synthetic_map()
    s = sch.make_scheduler(name)
    d = s(_ctx(m, 35.0, [140.0, 40.0]))
    assert d.hold is True


def test_deadband_releases_inside_the_band_and_holds_outside():
    m = synthetic_map()
    s = sch.DeadbandScheduler(deadband_m=20.0)
    assert s(_ctx(m, 35.0, [100.0, 0.0])).hold is True
    s2 = sch.DeadbandScheduler(deadband_m=20.0)
    s2(_ctx(m, 35.0, [100.0, 0.0]))
    assert s2(_ctx(m, 30.0, [5.0, 5.0])).note == "deadband"


def test_budget_holds_back_its_reserve():
    """
    Outside the endgame the budget law plans to spend at most (1 - reserve) of
    what is available, so its planned release is EARLIER than the release that
    would deliver the whole miss.
    """
    m = synthetic_map()
    miss = np.array([120.0, 0.0])
    plain, _, _, _ = m.hold_end_for(miss, 35.0)
    s = sch.BudgetScheduler(reserve=0.5, endgame_t_go=5.0)
    d = s(_ctx(m, 35.0, miss))
    assert d.hold is True
    assert d.planned_t_go_end > plain     # releases sooner
    assert s.plan["mode"] == "planned"
    assert s.plan["spend_m"] <= 0.5 * s.plan["avail_m"] + 1e-9


def test_budget_spends_the_reserve_in_the_endgame():
    """
    Inside the endgame the reserve has no future use, so the plan is for the
    whole of the miss rather than for (1 - reserve) of what is available.
    """
    m = synthetic_map()
    s = sch.BudgetScheduler(reserve=0.5, endgame_t_go=12.0)
    A, _ = m.remaining(10.0)
    need = 0.4 * m.reach_along(A, [1.0, 0.0])     # comfortably reachable
    s(_ctx(m, 10.0, [need, 0.0]))
    assert s.plan["mode"] == "endgame"
    assert s.plan["spend_m"] == pytest.approx(need, rel=1e-6)


def test_budget_saturates_rather_than_promising_the_unreachable():
    m = synthetic_map()
    s = sch.BudgetScheduler()
    d = s(_ctx(m, 35.0, [5000.0, 0.0]))
    assert d.hold is True and d.note == "saturated"
    assert d.planned_t_go_end == 0.0


def test_single_shot_decides_once_and_does_not_revise():
    m = synthetic_map()
    s = sch.SingleShotScheduler()
    d0 = s(_ctx(m, 35.0, [120.0, 20.0]))
    plan = dict(s.plan)
    # A wildly different miss on the next cycle must not move the plan.
    d1 = s(_ctx(m, 30.0, [-400.0, 300.0]))
    assert s.plan == plan
    assert d1.phi == pytest.approx(d0.phi)


def test_isotropic_ignores_the_direction_dependence():
    """
    The control experiment must actually be the control experiment: its
    `available_m` is the same for every miss direction, where the map's is
    not.
    """
    m = synthetic_map()
    iso, ani = [], []
    for deg in range(0, 360, 5):
        u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        iso.append(sch.IsotropicScheduler()(
            _ctx(m, 35.0, u * 50.0)).available_m)
        ani.append(sch.BudgetScheduler()(
            _ctx(m, 35.0, u * 50.0)).available_m)
    assert max(iso) - min(iso) < 1e-6
    # The anisotropic law's available authority spans exactly the axis ratio.
    assert max(ani) / min(ani) == pytest.approx(185.5 / 158.8, rel=0.02)


def test_pre_arm_commands_an_angle_but_is_not_counted_as_a_commit():
    """
    Before a steering force exists the servo still has to be given something
    to acquire -- the acquisition is what opens the stage-2 gate -- but the
    law must not treat those cycles as spending authority.
    """
    m = synthetic_map()
    for name in sch.SCHEDULERS:
        s = sch.make_scheduler(name)
        d = s(_ctx(m, 42.0, [100.0, 0.0], armed=False))
        assert d.hold is True and d.note.startswith("pre_arm")


# ===========================================================================
# The law: modes and the authority monitor
# ===========================================================================
class _StubPredictor:
    """Returns a scripted sequence of predicted impact points."""

    def __init__(self, seq, tof):
        self.seq = list(seq)
        self.tof = tof
        self.calls = 0
        self.derivative_calls = 0

    def predict(self, t, y):
        i = min(self.calls, len(self.seq) - 1)
        self.calls += 1
        r, d = self.seq[i]
        return r, d, self.tof, "impact"


def _law(mode="full", seq=None, tof=48.24, target=(0.0, 0.0), **kw):
    m = synthetic_map(t_imp=tof)
    cfg = gd.GuidanceConfig(mode=mode, **kw)
    p = _StubPredictor(seq or [(0.0, 0.0)], tof)
    s = sch.make_scheduler("budget")
    return gd.GuidanceLaw(cfg, m, p, s, target, m.deploy_time), m


def test_reversionary_never_holds():
    law, m = _law(mode="reversionary")
    for t in np.arange(m.deploy_time, m.deploy_time + 20.0, 1.0):
        law.sample(float(t), np.zeros(15), None)
    assert all(h is False for h in law.log["holding"])
    assert law.command(0.0)[1] is False


def test_degraded_holds_the_last_good_command():
    """
    Navigation invalid mid-flight: the commanded angle must FREEZE at its last
    valid value rather than track a prediction that no longer exists.
    """
    seq = [(100.0, 0.0)] * 6 + [(1e6, 1e6)] * 40
    law, m = _law(seq=seq, nav_fail_after_s=5.0, target=(0.0, 0.0))
    for t in np.arange(m.deploy_time, m.deploy_time + 20.0, 1.0):
        law.sample(float(t), np.zeros(15), None)
    assert law.nav_lost_time is not None
    phis = law.log["phi_deg"]
    i = law.log["note"].index("nav_lost")
    assert all(abs(p - phis[i]) < 1e-9 for p in phis[i:])


def test_inhibit_fires_beyond_the_threshold_and_latches():
    law, m = _law(seq=[(5000.0, 0.0)] * 40, inhibit_threshold_m=300.0,
                  target=(0.0, 0.0))
    for t in np.arange(m.deploy_time, m.deploy_time + 15.0, 1.0):
        law.sample(float(t), np.zeros(15), None)
    assert law.inhibited
    assert law.command(0.0)[1] is False
    assert law.log["note"][-1] == "inhibited"


def test_authority_monitor_fires_when_nothing_moves():
    """
    The stage-2 mechanical failure. The law holds, the map says the predicted
    impact should have moved by tens of metres, and it does not move at all.
    """
    law, m = _law(seq=[(400.0, 0.0)] * 60, target=(0.0, 0.0),
                  monitor_window_s=6.0)
    for t in np.arange(m.deploy_time, m.deploy_time + 30.0, 1.0):
        law.sample(float(t), np.zeros(15), None)
    assert law.authority_ok is False
    assert law.authority_fault_time is not None
    assert law.command(0.0)[1] is False


def test_authority_monitor_does_not_fire_when_the_correction_works():
    """
    The other half, and the one that matters: a healthy round must not be
    declared faulty. The predicted impact walks towards the target at the rate
    the map predicts.
    """
    m = synthetic_map()
    cfg = gd.GuidanceConfig()
    tof = m.impact_time
    seq, phi = [], math.radians(0.0)
    # Build a prediction sequence that moves exactly as the map says it will.
    t0 = m.deploy_time
    for k in range(60):
        t = t0 + k
        A, b = m.increment(tof - t0, max(tof - t, 0.0))
        w = np.array([math.cos(phi), math.sin(phi)])
        d = A @ w + b
        seq.append((400.0 + d[0], d[1]))
    p = _StubPredictor(seq, tof)
    law = gd.GuidanceLaw(cfg, m, p, sch.make_scheduler("budget"), (0.0, 0.0), t0)
    law._phi = phi
    for k in range(40):
        law.sample(t0 + k, np.zeros(15), None)
        law._phi = phi          # pin the command so the stub stays consistent
    assert law.authority_ok is True


def test_monitor_does_not_punish_a_law_for_converging():
    """
    A round whose miss is SMALLER than what holding promises will null it and
    then stop moving, because there is nothing left to remove. The first
    version of the monitor compared the observed movement against the map's
    promise alone and faulted 4 healthy rounds in 64 at the adopted engagement
    for exactly that. The expectation must be the smaller of the promise and
    the miss.
    """
    m = synthetic_map()
    cfg = gd.GuidanceConfig(monitor_window_s=6.0, monitor_floor_m=20.0)
    # Target 40 m away; the round removes it over four cycles and then sits.
    seq = [(40.0, 0.0), (30.0, 0.0), (20.0, 0.0), (10.0, 0.0)] + \
          [(0.0, 0.0)] * 60
    p = _StubPredictor(seq, m.impact_time)
    law = gd.GuidanceLaw(cfg, m, p, sch.make_scheduler("budget"), (0.0, 0.0),
                         m.deploy_time)
    for k in range(30):
        law.sample(m.deploy_time + k, np.zeros(15), None)
    assert law.authority_ok is True


def test_monitor_still_fires_on_a_large_unremoved_miss():
    """The other half: a big miss that does not move at all is a dead actuator."""
    m = synthetic_map()
    cfg = gd.GuidanceConfig(monitor_window_s=6.0, monitor_floor_m=20.0)
    p = _StubPredictor([(400.0, 0.0)] * 60, m.impact_time)
    law = gd.GuidanceLaw(cfg, m, p, sch.make_scheduler("budget"), (0.0, 0.0),
                         m.deploy_time)
    for k in range(30):
        law.sample(m.deploy_time + k, np.zeros(15), None)
    assert law.authority_ok is False


def test_monitor_is_skipped_below_its_resolution_floor():
    """
    Near impact the map predicts almost no movement, and a test on it would be
    a test on the prediction error. The monitor must decline to run there.
    """
    m = synthetic_map()
    cfg = gd.GuidanceConfig(monitor_floor_m=1e9)     # nothing is resolvable
    p = _StubPredictor([(400.0, 0.0)] * 60, m.impact_time)
    law = gd.GuidanceLaw(cfg, m, p, sch.make_scheduler("budget"), (0.0, 0.0),
                         m.deploy_time)
    for k in range(40):
        law.sample(m.deploy_time + k, np.zeros(15), None)
    assert law.authority_ok is True


def test_guidance_command_is_a_function_of_time_alone():
    """
    The step-2.5 restriction on the control seam is not weakened: `command`
    must return the latched value whatever time it is given, so that RK4's
    four evaluations inside one step all see the same command.
    """
    law, m = _law(seq=[(100.0, 0.0)] * 10)
    law.sample(m.deploy_time + 4.0, np.zeros(15), None)
    a = [law.command(t) for t in (0.0, 1.0, 1e6, -5.0)]
    assert len(set(a)) == 1


def test_guidance_samples_on_its_own_grid():
    """1 Hz means 1 Hz, whatever rate the integrator calls the hook at."""
    law, m = _law(seq=[(100.0, 0.0)] * 200)
    t = m.deploy_time
    for _ in range(20000):
        law.sample(t, np.zeros(15), None)
        t += 5.0e-4
    assert law.samples == pytest.approx(10, abs=1)
