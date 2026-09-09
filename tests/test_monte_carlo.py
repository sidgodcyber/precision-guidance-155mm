"""
Tests for the step-6 dispersion campaign.

Two things are asserted here that no other test in the project asserts, and
both of them are the sampling discipline rather than the physics:

  1. NOTHING IS REUSED ACROSS ROUNDS. Every per-round variable differs between
     rounds, including the sensor seed and the deployment phase. Step 4.5 and
     step 5.5 each lost a measurement to a violation of this and neither had a
     test that would have caught it.
  2. THE PAIRING HOLDS. The same round index is the same round under every
     configuration and every met message age, so a paired comparison really is
     paired. The first version of `sample_round` failed this and the failure
     was silent.

The physics changes step 6 made to `sim` and `models` are asserted to be
IDENTITIES when they are switched off, so that every number measured in steps
1 to 5.5 is still reproducible.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim import atmosphere as atm
from sim import aerodata, canards as cn, dynamics as dyn
from sim import integrate as ig, projectile as pr
from models import mpmm


# ===========================================================================
# The realised atmosphere
# ===========================================================================
def test_the_standard_profile_is_the_isa_exactly():
    """
    A MetProfile that says it is standard must return `isa_scalars` bit for
    bit, or every number measured before step 6 becomes irreproducible by a
    rounding error.
    """
    s = atm.MetProfile.standard()
    assert s.is_standard
    for h in (0.0, 1.0, 1234.5, 5000.0, 11000.0, 15999.0, 20000.0):
        assert s.scalars(h) == atm.isa_scalars(h)
        assert s.wind(h) == (0.0, 0.0, 0.0)


def test_a_realised_profile_is_thermodynamically_consistent():
    """p = rho R T at every altitude, however density and temperature moved."""
    p = atm.MetProfile.sample(np.random.default_rng(4))
    for h in np.linspace(0.0, 16000.0, 40):
        t, pres, rho, a = p.scalars(float(h))
        assert pres == pytest.approx(rho * atm.R_AIR * t, rel=1e-12)
        assert a == pytest.approx(math.sqrt(atm.GAMMA * atm.R_AIR * t),
                                  rel=1e-12)


def test_profile_sigmas_come_out_as_specified():
    """
    The generator must actually deliver the sigma schedule it is given -- the
    AR(1) recursion is easy to get wrong in a way that quietly halves the
    variance.
    """
    rng = np.random.default_rng(11)
    ps = [atm.MetProfile.sample(rng) for _ in range(400)]
    g = list(atm.MET_GRID)
    for h in (0.0, 6000.0, 12000.0):
        i = g.index(h)
        got = np.std([p.wind_north[i] for p in ps], ddof=1)
        want = atm._sigma_at(atm.WIND_SIGMA_ANCHORS, h)
        assert got == pytest.approx(want, rel=0.12), (h, got, want)


def test_profile_vertical_correlation_is_the_stated_length():
    rng = np.random.default_rng(12)
    ps = [atm.MetProfile.sample(rng) for _ in range(600)]
    g = list(atm.MET_GRID)
    i = g.index(4000.0)
    j = g.index(6000.0)
    a = np.array([p.wind_north[i] for p in ps])
    b = np.array([p.wind_north[j] for p in ps])
    got = float(np.corrcoef(a, b)[0, 1])
    want = math.exp(-(g[j] - g[i]) / atm.WIND_CORRELATION_M)
    assert got == pytest.approx(want, abs=0.08), (got, want)


def test_a_scaled_profile_scales():
    """`scale` must multiply every sigma, so results can be reported per unit
    sigma. scale=0 must return the standard atmosphere exactly."""
    assert atm.MetProfile.sample(np.random.default_rng(1), scale=0.0).is_standard
    rng = np.random.default_rng(13)
    a = [atm.MetProfile.sample(rng, scale=1.0) for _ in range(300)]
    rng = np.random.default_rng(13)
    b = [atm.MetProfile.sample(rng, scale=2.0) for _ in range(300)]
    i = list(atm.MET_GRID).index(8000.0)
    sa = np.std([p.wind_north[i] for p in a], ddof=1)
    sb = np.std([p.wind_north[i] for p in b], ddof=1)
    assert sb / sa == pytest.approx(2.0, rel=0.02)


def test_a_stale_message_decorrelates_at_the_stated_rate():
    """
    The message must be correlated exp(-dt/tau) with the truth and must carry
    the SAME marginal variance as the truth, not a shrunk one. A message
    modelled as a regression estimate rather than as a measurement of the past
    atmosphere would be less variable than the air, and it would flatter the
    fire-control solution.
    """
    i = list(atm.MET_GRID).index(8000.0)
    rng = np.random.default_rng(14)
    for age in (1.0, 6.0):
        truth, msg = [], []
        for _ in range(500):
            t = atm.MetProfile.sample(rng)
            m = t.message(rng, age)
            truth.append(t.wind_north[i])
            msg.append(m.wind_north[i])
        rho = float(np.corrcoef(truth, msg)[0, 1])
        want = math.exp(-age / atm.DECORRELATION_HOURS["wind"])
        # The fresh-message error decorrelates a little further, so the
        # observed correlation is at or below the pure staling value.
        assert rho <= want + 0.06
        assert rho == pytest.approx(want, abs=0.16), (age, rho, want)
        assert np.std(msg, ddof=1) >= 0.85 * np.std(truth, ddof=1)


def test_a_message_with_no_age_is_the_standard_atmosphere():
    p = atm.MetProfile.sample(np.random.default_rng(2))
    assert p.message(np.random.default_rng(3), None).is_standard


# ===========================================================================
# The atmosphere hook in the two models
# ===========================================================================
def _launch():
    return pr.LaunchConditions.from_mils(684.0, 525.3)


def test_the_flight_model_atmosphere_hook_is_an_identity_when_standard():
    """
    Passing the standard profile explicitly must give the same trajectory as
    passing nothing, to the last bit. This is what keeps every step-1 to
    step-5.5 number reproducible after the hook was added.
    """
    base = dyn.FlightModel(projectile=pr.M107, aero=aerodata.make_m107_table())
    hooked = dyn.FlightModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                             atmosphere=atm.STANDARD_MET.scalars)
    y0 = dyn.initial_state(pr.M107, _launch())
    # The project's own step. A coarser one does not integrate a 1308 rad/s
    # spin and the comparison would be between two divergences.
    a = ig.integrate(y0, base, dt=5e-4, log_every=10 ** 9, t_max=8.0,
                     stop_on_impact=False)
    b = ig.integrate(y0, hooked, dt=5e-4, log_every=10 ** 9, t_max=8.0,
                     stop_on_impact=False)
    assert np.allclose(a.impact_state, b.impact_state, rtol=0, atol=0)


def test_the_mpmm_atmosphere_hook_is_an_identity_when_standard():
    env = pr.Environment.from_degrees(45.0, include_coriolis=False)
    base = mpmm.MpmmModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                          environment=env)
    hooked = mpmm.MpmmModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                            environment=env,
                            atmosphere=atm.STANDARD_MET.scalars)
    y0 = mpmm.initial_state(pr.M107, _launch())
    a = mpmm.propagate_to_impact(y0, base, dt=0.1)
    b = mpmm.propagate_to_impact(y0, hooked, dt=0.1)
    assert a.range_m == pytest.approx(b.range_m, abs=1e-9)
    assert a.drift_m == pytest.approx(b.drift_m, abs=1e-9)


def test_a_denser_atmosphere_shortens_the_range():
    """The sign of the density perturbation, checked against physics rather
    than against itself."""
    n = len(atm.MET_GRID)
    dense = atm.MetProfile(atm.MET_GRID, (0.0,) * n, (0.0,) * n,
                           (1.05,) * n, (1.0,) * n, label="+5 % density")
    env = pr.Environment.from_degrees(45.0, include_coriolis=False)
    y0 = mpmm.initial_state(pr.M107, _launch())
    a = mpmm.propagate_to_impact(
        y0, mpmm.MpmmModel(projectile=pr.M107,
                           aero=aerodata.make_m107_table(), environment=env),
        dt=0.1)
    b = mpmm.propagate_to_impact(
        y0, mpmm.MpmmModel(projectile=pr.M107,
                           aero=aerodata.make_m107_table(), environment=env,
                           atmosphere=dense.scalars), dt=0.1)
    assert b.range_m < a.range_m - 50.0


def test_a_tail_wind_lengthens_the_range():
    """
    Wind is the VELOCITY OF THE AIR, so a POSITIVE north component on a
    northward shot is a tail wind and must add range. Getting this backwards
    would put every met correction the wrong way round, and the sign is the
    one thing in this module that a plausible-looking bug would hide.
    """
    n = len(atm.MET_GRID)
    tail = atm.MetProfile(atm.MET_GRID, (10.0,) * n, (0.0,) * n,
                          (1.0,) * n, (1.0,) * n, label="10 m/s tail wind")
    env = pr.Environment.from_degrees(45.0, include_coriolis=False)
    y0 = mpmm.initial_state(pr.M107, _launch())
    a = mpmm.propagate_to_impact(
        y0, mpmm.MpmmModel(projectile=pr.M107,
                           aero=aerodata.make_m107_table(), environment=env),
        dt=0.1)
    b = mpmm.propagate_to_impact(
        y0, mpmm.MpmmModel(projectile=pr.M107,
                           aero=aerodata.make_m107_table(), environment=env,
                           wind=tail.wind, atmosphere=tail.scalars), dt=0.1)
    assert b.range_m > a.range_m + 50.0


# ===========================================================================
# The canard C_Lalpha knob
# ===========================================================================
def test_cla_scale_is_an_identity_at_one_and_scales_the_force():
    from sim.canards import CanardGeometry
    from dataclasses import replace
    g = CanardGeometry(station_from_nose=0.025,
                       steering_deflection=math.radians(3.0))
    assert cn.CanardModel(geometry=g, projectile=pr.M107)._cla_scale == 1.0
    m13 = cn.CanardModel(geometry=replace(g, cla_scale=1.3),
                         projectile=pr.M107)
    assert m13._cla_scale == pytest.approx(1.3)


# ===========================================================================
# The sampling discipline -- the point of the whole module
# ===========================================================================
@pytest.fixture(scope="module")
def ctx():
    from analysis import guidance_cep as gc
    from analysis import monte_carlo as mc
    return mc.context(gc.load_maps(), "long")


def test_nothing_is_reused_across_rounds(ctx):
    """
    Every per-round variable must differ between rounds. This is the test step
    4.5 and step 5.5 each did not have.
    """
    from analysis import monte_carlo as mc
    ds = [mc.sample_round(i, ctx["base"]) for i in range(24)]
    assert len({d["seed"] for d in ds}) == 24
    assert len({d["dmv"] for d in ds}) == 24
    assert len({d["deploy_time"] for d in ds}) == 24
    assert len({d["projectile_overrides"]["mass"] for d in ds}) == 24
    assert len({d["met"].wind(6000.0) for d in ds}) == 24


def test_the_deployment_phase_is_uniform_on_the_circle(ctx):
    """
    The fuze setting increment is twenty roll periods, so the roll angle the
    kit deploys at must be uniform -- not clustered, and not a grid. Step 4.5
    measured a 35-point spread of retained authority across this phase.
    """
    from analysis import monte_carlo as mc
    ds = [mc.sample_round(i, ctx["base"]) for i in range(200)]
    # Spin at deployment is about 1308 rad/s; the phase is the fractional part
    # of the setting divided by the roll period.
    period = 2.0 * math.pi / 1308.0
    phase = np.array([(d["deploy_time"] % period) / period for d in ds])
    assert phase.min() < 0.06 and phase.max() > 0.94
    # A chi-squared uniformity check over eight bins, at 1 % significance.
    counts, _ = np.histogram(phase, bins=8, range=(0.0, 1.0))
    chi2 = ((counts - len(ds) / 8.0) ** 2 / (len(ds) / 8.0)).sum()
    assert chi2 < 18.5, (counts, chi2)


def test_pairing_is_invariant_to_the_met_message(ctx):
    """
    THE TEST THE FIRST VERSION OF `sample_round` FAILED.

    Round 37 must be the same round -- same muzzle velocity, same shell, same
    fuze offset, same sensor seed, same air -- at every met message age. Only
    the message and what fire control does with it may change. Without this,
    Task C's paired comparison is silently an unpaired comparison of two
    independent samples.
    """
    from analysis import monte_carlo as mc
    ages = ("perfect", 0.0, 1.0, 3.0, 6.0, None)
    ds = {a: mc.sample_round(37, ctx["base"], met_age_hours=a) for a in ages}
    ref = ds["perfect"]
    for a, d in ds.items():
        assert d["dmv"] == ref["dmv"], a
        assert d["seed"] == ref["seed"], a
        assert d["projectile_overrides"] == ref["projectile_overrides"], a
        assert d["_draw_scalars"]["fuze_offset_s"] == \
            ref["_draw_scalars"]["fuze_offset_s"], a
        assert d["met"].wind(7000.0) == ref["met"].wind(7000.0), a
    # and the message really does change
    assert ds[6.0]["met_message"].wind(7000.0) != ref["met"].wind(7000.0)
    assert ds[None]["met_message"].is_standard


def test_pairing_is_invariant_to_the_configuration(ctx):
    """
    The draw must not depend on how many configurations have already been
    flown, which is what makes common random numbers work without carrying
    state between campaigns.
    """
    from analysis import monte_carlo as mc
    a = mc.sample_round(9, ctx["base"])
    _ = [mc.sample_round(i, ctx["base"]) for i in range(50)]
    b = mc.sample_round(9, ctx["base"])
    assert a["dmv"] == b["dmv"] and a["deploy_time"] == b["deploy_time"]


def test_a_perfect_message_lands_the_lay_on_the_aim_point(ctx):
    """
    Fire control's job, checked: laid on a message that IS the atmosphere, the
    reduced-order model must land on the aim point. If it does not, every
    atmospheric result is measuring the solver rather than the message.
    """
    from analysis import monte_carlo as mc
    rng = np.random.default_rng(21)
    for _ in range(3):
        met = atm.MetProfile.sample(rng)
        sol = mc.lay_gun(ctx["base"], met)
        r, d = mc._mpmm_impact(ctx["base"], sol["dqe_mils"], sol["daz"], met)
        assert r == pytest.approx(ctx["base"]["uncorrected_range"], abs=1.0)
        assert d == pytest.approx(ctx["base"]["uncorrected_drift"], abs=1.0)


def test_the_nominal_lay_is_exactly_nominal(ctx):
    from analysis import monte_carlo as mc
    sol = mc.lay_gun(ctx["base"], None)
    assert sol["dqe_mils"] == 0.0 and sol["daz"] == 0.0


# ===========================================================================
# The statistics
# ===========================================================================
def test_the_bootstrap_interval_covers():
    """
    A confidence interval that does not cover is worse than none. Drawn from a
    known distribution, the nominal 95 % interval must contain the true median
    about 95 % of the time.
    """
    from analysis import monte_carlo as mc
    rng = np.random.default_rng(77)
    true = float(np.median(np.abs(rng.standard_normal(2_000_000))))
    hits = 0
    trials = 300
    for _ in range(trials):
        v = np.abs(rng.standard_normal(128))
        b = mc.bootstrap_cep(v, n_boot=600, seed=int(rng.integers(1 << 30)))
        hits += b["cep_lo_m"] <= true <= b["cep_hi_m"]
    assert 0.88 <= hits / trials <= 1.0, hits / trials


def test_a_paired_delta_resolves_better_than_two_absolutes():
    """
    The claim the comparisons rest on: pairing removes the common dispersion.
    A configuration that adds a small constant must be resolved by the paired
    difference and NOT by differencing two independently quoted CEPs.
    """
    from analysis import monte_carlo as mc
    rng = np.random.default_rng(5)
    common = np.abs(rng.standard_normal(160)) * 50.0
    a = [{"draw": i, "miss_m": float(v + 3.0), "miss_range_m": 0.0,
          "miss_defl_m": 0.0} for i, v in enumerate(common)]
    b = [{"draw": i, "miss_m": float(v), "miss_range_m": 0.0,
          "miss_defl_m": 0.0} for i, v in enumerate(common)]
    d = mc.paired_delta(a, b)
    assert d["delta_cep_m"] == pytest.approx(3.0, abs=0.6)
    # and the interval excludes zero, which two absolute CEPs at n=160 could
    # not do for a 3 m difference on a 34 m CEP
    assert d["delta_cep_lo_m"] > 0.0
    assert mc.bootstrap_cep(np.array([r["miss_m"] for r in a]))["cep_se_m"] \
        > 5.0 * d["delta_cep_se_m"]


def test_convergence_reports_failure_rather_than_asserting_success():
    """`stable_at` must be able to say no."""
    from analysis import monte_carlo as mc
    drifting = [{"n": n, "cep_m": 10.0 + n / 40.0,
                 "cep_lo_m": 9.0 + n / 40.0, "cep_hi_m": 11.0 + n / 40.0}
                for n in (16, 32, 64, 128, 256)]
    assert mc.stable_at(drifting)["converged"] is False
    flat = [{"n": n, "cep_m": 10.0, "cep_lo_m": 9.0, "cep_hi_m": 11.0}
            for n in (16, 32, 64, 128, 256)]
    assert mc.stable_at(flat)["converged"] is True


# ===========================================================================
# The monitor shadow
# ===========================================================================
def test_the_monitor_shadow_changes_nothing_about_the_flight():
    """
    Shadow mode must record the test statistic and never act on it, or the
    false-alarm rate it measures is the rate of a monitor that was interfering
    with the round it was measuring.
    """
    from analysis import guidance_cep as gc
    from analysis import nav_common as nc
    m = gc.load_maps()
    c = gc.engagement_context(m, "long")
    opts = gc.scheduler_options(m, "long")["proportional"]
    mon = {k: v for k, v in c["monitor"].items() if not k.startswith("_")}
    args = {"baseline": c["base"], "map": c["map"], **mon, "draw": 0,
            "scheduler": "proportional", "scheduler_opts": opts,
            "use_nav": False, "dmv": 4.0}
    off = nc.run_guided_nav({**args, "authority_monitor": False})
    shadow = nc.run_guided_nav({**args, "authority_monitor": False,
                                "monitor_shadow": True})
    assert shadow["miss_m"] == pytest.approx(off["miss_m"], abs=1e-9)
    assert shadow["g_authority_ok"] is True
    assert len(shadow["monitor_log"]) > 0


def test_would_fault_replays_the_recorded_statistic():
    from analysis import monte_carlo as mc
    log = [{"t": 10.0, "expected_m": 50.0, "promise_m": 60.0, "along_m": 40.0},
           {"t": 16.0, "expected_m": 50.0, "promise_m": 60.0, "along_m": 5.0}]
    # nothing fires while the observed movement clears the fraction
    assert mc.would_fault(log, 0.5, 20.0) == 16.0
    assert mc.would_fault(log, 0.05, 20.0) is None
    # and the floor suppresses the test entirely
    assert mc.would_fault(log, 0.5, 80.0) is None


# ===========================================================================
# The saturation measurement that replaced an artefact
# ===========================================================================
def test_saturation_is_read_from_the_scheduler_note_not_from_a_plan():
    """
    Until step 6, `guidance_cep.cep_of` read saturation off a `plan` field
    only two of the five schedulers write, so every proportional run reported
    0.0 whatever it did. The replacement must report a real fraction for a
    proportional round that saturates.
    """
    from analysis import guidance_cep as gc
    m = gc.load_maps()
    c = gc.engagement_context(m, "long")
    opts = gc.scheduler_options(m, "long")["proportional"]
    # A draw far outside the reachable set: it must hold to impact.
    r = gc.run_guided({"baseline": c["base"], "map": c["map"], "draw": 0,
                       "scheduler": "proportional", "scheduler_opts": opts,
                       "dmv": -12.0})
    assert r["g_plan"] is None, "proportional still writes no plan"
    assert r["g_armed_cycles"] > 0
    assert r["g_ever_saturated"] is True
    assert r["g_saturated_fraction"] > 0.0
    s = gc.cep_of([r])
    assert s["authority_limited_ever"] == 1.0
    assert not math.isnan(s["saturated_cycle_fraction"])


def test_cep_of_reports_nan_for_rows_that_predate_the_measurement():
    """
    Old campaign JSON has no saturation fields. It must come back as nan --
    a missing measurement -- and never as 0.0, which is what the artefact
    reported and what a reader would take for a measured absence.
    """
    from analysis import guidance_cep as gc
    rows = [{"miss_m": 10.0, "miss_range_m": 1.0, "miss_defl_m": 2.0}
            for _ in range(8)]
    s = gc.cep_of(rows)
    assert math.isnan(s["authority_limited_ever"])
    assert math.isnan(s["authority_limited_terminal"])



# ===========================================================================
# Task C's tag level
# ===========================================================================
# Until step 6's residual work Task C dispatched to `out["c"][engagement]`
# with no tag level, so `--tag` was accepted by the parser and silently
# ignored, and a second campaign at the same engagement replaced the first in
# place. These four tests are the mechanism that keeps it fixed. They assert
# STRUCTURE, not physics: none of them flies anything.
def _stub_campaign(sigma_range, inflate=None, n=8):
    """The smallest thing shaped like a Task C campaign."""
    return {
        "label": "long", "n": n, "met_scale": 1.0, "use_nav": False,
        "inflate": inflate,
        "ages": {
            "perfect": {"age_hours": "perfect", "n": n, "cep_m": 1.0,
                        "cep_lo_m": 0.5, "cep_hi_m": 2.0, "sd_range_m": 90.0,
                        "sd_defl_m": 20.0, "bias_range_m": -1.0},
            "2h": {"age_hours": 2.0, "n": n, "cep_m": 88.0,
                   "cep_lo_m": 69.0, "cep_hi_m": 117.0, "sd_range_m": 190.0,
                   "sd_defl_m": 42.0, "bias_range_m": -4.0,
                   "knowledge_term": {"n": n, "sigma_range_m": sigma_range,
                                      "sigma_defl_m": 36.8,
                                      "bias_range_m": 7.8}},
        },
    }


def test_task_c_is_tag_keyed(tmp_path, monkeypatch):
    """
    The dispatch writes under `d["c"][tag][engagement]`, and two tags coexist
    rather than one replacing the other.

    `task_c` is stubbed, so this exercises the dispatch and nothing else.
    """
    import json
    from analysis import monte_carlo as mc

    calls = []

    def fake_task_c(pool, mapdata, label, n, met_scale, **kw):
        calls.append((label, n, kw.get("inflate")))
        return _stub_campaign(100.0 + len(calls))

    monkeypatch.setattr(mc, "task_c", fake_task_c)
    out = tmp_path / "campaign.json"
    common = ["--tasks", "c", "--no-inflate", "--engagement", "long",
              "--n", "8", "--workers", "1", "--out", str(out)]

    assert mc.main(common + ["--tag", "headline"]) == 0
    assert mc.main(common + ["--tag", "physical"]) == 0

    d = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(d["c"]) == ["headline", "physical"], d["c"].keys()
    assert list(d["c"]["headline"]) == ["long"]
    assert list(d["c"]["physical"]) == ["long"]
    # the second run did not disturb the first
    assert (d["c"]["headline"]["long"]["ages"]["2h"]["knowledge_term"]
            ["sigma_range_m"]) == 101.0
    assert (d["c"]["physical"]["long"]["ages"]["2h"]["knowledge_term"]
            ["sigma_range_m"]) == 102.0
    assert len(calls) == 2


def test_the_migration_is_idempotent(tmp_path):
    """Twice equals once, and no value changes."""
    import json
    from analysis import migrate_c_tag as mig

    flat = {"config": {"x": 1},
            "c": {"long": _stub_campaign(148.04009757790277),
                  "middle": _stub_campaign(60.5)},
            "a": {"headline": {"long": {"guided": {"cep_m": 106.19}}}}}
    p = tmp_path / "campaign.json"
    p.write_text(json.dumps(flat, indent=1), encoding="utf-8")

    first = mig.migrate(str(p), backup=False)
    assert first["action"] == "migrated"
    after_one = json.loads(p.read_text(encoding="utf-8"))

    second = mig.migrate(str(p), backup=False)
    assert second["action"] == "none" and second["reason"] == "already tagged"
    after_two = json.loads(p.read_text(encoding="utf-8"))

    assert after_one == after_two
    # values moved, not rewritten
    assert after_one["c"]["headline"] == flat["c"]
    assert after_one["a"] == flat["a"] and after_one["config"] == flat["config"]
    assert (after_one["c"]["headline"]["long"]["ages"]["2h"]
            ["knowledge_term"]["sigma_range_m"]) == 148.04009757790277


def test_the_migration_refuses_a_mixed_shape(tmp_path):
    """Half-migrated is not a shape to guess at."""
    import json
    import pytest as _pytest
    from analysis import migrate_c_tag as mig

    p = tmp_path / "campaign.json"
    p.write_text(json.dumps(
        {"c": {"long": _stub_campaign(1.0),
               "headline": {"middle": _stub_campaign(2.0)}}}), encoding="utf-8")
    with _pytest.raises(SystemExit):
        mig.migrate(str(p), backup=False)


def test_the_migration_guards_the_published_number(tmp_path):
    """
    docs/CEP-FINAL.md rests on the 148 m term. If it does not read back after
    the move, the migration stops rather than writing.
    """
    import json
    import pytest as _pytest
    from analysis import migrate_c_tag as mig

    p = tmp_path / "campaign.json"
    p.write_text(json.dumps({"c": {"long": _stub_campaign(99.0)}}),
                 encoding="utf-8")
    with _pytest.raises(SystemExit):
        mig.migrate(str(p), backup=False)
    # and it did not write
    assert "headline" not in json.loads(p.read_text(encoding="utf-8"))["c"]


def test_the_report_defaults_to_headline():
    """
    With two tags present, the atmospheric table and the budget draw
    `headline`. The other tag appears only in a section that names it.
    """
    from analysis import monte_carlo_report as rep

    d = {"c": {"headline": {"long": _stub_campaign(148.0, inflate={
                    "sigma_mv": 7.895, "sigma_az": 0.00286})},
               "physical": {"long": _stub_campaign(77.7, inflate=None)}},
         "a": {"headline": {"long": {
             "guided": {"n": 8, "cep_m": 106.19, "cep_lo_m": 91.5,
                        "cep_hi_m": 130.1, "sd_range_m": 193.9,
                        "sd_defl_m": 49.8},
             "unguided": {"sigma_range_m": 273.6, "sigma_defl_m": 83.0}}}}}

    atm = "\n".join(rep.atmospheric(d))
    assert "148.00" in atm
    assert "77.70" in atm, "the other tag should still be rendered"
    # but only under a heading that names it, and after the published table
    assert atm.index("148.00") < atm.index("77.70")
    assert "tag `physical`" in atm
    assert "no dispersion top-up" in atm
    assert "not** the published" in atm

    bud = "\n".join(rep.budget(d))
    assert "148.0" in bud
    assert "77.7" not in bud, "the budget must not draw the control campaign"


def test_the_report_raises_on_a_missing_headline_tag():
    """
    No silent fallback. A campaign with only a control tag must raise rather
    than render the control under the published campaign's name -- the same
    failure `test_cep_of_reports_nan_for_rows_that_predate_the_measurement`
    prevents one level down.
    """
    import pytest as _pytest
    from analysis import monte_carlo_report as rep
    from analysis import monte_carlo_figures as figs

    d = {"c": {"physical": {"long": _stub_campaign(77.7)}}}
    with _pytest.raises(KeyError):
        rep.task_c_of(d)
    with _pytest.raises(KeyError):
        rep.atmospheric(d)
    with _pytest.raises(KeyError):
        figs._pick_tag(d["c"])
    # and no Task C at all is not an error, it is "not run yet"
    assert rep.task_c_of({}) == {}
    assert rep.atmospheric({}) == []


def test_pick_tag_is_not_pick():
    """
    `_pick` chooses an engagement and falls back; `_pick_tag` chooses a tag and
    does not. Conflating them is how the defect arose: `_pick` handed a
    tag-keyed dict walks past every engagement name and returns the first tag.
    """
    from analysis import monte_carlo_figures as figs

    tagged = {"physical": {"long": {}}, "headline": {"long": {}}}
    assert figs._pick(tagged) == "physical", (
        "_pick on a tag-keyed dict returns a TAG under an engagement's name; "
        "this is the bug _pick_tag exists to stop")
    assert figs._pick_tag(tagged) is tagged["headline"]
