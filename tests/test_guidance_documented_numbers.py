"""
Every number quoted in the step-3 prose against the JSON it came from.

The tables in `docs/guidance_tables.md` are generated, so they cannot drift.
The prose in `docs/INVERSE-MAP.md`, `docs/GUIDANCE-DESIGN.md`,
`docs/AIM-OFF.md`, `docs/DEGRADATION-LADDER.md` and
`docs/CEP-CLOSED-LOOP.md` is written by hand, and this is what stops it
drifting: it is a transcription of the figures those documents assert, checked
against `docs/guidance_map.json` and `docs/guidance_cep.json`.

If a campaign is re-run and a number moves, this fails and names it, and the
document has to be edited rather than quietly contradicted.

Skipped when the campaign outputs are absent, so a fresh clone still passes.
"""

from __future__ import annotations

import json
import os

import pytest

MAP = "docs/guidance_map.json"
CEP = "docs/guidance_cep.json"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(MAP) and os.path.exists(CEP)),
    reason="step-3 campaign outputs not present; run analysis.guidance_authority "
           "and analysis.guidance_cep")


@pytest.fixture(scope="module")
def data():
    with open(MAP) as fh:
        m = json.load(fh)
    with open(CEP) as fh:
        c = json.load(fh)
    return m, c


def close(got, want, tol=0.02):
    return abs(got - want) <= tol * max(abs(want), 1e-9)


# ===========================================================================
# docs/INVERSE-MAP.md
# ===========================================================================
def test_inverse_map_headline_numbers(data):
    m, _ = data
    e = m["engagements"]["long"]
    f = e["full_hold_ellipse"]
    assert close(f["semi_major_m"], 184.0, 0.005)
    assert close(f["semi_minor_m"], 166.8, 0.005)
    assert close(f["axis_ratio"], 1.10, 0.01)
    assert close(e["nodes"][-1]["residual"], 0.020, 0.06)
    assert close(100 * e["retained_rms"], 73.2, 0.01)
    assert close(e["ideal"]["ellipse"]["semi_major_m"], 260.3, 0.005)
    # 20 nodes x 6 floats = 120 floats = 480 bytes single precision
    assert 6 * len(e["nodes"]) == 120


def test_inverse_map_pointing_residual(data):
    m, _ = data
    s = m["engagements"]["long"]["pointing_validation"]["summary"]
    assert close(s["1.0"]["pointing_rms_deg"], 0.84, 0.02)
    assert close(s["1.0"]["pointing_max_deg"], 1.61, 0.02)
    assert close(s["0.55"]["pointing_rms_deg"], 0.78, 0.02)
    assert close(s["1.0"]["cross_rms_m"], 2.58, 0.02)
    assert close(100 * s["1.0"]["magnitude_rms_frac"], 1.3, 0.05)
    short = m["engagements"]["short"]["pointing_validation"]["summary"]
    assert close(short["1.0"]["pointing_rms_deg"], 4.57, 0.02)


def test_inverse_map_rotation_and_reach_bands(data):
    m, _ = data
    p = m["engagements"]["long"]["pointing"]
    rot = [x["rotation_deg"] for x in p]
    reach = [x["reach_m"] for x in p]
    assert close(min(rot), -160.2, 0.01) and close(max(rot), -154.7, 0.01)
    assert close(min(reach), 166.9, 0.01) and close(max(reach), 183.9, 0.01)


def test_mach_is_double_valued_over_the_guided_phase(data):
    """The reason the map is indexed on time to go. INVERSE-MAP.md section 4."""
    m, _ = data
    mach = m["engagements"]["long"]["schedule_mach"]
    assert close(mach[0], 1.521, 0.005)
    assert close(min(mach), 0.864, 0.005)
    assert close(mach[-1], 0.917, 0.005)
    # and it really is double-valued: the minimum is interior
    assert 0 < mach.index(min(mach)) < len(mach) - 1


# ===========================================================================
# docs/GUIDANCE-DESIGN.md
# ===========================================================================
@pytest.mark.parametrize("law,cep", [("proportional", 36.3), ("deadband", 39.4),
                                     ("budget", 36.3), ("single_shot", 47.9),
                                     ("isotropic", 43.0)])
def test_task_c_ceps(data, law, cep):
    _, c = data
    assert close(c["task_c"]["schedulers"][law]["cep"]["cep_m"], cep, 0.01)


def test_task_c_paired_differences(data):
    _, c = data
    p = c["task_c"]["paired_vs_best"]
    assert close(p["single_shot"]["median_delta_m"], 13.0, 0.02)
    assert close(p["isotropic"]["median_delta_m"], 1.72, 0.02)
    # The budget law is IDENTICAL to the best one -- the perishable-budget
    # finding of GUIDANCE-DESIGN.md section 6.1.
    assert p["budget"]["median_delta_m"] == 0.0
    assert p["budget"]["se_median_m"] == 0.0


def test_every_round_holds_to_impact_at_the_unreachable_dispersion(data):
    _, c = data
    assert c["task_c"]["never_released"] == 1.0
    assert c["task_c2"]["never_released"] == 1.0


@pytest.mark.parametrize("law,cep", [("proportional", 0.4), ("deadband", 17.8),
                                     ("budget", 0.4), ("single_shot", 36.3),
                                     ("isotropic", 1.4)])
def test_task_c2_ceps(data, law, cep):
    _, c = data
    assert close(c["task_c2"]["schedulers"][law]["cep"]["cep_m"], cep, 0.06)


def test_guidance_rate(data):
    _, c = data
    r = c["task_r"]["rates"]
    assert close(r["0.5"]["paired_vs_1hz"]["median_delta_m"], -0.03, 0.5)
    assert close(r["2.0"]["paired_vs_1hz"]["median_delta_m"], 0.08, 0.5)
    assert close(r["5.0"]["paired_vs_1hz"]["median_delta_m"], 1.633, 0.02)
    assert close(100 * r["5.0"]["paired_vs_1hz"]["fraction_worse"], 69, 0.02)
    # and 5 Hz costs five times the propagations
    assert r["5.0"]["predictor_calls"] > 4.5 * r["1.0"]["predictor_calls"]


def test_prediction_error_quoted_rows(data):
    m, _ = data
    rows = {round(x["t_end_offset"], 2): x
            for x in m["engagements"]["long"]["prediction_error"]["rows"]}
    for off, bias_r, rms in ((0.0, -49.91, 57.62), (3.0, -8.41, 23.34),
                             (6.4, -0.89, 15.83), (22.2, -3.84, 5.71)):
        k = min(rows, key=lambda z: abs(z - off))
        assert close(rows[k]["bias_range_m"], bias_r, 0.02), off
        assert close(rows[k]["rms_m"], rms, 0.02), off


def test_prediction_error_is_bias_not_scatter(data):
    """GUIDANCE-DESIGN.md section 3.2: the spread over commanded angles is far
    smaller than the bias, early in flight."""
    m, _ = data
    rows = m["engagements"]["long"]["prediction_error"]["rows"]
    early = [r for r in rows if r["t_end_offset"] <= 2.0]
    for r in early:
        assert abs(r["bias_range_m"]) > 5.0 * max(r["sd_range_m"], 1e-9) or \
               r["sd_range_m"] == 0.0


# ===========================================================================
# docs/AIM-OFF.md
# ===========================================================================
@pytest.mark.parametrize("off,cep,bias", [("0.0", 76.3, -111.2), ("100.0", 44.6, -52.6),
                                          ("175.0", 31.2, -10.9), ("224.0", 37.0, 14.5),
                                          ("275.0", 44.4, 43.4), ("350.0", 51.5, 87.5)])
def test_aim_off_sweep(data, off, cep, bias):
    _, c = data
    r = c["task_d"]["offsets"][off]["cep"]
    assert close(r["cep_m"], cep, 0.01)
    assert close(r["bias_range_m"], bias, 0.02)


def test_two_dimensional_aim_off(data):
    _, c = data
    t = c["task_d"]["two_d"]
    assert close(t["cep"]["cep_m"], 29.7, 0.01)
    # The SIGN: the offset must be MINUS the measured bias, or it doubles it.
    assert t["aim_off_deflection_m"] == pytest.approx(-t["measured_defl_bias_m"])
    assert abs(t["cep"]["bias_defl_m"]) < abs(t["measured_defl_bias_m"])


def test_the_three_drag_biases(data):
    """AIM-OFF.md section 1. The steering pair's first 4 s cost 90 m."""
    m, _ = data
    e = m["engagements"]["long"]
    b = e["baseline"]
    free_nose = b["free_range"] - b["uncorrected_range"]
    staged_free = e["reference"]["d_range_m"]
    staged_held = staged_free + e["nodes"][-1]["centre_range_m"]
    assert close(free_nose, -226.6, 0.01)
    assert close(staged_free, -136.4, 0.01)
    assert close(staged_held, -186.7, 0.01)
    assert close(staged_free - free_nose, 90.3, 0.02)


# ===========================================================================
# docs/DEGRADATION-LADDER.md
# ===========================================================================
@pytest.mark.parametrize("rung,cep", [("full", 36.3), ("degraded_25pc", 59.1),
                                      ("degraded_60pc", 36.3), ("reversionary", 212.2),
                                      ("inhibit", 36.3), ("stage2_fail", 216.0),
                                      ("stage2_fail_monitor_off", 216.0)])
def test_ladder_ceps(data, rung, cep):
    _, c = data
    assert close(c["task_e"]["rungs"][rung]["cep"]["cep_m"], cep, 0.01)


def test_losing_navigation_late_costs_nothing_at_the_median(data):
    _, c = data
    r = c["task_e"]["rungs"]
    assert close(r["degraded_60pc"]["cep"]["cep_m"],
                 r["full"]["cep"]["cep_m"], 0.01)
    assert r["degraded_25pc"]["cep"]["cep_m"] > 1.5 * r["full"]["cep"]["cep_m"]


def test_losing_navigation_late_costs_the_tail_that_matters(data):
    """
    DEGRADATION-LADDER.md section 2. The CEP is untouched because the median
    round is unreachable; the rounds the kit CAN reach lose their sub-metre
    residuals, and that is what a "costs nothing" claim would hide.
    """
    import numpy as np
    _, c = data
    r = c["task_e"]["rungs"]
    full = np.array([x["miss_m"] for x in r["full"]["rows"]])
    deg = np.array([x["miss_m"] for x in r["degraded_60pc"]["rows"]])
    assert (full <= 5.0).mean() > 0.30
    assert (deg <= 5.0).mean() < 0.05
    assert np.percentile(deg, 25) > 10.0 * np.percentile(full, 25)


def test_inhibit_leaves_the_median_and_wrecks_the_tail(data):
    """DEGRADATION-LADDER.md section 4: inhibiting on an accuracy criterion is
    strictly worse than continuing, because the correction is monotone."""
    _, c = data
    full = c["task_e"]["rungs"]["full"]["cep"]
    inh = c["task_e"]["rungs"]["inhibit"]["cep"]
    assert close(inh["cep_m"], full["cep_m"], 0.01)
    assert inh["p90_m"] > 1.5 * full["p90_m"]
    assert inh["mean_miss_m"] > 1.4 * full["mean_miss_m"]


def test_the_monitor_does_not_fire_on_healthy_rounds(data):
    _, c = data
    for t in ("task_c", "task_c2"):
        for law, r in c[t]["schedulers"].items():
            n = sum(1 for x in r["rows"]
                    if x.get("g_authority_fault_time") is not None)
            # `isotropic` at the reachable dispersion is the single exception
            # and it fires once; everything else is clean.
            assert n <= 1, (t, law, n)


def test_stage2_detection_and_the_negative_result_on_the_start_time(data):
    _, c = data
    assert c["task_e"]["stage2_detection"]["detected"] == 44
    assert c["task_e"]["stage2_detection"]["of"] == 64
    e2 = c["task_e2"]["rungs"]
    # Three more detections, two false alarms, and 3.3 m of CEP. Not adopted.
    assert e2["stage2_fail"]["authority_faults"] == 46
    assert e2["full"]["authority_faults"] == 2
    assert e2["full"]["cep"]["cep_m"] > c["task_e"]["rungs"]["full"]["cep"]["cep_m"]


def test_detecting_the_stage2_failure_changes_no_cep(data):
    """The null result IS the point: the monitor makes the fault observable,
    not survivable."""
    _, c = data
    on = c["task_e"]["rungs"]["stage2_fail"]["cep"]["cep_m"]
    off = c["task_e"]["rungs"]["stage2_fail_monitor_off"]["cep"]["cep_m"]
    assert close(on, off, 0.01)


# ===========================================================================
# docs/CEP-CLOSED-LOOP.md
# ===========================================================================
@pytest.mark.parametrize("lbl,guided,unguided", [
    ("short", 24.2, 23.9), ("short2", 41.7, 50.7), ("middle", 61.2, 119.8),
    ("mid2", 64.5, 170.3), ("long", 36.3, 212.4)])
def test_firing_table(data, lbl, guided, unguided):
    _, c = data
    r = c["task_f"]["engagements"][lbl]
    assert close(r["cep"]["cep_m"], guided, 0.01)
    assert close(r["unguided_cep"]["cep_m"], unguided, 0.01)


def test_the_design_does_nothing_at_two_kilometres(data):
    _, c = data
    m, _ = data
    r = c["task_f"]["engagements"]["short"]
    assert r["cep"]["cep_m"] >= r["unguided_cep"]["cep_m"]
    assert close(m["engagements"]["short"]["full_hold_ellipse"]["semi_major_m"],
                 3.2, 0.03)


def test_the_semi_minor_orders_the_firing_table_and_the_semi_major_does_not(data):
    """
    CEP-CLOSED-LOOP.md section 2. The guided-to-unguided ratio is ordered
    perfectly by b / CEP_unc and imperfectly by a / CEP_unc -- middle and mid2
    have nearly the same semi-major relative to their dispersion and very
    different semi-minors, and the semi-minor is the one that predicts.
    """
    _, c = data
    rows = []
    for lbl, r in c["task_f"]["engagements"].items():
        E, d = r["ellipse"], r["dispersion"]
        rows.append((E["semi_major_m"] / d["cep_m"], E["semi_minor_m"] / d["cep_m"],
                     r["cep"]["cep_m"] / r["unguided_cep"]["cep_m"], lbl))

    def monotone(key):
        s = sorted(rows, key=lambda x: x[key])
        v = [x[2] for x in s]
        return all(v[i] >= v[i + 1] for i in range(len(v) - 1))

    assert monotone(1), rows          # semi-minor: perfect
    assert not monotone(0), rows      # semi-major: not


def test_containment(data):
    _, c = data
    g = {(x["dispersion"], x["set"]): x for x in c["task_g"]["containment"]}
    staged_rd = g[("range_dominated", "staged_closed_loop")]
    staged_ci = g[("circular", "staged_closed_loop")]
    assert close(100 * staged_rd["p_contains"], 41.8, 0.01)
    assert close(100 * staged_ci["p_contains"], 41.2, 0.01)
    # The eccentricity question: it barely matters, because the reachable set
    # is nearly circular. CEP-CLOSED-LOOP.md section 5.2.
    assert abs(staged_rd["p_contains"] - staged_ci["p_contains"]) < 0.01
    assert close(staged_rd["scale_for_50pc"], 1.17, 0.02)


def test_the_published_criterion_is_axis_ratio_dependent(data):
    """
    CEP-CLOSED-LOOP.md section 5.1, and the reason STAGED-DEPLOYMENT.md section
    8.2's verdict does not survive. 186.3 m is axis ratio 1.20; the adopted
    configuration is 1.10.
    """
    from analysis import cep_projection as cp
    sr = sd = 169.9                       # circular, uncorrected CEP 200 m
    assert close(cp.required_authority(sr, sd, 30.0, axis_ratio=1.20), 186.3, 0.005)
    assert close(cp.required_authority(sr, sd, 30.0, axis_ratio=1.10), 178.7, 0.005)
    assert close(cp.required_authority(sr, sd, 30.0, axis_ratio=1.44), 206.3, 0.005)
    m, _ = data
    a = m["engagements"]["long"]["full_hold_ellipse"]["semi_major_m"]
    assert a > cp.required_authority(sr, sd, 30.0, axis_ratio=1.10)
    assert a < cp.required_authority(sr, sd, 30.0, axis_ratio=1.20)
