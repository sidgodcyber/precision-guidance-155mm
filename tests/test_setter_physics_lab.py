"""
Physics Lab tests (Control Room spec Part B5, Step 8 first half).

Every card is a single closed-form evaluation of an already-public
`sim.canards`/`gnc.roll_control` function, or a read of already-stored
JSON -- nothing here flies a new 6-DOF trajectory, so the whole page should
load in low seconds, not the ~33 s `analysis.roll_servo.baseline()` costs.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from setter import physics_lab_data as pld
from setter.config import DOCS_DIR, MET_AGE_BUCKETS

APP_PATH = Path(__file__).resolve().parent.parent / "setter" / "app.py"


def _fresh_physics_lab() -> AppTest:
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, at.exception
    at.switch_page("pages/physics_lab.py")
    at.run(timeout=60)
    assert not at.exception, at.exception
    return at


# ===========================================================================
# Reachability / registration
# ===========================================================================
def test_physics_lab_reachable_without_firing():
    at = _fresh_physics_lab()
    assert at.title[0].value == "Physics Lab"


def test_physics_lab_registered_unconditionally_in_navigation():
    """Like Error Budget/Archive -- no `fired_round` gate."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert "fired_round" not in at.session_state
    at.switch_page("pages/physics_lab.py")
    at.run(timeout=60)
    assert not at.exception, at.exception


def test_physics_lab_page_load_is_fast():
    """Regression guard: a future card that accidentally calls a
    campaign-class function live (e.g. analysis.roll_servo.baseline(),
    measured at ~33 s) instead of reading its stored output would blow this
    budget by more than 2x."""
    t0 = time.perf_counter()
    _fresh_physics_lab()
    elapsed = time.perf_counter() - t0
    assert elapsed < 15.0, f"Physics Lab took {elapsed:.1f} s to load"


# ===========================================================================
# Ground-truth cross-checks -- independent of the page
# ===========================================================================
def test_adopted_deployment_condition_matches_roll_servo_json():
    with open(DOCS_DIR / "roll_servo.json", encoding="utf-8") as fh:
        stored = json.load(fh)["baseline"]
    c = pld.adopted_deployment_condition()
    assert c.mach == pytest.approx(stored["mach_at_deploy"])
    assert c.qbar_pa == pytest.approx(stored["qbar_at_deploy"])
    assert c.spin_rad_s == pytest.approx(stored["spin_at_deploy"])
    assert c.time_s == pytest.approx(stored["deploy_time"])
    # Round-trip check: the derived (altitude, airspeed) pair must itself
    # reproduce the stored (mach, qbar) through the same ISA relation.
    from sim.atmosphere import GAMMA, isa_scalars
    _, p, _, a = isa_scalars(c.altitude_m)
    assert c.airspeed_ms == pytest.approx(c.mach * a, rel=1e-9)
    assert 2.0 * c.qbar_pa == pytest.approx(GAMMA * c.mach ** 2 * p, rel=1e-9)


def test_authority_phi_sweep_has_exactly_12_stored_points_matching_json():
    with open(DOCS_DIR / "authority_results.json", encoding="utf-8") as fh:
        stored = json.load(fh)["main"]["c8_qe525.3"]["rows"]
    sweep = pld.authority_phi_sweep("c8_qe525.3")
    assert len(sweep) == 12 == len(stored)
    for (phi, dr, dd), row in zip(sweep, stored):
        assert phi == pytest.approx(row["phi_deg"])
        assert dr == pytest.approx(row["d_range"])
        assert dd == pytest.approx(row["d_deflection"])


def test_authority_deploy_sweep_has_3_tags_of_6_fractions():
    sweep = pld.authority_deploy_sweep()
    assert len(sweep) == 18
    tags = {k.rsplit("_f", 1)[0] for k in sweep}
    assert tags == {"c4_qe97.2", "c6_qe258.4", "c8_qe525.3"}
    for entry in sweep.values():
        assert len(entry["rows"]) == 12
        assert "semi_axis_major_m" in entry["stats"]


# ===========================================================================
# The rendered page
# ===========================================================================
def test_physics_lab_shows_all_six_cards():
    at = _fresh_physics_lab()
    subheaders = [s.value for s in at.subheader]
    for n in range(1, 7):
        assert any(s.startswith(f"{n}.") for s in subheaders), f"card {n} missing"


def test_card1_spin_slider_changes_the_displayed_cancellation():
    at = _fresh_physics_lab()
    sliders = {s.key: s for s in at.slider}
    sliders["pl_spin"].set_value(0.0).run(timeout=60)
    assert not at.exception, at.exception
    low_spin = " ".join(c.value for c in at.caption if "cancelled" in c.value)
    sliders = {s.key: s for s in at.slider}
    sliders["pl_spin"].set_value(1308.0).run(timeout=60)
    assert not at.exception, at.exception
    high_spin = " ".join(c.value for c in at.caption if "cancelled" in c.value)
    assert "0.0% cancelled" in low_spin  # zero spin: no cancellation at all
    assert "99." in high_spin  # near-total cancellation at realistic spin


def test_card2_brake_slider_changes_equilibrium_rate():
    at = _fresh_physics_lab()
    sliders = {s.key: s for s in at.slider}
    sliders["pl_brake_2"].set_value(0.0).run(timeout=60)
    assert not at.exception, at.exception
    sliders = {s.key: s for s in at.slider}
    sliders["pl_brake_2"].set_value(1.0).run(timeout=60)
    assert not at.exception, at.exception


def test_card3_phi_slider_changes_the_correction_vector():
    at = _fresh_physics_lab()
    sliders = {s.key: s for s in at.slider}
    sliders["pl_phi"].set_value(0.0).run(timeout=60)
    assert not at.exception, at.exception
    sliders = {s.key: s for s in at.slider}
    sliders["pl_phi"].set_value(180.0).run(timeout=60)
    assert not at.exception, at.exception


def test_card5_select_slider_options_are_exactly_the_6_stored_deploy_fractions():
    at = _fresh_physics_lab()
    select_sliders = {s.key: s for s in at.select_slider}
    options = select_sliders["pl_deploy_frac"].proto.options
    assert len(options) == 6


def test_card6_scatter_length_matches_stored_n_at_2h():
    from setter import campaign
    scatter = campaign.task_c_scatter("2h", tag="headline", engagement="long")
    point = campaign.task_c_point("2h", tag="headline", engagement="long")
    assert point.available
    assert len(scatter) == point.n == 128


def test_card6_met_age_options_match_config():
    at = _fresh_physics_lab()
    select_sliders = {s.key: s for s in at.select_slider}
    options = select_sliders["pl_met_age"].proto.options
    assert tuple(options) == MET_AGE_BUCKETS


def test_every_card_cites_a_real_repository_function():
    at = _fresh_physics_lab()
    captions = " ".join(c.value for c in at.caption)
    for needle in ("sim/canards.py", "gnc/roll_control.py",
                   "analysis/authority.py", "docs/authority_results.json",
                   "setter/campaign.py"):
        assert needle in captions, f"missing citation: {needle}"
