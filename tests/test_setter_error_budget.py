"""
Error Budget page tests (Control Room spec Part C, L3 / Step 6 second half).

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from setter import campaign

APP_PATH = Path(__file__).resolve().parent.parent / "setter" / "app.py"
PAIRED_PATH = Path(__file__).resolve().parent.parent / "docs" / "setter_validation" / "paired_age_comparison.json"


def _fresh_error_budget() -> AppTest:
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, at.exception
    at.switch_page("pages/error_budget.py")
    at.run(timeout=60)
    assert not at.exception, at.exception
    return at


# ===========================================================================
# Reachable without a fired round -- unconditional, unlike Flight Deck
# ===========================================================================
def test_error_budget_reachable_without_firing():
    at = _fresh_error_budget()
    assert at.title[0].value == "Error Budget"


def test_error_budget_registered_unconditionally_in_navigation():
    """Unlike Flight Deck, Error Budget needs no fired round -- it must be
    switchable to from a completely fresh session."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert "fired_round" not in at.session_state
    at.switch_page("pages/error_budget.py")
    at.run(timeout=60)
    assert not at.exception, at.exception


# ===========================================================================
# Ranked contributions -- no new computation
# ===========================================================================
def test_ranked_contributions_dominant_term_is_met_knowledge():
    terms = campaign.error_budget()
    assert terms, "no error-budget terms available"
    assert terms[0].name == "meteorological knowledge"
    # "dominant term obvious at a glance": comfortably ahead of #2
    assert terms[0].sigma_range_m > terms[1].sigma_range_m * 1.5


def test_ranked_contributions_every_term_carries_n():
    for t in campaign.error_budget():
        assert t.n and t.n > 0


def test_error_budget_page_shows_all_five_named_terms():
    at = _fresh_error_budget()
    markdown_text = " ".join(m.value for m in at.markdown)
    for name in ("meteorological knowledge", "muzzle velocity", "laying",
                "model error", "navigation"):
        assert name in markdown_text


# ===========================================================================
# Staleness curve -- confidence intervals present, non-monotonicity visible
# ===========================================================================
def test_task_c_curve_carries_confidence_intervals():
    curve = campaign.task_c_curve()
    available = [p for p in curve if p.available]
    assert available
    for p in available:
        assert p.cep_lo_m is not None and p.cep_hi_m is not None
        assert p.cep_lo_m <= p.cep_m <= p.cep_hi_m


def test_task_c_curve_shows_the_post_3h_non_monotonicity():
    """CEP must rise 3h -> 6h and then fall 6h -> none in the stored data
    -- the exact shape the page labels as unexplained."""
    curve = {p.age: p for p in campaign.task_c_curve() if p.available}
    assert curve["6h"].cep_m > curve["3h"].cep_m
    assert curve["none"].cep_m < curve["6h"].cep_m


def test_error_budget_page_labels_the_non_monotonicity_unexplained():
    at = _fresh_error_budget()
    captions = " ".join(c.value for c in at.caption)
    assert "UNEXPLAINED" in captions
    assert "not smoothed" in captions


# ===========================================================================
# The paired result -- the one claim that needs the bootstrap artifact
# ===========================================================================
def test_paired_artifact_shows_3h_to_6h_real_and_6h_to_none_not_established():
    """Read the artifact directly, independent of the page, as the ground
    truth this test then checks the rendered page against."""
    data = json.loads(PAIRED_PATH.read_text())
    comps = data["comparisons"]
    assert comps["3h_vs_6h"]["interpretation"]["cep_interval_excludes_zero"] is True
    assert comps["3h_vs_6h"]["variance_bias_decomposition"]["category"] == "dispersion"
    assert comps["6h_vs_none"]["interpretation"]["cep_interval_excludes_zero"] is False


def test_error_budget_page_displays_the_right_paired_conclusion():
    at = _fresh_error_budget()
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "3h" in markdown_text and "6h" in markdown_text
    assert "real, dispersion-driven" in markdown_text
    assert "not established" in markdown_text


def test_error_budget_page_paired_numbers_match_the_artifact_exactly():
    at = _fresh_error_budget()
    data = json.loads(PAIRED_PATH.read_text())
    cep_36 = data["comparisons"]["3h_vs_6h"]["cep_difference"]
    captions = " ".join(c.value for c in at.caption)
    assert f"{cep_36['delta_m']:+.1f} m" in captions
    assert f"n={cep_36['n']} shared" in captions


# ===========================================================================
# Plain-language summary
# ===========================================================================
def test_summary_paragraph_present_and_unhedged():
    at = _fresh_error_budget()
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "limited by how well the gun knows the air" in markdown_text
    assert "upload a fresh meteorological message" in markdown_text.lower() or \
        "upload a fresh" in markdown_text.lower()


# ===========================================================================
# Navigation -- reachable from Overview and Mission Control
# ===========================================================================
def test_reachable_from_overview_unresolved_tile():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    labels = [l.value for l in at.get("page_link")]
    pages = [l.proto.page for l in at.get("page_link")]
    assert any("Error Budget" in l for l in labels)
    assert "error-budget" in pages


def test_reachable_from_mission_control_campaign_accuracy_section():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    at.switch_page("pages/mission_control.py")
    at.run(timeout=60)
    labels = [l.value for l in at.get("page_link")]
    pages = [l.proto.page for l in at.get("page_link")]
    assert any("Error Budget" in l for l in labels)
    assert "error-budget" in pages
