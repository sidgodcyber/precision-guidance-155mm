"""
L1 Overview page tests (Control Room spec Part C / Step 5).

Run:  python -m pytest tests -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from setter import campaign

APP_PATH = Path(__file__).resolve().parent.parent / "setter" / "app.py"


def _fresh_overview() -> AppTest:
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, at.exception
    return at


# ===========================================================================
# Overview is the default landing page
# ===========================================================================
def test_overview_is_the_default_landing_page():
    """Part A1: L1 is where you start. A plain run of app.py, with no
    navigation, must land on Overview -- not Mission Control."""
    at = _fresh_overview()
    titles = [t.value for t in at.title]
    assert any("Overview" in t for t in titles)
    assert not any(t == "SIMULATION SETTER" for t in titles)  # Mission Control's exact title


def test_overview_has_five_status_tiles_and_no_widgets():
    """"Status tiles only" -- Overview must not contain interactive
    mission-planning controls (those belong to Mission Control)."""
    at = _fresh_overview()
    assert len(list(at.selectbox)) == 0
    assert len(list(at.select_slider)) == 0
    assert len(list(at.number_input)) == 0


# ===========================================================================
# Current mission tile -- session state, with sensible defaults
# ===========================================================================
def test_current_mission_tile_defaults_when_mission_control_unvisited():
    at = _fresh_overview()
    captions = [c.value for c in at.caption]
    assert any("not yet visited this session" in c for c in captions)
    markdowns = [m.value for m in at.markdown]
    assert any("long" in m for m in markdowns)  # the documented default engagement


def test_current_mission_tile_reflects_mission_control_after_a_visit():
    at = _fresh_overview()
    at.switch_page("pages/mission_control.py")
    at.run(timeout=60)
    box = [s for s in at.selectbox if s.label == "Engagement"][0]
    box.set_value("short").run(timeout=60)
    assert not at.exception, at.exception

    at.switch_page("pages/overview.py")
    at.run(timeout=60)
    assert not at.exception, at.exception
    markdowns = [m.value for m in at.markdown]
    assert any("short" == m.strip("*") for m in markdowns)
    captions = [c.value for c in at.caption]
    assert not any("not yet visited this session" in c for c in captions)


# ===========================================================================
# Predicted accuracy tile -- same source as Mission Control, not recomputed
# ===========================================================================
def test_predicted_accuracy_matches_mission_control_default():
    """The default mission is long/2h, Task A's own headline point --
    Overview's number must match campaign.task_a_by_age exactly, since it
    is either that same lookup or Mission Control's cached copy of it."""
    at = _fresh_overview()
    expected = campaign.task_a_by_age("long", 2.0)
    markdowns = " ".join(m.value for m in at.markdown)
    assert f"{expected.cep_m:.1f} m" in markdowns


# ===========================================================================
# Engine status tile
# ===========================================================================
def test_engine_status_tile_shows_test_count_date_and_commit():
    at = _fresh_overview()
    captions = [c.value for c in at.caption]
    markdowns = [m.value for m in at.markdown]
    assert any("test functions" in m for m in markdowns)
    assert any(c.startswith("last campaign ") for c in captions)
    assert any(c.startswith("commit `") for c in captions)


# ===========================================================================
# Artifact freshness tile -- reuses setter.provenance, not rebuilt
# ===========================================================================
def test_artifact_freshness_tile_matches_provenance_directly():
    from setter import provenance, registry
    at = _fresh_overview()
    expected_stale = sum(1 for e in registry.ENTRIES if provenance.check(e).stale is True)
    markdowns = " ".join(m.value for m in at.markdown)
    assert f"of {len(registry.ENTRIES)} stale" in markdowns
    if expected_stale:
        assert f"{expected_stale}" in markdowns
    else:
        assert "**0** of" in markdowns


# ===========================================================================
# Unresolved / amber tile
# ===========================================================================
def test_unresolved_tile_flags_mid2_non_monotonicity():
    """CLAUDE.md's headline anomaly must show up here, in amber, checked
    live against the stored campaign rather than asserted as fixed prose."""
    at = _fresh_overview()
    table = campaign.task_a_headline_table()
    mid2, middle, long_ = table["mid2"], table["middle"], table["long"]
    is_flagged = mid2.available and middle.available and long_.available and \
        mid2.cep_m > middle.cep_m and mid2.cep_m > long_.cep_m
    markdowns = " ".join(m.value for m in at.markdown)
    if is_flagged:
        assert "non-monotonic" in markdowns
    else:
        assert "non-monotonic" not in markdowns


# ===========================================================================
# Navigation -- tiles link to real screens, and only to real screens
# ===========================================================================
def test_tiles_link_to_mission_control_and_archive():
    at = _fresh_overview()
    labels = [link.value for link in at.get("page_link")]
    pages = [link.proto.page for link in at.get("page_link")]
    assert any("Mission Control" in l for l in labels)
    assert any("Archive" in l for l in labels)
    assert "mission-control" in pages
    assert "archive" in pages


def test_unbuilt_l3_screens_are_noted_not_linked():
    """Physics Lab / The Engine don't exist yet -- Overview must say so
    rather than link to nothing or guess at a page that isn't built.
    Error Budget is now built (Step 6) and IS linked -- see
    test_error_budget_tile_links_to_error_budget below. Flight Deck is
    conditionally linked only after a fire (test_setter_flight_deck.py),
    so it's correctly absent from a fresh, un-fired session too."""
    at = _fresh_overview()
    captions = " ".join(c.value for c in at.caption)
    assert "not built yet" in captions
    link_labels = [link.value for link in at.get("page_link")]
    assert not any("The Engine" in l or "Physics Lab" in l for l in link_labels)


def test_error_budget_tile_links_to_error_budget():
    at = _fresh_overview()
    labels = [link.value for link in at.get("page_link")]
    pages = [link.proto.page for link in at.get("page_link")]
    assert any("Error Budget" in l for l in labels)
    assert "error-budget" in pages


# ===========================================================================
# Nothing regressed
# ===========================================================================
def test_mission_control_still_reachable_and_unexceptional():
    at = _fresh_overview()
    at.switch_page("pages/mission_control.py")
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert at.title[0].value == "SIMULATION SETTER"


def test_archive_still_reachable_and_unexceptional():
    at = _fresh_overview()
    at.switch_page("pages/archive.py")
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert len(list(at.button)) == 20
