"""
L1 -- Overview (Control Room spec Part C).

One screen, scannable in three seconds, no scrolling, status tiles only.
"Is everything all right, and where do I look next" -- nothing here is a
control; every tile either states a fact or points at the screen where you
would act on it.

Registered in `setter/app.py` as a FILE-based `st.Page` and the app's
DEFAULT page (see the note in `setter/pages/mission_control.py` on why
file-based). `st.page_link` accepts the same relative file-path strings
`st.navigation` was given in `setter/app.py`, so this module needs no
reference to the `st.Page` objects themselves.

DATA SOURCES, AND WHY EACH IS CHEAP (Part A4: this page must open in under
a second, "almost entirely cached reads, no fresh computation"):

  current mission       `st.session_state` -- written by Mission Control's
                         own widgets (`key="engagement"`, `key="mc_met_age"`
                         via `met_age_bucket`, `key="mc_fuze_mode"`), or a
                         sensible default (matching Mission Control's own
                         widget defaults) if that page hasn't run yet this
                         session.
  predicted accuracy    `st.session_state["predicted_cep"]`, written by
                         Mission Control right after IT computes the same
                         `campaign.task_a_by_age` lookup -- read back, not
                         recomputed, when available; falls back to the same
                         cheap cached JSON lookup (not a live result, not a
                         simulation) at the default mission otherwise.
  engine status          a static grep of `def test_` across tests/*.py (no
                         pytest collection -- that imports the whole
                         engine and costs seconds, not the budget here),
                         `docs/monte_carlo.json`'s own mtime, and
                         `setter.characterization.git_commit()` (a single
                         `git rev-parse`). All three cached for the process.
  artifact freshness    `setter.provenance.check` over every
                         `setter.registry` entry -- exactly Step 4's
                         staleness logic, not rebuilt; cached briefly since
                         it's ~35 stat calls, not zero-cost.
  unresolved / amber     the mid2 non-monotonicity, checked LIVE against
                         `campaign.task_a_headline_table()` rather than
                         asserted as fixed prose, so this tile stops
                         claiming it if the stored campaign ever changes;
                         plus the current mission's own availability from
                         the predicted-accuracy lookup above.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from setter import campaign, provenance, registry
from setter.characterization import git_commit
from setter.config import MONTE_CARLO_PATH, SUPPORTED_ENGAGEMENTS
from setter.palette import ACCENT, AMBER

_MISSION_CONTROL_PATH = "pages/mission_control.py"
_ARCHIVE_PATH = "pages/archive.py"
_FLIGHT_DECK_PATH = "pages/flight_deck.py"
_ERROR_BUDGET_PATH = "pages/error_budget.py"


def _amber(text: str) -> str:
    return f"<span style='color:{AMBER}'>{text}</span>"

_DEFAULT_ENGAGEMENT = SUPPORTED_ENGAGEMENTS[-1]  # "long" -- matches Mission Control's own default
_DEFAULT_MET_AGE = "2h"  # matches Mission Control's slider default
_DEFAULT_FUZE_MODE = "time"  # matches Mission Control's selectbox default (index 0)


@st.cache_data(ttl=300)
def _test_function_count() -> int:
    """A static count of `def test_` across tests/*.py -- NOT the same
    number `python -m pytest` reports (parametrized tests multiply into
    many cases from one function), and deliberately not run here: pytest
    collection imports the whole engine and costs seconds, which this
    page's budget does not have. Labelled "test functions" on screen for
    exactly that reason."""
    total = 0
    for f in Path("tests").glob("*.py"):
        total += f.read_text(encoding="utf-8").count("def test_")
    return total


@st.cache_data(ttl=300)
def _repo_commit_short() -> str:
    commit = git_commit()
    return commit[:8] if commit else "unknown"


@st.cache_data(ttl=300)
def _last_campaign_mtime() -> float:
    return os.stat(MONTE_CARLO_PATH).st_mtime


@st.cache_data(ttl=60)
def _stale_count() -> tuple:
    """`(n_stale, n_total)` over every script-registry entry, via
    `setter.provenance.check` -- Step 4's staleness logic, reused exactly,
    not rebuilt for this page."""
    stale = [e for e in registry.ENTRIES if provenance.check(e).stale is True]
    return len(stale), len(registry.ENTRIES)


@st.cache_data(ttl=60)
def _mid2_is_non_monotonic() -> bool:
    """Live check against the stored campaign, not asserted prose: is
    mid2's 2h headline CEP still larger than both its range-neighbours'?
    (the anomaly CLAUDE.md and Mission Control's own warning both name)."""
    table = campaign.task_a_headline_table()
    mid2, middle, long_ = table.get("mid2"), table.get("middle"), table.get("long")
    if not (mid2 and middle and long_ and mid2.available and middle.available and long_.available):
        return False
    return mid2.cep_m > middle.cep_m and mid2.cep_m > long_.cep_m


def _current_mission() -> dict:
    return {
        "engagement": st.session_state.get("engagement", _DEFAULT_ENGAGEMENT),
        "met_age_bucket": st.session_state.get("met_age_bucket", _DEFAULT_MET_AGE),
        "fuze_mode": st.session_state.get("mc_fuze_mode", _DEFAULT_FUZE_MODE),
        "from_session": "engagement" in st.session_state,
    }


def _predicted_accuracy(mission: dict) -> dict:
    cached = st.session_state.get("predicted_cep")
    if cached and cached["engagement"] == mission["engagement"] and cached["age_bucket"] == mission["met_age_bucket"]:
        return cached
    # Mission Control hasn't run yet this session (or ran for a different
    # mission) -- the SAME cheap, cached campaign lookup it itself uses,
    # not a new kind of number.
    hours = {"perfect": None, "0h": 0.0, "1h": 1.0, "2h": 2.0,
             "3h": 3.0, "6h": 6.0, "none": None}.get(mission["met_age_bucket"])
    if hours is None:
        return {"available": False, "cep_m": None, "n": None,
                "reason": f"Task A has no stored campaign point for met-age bucket "
                          f"{mission['met_age_bucket']!r}."}
    point = campaign.task_a_by_age(mission["engagement"], hours)
    return {"available": point.available, "cep_m": point.cep_m, "n": point.n, "reason": point.reason}


def render() -> None:
    st.title("SIMULATION SETTER — Overview")
    st.caption(
        "Is everything all right, and where do I look next. Status tiles "
        "only -- nothing on this screen is a control.")

    mission = _current_mission()
    accuracy = _predicted_accuracy(mission)
    n_stale, n_total = _stale_count()
    mid2_flagged = _mid2_is_non_monotonic()

    tiles = st.columns(5)

    # -----------------------------------------------------------------
    # 1. Current mission
    # -----------------------------------------------------------------
    with tiles[0]:
        with st.container(border=True):
            st.caption("CURRENT MISSION")
            st.markdown(f"**{mission['engagement']}**")
            st.caption(f"met age {mission['met_age_bucket']} · fuze mode {mission['fuze_mode']}")
            if not mission["from_session"]:
                st.caption("_default -- Mission Control not yet visited this session_")
            st.page_link(_MISSION_CONTROL_PATH, label="Open Mission Control")
            if "fired_round" in st.session_state:
                st.page_link(_FLIGHT_DECK_PATH, label="Open Flight Deck (last fired round)")
            else:
                st.caption("Flight Deck — fire a round from Mission Control first")

    # -----------------------------------------------------------------
    # 2. Predicted accuracy -- the one accent-coloured number on the page
    # -----------------------------------------------------------------
    with tiles[1]:
        with st.container(border=True):
            st.caption("PREDICTED ACCURACY")
            if accuracy["available"]:
                st.markdown(
                    f"<span style='font-size:2.2rem;font-weight:700;color:{ACCENT}'>"
                    f"{accuracy['cep_m']:.1f} m</span>",
                    unsafe_allow_html=True)
                st.caption(f"n={accuracy['n']} · stored Task A campaign statistic")
            else:
                st.markdown("**unavailable**")
                st.caption(accuracy["reason"] or "No stored campaign point for this mission.")
            st.page_link(_MISSION_CONTROL_PATH, label="Open Mission Control")

    # -----------------------------------------------------------------
    # 3. Engine status
    # -----------------------------------------------------------------
    with tiles[2]:
        with st.container(border=True):
            st.caption("ENGINE STATUS")
            st.markdown(f"**{_test_function_count()}** test functions")
            campaign_date = datetime.fromtimestamp(_last_campaign_mtime()).strftime("%Y-%m-%d")
            st.caption(f"last campaign {campaign_date}")
            st.caption(f"commit `{_repo_commit_short()}`")
            st.caption("The Engine — not built yet (Step 8)")

    # -----------------------------------------------------------------
    # 4. Artifact freshness
    # -----------------------------------------------------------------
    with tiles[3]:
        with st.container(border=True):
            st.caption("ARTIFACT FRESHNESS")
            if n_stale:
                st.markdown(_amber(f"<b>{n_stale}</b> of {n_total} stale"), unsafe_allow_html=True)
            else:
                st.markdown(f"**0** of {n_total} stale")
            st.caption("script outputs vs. their inputs")
            st.page_link(_ARCHIVE_PATH, label="Open Archive")

    # -----------------------------------------------------------------
    # 5. Unresolved -- amber only, per Part A2
    # -----------------------------------------------------------------
    with tiles[4]:
        with st.container(border=True):
            st.caption("UNRESOLVED")
            any_flag = False
            if mid2_flagged:
                any_flag = True
                st.markdown(_amber("mid2 CEP non-monotonic with range"), unsafe_allow_html=True)
            if not accuracy["available"]:
                any_flag = True
                st.markdown(_amber("no campaign point for current mission"), unsafe_allow_html=True)
            if not any_flag:
                st.caption("nothing flagged for the current mission")
            st.page_link(_ERROR_BUDGET_PATH, label="Open Error Budget")


render()
