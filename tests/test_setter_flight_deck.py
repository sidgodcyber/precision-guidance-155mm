"""
Flight Deck tests (Control Room spec Part C, L3 / Step 6).

One of these fires a real round (~30 s, more under load) to get genuine
`fired_round` data to replay -- consistent with test_setter_fire.py's
existing practice.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "setter" / "app.py"


def _session_get(at: AppTest, key: str):
    return at.session_state[key] if key in at.session_state else None


def _fresh_app() -> AppTest:
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, at.exception
    return at


def _fire_a_round(at: AppTest, timeout_s: float = 240.0) -> dict:
    at.switch_page("pages/mission_control.py")
    at.run(timeout=60)
    assert not at.exception, at.exception
    fire_buttons = [b for b in at.button if b.label == "FIRE"]
    assert len(fire_buttons) == 1
    fire_buttons[0].click().run(timeout=60)
    assert not at.exception, at.exception

    deadline = time.time() + timeout_s
    while _session_get(at, "fire_job") is not None:
        assert time.time() < deadline, "FIRE did not complete in time"
        time.sleep(2)
        at.run(timeout=60)
        assert not at.exception, at.exception

    fired = _session_get(at, "fired_round")
    assert fired is not None and fired["ok"]
    return fired


# ===========================================================================
# Not reachable without a fired round
# ===========================================================================
def test_flight_deck_not_in_navigation_before_any_fire():
    at = _fresh_app()
    with pytest.raises(ValueError):
        at.switch_page("pages/flight_deck.py")


def test_flight_deck_note_on_overview_before_any_fire():
    at = _fresh_app()
    captions = " ".join(c.value for c in at.caption)
    assert "fire a round from Mission Control first" in captions


# ===========================================================================
# End to end -- fire, then replay
# ===========================================================================
def test_flight_deck_renders_after_a_real_fire():
    at = _fresh_app()
    fired = _fire_a_round(at)

    at.switch_page("pages/flight_deck.py")
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert at.title[0].value == "Flight Deck"

    # -- the scrubber's range matches the stored trajectory exactly
    sliders = list(at.slider)
    assert len(sliders) == 1
    ts = fired["state_trajectory"]["t"]
    assert sliders[0].min == pytest.approx(ts[0], abs=1e-6)
    assert sliders[0].max == pytest.approx(ts[-1], abs=1e-6)
    assert sliders[0].min == pytest.approx(fired["t_dep_actual"], abs=1e-6)

    # -- state panel present and sane at the initial (deployment) position
    metrics = {m.label: m.value for m in at.metric}
    for label in ("Time of flight", "Downrange", "Altitude", "Mach", "Body spin rate"):
        assert label in metrics

    # -- the two-panel ground track/altitude figure rendered
    assert len(list(at.get("image"))) >= 1

    # -- fuze state panel present, one of the four named states
    subheaders = [s.value for s in at.subheader]
    assert "Fuze state" in subheaders
    assert "Authority" in subheaders
    assert "State" in subheaders


def test_flight_deck_scrubber_updates_state_panel():
    at = _fresh_app()
    fired = _fire_a_round(at)
    at.switch_page("pages/flight_deck.py")
    at.run(timeout=60)

    sl = at.slider[0]
    before = {m.label: m.value for m in at.metric}
    mid_t = (sl.min + sl.max) / 2
    sl.set_value(mid_t).run(timeout=60)
    assert not at.exception, at.exception
    after = {m.label: m.value for m in at.metric}

    assert before["Time of flight"] != after["Time of flight"]
    assert after["Time of flight"] == f"{mid_t:.2f} s"


def test_flight_deck_phase_strip_separates_ballistic_from_guided():
    """The phase strip must show BALLISTIC (no data) distinctly from the
    guided phases -- the pre-deployment leg genuinely has no data here."""
    at = _fresh_app()
    fired = _fire_a_round(at)
    at.switch_page("pages/flight_deck.py")
    at.run(timeout=60)

    all_text = " ".join(c.value for c in at.caption) + " " + " ".join(m.value for m in at.markdown)
    assert "BALLISTIC" in all_text
    assert "no data" in all_text
    assert "DEPLOYMENT" in all_text
    assert "CORRECTION" in all_text
    assert "TERMINAL" in all_text
    assert "GUIDED PHASE ONLY" in all_text


def test_flight_deck_guidance_convergence_panel_present():
    """The predicted-impact trail/marker -- "the visual centre of the
    page" -- must appear once g_log data exists for the scrubber position."""
    at = _fresh_app()
    fired = _fire_a_round(at)
    assert fired.get("g_log") is not None
    at.switch_page("pages/flight_deck.py")
    at.run(timeout=60)

    sl = at.slider[0]
    sl.set_value(sl.max - 1.0).run(timeout=60)
    assert not at.exception, at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "predicted miss" in captions


def test_flight_deck_never_appears_on_overview_or_archive_headers():
    """Flight Deck's own content (its title, its panels) must never render
    inside Overview or Archive -- it is its own page, reached only via
    navigation, not inlined elsewhere."""
    at = _fresh_app()
    _fire_a_round(at)
    at.switch_page("pages/overview.py")
    at.run(timeout=60)
    assert "Flight Deck" != (at.title[0].value if at.title else None)

    at.switch_page("pages/archive.py")
    at.run(timeout=60)
    assert "Flight Deck" != (at.title[0].value if at.title else None)
