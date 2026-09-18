"""
FIRE tests (Control Room spec Part C "Bottom -- FIRE" / B3).

Two of these actually fire a real guided round through the 6-DOF engine
with navigation in the loop (~30-40 s each) -- consistent with the
project's existing practice of calling `analysis.nav_common.run_guided_nav`
directly in tests/test_navigation.py and tests/test_monte_carlo.py, not a
new kind of slowness this suite hasn't already accepted.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "setter" / "app.py"
REPO_ROOT = APP_PATH.parent.parent


def _fresh_mission_control() -> AppTest:
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, at.exception
    at.switch_page("pages/mission_control.py")
    at.run(timeout=60)
    assert not at.exception, at.exception
    return at


def _session_get(at: AppTest, key: str):
    return at.session_state[key] if key in at.session_state else None


# ===========================================================================
# Worker CLI -- fast negative case (no real round flown)
# ===========================================================================
def test_fire_worker_rejects_unknown_engagement(tmp_path):
    out_path = tmp_path / "fire_result.json"
    result = subprocess.run(
        [sys.executable, "-m", "setter.fire_worker",
         "--engagement", "not_a_real_engagement", "--met-age", "2h",
         "--out", str(out_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0  # the worker itself always exits 0 -- errors go in the JSON
    data = json.loads(out_path.read_text())
    assert data["ok"] is False
    assert "not_a_real_engagement" in data["error"]


# ===========================================================================
# Cost stated before starting
# ===========================================================================
def test_fire_states_cost_before_starting():
    at = _fresh_mission_control()
    captions = " ".join(c.value for c in at.caption)
    assert "Runs ONE real round" in captions
    assert "28-31 s" in captions or "s for one round" in captions
    assert "subprocess" in captions


# ===========================================================================
# Never on Overview or Archive
# ===========================================================================
def test_fire_does_not_appear_on_overview():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert "FIRE" not in [h.value for h in at.header]
    assert not any(b.value == "FIRE" for b in at.button)


def test_fire_does_not_appear_on_archive():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    at.switch_page("pages/archive.py")
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert "FIRE" not in [h.value for h in at.header]
    assert not any(b.value == "FIRE" for b in at.button)


# ===========================================================================
# End to end -- real subprocess, real round, real completion
# ===========================================================================
def test_fire_end_to_end_via_ui():
    at = _fresh_mission_control()
    fire_buttons = [b for b in at.button if b.label == "FIRE"]
    assert len(fire_buttons) == 1
    fire_buttons[0].click().run(timeout=60)
    assert not at.exception, at.exception

    job = _session_get(at, "fire_job")
    assert job is not None, "FIRE did not launch a subprocess"
    proc = job["proc"]

    deadline = time.time() + 90
    while _session_get(at, "fire_job") is not None:
        assert time.time() < deadline, "FIRE did not complete within 90 s"
        time.sleep(2)
        at.run(timeout=60)
        assert not at.exception, at.exception

    assert proc.poll() is not None, "worker subprocess should have exited"
    fired = _session_get(at, "fired_round")
    assert fired is not None
    assert fired["ok"] is True
    assert fired["engagement"] == "long"  # the default mission
    assert fired["miss_m"] >= 0.0
    assert fired["state_trajectory_available"] is True
    assert "g_log" in fired and fired["g_log"] is not None
    tr = fired["state_trajectory"]
    assert tr is not None
    assert len(tr["t"]) > 0
    assert len(tr["position"]) == len(tr["t"])
    assert len(tr["mach"]) == len(tr["t"])
    # guided phase only: starts at deployment, not at the muzzle
    assert tr["t"][0] == pytest.approx(fired["t_dep_actual"], abs=0.5)

    metrics = {m.label: m.value for m in at.metric}
    assert "Miss distance" in metrics
    assert metrics["Miss distance"] == f"{fired['miss_m']:.1f} m"

    assert len(list(at.get("image"))) >= 1  # the fire_result_figure rendered


# ===========================================================================
# Cancel actually kills the subprocess
# ===========================================================================
def test_fire_cancel_terminates_the_subprocess():
    at = _fresh_mission_control()
    fire_buttons = [b for b in at.button if b.label == "FIRE"]
    fire_buttons[0].click().run(timeout=60)
    assert not at.exception, at.exception

    job = _session_get(at, "fire_job")
    assert job is not None
    proc = job["proc"]
    assert proc.poll() is None, "subprocess should still be running"

    time.sleep(2)
    at.run(timeout=60)
    cancel_buttons = [b for b in at.button if b.label == "Cancel"]
    assert len(cancel_buttons) == 1
    cancel_buttons[0].click().run(timeout=60)
    assert not at.exception, at.exception

    # give the OS a moment to finish tearing the process down
    for _ in range(20):
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    assert proc.poll() is not None, "Cancel did not terminate the subprocess"
    assert _session_get(at, "fire_job") is None
    assert any("Cancelled" in w.value for w in at.warning)
