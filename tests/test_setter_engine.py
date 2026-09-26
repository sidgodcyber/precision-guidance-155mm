"""
The Engine tests (Control Room spec Part B6, Step 8 second half).

Two of these run the live model-comparison worker end to end (~2-5 s for
the cheapest engagement) -- consistent with the project's existing practice
of exercising real subprocess workers in tests/test_setter_fire.py.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from setter import simulation_adapter as sa
from setter import validation_data as vd
from setter.config import DOCS_DIR, SUPPORTED_ENGAGEMENTS

APP_PATH = Path(__file__).resolve().parent.parent / "setter" / "app.py"
REPO_ROOT = APP_PATH.parent.parent


def _fresh_engine() -> AppTest:
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, at.exception
    at.switch_page("pages/engine.py")
    at.run(timeout=60)
    assert not at.exception, at.exception
    return at


def _session_get(at: AppTest, key: str):
    return at.session_state[key] if key in at.session_state else None


# ===========================================================================
# Reachability / registration
# ===========================================================================
def test_engine_reachable_without_firing():
    at = _fresh_engine()
    assert at.title[0].value == "The Engine"


def test_engine_registered_unconditionally_in_navigation():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert "fired_round" not in at.session_state
    at.switch_page("pages/engine.py")
    at.run(timeout=60)
    assert not at.exception, at.exception


# ===========================================================================
# Ground-truth cross-checks -- independent of the page
# ===========================================================================
def test_firing_table_rms_matches_independently_computed_ground_truth():
    with open(DOCS_DIR / "validation_results.json", encoding="utf-8") as fh:
        rows = json.load(fh)["firing_table"]
    expected = math.sqrt(sum(r["range_err_pct"] ** 2 for r in rows) / len(rows))
    assert vd.firing_table_range_rms_pct() == pytest.approx(expected)
    assert vd.firing_table_range_rms_pct() == pytest.approx(0.477, abs=0.01)

    at = _fresh_engine()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Range RMS error vs firing table, 15 cases"] == "0.48 %"


def test_model_error_summary_matches_model_error_md_recommended_entries():
    summary = vd.model_error_summary()
    assert summary["range"]["rms_m"] == pytest.approx(0.65, abs=0.02)
    assert summary["deflection"]["rms_m"] == pytest.approx(0.06, abs=0.01)
    assert summary["recommended_budget_m"]["range_1sigma"] == 0.7
    assert summary["recommended_budget_m"]["deflection_1sigma"] == 0.1


def test_engagement_to_firing_table_index_mapping_matches_baseline_charges():
    from analysis.mpmm_compare import FIRING_TABLE
    for e in SUPPORTED_ENGAGEMENTS:
        idx = vd.ENGAGEMENT_TO_FIRING_TABLE_INDEX[e]
        charge, mv, qe_mils, *_ = FIRING_TABLE[idx]
        base = sa.baseline_for(e)
        assert base["charge"] == charge
        assert base["qe_mils"] == pytest.approx(qe_mils)


def test_mpmm_point_reproduces_stored_json_for_every_engagement():
    with open(DOCS_DIR / "mpmm_results.json", encoding="utf-8") as fh:
        rows = json.load(fh)
    for e in SUPPORTED_ENGAGEMENTS:
        idx = vd.ENGAGEMENT_TO_FIRING_TABLE_INDEX[e]
        point = vd.mpmm_point(e)
        assert point.available
        assert point.max_divergence_m == pytest.approx(rows[idx]["max_divergence_m"])


# ===========================================================================
# The rendered page -- static content
# ===========================================================================
def test_engine_shows_four_chain_links():
    at = _fresh_engine()
    assert len(list(at.expander)) == 4


def test_guns_to_tables_link_is_marked_as_citation_not_measured():
    at = _fresh_engine()
    mds = " ".join(m.value for m in at.markdown)
    assert "Given, not measured in this repository" in mds


def test_reduced_order_to_flight_computer_is_validated_by_construction():
    at = _fresh_engine()
    mds = " ".join(m.value for m in at.markdown)
    assert "Validated by construction, not by comparison" in mds


def test_engine_page_does_not_contradict_error_budget_model_error_term():
    at = _fresh_engine()
    warnings = " ".join(w.value for w in at.warning)
    assert "Task U" in warnings
    assert "different, unrelated quantity" in warnings

    from setter.pages import error_budget as eb_mod
    assert "different, unrelated quantity" in eb_mod.campaign.error_budget.__doc__


def test_module_diagram_line_counts_match_live_file_read():
    at = _fresh_engine()
    metrics = {m.label: m.value for m in at.metric}
    total = 0
    for rel in ("models/mpmm.py", "gnc/guidance.py", "gnc/inverse_map.py",
                "gnc/navigation.py", "gnc/roll_control.py", "gnc/scheduler.py"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        total += sum(1 for ln in text.splitlines() if ln.strip())
    assert metrics["Total"] == f"{total:,} lines"
    assert 4500 <= total <= 5200


def test_sensors_boundary_quote_present():
    at = _fresh_engine()
    infos = " ".join(i.value for i in at.info)
    assert "truth_at" in infos
    assert "boundary between the simulator and the sensor models" in infos


# ===========================================================================
# Live model-comparison worker -- fast negative case
# ===========================================================================
def test_engine_worker_rejects_unknown_engagement(tmp_path):
    out_path = tmp_path / "engine_result.json"
    result = subprocess.run(
        [sys.executable, "-m", "setter.engine_worker",
         "--engagement", "not_a_real_engagement", "--out", str(out_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    data = json.loads(out_path.read_text())
    assert data["ok"] is False
    assert "not_a_real_engagement" in data["error"]


# ===========================================================================
# Live model-comparison -- end to end, real subprocess, cheapest engagement
# ===========================================================================
def test_engine_live_comparison_end_to_end():
    at = _fresh_engine()
    selectboxes = {s.key: s for s in at.selectbox}
    selectboxes["engine_engagement"].set_value("short").run(timeout=60)
    assert not at.exception, at.exception

    buttons = {b.key: b for b in at.button}
    buttons["engine_run_live"].click().run(timeout=60)
    assert not at.exception, at.exception

    job = _session_get(at, "engine_job")
    assert job is not None, "live comparison did not launch a subprocess"

    deadline = time.time() + 60
    while _session_get(at, "engine_job") is not None:
        assert time.time() < deadline, "live comparison did not complete within 60 s"
        time.sleep(1)
        at.run(timeout=60)
        assert not at.exception, at.exception

    result = _session_get(at, "engine_result")
    assert result is not None
    assert result["ok"] is True
    assert result["engagement"] == "short"
    assert result["max_divergence_m"] == pytest.approx(
        result["stored_comparison"]["max_divergence_m"], abs=1e-6)
    assert len(result["divergence"]["distance_m"]) > 0
    assert len(list(at.get("image"))) >= 3  # chain figures + comparison figure


def test_engine_live_comparison_cancel_terminates_subprocess():
    at = _fresh_engine()
    selectboxes = {s.key: s for s in at.selectbox}
    selectboxes["engine_engagement"].set_value("short").run(timeout=60)
    assert not at.exception, at.exception

    buttons = {b.key: b for b in at.button}
    buttons["engine_run_live"].click().run(timeout=60)
    assert not at.exception, at.exception

    job = _session_get(at, "engine_job")
    assert job is not None
    proc = job["proc"]

    buttons = {b.key: b for b in at.button}
    assert "engine_cancel" in buttons
    buttons["engine_cancel"].click().run(timeout=60)
    assert not at.exception, at.exception

    for _ in range(20):
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    assert proc.poll() is not None, "Cancel did not terminate the subprocess"
    assert _session_get(at, "engine_job") is None
