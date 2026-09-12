"""
Phase 2 characterization: Streamlit demonstrator validation via
`streamlit.testing.v1.AppTest`.

Exercises application startup, every engagement, every met-age bucket, every
event-mode, trajectory rendering, configuration validation, JSON generation,
message export, and out-of-range widget input -- without a browser. Looks
specifically for exceptions during widget interaction, Task A/Task C label
consistency, and provenance clarity (precomputed vs. live values).

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from setter import campaign
from setter.config import MET_AGE_BUCKETS, SUPPORTED_ENGAGEMENTS

APP_PATH = Path(__file__).resolve().parent.parent / "setter" / "app.py"
REPO_ROOT = APP_PATH.parent.parent


def _fresh_app() -> AppTest:
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, at.exception
    return at


# ===========================================================================
# 1. Application startup
# ===========================================================================
def test_app_starts_without_exception():
    at = _fresh_app()
    assert at.title[0].value == "SIMULATION SETTER"


def test_app_importable_under_real_streamlit_run_sys_path():
    """Regression test for the launch defect found in Phase 3:
    `streamlit run setter/app.py` execs the file directly and puts only its
    OWN directory on `sys.path` (`streamlit.web.bootstrap._fix_sys_path`) --
    never the repository root. `AppTest`/pytest can never catch this,
    because the process running them already has the repo root on
    `sys.path` for unrelated reasons (pytest's own rootdir handling, or
    `python -c`'s implicit cwd entry). This test runs in an isolated
    subprocess with a `sys.path` built the same way Streamlit's bootstrap
    builds it -- script directory only, no cwd, no inherited PYTHONPATH --
    and simply imports app.py's module-level code up to (not including) the
    first Streamlit-runtime call, to confirm `setter`/`fuze` resolve."""
    probe = f"""
import sys
# `python -c` auto-adds cwd as '' at sys.path[0]; strip that AND the repo
# root out first, then reproduce exactly what
# streamlit.web.bootstrap._fix_sys_path does: insert only the script's own
# directory. What remains (stdlib, site-packages) is untouched.
_repo_root = {str(REPO_ROOT)!r}
sys.path = [p for p in sys.path if p not in ('', _repo_root)]
sys.path.insert(0, {str(APP_PATH.parent)!r})

import ast
app_path = {str(APP_PATH)!r}
source = open(app_path).read()
tree = ast.parse(source)
import_nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
# Exec the file's source up to and including the LAST top-level import --
# not just the import nodes in isolation -- so any non-import statement
# sitting between imports (such as the sys.path repair itself) still runs.
# This is what would actually execute before app.py reaches its first
# Streamlit-runtime call, which is as far as this probe needs to go.
cutoff_line = import_nodes[-1].end_lineno
prefix_source = \"\\n\".join(source.splitlines()[:cutoff_line])
exec(compile(prefix_source, app_path, "exec"), {{"__name__": "__main__", "__file__": app_path}})
print("IMPORTS_OK")
"""
    import os
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=60)
    assert "IMPORTS_OK" in result.stdout, (
        f"app.py's imports failed under a real streamlit-run-style sys.path.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}")


# ===========================================================================
# 2. Engagement selection -- all five, no exceptions
# ===========================================================================
@pytest.mark.parametrize("engagement", SUPPORTED_ENGAGEMENTS)
def test_engagement_selection_all_five(engagement):
    at = _fresh_app()
    box = [s for s in at.selectbox if s.label == "Engagement"][0]
    box.set_value(engagement).run(timeout=60)
    assert not at.exception, at.exception


# ===========================================================================
# 3. Met-age selection -- all seven buckets, no exceptions
# ===========================================================================
@pytest.mark.parametrize("age_bucket", MET_AGE_BUCKETS)
def test_met_age_selection_all_buckets(age_bucket):
    at = _fresh_app()
    slider = at.select_slider[0]
    slider.set_value(age_bucket).run(timeout=60)
    assert not at.exception, at.exception


@pytest.mark.parametrize("age_bucket", MET_AGE_BUCKETS)
def test_met_age_ui_readout_matches_the_data_layer(age_bucket):
    """The UI's Task C readout for a selected met-age bucket must retrieve
    the SAME value `setter.campaign.task_c_point` reports -- there is no
    separate copy of the number living in the UI layer."""
    at = _fresh_app()
    slider = at.select_slider[0]
    slider.set_value(age_bucket).run(timeout=60)
    assert not at.exception, at.exception

    expected = campaign.task_c_point(age_bucket)
    texts = [w.value for w in at.get("markdown")]
    if expected.available:
        needle = f"At **{age_bucket}**: CEP {expected.cep_m:.2f} m"
        assert any(needle in t for t in texts), texts
    else:
        infos = [w.value for w in at.info]
        assert any(expected.reason in t for t in infos), infos


# ===========================================================================
# 4. Event-mode selection
# ===========================================================================
@pytest.mark.parametrize("mode", ["time", "motion", "proximity", "combined"])
def test_event_mode_selection(mode):
    at = _fresh_app()
    box = [s for s in at.selectbox if s.label.startswith("Event engine")][0]
    box.set_value(mode).run(timeout=60)
    assert not at.exception, at.exception


# ===========================================================================
# 5. Trajectory rendering
# ===========================================================================
def test_trajectory_and_knowledge_term_plots_render():
    """`st.pyplot` figures surface as `image` elements in AppTest's tree --
    at least the ground-track/altitude figure and the knowledge-term figure
    must both have rendered without raising."""
    at = _fresh_app()
    assert not at.exception
    assert len(list(at.get("image"))) >= 2


# ===========================================================================
# 6. Configuration validation
# ===========================================================================
def test_configuration_validates_by_default():
    at = _fresh_app()
    assert [s.value for s in at.success] == ["Configuration message is valid."]
    assert list(at.error) == []


# ===========================================================================
# 7. JSON generation / 8. message export
# ===========================================================================
def test_serialized_json_is_present_and_parses():
    at = _fresh_app()
    blocks = list(at.code)
    assert blocks, "expected a st.code block with the serialized message"
    data = json.loads(blocks[0].value)
    assert data["schema"] == "SIM-SETTER-V1"


def test_export_button_present_and_enabled_when_valid():
    at = _fresh_app()
    downloads = list(at.get("download_button"))
    assert len(downloads) == 1
    assert downloads[0].disabled is False


def test_message_size_and_checksum_metrics_present():
    at = _fresh_app()
    labels = {m.label: m.value for m in at.metric}
    assert "Message size" in labels and labels["Message size"].endswith("bytes")
    assert "CRC32" in labels
    assert labels["Round-trip check"] == "OK"


# ===========================================================================
# 9. Invalid configuration handling
# ===========================================================================
def test_numeric_widgets_reject_out_of_range_input():
    """The dashboard's own numeric widgets are declared with `min_value`,
    so an out-of-range value cannot be driven into them through normal
    interaction: AppTest confirms the widget simply does not accept it
    (value is left unchanged) rather than the app crashing or silently
    building an invalid message from it."""
    at = _fresh_app()
    threshold = [n for n in at.number_input if n.label == "motion_event.threshold"][0]
    original = threshold.value
    threshold.set_value(-5.0).run(timeout=60)
    assert not at.exception, at.exception
    still = [n for n in at.number_input if n.label == "motion_event.threshold"][0]
    assert still.value == original
    assert [s.value for s in at.success] == ["Configuration message is valid."]


# ===========================================================================
# Task A / Task C label consistency and provenance clarity
# ===========================================================================
def test_task_a_and_task_c_labels_are_distinct_in_the_ui():
    at = _fresh_app()
    headers = [h.value for h in at.header] + [h.value for h in at.subheader]
    joined = " ".join(headers)
    assert "Navigation-in-loop campaign reference" in joined
    assert "atmospheric" in joined.lower() and "knowledge" in joined.lower()


def test_ui_marks_trajectory_as_visualization_not_campaign_result():
    at = _fresh_app()
    captions = [c.value for c in at.caption]
    assert any("not a new campaign result" in c for c in captions)


def test_ui_marks_met_age_control_as_not_launching_a_new_campaign():
    at = _fresh_app()
    slider = at.select_slider[0]
    assert slider.help is not None
    assert "new Monte Carlo campaign" in slider.help


# ===========================================================================
# Stale-cache / recomputation characterization
# ===========================================================================
def test_repeated_reruns_of_same_engagement_are_stable():
    """Re-running the script for the SAME engagement selection must not
    change the campaign readouts (they are precomputed lookups) even though
    the script re-executes top to bottom on every Streamlit interaction."""
    at = _fresh_app()
    metrics_1 = {m.label: m.value for m in at.metric}
    at.run(timeout=60)
    metrics_2 = {m.label: m.value for m in at.metric}
    for label in ("Fresh met (long, upload-at-fuze-setting)",
                  "2h met (long, headline)",
                  "Physical control (long, no dispersion top-up)"):
        assert metrics_1[label] == metrics_2[label]
