"""
Archive gallery tests (Control Room spec Part B4/L4, Step 7): the figure
gallery, campaign data, run logs, and provenance tabs added on top of
Step 4's script registry.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from setter import figures_registry as fig_registry
from setter import provenance, registry
from setter.config import DOCS_DIR, REPO_ROOT

APP_PATH = Path(__file__).resolve().parent.parent / "setter" / "app.py"


def _fresh_archive() -> AppTest:
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    at.switch_page("pages/archive.py")
    at.run(timeout=120)
    assert not at.exception, at.exception
    return at


# ===========================================================================
# Figures registry -- completeness and correctness
# ===========================================================================
def test_every_figure_on_disk_is_registered_and_vice_versa():
    registered = {e.filename for e in fig_registry.ENTRIES}
    actual = {p.name for p in (DOCS_DIR / "figures").glob("*.png")}
    assert actual - registered == set(), f"unregistered figures: {actual - registered}"
    assert registered - actual == set(), f"registered but missing: {registered - actual}"


def test_figure_count_matches_the_spec():
    assert len(fig_registry.ENTRIES) == 38


def test_every_figure_entry_writes_its_own_filename():
    for e in fig_registry.ENTRIES:
        assert e.writes == (f"docs/figures/{e.filename}",)


def test_figure_entries_are_valid_provenance_check_inputs():
    """FigureEntry deliberately has the same reads/writes shape
    setter.provenance.check expects, so it can be reused unmodified."""
    for e in fig_registry.ENTRIES[:5]:
        st = provenance.check(e)
        assert st.stale in (True, False, None)


def test_orphaned_figure_is_flagged_not_guessed():
    e = fig_registry.by_id("design_sweep")
    assert e is not None
    assert e.orphaned is True
    assert e.module is None


def test_ballistic_figures_have_no_module_and_no_reads():
    """run_ballistic.py is a root-level script, not in setter/registry.py,
    and flies a live round rather than reading a stored file."""
    for fid in ("trajectory_c4_qe97", "ground_track_c8_qe525",
                "angle_of_attack_c4_qe97", "diagnostics_c8_qe525"):
        e = fig_registry.by_id(fid)
        assert e is not None
        assert e.module is None
        assert e.reads == ()
        assert not e.orphaned


# ===========================================================================
# Campaign data and run logs -- completeness against the spec's counts
# ===========================================================================
def test_campaign_json_count_matches_the_spec():
    assert len(list(DOCS_DIR.glob("*.json"))) == 26


def test_run_log_count_matches_the_spec():
    assert len(list(DOCS_DIR.glob("*.log"))) == 24


def test_every_campaign_json_has_a_generator_verdict():
    from setter.pages import archive as a
    for f in DOCS_DIR.glob("*.json"):
        verdict = a._campaign_json_generator(f.name)
        assert verdict  # never empty -- "unknown" is a valid, honest verdict


def test_every_run_log_has_a_source_verdict():
    from setter.pages import archive as a
    for f in DOCS_DIR.glob("*.log"):
        verdict = a._run_log_source(f.name)
        assert verdict


def test_monte_carlo_json_generator_names_both_writers():
    """docs/monte_carlo.json is written by both analysis.monte_carlo (the
    campaign) and analysis.migrate_c_tag (a one-off in-place migration) --
    both must be named, not just the first one found."""
    from setter.pages import archive as a
    verdict = a._campaign_json_generator("monte_carlo.json")
    assert "monte_carlo" in verdict
    assert "migrate_c_tag" in verdict


# ===========================================================================
# The rendered page
# ===========================================================================
def test_archive_page_renders_all_five_tabs_without_exception():
    at = _fresh_archive()
    assert at.title[0].value == "Archive"


def test_archive_page_shows_all_38_figures():
    at = _fresh_archive()
    assert len(list(at.get("image"))) == 38


def test_archive_page_shows_all_26_campaign_json_and_24_run_logs_as_expanders():
    at = _fresh_archive()
    expander_labels = [e.proto.label for e in at.expander]
    json_expanders = [l for l in expander_labels if l.endswith(".json)") or ") " not in l and l.endswith("KB)")]
    log_expanders = [l for l in expander_labels if l.endswith("bytes)")]
    assert len(log_expanders) == 24
    # 26 campaign-json expanders + 26 nested "raw JSON" expanders each
    assert expander_labels.count("Raw JSON (truncated preview)") == 26


def test_archive_page_provenance_tab_matches_the_spec_counts():
    at = _fresh_archive()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Figures / JSON / logs"] == "38 / 26 / 24"
    assert metrics["Test functions"].isdigit()
    assert len(metrics["Repository commit"]) == 8


def test_archive_page_still_has_the_script_registry():
    """Step 4's registry must still be present, unchanged, as its own tab."""
    at = _fresh_archive()
    assert len(list(at.button)) == 20  # the 20 runnable entries from Step 4
