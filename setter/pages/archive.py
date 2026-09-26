"""
L4 -- Archive (Control Room spec Part B4 / Step 7): script registry,
figure gallery, campaign data, run logs, and provenance.

Step 4 shipped only the script registry ("cheap, and it makes everything
else browsable"). Step 7 adds the rest of the L4 spec: all 38 figures with
their generating script and source data, all 26 campaign JSON files with a
readable summary and a raw view, all 24 run logs, and a provenance panel.

Registered in `setter/app.py` as a FILE-based `st.Page`; `render()` stays a
plain function so it can also be unit-tested by calling it directly.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import streamlit as st

from setter import figures_registry as fig_registry
from setter import provenance, registry, runner
from setter.config import DOCS_DIR, REPO_ROOT

_SUBSYSTEM_TITLES = {
    "ballistics_validation": "Ballistics validation",
    "authority": "Correction authority",
    "guidance": "Guidance",
    "roll_servo": "Roll servo",
    "navigation": "Navigation",
    "monte_carlo": "Monte Carlo campaign",
}

_COST_CLASS_LABEL = {
    "reader": "Reader · seconds",
    "figure": "Figure · seconds",
    "analysis": "Analysis · seconds to a minute",
    "campaign": "Campaign · minutes to hours",
}


def _format_age(seconds) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 120:
        return f"{seconds:.0f} s ago"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h ago"
    return f"{seconds / 86400:.1f} d ago"


# ===========================================================================
# Tab 1 -- Script registry (Step 4, unchanged)
# ===========================================================================
@st.fragment
def _entry_card(entry: registry.ScriptEntry) -> None:
    """One entry, one fragment: clicking Run on one script reruns only its
    own card -- not the other 34 -- and only this card re-stats its own
    files on that rerun."""
    stale = provenance.check(entry)

    with st.container(border=True):
        top = st.columns([3, 1])
        top[0].markdown(f"**{entry.title}**")
        top[0].caption(f"`python -m {entry.module}`")
        top[1].caption(_COST_CLASS_LABEL[entry.cost_class])

        st.caption(entry.description)

        if entry.reads:
            st.caption("reads: " + ", ".join(f"`{r}`" for r in entry.reads))
        if entry.writes:
            st.caption("writes: " + ", ".join(f"`{w}`" for w in entry.writes))
        if entry.notes:
            st.caption(entry.notes)

        if stale.stale is True:
            st.warning(
                f"Stale -- an input changed after the output was last written "
                f"(output {_format_age(stale.oldest_write_age_s)}, "
                f"newest input {_format_age(stale.newest_read_age_s)}).")
        elif stale.stale is False:
            st.caption(f"Output current, last written {_format_age(stale.oldest_write_age_s)}.")
        if stale.writes_missing:
            st.caption(f"Not yet generated on this checkout: {', '.join(stale.writes_missing)}")
        if stale.reads_missing:
            st.caption(f"Missing input on this checkout: {', '.join(stale.reads_missing)}")

        if runner.can_run(entry):
            if st.button("Run", key=f"run_{entry.id}"):
                st.caption(f"Running `python -m {entry.module}` -- typically {entry.cost}.")
                with st.status(f"Running {entry.title}…", expanded=True) as status:
                    result = runner.run_entry(entry)
                    if result.stdout.strip():
                        st.code(result.stdout[-4000:], language="text")
                    if result.stderr.strip():
                        st.code(result.stderr[-2000:], language="text")
                    if result.timed_out:
                        status.update(label=f"{entry.title}: timed out", state="error")
                    elif result.ok:
                        status.update(label=f"{entry.title}: complete", state="complete")
                    else:
                        status.update(
                            label=f"{entry.title}: failed (exit {result.returncode})",
                            state="error")
        else:
            st.caption(f"Runs from a terminal only — {entry.cost}.")


def _render_registry_tab() -> None:
    st.caption(
        "Every analysis/*.py script with a command-line entry point, what "
        "each reads and writes, and whether its output is current against "
        "its input. Reader/Figure/Analysis scripts can be re-run from "
        "here; Campaign scripts (minutes to hours) and anything that "
        "writes a campaign data file in place show their command line and "
        "run from a terminal only -- never from this page.")

    entries_by_subsystem = registry.by_subsystem()
    n_runnable = sum(1 for e in registry.ENTRIES if runner.can_run(e))
    st.caption(f"{len(registry.ENTRIES)} scripts · {n_runnable} runnable from this page.")

    for subsystem in registry.SUBSYSTEMS:
        entries = entries_by_subsystem.get(subsystem, [])
        if not entries:
            continue
        st.subheader(_SUBSYSTEM_TITLES.get(subsystem, subsystem))
        for entry in entries:
            _entry_card(entry)


# ===========================================================================
# Tab 2 -- Figures
# ===========================================================================
def _figure_card(entry: fig_registry.FigureEntry) -> None:
    stale = provenance.check(entry)
    with st.container(border=True):
        cols = st.columns([1, 1.6])
        with cols[0]:
            st.image(str(DOCS_DIR / "figures" / entry.filename), width="stretch")
        with cols[1]:
            st.markdown(f"**{entry.filename}**")
            st.caption(entry.caption)
            if entry.orphaned:
                st.warning(f"No current generator found. {entry.generator_note}")
            elif entry.module:
                st.caption(f"generated by `python -m {entry.module}`")
                if entry.reads:
                    st.caption("reads: " + ", ".join(f"`{r}`" for r in entry.reads))
            else:
                st.caption(entry.generator_note)

            if stale.stale is True:
                st.warning(
                    f"Stale (output {_format_age(stale.oldest_write_age_s)}, "
                    f"newer input {_format_age(stale.newest_read_age_s)}).")
            elif stale.stale is False and not entry.orphaned:
                st.caption(f"Written {_format_age(stale.oldest_write_age_s)}.")


def _render_figures_tab() -> None:
    st.caption(
        "All 38 files in docs/figures/, each with its caption (its "
        "generating function's own docstring), its generating script, and "
        "its source data. Two of these -- see the ballistics-validation "
        "group and 'design_sweep' below -- don't fit the usual "
        "script-reads-JSON-writes-PNG shape, and say so rather than "
        "guessing: 8 come from a root-level Step 1 script that flies a "
        "live 6-DOF round instead of reading a stored file, and one has no "
        "current generator at all.")

    n_orphaned = sum(1 for e in fig_registry.ENTRIES if e.orphaned)
    st.caption(f"{len(fig_registry.ENTRIES)} figures · {n_orphaned} with no current generator.")

    by_subsystem = {}
    for e in fig_registry.ENTRIES:
        by_subsystem.setdefault(e.subsystem, []).append(e)

    for subsystem in registry.SUBSYSTEMS:
        entries = by_subsystem.get(subsystem, [])
        if not entries:
            continue
        st.subheader(_SUBSYSTEM_TITLES.get(subsystem, subsystem))
        for entry in entries:
            _figure_card(entry)


# ===========================================================================
# Tab 3 -- Campaign data
# ===========================================================================
#: docs/*.json files registry.py's own `writes` fields don't cover --
#: confirmed by grep + `git log --diff-filter=A` (same method as
#: setter/figures_registry.py's design_sweep.png finding), not guessed.
_CAMPAIGN_JSON_OVERRIDES = {
    "design_sweep_best.json": "analysis.design_sweep (--best variant, explicit --out override; not the registry's default invocation)",
    "design_sweep_combos.json": "analysis.design_sweep (--combos variant, explicit --out override; not the registry's default invocation)",
    "validation_results.json": "run_validation.py (root-level script, Step 1; not in setter/registry.py)",
    "nav_coast.json": "unknown — no current generator found (same pattern as docs/figures/design_sweep.png; see setter/figures_registry.py)",
    "nav_tuning.json": "unknown — no current generator found (same pattern as docs/figures/design_sweep.png; see setter/figures_registry.py)",
}


def _campaign_json_generator(filename: str) -> str:
    if filename in _CAMPAIGN_JSON_OVERRIDES:
        return _CAMPAIGN_JSON_OVERRIDES[filename]
    writers = [e.id for e in registry.ENTRIES if any(w.endswith(filename) for w in e.writes)]
    if not writers:
        return "unknown"
    if len(writers) == 1:
        return f"`python -m {registry.by_id(writers[0]).module}`"
    return " and ".join(f"`python -m {registry.by_id(w).module}`" for w in writers)


def _shape(v) -> str:
    if isinstance(v, dict):
        return f"dict, {len(v)} keys"
    if isinstance(v, list):
        return f"list, {len(v)} items"
    return type(v).__name__


@st.cache_data(ttl=300)
def _json_file_summary(filename: str):
    p = DOCS_DIR / filename
    size = p.stat().st_size
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        shape = {k: _shape(v) for k, v in data.items()}
    else:
        shape = {"(top level)": _shape(data)}
    return size, shape, data


def _render_campaign_data_tab() -> None:
    st.caption(
        "All 26 JSON files in docs/, each with a readable top-level summary "
        "and a raw view. These are the same files setter.campaign already "
        "reads for Mission Control/Error Budget -- this tab is for looking "
        "at everything else in them.")

    files = sorted(f.name for f in DOCS_DIR.glob("*.json"))
    st.caption(f"{len(files)} files.")

    for filename in files:
        size, shape, data = _json_file_summary(filename)
        with st.expander(f"{filename}  ({size / 1024:.0f} KB)"):
            st.caption(f"generated by: {_campaign_json_generator(filename)}")
            st.markdown("**top-level shape**")
            for k, v in shape.items():
                st.caption(f"`{k}`: {v}")
            with st.expander("Raw JSON (truncated preview)"):
                raw = json.dumps(data, indent=1)
                preview = raw[:5000]
                st.code(preview, language="json")
                if len(raw) > 5000:
                    st.caption(f"truncated — {len(raw)} characters total. "
                              f"Read the file directly for the full content: "
                              f"`docs/{filename}`.")


# ===========================================================================
# Tab 4 -- Run logs
# ===========================================================================
#: filename -> what produced it, by the same grep+git-log method as the
#: figures/campaign-data overrides above. Not all 24 map cleanly onto a
#: single registry entry (some are numbered re-runs of the same script;
#: two are root-level Step 1 scripts; one has no current generator).
_RUN_LOG_SOURCE = {
    "authority_run.log": "authority",
    "guidance_cep_run.log": "guidance_cep",
    "guidance_cep_run2.log": "guidance_cep",
    "guidance_map_run.log": "guidance_authority",
    "guidance_pointing_run.log": "guidance_authority (--tasks pointing)",
    "monte_carlo_run.log": "monte_carlo",
    "monte_carlo_run2.log": "monte_carlo",
    "monte_carlo_run3.log": "monte_carlo",
    "mpmm_compare.log": "mpmm_compare",
    "mpmm_compute.log": "mpmm_compute",
    "nav_ablation_run.log": "nav_ablation",
    "nav_ablation_d_run.log": "nav_ablation (--tasks d)",
    "nav_antenna_run.log": "nav_antenna",
    "nav_cep_run.log": "nav_cep",
    "nav_consistency_run.log": "nav_consistency",
    "nav_drivers_run.log": "nav_drivers",
    "nav_sensors_run.log": "nav_sensors",
    "nav_tuning_run.log": "unknown — no current generator found (see docs/nav_tuning.json)",
    "roll_robustness_run.log": "roll_robustness",
    "roll_servo_run.log": "roll_servo",
    "run_ballistic_c4.log": "run_ballistic.py (root-level script, Step 1)",
    "run_ballistic_c8.log": "run_ballistic.py (root-level script, Step 1)",
    "staged_deployment_run.log": "staged_deployment",
    "validation_run.log": "run_validation.py (root-level script, Step 1)",
}


def _run_log_source(filename: str) -> str:
    """`_RUN_LOG_SOURCE`'s values are either a bare registry id (look up
    and format as a real command) or an already-written description
    (root-level script, unknown, or a registry id plus a parenthetical
    note on which flags that particular run used) -- shown as-is."""
    src = _RUN_LOG_SOURCE.get(filename, "unknown")
    entry = registry.by_id(src)
    if entry:
        return f"`python -m {entry.module}`"
    base = src.split(" (")[0]
    base_entry = registry.by_id(base)
    if base_entry:
        note = src[len(base):].strip()
        return f"`python -m {base_entry.module}` {note}"
    return src


def _render_run_logs_tab() -> None:
    st.caption(
        "All 24 run logs in docs/ -- plain stdout captures from the "
        "terminal invocations that produced the campaign data, kept for "
        "provenance. Each is matched to the script that produced it where "
        "the filename makes that unambiguous.")

    files = sorted(Path(f).name for f in DOCS_DIR.glob("*.log"))
    st.caption(f"{len(files)} files.")

    for filename in files:
        p = DOCS_DIR / filename
        size = p.stat().st_size
        with st.expander(f"{filename}  ({size} bytes)"):
            st.caption(f"from: {_run_log_source(filename)}")
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                text = f"(could not read: {exc})"
            st.code(text, language="text")


# ===========================================================================
# Tab 5 -- Provenance
# ===========================================================================
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


@st.cache_data(ttl=300)
def _test_function_count() -> int:
    total = 0
    for f in (REPO_ROOT / "tests").glob("*.py"):
        total += f.read_text(encoding="utf-8").count("def test_")
    return total


def _render_provenance_tab() -> None:
    st.caption("Repository state at the time this page was loaded.")
    commit = _git_commit()
    col1, col2, col3 = st.columns(3)
    col1.metric("Repository commit", commit[:8] if commit != "unknown" else "unknown")
    col2.metric("Test functions", _test_function_count())
    col3.metric("Figures / JSON / logs", f"{len(fig_registry.ENTRIES)} / "
               f"{len(list(DOCS_DIR.glob('*.json')))} / {len(list(DOCS_DIR.glob('*.log')))}")
    st.caption(
        "Test function count is a static grep of `def test_` across "
        "tests/*.py, not a live pytest collection (that imports the whole "
        "engine and costs seconds) -- see setter/pages/overview.py's "
        "identical convention. It undercounts pytest's own reported total "
        "because parametrized tests multiply one function into many cases.")


# ===========================================================================
def render() -> None:
    st.title("Archive")
    st.caption(
        "Script registry, figure gallery, campaign data, run logs, and "
        "provenance -- everything precomputed in this repository, made "
        "browsable rather than requiring a source read.")

    tabs = st.tabs(["Script registry", "Figures", "Campaign data", "Run logs", "Provenance"])
    with tabs[0]:
        _render_registry_tab()
    with tabs[1]:
        _render_figures_tab()
    with tabs[2]:
        _render_campaign_data_tab()
    with tabs[3]:
        _render_run_logs_tab()
    with tabs[4]:
        _render_provenance_tab()


render()
