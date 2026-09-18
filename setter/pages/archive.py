"""
L4 -- Archive: the script registry (Control Room spec Part B4).

Step 4 of the build order ships only the registry piece of L4 -- "cheap,
and it makes everything else browsable." The figures/campaign-data/run-log
galleries the full L4 spec also names are Step 7.

Every card here is read-only data from `setter.registry` plus one `stat`
call per path (`setter.provenance`); running a script goes through
`setter.runner`, which refuses anything `runner.can_run` doesn't clear --
Campaign-class entries and `migrate_c_tag` (fast, but writes the protected
`docs/monte_carlo.json` in place) show their command line instead of a
button, regardless of what this page's own logic does.

Registered in `setter/app.py` as a FILE-based `st.Page` (see the note in
`setter/pages/mission_control.py` on why); `render()` still calls
unconditionally at the bottom of this file, but stays a plain function so
it can also be unit-tested by calling it directly.
"""

from __future__ import annotations

import streamlit as st

from setter import provenance, registry, runner

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


def render() -> None:
    st.title("Archive — script registry")
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
        st.header(_SUBSYSTEM_TITLES.get(subsystem, subsystem))
        for entry in entries:
            _entry_card(entry)


render()
