"""
Unit tests for `setter.registry`, `setter.provenance` and `setter.runner`
(Control Room spec Part B4 -- the script registry).

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import pytest

from setter import provenance, registry, runner


def test_entries_have_unique_ids():
    ids = [e.id for e in registry.ENTRIES]
    assert len(ids) == len(set(ids))


def test_every_entry_is_in_a_known_subsystem():
    for e in registry.ENTRIES:
        assert e.subsystem in registry.SUBSYSTEMS, e.id


def test_every_entry_module_is_actually_importable():
    """The registry's whole point is provenance the app can trust -- a
    `module` string that doesn't resolve to a real analysis/*.py script
    would silently break every run button and staleness check for it."""
    import importlib
    for e in registry.ENTRIES:
        importlib.import_module(e.module)


def test_cost_class_is_one_of_the_four_named_in_the_spec():
    for e in registry.ENTRIES:
        assert e.cost_class in ("reader", "figure", "analysis", "campaign"), e.id


def test_campaign_class_entries_are_never_runnable():
    for e in registry.ENTRIES:
        if e.cost_class == "campaign":
            assert not e.runnable, e.id
            assert not runner.can_run(e), e.id


def test_migrate_c_tag_is_not_runnable_despite_being_fast():
    """The one entry whose gate is NOT about cost: it mutates
    docs/monte_carlo.json in place, which this app is never allowed to
    write, regardless of how fast the migration itself is."""
    e = registry.by_id("migrate_c_tag")
    assert e is not None
    assert e.cost_class != "campaign"
    assert not e.runnable
    assert not runner.can_run(e)


def test_by_id_returns_none_for_unknown_id():
    assert registry.by_id("not_a_real_script") is None


def test_by_subsystem_covers_every_entry_exactly_once():
    grouped = registry.by_subsystem()
    total = sum(len(v) for v in grouped.values())
    assert total == len(registry.ENTRIES)


# ===========================================================================
# runner -- subprocess safety
# ===========================================================================
def test_run_entry_refuses_a_non_runnable_entry():
    e = registry.by_id("monte_carlo")
    with pytest.raises(ValueError):
        runner.run_entry(e)


def test_run_entry_executes_a_real_readonly_script():
    """compare_c_tags is explicitly read-only ("Writes nothing.") per its
    own docstring -- safe to actually execute in a test."""
    e = registry.by_id("compare_c_tags")
    assert runner.can_run(e)
    result = runner.run_entry(e, timeout_s=60)
    assert result.ok, result.stderr
    assert result.returncode == 0
    assert not result.timed_out


# ===========================================================================
# provenance -- staleness / file resolution
# ===========================================================================
def test_provenance_check_runs_for_every_entry_without_raising():
    for e in registry.ENTRIES:
        st = provenance.check(e)
        assert st.stale in (True, False, None)


def test_provenance_finds_the_checked_in_monte_carlo_json():
    e = registry.by_id("monte_carlo_report")
    st = provenance.check(e)
    assert "docs/monte_carlo.json" in st.reads_found
    assert not st.reads_missing


def test_provenance_glob_resolves_figure_outputs():
    e = registry.by_id("monte_carlo_figures")
    st = provenance.check(e)
    # docs/figures/mc_*.png is checked into the repo (Step 6 ran it) --
    # the glob pattern should resolve to at least one real file.
    assert st.writes_found == ("docs/figures/mc_*.png",)


def test_provenance_reports_unknown_stale_when_entry_writes_nothing():
    e = registry.by_id("compare_c_tags")
    st = provenance.check(e)
    assert st.stale is None
