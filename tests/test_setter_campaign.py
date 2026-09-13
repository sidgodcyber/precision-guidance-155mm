"""
Unit tests for `setter.campaign` -- read-only access to
`docs/monte_carlo.json`'s Task A (navigation-in-loop) and Task C (truth-fed,
atmospheric-knowledge) campaign data.

Verifies the known headline numbers directly against the repository's own
JSON rather than hard-coding them in application logic.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import json

import pytest

from setter import campaign
from setter.config import MONTE_CARLO_PATH


pytestmark = pytest.mark.skipif(
    not MONTE_CARLO_PATH.exists(),
    reason="docs/monte_carlo.json not present")


@pytest.fixture(scope="module")
def raw():
    with open(MONTE_CARLO_PATH) as fh:
        return json.load(fh)


# ===========================================================================
# Task A lookup
# ===========================================================================
def test_task_a_headline_long_matches_repository_json(raw):
    expected = raw["a"]["headline"]["long"]["guided"]["cep_m"]
    got = campaign.task_a_point("long", "headline")
    assert got.available
    assert got.cep_m == pytest.approx(expected)
    assert got.cep_m == pytest.approx(106.19, abs=0.01)
    assert got.n == 192


def test_task_a_fresh_met_long_matches_repository_json(raw):
    expected = raw["a"]["fresh_met"]["long"]["guided"]["cep_m"]
    got = campaign.task_a_point("long", "fresh_met")
    assert got.available
    assert got.cep_m == pytest.approx(expected)
    assert got.cep_m == pytest.approx(27.51, abs=0.01)


def test_task_a_physical_long(raw):
    expected = raw["a"]["physical"]["long"]["guided"]["cep_m"]
    got = campaign.task_a_point("long", "physical")
    assert got.available
    assert got.cep_m == pytest.approx(expected)
    assert got.cep_m == pytest.approx(59.50, abs=0.01)


def test_task_a_headline_table_has_all_five_engagements():
    table = campaign.task_a_headline_table()
    assert set(table) == set(campaign.SUPPORTED_ENGAGEMENTS)
    assert all(p.available for p in table.values())


def test_task_a_headline_anomaly_is_not_monotonic_in_range():
    """The 2h headline CEP does not order by engagement range -- CLAUDE.md's
    reported anomaly. This test pins the raw shape, not an explanation."""
    table = campaign.task_a_headline_table()
    cep = {e: p.cep_m for e, p in table.items()}
    assert cep["mid2"] > cep["middle"]
    assert cep["mid2"] > cep["long"]


def test_unsupported_engagement_rejected():
    with pytest.raises(ValueError):
        campaign.task_a_point("nonexistent")


def test_unsupported_task_a_tag_rejected():
    with pytest.raises(ValueError):
        campaign.task_a_point("long", "not_a_tag")


# ===========================================================================
# "Do not fabricate intermediate Task A points" / unsupported Task A age
# ===========================================================================
def test_task_a_fresh_met_unavailable_for_non_long_engagement():
    got = campaign.task_a_point("short", "fresh_met")
    assert not got.available
    assert got.cep_m is None
    assert "short" in got.reason


def test_task_a_by_age_reports_unsupported_age_explicitly():
    got = campaign.task_a_by_age("long", 1.0)
    assert not got.available
    assert got.cep_m is None
    assert got.reason


def test_task_a_by_age_finds_known_ages():
    assert campaign.task_a_by_age("long", 0.0).cep_m == pytest.approx(27.51, abs=0.01)
    assert campaign.task_a_by_age("long", 2.0).cep_m == pytest.approx(106.19, abs=0.01)


# ===========================================================================
# Task C lookup
# ===========================================================================
def test_task_c_all_seven_ages_present(raw):
    expected_ages = set(raw["c"]["headline"]["long"]["ages"])
    assert expected_ages == set(campaign.MET_AGE_BUCKETS)
    curve = campaign.task_c_curve()
    assert all(p.available for p in curve)


def test_task_c_2h_matches_repository_json(raw):
    expected = raw["c"]["headline"]["long"]["ages"]["2h"]["cep_m"]
    got = campaign.task_c_point("2h")
    assert got.available
    assert got.cep_m == pytest.approx(expected)


def test_task_c_perfect_has_no_knowledge_term_contribution():
    got = campaign.task_c_point("perfect")
    assert got.available
    assert got.cep_contribution_m is None


def test_unsupported_task_c_age_rejected():
    with pytest.raises(ValueError):
        campaign.task_c_point("4h")


def test_task_c_physical_only_has_perfect_and_2h():
    assert campaign.task_c_point("perfect", tag="physical").available
    assert campaign.task_c_point("2h", tag="physical").available
    assert not campaign.task_c_point("1h", tag="physical").available


# ===========================================================================
# Task A / Task C separation
# ===========================================================================
def test_task_a_and_task_c_are_distinct_series():
    """Task A (navigation in loop) and Task C (truth-fed, navigation
    excluded) must not be merged into one generic CEP curve."""
    task_a_2h = campaign.task_a_point("long", "headline").cep_m
    task_c_2h = campaign.task_c_point("2h").cep_m
    assert task_a_2h != pytest.approx(task_c_2h)
