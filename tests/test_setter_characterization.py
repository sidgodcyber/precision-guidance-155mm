"""
Phase 2: the checked-in `docs/setter_validation/*.json` artifacts against the
live data/code that generated them -- so if `docs/monte_carlo.json` or the
setter package changes and the artifacts are not regenerated
(`python -m setter.characterization`), this fails and names it, the same
role `tests/test_guidance_documented_numbers.py` plays for the step-3 prose.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import json

import pytest

from setter import campaign, characterization
from setter.config import SUPPORTED_ENGAGEMENTS

ARTIFACT_DIR = characterization.OUT_DIR

pytestmark = pytest.mark.skipif(
    not ARTIFACT_DIR.exists(),
    reason="docs/setter_validation not generated; run python -m setter.characterization")


def _load(name: str) -> dict:
    with open(ARTIFACT_DIR / name) as fh:
        return json.load(fh)


# ===========================================================================
# campaign_summary.json matches the live campaign module
# ===========================================================================
def test_campaign_summary_headline_matches_live_lookup():
    data = _load("campaign_summary.json")
    for engagement in SUPPORTED_ENGAGEMENTS:
        live = campaign.task_a_point(engagement, "headline")
        stored = data["task_a"]["points"]["headline"][engagement]
        assert stored["available"] == live.available
        assert stored["cep_m"] == pytest.approx(live.cep_m)
        assert stored["n"] == live.n


def test_campaign_summary_preserves_unavailable_combinations_explicitly():
    data = _load("campaign_summary.json")
    fresh_met = data["task_a"]["points"]["fresh_met"]
    for engagement in SUPPORTED_ENGAGEMENTS:
        if engagement == "long":
            assert fresh_met[engagement]["available"] is True
        else:
            assert fresh_met[engagement]["available"] is False
            assert "reason" in fresh_met[engagement]


def test_campaign_summary_task_a_and_task_c_stay_separate_keys():
    data = _load("campaign_summary.json")
    assert set(data) >= {"task_a", "task_c"}
    assert data["task_a"]["description"] != data["task_c"]["description"]


def test_campaign_summary_anomaly_matches_live_headline_table():
    data = _load("campaign_summary.json")
    stored = data["task_a"]["anomaly"]["headline_2h_cep_m_by_engagement"]
    live = campaign.task_a_headline_table()
    for engagement in SUPPORTED_ENGAGEMENTS:
        assert stored[engagement] == pytest.approx(live[engagement].cep_m)
    assert stored["mid2"] > stored["middle"]
    assert stored["mid2"] > stored["long"]


# ===========================================================================
# validation_summary.json: every engagement's representative message is
# actually valid, and the recorded rejection checks are all true
# ===========================================================================
def test_validation_summary_all_engagements_valid():
    data = _load("validation_summary.json")
    for engagement in SUPPORTED_ENGAGEMENTS:
        row = data["per_engagement"][engagement]
        assert row["validation_status"] == "valid"
        assert row["round_trip_ok"] is True
        assert row["checksum_verified"] is True
        assert row["met_profile_fields_complete"] is True


def test_validation_summary_rejection_checks_all_true():
    data = _load("validation_summary.json")
    assert all(data["rejection_checks"].values())
    assert data["sensitivity"]["single_field_change_changes_checksum"] is True


# ===========================================================================
# runtime_summary.json: structurally sane, and the cache characterization
# shows the expected cold > warm relationship
# ===========================================================================
def test_runtime_summary_covers_all_engagements():
    data = _load("runtime_summary.json")
    assert set(data["single_trajectory"]) == set(SUPPORTED_ENGAGEMENTS)
    for row in data["single_trajectory"].values():
        assert row["median_s"] > 0.0
        assert row["max_s"] >= row["median_s"] >= row["min_s"]


def test_runtime_summary_engagement_context_cache_is_effective():
    data = _load("runtime_summary.json")
    cache = data["engagement_context_cache"]
    assert cache["warm_s"] < cache["cold_s"]


# ===========================================================================
# The characterization functions themselves are stable (aside from
# provenance timestamps) across repeated calls -- no hidden randomness.
# ===========================================================================
def test_campaign_summary_function_is_stable_across_calls():
    a = characterization.campaign_summary()
    b = characterization.campaign_summary()
    a.pop("provenance")
    b.pop("provenance")
    assert a == b


def test_validation_summary_function_is_stable_across_calls():
    a = characterization.validation_summary()
    b = characterization.validation_summary()
    a.pop("provenance")
    b.pop("provenance")
    assert a == b
