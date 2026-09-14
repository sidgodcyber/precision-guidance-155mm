"""
Read-only access to the precomputed campaign data in `docs/monte_carlo.json`.

Two campaigns live in that one file and this module keeps them apart on
purpose (see CLAUDE.md and the task brief, section 4):

  Task A (`a`)  -- the primary navigation-in-loop campaign. Only the
                   engagement/tag combinations actually flown are present;
                   `task_a_point` reports an unavailable point explicitly
                   rather than fabricating or interpolating one.
  Task C (`c`)  -- truth-fed, navigation excluded, isolating the atmospheric
                   knowledge term across met-message age. This is the series
                   behind the interactive met-age visualization.

Nothing here runs a new Monte Carlo campaign; it only reads the JSON that
step 6 already produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from setter.config import MET_AGE_BUCKETS, MONTE_CARLO_PATH, SUPPORTED_ENGAGEMENTS, TASK_A_TAGS

__all__ = [
    "CampaignPoint", "KnowledgeTermPoint",
    "task_a_point", "task_a_by_age", "task_a_headline_table",
    "task_a_scatter", "task_a_max_miss_m",
    "task_c_point", "task_c_curve",
]

#: Task A does not vary met age continuously -- it has exactly two stored
#: met-age points for the "long" engagement (fresh_met at upload-at-fuze-
#: setting, headline at the 2-hour message) plus the "physical" no-inflation
#: control at the headline age. This maps a requested age to the tag that
#: carries it; any other age is reported unavailable rather than interpolated.
_TASK_A_AGE_TO_TAG = {0.0: "fresh_met", 2.0: "headline"}


@lru_cache(maxsize=1)
def _data() -> dict:
    with open(MONTE_CARLO_PATH) as fh:
        return json.load(fh)


@dataclass(frozen=True)
class CampaignPoint:
    """One Task A campaign statistic, or an explicit report that none is
    stored for the requested engagement/tag."""

    available: bool
    engagement: str
    tag: str
    cep_m: Optional[float] = None
    n: Optional[int] = None
    cep_se_m: Optional[float] = None
    met_age_hours: Optional[float] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class KnowledgeTermPoint:
    """One Task C atmospheric-knowledge-term statistic at one met-message
    age, or an explicit report that none is stored."""

    available: bool
    age: str
    tag: str
    engagement: str
    cep_m: Optional[float] = None
    n: Optional[int] = None
    cep_contribution_m: Optional[float] = None
    reason: Optional[str] = None


def _check_engagement(engagement: str) -> None:
    if engagement not in SUPPORTED_ENGAGEMENTS:
        raise ValueError(
            f"unsupported engagement {engagement!r}; supported: {SUPPORTED_ENGAGEMENTS}")


def task_a_point(engagement: str, tag: str = "headline") -> CampaignPoint:
    """Task A's guided-round summary for `engagement`/`tag`.

    `tag` must be one of "headline" (all five engagements), "fresh_met" or
    "physical" (both stored only for "long"). A combination the campaign
    never flew comes back with `available=False` and a `reason`, never a
    fabricated or interpolated number.
    """
    _check_engagement(engagement)
    if tag not in TASK_A_TAGS:
        raise ValueError(f"unsupported Task A tag {tag!r}; supported: {TASK_A_TAGS}")
    entry = _data()["a"].get(tag, {}).get(engagement)
    if entry is None:
        return CampaignPoint(
            available=False, engagement=engagement, tag=tag,
            reason=f"Task A has no stored {tag!r} point for engagement {engagement!r}")
    g = entry["guided"]
    return CampaignPoint(
        available=True, engagement=engagement, tag=tag,
        cep_m=g["cep_m"], n=g["n"], cep_se_m=g.get("cep_se_m"),
        met_age_hours=entry.get("met_age_hours"))


def task_a_by_age(engagement: str, met_age_hours: float) -> CampaignPoint:
    """Task A lookup by met-message age in hours, for engagements/ages the
    campaign actually flew. Any other age reports unavailable."""
    _check_engagement(engagement)
    tag = _TASK_A_AGE_TO_TAG.get(float(met_age_hours))
    if tag is None:
        return CampaignPoint(
            available=False, engagement=engagement, tag="",
            met_age_hours=met_age_hours,
            reason=f"Task A has no stored campaign point at met age {met_age_hours} h")
    return task_a_point(engagement, tag)


def task_a_scatter(engagement: str, tag: str = "headline") -> list:
    """Per-round (range-miss, deflection-miss) pairs, metres, for one Task A
    engagement/tag -- the campaign's own stored impact scatter, drawn for
    the CEP-circle view. Not a live simulation result; empty (not
    fabricated) for a combination the campaign never flew.
    """
    _check_engagement(engagement)
    if tag not in TASK_A_TAGS:
        raise ValueError(f"unsupported Task A tag {tag!r}; supported: {TASK_A_TAGS}")
    entry = _data()["a"].get(tag, {}).get(engagement)
    if entry is None:
        return []
    return [(row["miss_range_m"], row["miss_defl_m"]) for row in entry.get("rows", [])]


def task_a_max_miss_m() -> float:
    """The single largest stored Task A round miss distance, across every
    engagement and tag -- a fixed reference radius so a CEP-circle plot's
    axes never rescale between engagements or met ages (a circle that grows
    against a shrinking frame is not to scale)."""
    worst = 0.0
    for tag_entries in _data()["a"].values():
        for entry in tag_entries.values():
            worst = max(worst, entry.get("guided", {}).get("max_m", 0.0))
    return worst


def task_a_headline_table() -> dict:
    """The five-engagement 2-hour headline table.

    Includes the measured non-monotonicity across engagements (middle >
    mid2's neighbours do not order the way range does) -- reported as-is,
    not smoothed or refit. See docs/DEGRADATION-LADDER.md and CLAUDE.md.
    """
    return {e: task_a_point(e, "headline") for e in SUPPORTED_ENGAGEMENTS}


def task_c_point(age: str, tag: str = "headline", engagement: str = "long") -> KnowledgeTermPoint:
    """Task C's atmospheric-knowledge-term summary at one met-message age.

    Task C's stored engagement is effectively just "long" -- it isolates the
    atmospheric term rather than comparing engagements, so `engagement`
    exists for forward compatibility but defaults to the only series present.
    """
    if age not in MET_AGE_BUCKETS:
        raise ValueError(f"unsupported Task C met age {age!r}; supported: {MET_AGE_BUCKETS}")
    entry = _data()["c"].get(tag, {}).get(engagement)
    if entry is None:
        return KnowledgeTermPoint(
            available=False, age=age, tag=tag, engagement=engagement,
            reason=f"Task C has no {tag!r}/{engagement!r} series")
    row = entry.get("ages", {}).get(age)
    if row is None:
        return KnowledgeTermPoint(
            available=False, age=age, tag=tag, engagement=engagement,
            reason=f"Task C {tag!r}/{engagement!r} has no {age!r} case")
    kt = row.get("knowledge_term", {})
    return KnowledgeTermPoint(
        available=True, age=age, tag=tag, engagement=engagement,
        cep_m=row.get("cep_m"), n=row.get("n"),
        cep_contribution_m=kt.get("cep_contribution_m"))


def task_c_curve(tag: str = "headline", engagement: str = "long") -> list:
    """All seven Task C age points, in `MET_AGE_BUCKETS` order, for the
    met-age visualization. Ages the tag/engagement doesn't carry come back
    as explicit unavailable points rather than being omitted silently."""
    return [task_c_point(age, tag, engagement) for age in MET_AGE_BUCKETS]
