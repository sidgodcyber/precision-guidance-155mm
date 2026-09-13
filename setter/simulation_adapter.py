"""
The boundary between the setter/dashboard and the frozen simulation engine.

Everything the UI or the message schema needs from `sim/`, `analysis/`,
`models/`, or `gnc/` goes through this module. It calls the existing public
functions as-is (`analysis.monte_carlo.lay_gun`, `.fuze_setting`, `.context`;
`analysis.guidance_cep.load_maps`); it does not reimplement or refactor any
of them.

`engagement_context()` is the one expensive call (~700-750 ms cold, per the
Phase 0 measurement). It and the guidance-map load behind it are wrapped in
`st.cache_resource` rather than `functools.lru_cache`: both return objects
that are read-only downstream (never mutated by callers) and constant for a
given key, which is exactly what `cache_resource` is for -- and unlike
`lru_cache`, it is shared across Streamlit sessions in the same process and
clearable from the UI (see `st.cache_resource.clear()`). `st.cache_resource`
degrades to a plain in-process cache when there is no active Streamlit
runtime (e.g. under pytest or `characterization.py`), so it works the same
way here as it did under `lru_cache`.

Use `.clear()` (the `cache_resource` API), not `functools`' `.cache_clear()`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st

from analysis import guidance_cep as gc
from analysis import monte_carlo as mc
from fuze.trajectory_adapter import TrajectorySample, generate_trajectory
from setter.config import GUIDANCE_MAP_PATH, MET_AGE_BUCKETS, SUPPORTED_ENGAGEMENTS
from sim.atmosphere import MetProfile

__all__ = [
    "supported_engagements", "engagement_context", "baseline_for",
    "lay_gun", "fuze_setting", "single_trajectory", "met_profile_for_age",
    "age_bucket_to_hours",
]

#: Task C's age buckets, in hours, for `MetProfile.message()`. "perfect"
#: hands the truth profile straight through with no message error at all;
#: "none" is the "no met message" case, i.e. `MetProfile.standard()`.
_AGE_HOURS = {"perfect": None, "0h": 0.0, "1h": 1.0, "2h": 2.0, "3h": 3.0,
              "6h": 6.0, "none": None}


@st.cache_resource
def _mapdata() -> dict:
    return gc.load_maps(str(GUIDANCE_MAP_PATH))


def supported_engagements() -> tuple:
    return SUPPORTED_ENGAGEMENTS


def _check_engagement(label: str) -> None:
    if label not in SUPPORTED_ENGAGEMENTS:
        raise ValueError(f"unsupported engagement {label!r}; supported: {SUPPORTED_ENGAGEMENTS}")


@st.cache_resource
def engagement_context(label: str) -> dict:
    """`analysis.monte_carlo.context(mapdata, label)`, cached per label."""
    _check_engagement(label)
    return mc.context(_mapdata(), label)


def baseline_for(label: str) -> dict:
    """The engagement's baseline dict (qe_mils, muzzle_velocity, ranges,
    timings) -- display/reference information, not a live target solve."""
    return dict(engagement_context(label)["base"])


def lay_gun(label: str, met: Optional[MetProfile]) -> dict:
    """The quadrant-elevation/azimuth correction fire control would order
    for this engagement given `met` (None = standard atmosphere, nominal
    lay). Thin call-through to `analysis.monte_carlo.lay_gun`."""
    return mc.lay_gun(baseline_for(label), met)


def fuze_setting(label: str, met: Optional[MetProfile], dqe_mils: float) -> float:
    """The deployment time fire control would order, from its own solution
    in `met`'s atmosphere. Thin call-through to
    `analysis.monte_carlo.fuze_setting`."""
    return mc.fuze_setting(baseline_for(label), met, dqe_mils)


def single_trajectory(label: str, met: Optional[MetProfile] = None) -> TrajectorySample:
    """One lightweight reduced-order trajectory for ground-track
    visualization -- not a campaign result. See
    `fuze.trajectory_adapter.generate_trajectory`."""
    return generate_trajectory(baseline_for(label), met=met)


def met_profile_for_age(age_bucket: str, seed: int = 0) -> MetProfile:
    """A demonstration `MetProfile` for one of Task C's met-age buckets,
    built entirely from the repository's own `MetProfile.sample()` /
    `.message()` -- not a second atmospheric model. Deterministic in `seed`,
    so re-selecting the same bucket in the UI reproduces the same profile
    rather than resampling on every rerun.
    """
    if age_bucket not in MET_AGE_BUCKETS:
        raise ValueError(f"unsupported met age bucket {age_bucket!r}; supported: {MET_AGE_BUCKETS}")
    rng = np.random.default_rng(seed)
    truth = MetProfile.sample(rng, scale=1.0, label="demonstration truth atmosphere")
    if age_bucket == "perfect":
        return truth
    return truth.message(rng, _AGE_HOURS[age_bucket])


def age_bucket_to_hours(age_bucket: str):
    """The numeric met-message age (hours) behind a Task C age bucket, or
    None for "perfect"/"none" (see `met_profile_for_age`)."""
    if age_bucket not in MET_AGE_BUCKETS:
        raise ValueError(f"unsupported met age bucket {age_bucket!r}; supported: {MET_AGE_BUCKETS}")
    return _AGE_HOURS[age_bucket]
