"""
Unit tests for `setter.simulation_adapter` -- the boundary between the setter
and the frozen simulation engine (`analysis.monte_carlo`,
`analysis.guidance_cep`, `sim.atmosphere`, `fuze.trajectory_adapter`).

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import pytest

from fuze.trajectory_adapter import TrajectorySample
from setter import simulation_adapter as sim_adapter
from sim.atmosphere import MetProfile


def test_supported_engagements_are_the_five_named_scenarios():
    assert sim_adapter.supported_engagements() == (
        "short", "short2", "middle", "mid2", "long")


def test_unsupported_engagement_rejected():
    with pytest.raises(ValueError):
        sim_adapter.baseline_for("nonexistent")


def test_baseline_for_long_matches_guidance_map():
    base = sim_adapter.baseline_for("long")
    assert base["label"] == "long"
    assert base["charge"] == 8
    assert base["qe_mils"] == pytest.approx(525.3)


def test_engagement_context_is_cached():
    """`functools.lru_cache` on `engagement_context` -- repeated calls for
    the same label return the identical (not just equal) object."""
    ctx1 = sim_adapter.engagement_context("middle")
    ctx2 = sim_adapter.engagement_context("middle")
    assert ctx1 is ctx2
    info = sim_adapter.engagement_context.cache_info()
    assert info.hits >= 1


def test_lay_gun_standard_atmosphere_is_nominal_lay():
    result = sim_adapter.lay_gun("long", None)
    assert result["dqe_mils"] == 0.0
    assert result["daz"] == 0.0


def test_lay_gun_with_met_profile_returns_finite_correction():
    met = MetProfile.sample(__import__("numpy").random.default_rng(7), label="test")
    result = sim_adapter.lay_gun("long", met)
    assert result["dqe_mils"] == result["dqe_mils"]  # not NaN
    assert result["lay_residual_m"] < 5.0


def test_fuze_setting_standard_atmosphere_matches_base_deploy_time():
    base = sim_adapter.baseline_for("long")
    deploy_time = sim_adapter.fuze_setting("long", None, 0.0)
    assert deploy_time == pytest.approx(base["deploy_time"])


def test_met_profile_for_age_unsupported_bucket_rejected():
    with pytest.raises(ValueError):
        sim_adapter.met_profile_for_age("4h")


def test_met_profile_for_age_is_deterministic_for_same_seed():
    a = sim_adapter.met_profile_for_age("2h", seed=42)
    b = sim_adapter.met_profile_for_age("2h", seed=42)
    assert a.wind_north == b.wind_north
    assert a.density_ratio == b.density_ratio


def test_met_profile_for_age_none_bucket_is_standard():
    met = sim_adapter.met_profile_for_age("none", seed=1)
    assert met.is_standard


def test_met_profile_for_age_perfect_is_not_standard():
    met = sim_adapter.met_profile_for_age("perfect", seed=1)
    assert not met.is_standard


def test_age_bucket_to_hours():
    assert sim_adapter.age_bucket_to_hours("2h") == 2.0
    assert sim_adapter.age_bucket_to_hours("none") is None
    assert sim_adapter.age_bucket_to_hours("perfect") is None


# ===========================================================================
# Trajectory adapter structure
# ===========================================================================
def test_single_trajectory_returns_expected_structure():
    sample = sim_adapter.single_trajectory("short")
    assert isinstance(sample, TrajectorySample)
    assert sample.t.size > 1
    assert sample.duration_s > 0.0
    assert sample.range_m > 0.0
    # monotonically non-decreasing simulation clock
    assert (sample.t[1:] >= sample.t[:-1]).all()
    rows = sample.as_rows()
    assert len(rows) == sample.t.size
    assert set(rows[0]) == {
        "t", "downrange_m", "crossrange_m", "altitude_m",
        "velocity_ms", "acceleration_ms2", "spin_rad_s", "alpha_e_rad", "mach",
    }


def test_single_trajectory_with_met_profile_runs():
    met = sim_adapter.met_profile_for_age("2h", seed=3)
    sample = sim_adapter.single_trajectory("long", met=met)
    assert sample.range_m > 0.0
