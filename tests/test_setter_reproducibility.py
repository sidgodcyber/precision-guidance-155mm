"""
Phase 2 characterization: reproducibility of the setter/fuze demonstration
path under identical inputs.

Running the setter with the same engagement, met profile (same seed), met
age, and simulation configuration must yield: identical serialization,
identical CRC32, identical simulated sensor readings, identical trajectory
output, and identical campaign lookups. This file pins that property with
regression tests rather than asserting it only in prose.

One issue was found and fixed while characterizing this: `generated_at` (a
wall-clock generation timestamp) was included in the checksummed body, so an
otherwise identical configuration produced a different CRC32 a moment later.
`setter.message_codec` now excludes `generated_at` from the checksum body;
`test_identical_configuration_has_identical_checksum_across_time` pins the
fix.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fuze.sensors import AltitudeSensor, MotionSensor, TimeSensor
from setter import campaign, message_codec, simulation_adapter as sa
from setter.schemas import (EventConfiguration, MetProfileMessage, Position,
                             SetterMessage, SimulationConfig)


def _build_message(engagement: str = "long", age_bucket: str = "2h", seed: int = 11) -> SetterMessage:
    base = sa.baseline_for(engagement)
    met = sa.met_profile_for_age(age_bucket, seed=seed)
    lay = sa.lay_gun(engagement, met)
    deploy = sa.fuze_setting(engagement, met, lay["dqe_mils"])
    return SetterMessage(
        scenario_id=engagement,
        reference_position=Position(),
        target_position=Position(x_m=base["uncorrected_range"], y_m=base["uncorrected_drift"]),
        met_profile=MetProfileMessage.from_met_profile(met, profile_id=f"{engagement}-{age_bucket}"),
        met_age_hours=sa.age_bucket_to_hours(age_bucket),
        simulation_config=SimulationConfig(
            scenario_id=engagement, charge=base["charge"], qe_mils=base["qe_mils"],
            muzzle_velocity_ms=base["muzzle_velocity"], dqe_mils=lay["dqe_mils"],
            daz_mils=lay["daz"], deploy_time_s=deploy, guided_phase_s=base["guided_phase_s"]),
        event_configuration=EventConfiguration(mode="time", parameters={"event_time_s": 30.0}),
    )


# ===========================================================================
# 1. Configuration serialization is identical
# ===========================================================================
def test_identical_inputs_produce_identical_canonical_body():
    m1 = _build_message()
    m2 = _build_message()
    assert message_codec.canonical_body(m1) == message_codec.canonical_body(m2)


# ===========================================================================
# 2. CRC32 is identical -- including across wall-clock time
# ===========================================================================
def test_identical_configuration_has_identical_checksum_across_time():
    m1 = _build_message()
    time.sleep(1.05)
    m2 = _build_message()
    assert m1.generated_at != m2.generated_at, "test is only meaningful if timestamps actually differ"
    assert message_codec.compute_checksum(m1) == message_codec.compute_checksum(m2)


def test_different_engagement_changes_checksum():
    m_long = _build_message(engagement="long")
    m_short = _build_message(engagement="short")
    assert message_codec.compute_checksum(m_long) != message_codec.compute_checksum(m_short)


# ===========================================================================
# 3. Simulated sensor readings are identical for identical seeds
# ===========================================================================
def test_time_sensor_identical_for_identical_seed():
    a = TimeSensor(rng=np.random.default_rng(5), jitter_s=0.01)
    b = TimeSensor(rng=np.random.default_rng(5), jitter_s=0.01)
    for t in (0.0, 1.0, 2.0, 3.0):
        ra, rb = a.read(t), b.read(t)
        assert ra.value == rb.value
        assert ra.valid == rb.valid


def test_motion_sensor_identical_for_identical_seed():
    a = MotionSensor(rng=np.random.default_rng(9), noise_std=2.0, dropout_prob=0.2)
    b = MotionSensor(rng=np.random.default_rng(9), noise_std=2.0, dropout_prob=0.2)
    for i in range(20):
        t = i * 0.1
        ra = a.read(t_true=t, true_value=50.0 + i)
        rb = b.read(t_true=t, true_value=50.0 + i)
        assert ra.value == rb.value
        assert ra.valid == rb.valid
        assert ra.stale == rb.stale


def test_altitude_sensor_differs_with_different_seed():
    """Reproducibility is SEED-conditioned, not unconditional -- different
    seeds must legitimately give different noise realisations."""
    a = AltitudeSensor(rng=np.random.default_rng(1), noise_std=5.0)
    b = AltitudeSensor(rng=np.random.default_rng(2), noise_std=5.0)
    readings_a = [a.read(t_true=float(i), true_value=100.0).value for i in range(10)]
    readings_b = [b.read(t_true=float(i), true_value=100.0).value for i in range(10)]
    assert readings_a != readings_b


# ===========================================================================
# 4. Trajectory output is identical within the engine's own tolerance
# ===========================================================================
def test_single_trajectory_is_bitwise_identical_across_repeated_runs():
    """The MPMM trajectory has no random draw in it (dispersion is a Task-6
    campaign concept, not part of a single reduced-order run), so repeated
    calls for the same engagement/met must match exactly, not just within
    tolerance."""
    met = sa.met_profile_for_age("2h", seed=3)
    s1 = sa.single_trajectory("middle", met=met)
    s2 = sa.single_trajectory("middle", met=met)
    assert np.array_equal(s1.t, s2.t)
    assert np.array_equal(s1.downrange, s2.downrange)
    assert np.array_equal(s1.altitude, s2.altitude)
    assert s1.range_m == s2.range_m


def test_single_trajectory_reproducible_across_engagements():
    for engagement in sa.supported_engagements():
        met = sa.met_profile_for_age("perfect", seed=1)
        s1 = sa.single_trajectory(engagement, met=met)
        s2 = sa.single_trajectory(engagement, met=met)
        assert np.array_equal(s1.velocity_ms, s2.velocity_ms), engagement


# ===========================================================================
# 5. Campaign lookups return the same values
# ===========================================================================
def test_campaign_lookups_are_stable_across_repeated_calls():
    a1 = campaign.task_a_point("long", "headline")
    a2 = campaign.task_a_point("long", "headline")
    assert a1 == a2

    c1 = campaign.task_c_point("2h")
    c2 = campaign.task_c_point("2h")
    assert c1 == c2


def test_campaign_data_loader_is_cached_to_a_single_object():
    """`setter.campaign._data()` is `lru_cache`d -- repeated lookups must not
    re-read docs/monte_carlo.json from disk."""
    assert campaign._data() is campaign._data()
