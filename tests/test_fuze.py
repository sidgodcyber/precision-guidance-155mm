"""
Unit tests for the generic multi-mode simulation event engine (`fuze/`).

Everything here is an abstract software state machine over simulation
fixtures -- there is no real-world arming, impact, or proximity threshold
anywhere in this module. Thresholds are chosen only to exercise the state
machine and detector logic deterministically.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from fuze.arbitration import arbitrate
from fuze.config import (ArmingConfig, FuzeEngineConfig, MotionEventConfig,
                          ProximityEventConfig, TimeEventConfig)
from fuze.events import EventType, ModeEvent, MotionDetector, detect_proximity_event, detect_time_event
from fuze.sensors import AltitudeSensor, MotionSensor, SensorReading, TimeSensor
from fuze.state_machine import EventStateMachine, InvalidTransitionError, SimState
from fuze.validation import FuzeConfigError, build_engine_config


# ===========================================================================
# State machine: valid / invalid transitions, minimum delay, terminal state
# ===========================================================================
def test_starts_safe():
    m = EventStateMachine(ArmingConfig(arming_delay_s=2.0, require_startup_ok=False))
    assert m.state is SimState.SAFE


def test_valid_transition_sequence():
    m = EventStateMachine(ArmingConfig(arming_delay_s=2.0, require_startup_ok=False))
    m.begin_arming(t=0.0)
    assert m.state is SimState.ARMING
    m.update(t=2.0)
    assert m.state is SimState.ARMED
    fired = m.function(t=3.0, event=ModeEvent(EventType.TIME_EVENT, 3.0, "time_sensor"))
    assert fired is True
    assert m.state is SimState.FUNCTION_EVENT
    assert m.terminal_event.kind is EventType.TIME_EVENT


def test_begin_arming_requires_startup_confirmation():
    m = EventStateMachine(ArmingConfig(arming_delay_s=1.0, require_startup_ok=True))
    with pytest.raises(InvalidTransitionError):
        m.begin_arming(t=0.0)
    m.confirm_startup()
    m.begin_arming(t=0.0)
    assert m.state is SimState.ARMING


def test_invalid_transition_begin_arming_twice():
    m = EventStateMachine(ArmingConfig(arming_delay_s=1.0, require_startup_ok=False))
    m.begin_arming(t=0.0)
    with pytest.raises(InvalidTransitionError):
        m.begin_arming(t=0.5)


def test_invalid_transition_function_before_armed():
    m = EventStateMachine(ArmingConfig(arming_delay_s=1.0, require_startup_ok=False))
    with pytest.raises(InvalidTransitionError):
        m.function(t=0.0, event=ModeEvent(EventType.TIME_EVENT, 0.0, "time_sensor"))
    m.begin_arming(t=0.0)
    with pytest.raises(InvalidTransitionError):
        m.function(t=0.1, event=ModeEvent(EventType.TIME_EVENT, 0.1, "time_sensor"))


def test_minimum_delay_invariant_not_yet_elapsed():
    m = EventStateMachine(ArmingConfig(arming_delay_s=2.0, require_startup_ok=False))
    m.begin_arming(t=0.0)
    m.update(t=1.9)
    assert m.state is SimState.ARMING


def test_minimum_delay_invariant_elapsed():
    m = EventStateMachine(ArmingConfig(arming_delay_s=2.0, require_startup_ok=False))
    m.begin_arming(t=0.0)
    m.update(t=2.0)
    assert m.state is SimState.ARMED


def test_terminal_state_repeated_function_is_deterministic_noop():
    m = EventStateMachine(ArmingConfig(arming_delay_s=0.0, require_startup_ok=False))
    m.begin_arming(t=0.0)
    m.update(t=0.0)
    event = ModeEvent(EventType.PROXIMITY_EVENT, 1.0, "altitude_sensor")
    assert m.function(t=1.0, event=event) is True
    history_len = len(m.history)
    assert m.function(t=2.0, event=event) is False
    assert m.state is SimState.FUNCTION_EVENT
    assert len(m.history) == history_len
    assert m.terminal_event is event


# ===========================================================================
# Time event
# ===========================================================================
def test_time_event_fires_at_configured_time():
    cfg = TimeEventConfig(enabled=True, event_time_s=10.0, min_time_s=0.0, max_time_s=60.0)
    assert detect_time_event(9.9, cfg) is None
    fired = detect_time_event(10.0, cfg)
    assert fired is not None
    assert fired.kind is EventType.TIME_EVENT


def test_time_event_disabled_mode_never_fires():
    cfg = TimeEventConfig(enabled=False, event_time_s=10.0, min_time_s=0.0, max_time_s=60.0)
    assert detect_time_event(100.0, cfg) is None


def test_time_event_outside_valid_range_does_not_fire():
    cfg = TimeEventConfig(enabled=True, event_time_s=10.0, min_time_s=0.0, max_time_s=15.0)
    assert detect_time_event(20.0, cfg) is None


# ===========================================================================
# Motion/impact-like event: threshold, persistence, hysteresis, dropout
# ===========================================================================
def test_motion_event_requires_persistence():
    cfg = MotionEventConfig(enabled=True, threshold=50.0, persistence_samples=3, hysteresis=5.0)
    detector = MotionDetector(cfg)
    readings = [SensorReading(t=float(i) * 0.1, value=60.0, valid=True) for i in range(3)]
    assert detector.update(readings[0]) is None
    assert detector.update(readings[1]) is None
    fired = detector.update(readings[2])
    assert fired is not None
    assert fired.kind is EventType.IMPACT_EVENT


def test_motion_event_below_threshold_never_fires():
    cfg = MotionEventConfig(enabled=True, threshold=50.0, persistence_samples=2, hysteresis=5.0)
    detector = MotionDetector(cfg)
    for i in range(5):
        assert detector.update(SensorReading(t=float(i), value=10.0, valid=True)) is None


def test_motion_event_hysteresis_keeps_detector_armed():
    """Once armed above threshold, a dip that stays above threshold-hysteresis
    must not reset the persistence count."""
    cfg = MotionEventConfig(enabled=True, threshold=50.0, persistence_samples=3, hysteresis=10.0)
    detector = MotionDetector(cfg)
    assert detector.update(SensorReading(t=0.0, value=55.0, valid=True)) is None
    assert detector.update(SensorReading(t=0.1, value=42.0, valid=True)) is None  # >= 50-10
    fired = detector.update(SensorReading(t=0.2, value=55.0, valid=True))
    assert fired is not None


def test_motion_event_disabled_mode_never_fires():
    cfg = MotionEventConfig(enabled=False, threshold=1.0, persistence_samples=1, hysteresis=0.5)
    detector = MotionDetector(cfg)
    assert detector.update(SensorReading(t=0.0, value=1000.0, valid=True)) is None


# ===========================================================================
# Proximity/altitude-like event
# ===========================================================================
def test_proximity_event_fires_at_or_below_trigger():
    cfg = ProximityEventConfig(enabled=True, trigger_value=5.0)
    assert detect_proximity_event(SensorReading(t=1.0, value=10.0, valid=True), cfg) is None
    fired = detect_proximity_event(SensorReading(t=2.0, value=5.0, valid=True), cfg)
    assert fired is not None
    assert fired.kind is EventType.PROXIMITY_EVENT


def test_proximity_event_disabled_mode_never_fires():
    cfg = ProximityEventConfig(enabled=False, trigger_value=5.0)
    assert detect_proximity_event(SensorReading(t=1.0, value=0.0, valid=True), cfg) is None


# ===========================================================================
# Failed / stale sensors
# ===========================================================================
def test_proximity_event_ignores_invalid_reading():
    cfg = ProximityEventConfig(enabled=True, trigger_value=5.0)
    assert detect_proximity_event(SensorReading(t=1.0, value=None, valid=False), cfg) is None
    assert detect_proximity_event(None, cfg) is None


def test_proximity_event_ignores_stale_reading():
    cfg = ProximityEventConfig(enabled=True, trigger_value=5.0)
    stale = SensorReading(t=1.0, value=1.0, valid=True, stale=True)
    assert detect_proximity_event(stale, cfg) is None


def test_motion_detector_resets_on_dropout():
    cfg = MotionEventConfig(enabled=True, threshold=50.0, persistence_samples=2, hysteresis=5.0)
    detector = MotionDetector(cfg)
    assert detector.update(SensorReading(t=0.0, value=60.0, valid=True)) is None
    assert detector.update(SensorReading(t=0.1, value=None, valid=False)) is None
    fired = detector.update(SensorReading(t=0.2, value=60.0, valid=True))
    assert fired is None  # persistence restarted, only one consecutive sample so far


def test_altitude_sensor_reports_dropout_as_invalid():
    sensor = AltitudeSensor(rng=np.random.default_rng(1), dropout_prob=1.0)
    reading = sensor.read(t_true=0.0, true_value=100.0)
    assert reading.valid is False


def test_altitude_sensor_reports_staleness():
    sensor = AltitudeSensor(rng=np.random.default_rng(2), sample_period_s=10.0,
                             stale_after_s=0.5)
    first = sensor.read(t_true=0.0, true_value=100.0)
    assert first.valid and not first.stale
    later = sensor.read(t_true=1.0, true_value=100.0)
    assert later.valid and later.stale


def test_motion_sensor_latency_delays_delivery():
    sensor = MotionSensor(rng=np.random.default_rng(3), latency_s=0.2)
    immediate = sensor.read(t_true=0.0, true_value=42.0)
    assert immediate.valid is False
    delayed = sensor.read(t_true=0.2, true_value=42.0)
    assert delayed.valid is True


def test_time_sensor_dropout():
    sensor = TimeSensor(rng=np.random.default_rng(4), dropout_prob=1.0)
    assert sensor.read(t_true=5.0).valid is False


# ===========================================================================
# Arbitration: precedence, simultaneous events, disabled/failed, no event
# ===========================================================================
def test_arbitration_no_candidates_returns_none():
    assert arbitrate([None, None]) is None


def test_arbitration_simultaneous_events_precedence():
    impact = ModeEvent(EventType.IMPACT_EVENT, 5.0, "motion_sensor")
    proximity = ModeEvent(EventType.PROXIMITY_EVENT, 5.0, "altitude_sensor")
    time_ev = ModeEvent(EventType.TIME_EVENT, 5.0, "time_sensor")
    chosen = arbitrate([time_ev, proximity, impact])
    assert chosen is impact


def test_arbitration_ties_within_kind_resolve_by_earliest_time():
    earlier = ModeEvent(EventType.TIME_EVENT, 1.0, "time_sensor")
    later = ModeEvent(EventType.TIME_EVENT, 2.0, "time_sensor")
    assert arbitrate([later, earlier]) is earlier


def test_arbitration_disabled_mode_contributes_no_candidate():
    """A disabled mode never produces a ModeEvent (see detect_* functions),
    so arbitration simply never sees one for it."""
    cfg = TimeEventConfig(enabled=False, event_time_s=0.0, min_time_s=0.0, max_time_s=10.0)
    candidate = detect_time_event(5.0, cfg)
    assert arbitrate([candidate]) is None


def test_arbitration_is_deterministic_across_repeated_calls():
    candidates = [
        ModeEvent(EventType.PROXIMITY_EVENT, 3.0, "altitude_sensor"),
        ModeEvent(EventType.TIME_EVENT, 1.0, "time_sensor"),
    ]
    results = {arbitrate(list(candidates)).kind for _ in range(10)}
    assert results == {EventType.PROXIMITY_EVENT}


def test_repeated_terminal_event_attempts_are_idempotent_through_arbitration():
    m = EventStateMachine(ArmingConfig(arming_delay_s=0.0, require_startup_ok=False))
    m.begin_arming(t=0.0)
    m.update(t=0.0)
    event = arbitrate([ModeEvent(EventType.IMPACT_EVENT, 1.0, "motion_sensor")])
    assert m.function(1.0, event) is True
    assert m.function(1.1, event) is False
    assert m.function(1.2, event) is False


# ===========================================================================
# Configuration: malformed / out-of-range
# ===========================================================================
def test_malformed_configuration_rejected_not_a_mapping():
    with pytest.raises(FuzeConfigError):
        build_engine_config(["not", "a", "mapping"])


def test_malformed_configuration_unknown_field_rejected():
    with pytest.raises(FuzeConfigError):
        build_engine_config({"arming": {"arming_delay_s": 1.0, "bogus_field": True}})


def test_out_of_range_arming_delay_rejected():
    with pytest.raises(ValidationError):
        ArmingConfig(arming_delay_s=-1.0)


def test_out_of_range_time_event_rejected():
    with pytest.raises(ValidationError):
        TimeEventConfig(event_time_s=100.0, min_time_s=0.0, max_time_s=50.0)


def test_hysteresis_must_be_smaller_than_threshold():
    with pytest.raises(ValidationError):
        MotionEventConfig(threshold=10.0, hysteresis=10.0)


def test_precedence_must_not_repeat_a_kind():
    with pytest.raises(ValidationError):
        FuzeEngineConfig(precedence=("TIME_EVENT", "TIME_EVENT", "IMPACT_EVENT"))


def test_build_engine_config_accepts_full_valid_configuration():
    cfg = build_engine_config({
        "arming": {"arming_delay_s": 2.0},
        "time_event": {"event_time_s": 30.0, "min_time_s": 0.0, "max_time_s": 60.0},
        "motion_event": {"threshold": 50.0, "hysteresis": 5.0, "persistence_samples": 3},
        "proximity_event": {"trigger_value": 5.0},
    })
    assert isinstance(cfg, FuzeEngineConfig)
    assert cfg.arming.arming_delay_s == 2.0
