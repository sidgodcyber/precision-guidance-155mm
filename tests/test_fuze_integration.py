"""
Phase 2 characterization: integration-level behavior of the `fuze/` event
engine that the Phase 1 unit tests (`tests/test_fuze.py`) don't already
cover -- running the full sensor -> detector -> arbitration -> state-machine
pipeline end to end, twice, with identical fixtures, and confirming a
sensor's behavior after it recovers from a temporary dropout.

Everything here is a software state machine over simulation fixtures; no
value is derived from an operational specification.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from fuze.arbitration import arbitrate
from fuze.config import ArmingConfig, FuzeEngineConfig, MotionEventConfig, ProximityEventConfig, TimeEventConfig
from fuze.events import MotionDetector, detect_proximity_event, detect_time_event
from fuze.sensors import AltitudeSensor, MotionSensor, TimeSensor
from fuze.state_machine import EventStateMachine, SimState


def _run_pipeline(seed: int, engine_config: FuzeEngineConfig):
    """One deterministic simulated run: a fixed synthetic trajectory (a
    ramp in simulated motion and a descending altitude) driven through
    sensors, detectors, arbitration, and the state machine to a terminal
    state or exhaustion of the fixture."""
    rng = np.random.default_rng(seed)
    motion_sensor = MotionSensor(rng=rng, noise_std=1.0, dropout_prob=0.05)
    altitude_sensor = AltitudeSensor(rng=rng, noise_std=0.5, dropout_prob=0.05)
    time_sensor = TimeSensor(rng=rng, jitter_s=0.0)
    motion_detector = MotionDetector(engine_config.motion_event)

    m = EventStateMachine(engine_config.arming)
    m.confirm_startup()
    m.begin_arming(t=0.0)

    dt = 0.1
    steps = 200
    true_altitude0 = 500.0
    true_descent_rate = 20.0
    true_motion0 = 0.0
    true_motion_rate = 8.0

    log = []
    for i in range(steps):
        t = time_sensor.read(i * dt).value
        m.update(t)

        true_altitude = max(0.0, true_altitude0 - true_descent_rate * t)
        true_motion = true_motion0 + true_motion_rate * t

        alt_reading = altitude_sensor.read(t_true=t, true_value=true_altitude)
        motion_reading = motion_sensor.read(t_true=t, true_value=true_motion)

        candidates = [
            detect_time_event(t, engine_config.time_event),
            detect_proximity_event(alt_reading, engine_config.proximity_event),
            motion_detector.update(motion_reading),
        ]
        chosen = arbitrate(candidates, engine_config.precedence)
        log.append((t, chosen.kind.value if chosen else None))

        if m.state is SimState.ARMED and chosen is not None:
            m.function(t, chosen)
        if m.state is SimState.FUNCTION_EVENT:
            break

    return m, log


def _engine_config() -> FuzeEngineConfig:
    return FuzeEngineConfig(
        arming=ArmingConfig(arming_delay_s=0.5, require_startup_ok=True),
        time_event=TimeEventConfig(enabled=True, event_time_s=15.0, min_time_s=0.0, max_time_s=20.0),
        motion_event=MotionEventConfig(enabled=True, threshold=50.0, persistence_samples=3, hysteresis=5.0),
        proximity_event=ProximityEventConfig(enabled=True, trigger_value=10.0),
    )


# ===========================================================================
# Repeated execution reproducibility (full pipeline)
# ===========================================================================
def test_full_pipeline_is_reproducible_for_identical_seed():
    m1, log1 = _run_pipeline(seed=123, engine_config=_engine_config())
    m2, log2 = _run_pipeline(seed=123, engine_config=_engine_config())

    assert log1 == log2
    assert m1.state is m2.state
    assert m1.history == m2.history
    if m1.terminal_event is not None:
        assert m1.terminal_event.kind == m2.terminal_event.kind
        assert m1.terminal_event.t == m2.terminal_event.t


def test_full_pipeline_reaches_a_terminal_state_from_this_fixture():
    """Confirms the fixture is meaningful (it actually exercises
    FUNCTION_EVENT) rather than trivially reproducible by never firing."""
    m, _ = _run_pipeline(seed=123, engine_config=_engine_config())
    assert m.state is SimState.FUNCTION_EVENT


def test_full_pipeline_reproducibility_holds_for_a_second_seed_too():
    """Reproducibility is a property of the pipeline, not a fluke of one
    particular seed -- repeat the same-seed check with a different seed."""
    m_a, log_a = _run_pipeline(seed=99, engine_config=_engine_config())
    m_b, log_b = _run_pipeline(seed=99, engine_config=_engine_config())
    assert log_a == log_b
    assert m_a.history == m_b.history


def test_sensor_noise_realisation_is_seed_dependent():
    """The pipeline's reproducibility is CONDITIONED on the seed: two
    different seeds must legitimately give different noise realisations,
    or the "identical seed -> identical run" check above would be vacuous."""
    motion_a = MotionSensor(rng=np.random.default_rng(1), noise_std=3.0)
    motion_b = MotionSensor(rng=np.random.default_rng(2), noise_std=3.0)
    ra = [motion_a.read(t_true=float(i), true_value=10.0).value for i in range(20)]
    rb = [motion_b.read(t_true=float(i), true_value=10.0).value for i in range(20)]
    assert ra != rb


# ===========================================================================
# Sensor recovery after temporary failure
# ===========================================================================
def test_altitude_sensor_recovers_after_forced_dropout_window():
    """A sensor that drops every reading for a window, then resumes,
    must deliver fresh (non-stale) readings again once true dropout stops --
    recovery is not permanently blocked by the earlier failure."""
    sensor = AltitudeSensor(rng=np.random.default_rng(4), noise_std=0.0,
                             dropout_prob=0.0, stale_after_s=0.5)

    # Force a dropout window, then restore normal operation.
    sensor.dropout_prob = 1.0
    for i in range(5):
        r = sensor.read(t_true=i * 0.1, true_value=100.0)
        assert r.valid is False
    sensor.dropout_prob = 0.0
    recovered = sensor.read(t_true=0.6, true_value=100.0)
    assert recovered.valid is True
    assert recovered.stale is False


def test_sensor_module_does_not_import_state_machine():
    """Sensor generation must stay independent of state-transition logic
    (spec section 16): `fuze.sensors` should not import `fuze.state_machine`
    or `fuze.events` at all."""
    import fuze.sensors as sensors_module
    assert "state_machine" not in sensors_module.__dict__
    assert not hasattr(sensors_module, "EventStateMachine")
    assert not hasattr(sensors_module, "ModeEvent")


def test_motion_detector_persistence_recovers_after_dropout_then_reaccumulates():
    cfg = MotionEventConfig(enabled=True, threshold=50.0, persistence_samples=2, hysteresis=5.0)
    detector = MotionDetector(cfg)
    from fuze.sensors import SensorReading

    assert detector.update(SensorReading(t=0.0, value=60.0, valid=True)) is None
    assert detector.update(SensorReading(t=0.1, value=None, valid=False)) is None  # dropout
    assert detector.update(SensorReading(t=0.2, value=60.0, valid=True)) is None  # persistence restarted
    fired = detector.update(SensorReading(t=0.3, value=60.0, valid=True))
    assert fired is not None  # recovers and reaches persistence again
