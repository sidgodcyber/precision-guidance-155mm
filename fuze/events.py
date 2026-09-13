"""
Abstract event types and the three generic mode detectors.

Each detector consumes `fuze.sensors.SensorReading` objects and this
module's own config dataclasses from `fuze.config`; it never talks to the
state machine and never generates its own sensor data. A disabled mode, an
invalid reading, or a stale reading all simply fail to produce an event --
none of that is treated as an error here, since arbitration (fuze.arbitration)
is where "no event occurred" is a normal, deterministic outcome.

The terminal result of a detector firing is an abstract `ModeEvent`. Nothing
downstream of it carries physical or destructive meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from fuze.config import MotionEventConfig, ProximityEventConfig, TimeEventConfig
from fuze.sensors import SensorReading

__all__ = [
    "EventType", "ModeEvent",
    "detect_time_event", "detect_proximity_event", "MotionDetector",
]


class EventType(str, Enum):
    TIME_EVENT = "TIME_EVENT"
    IMPACT_EVENT = "IMPACT_EVENT"       # abstract motion/impact-like event
    PROXIMITY_EVENT = "PROXIMITY_EVENT"


@dataclass(frozen=True)
class ModeEvent:
    """One abstract candidate event, before arbitration."""

    kind: EventType
    t: float
    source: str
    detail: dict = field(default_factory=dict)


def detect_time_event(t: float, config: TimeEventConfig) -> Optional[ModeEvent]:
    """Fires once the simulation clock reaches `config.event_time_s`, within
    `[min_time_s, max_time_s]`."""
    if not config.enabled:
        return None
    if not (config.min_time_s <= t <= config.max_time_s):
        return None
    if t >= config.event_time_s:
        return ModeEvent(EventType.TIME_EVENT, t, "time_sensor",
                          {"event_time_s": config.event_time_s})
    return None


def detect_proximity_event(reading: Optional[SensorReading],
                            config: ProximityEventConfig) -> Optional[ModeEvent]:
    """Fires when an altitude-like reading falls to or below the trigger
    value. A missing, invalid, or stale reading never fires."""
    if not config.enabled:
        return None
    if reading is None or not reading.valid or reading.stale:
        return None
    if reading.value is None:
        return None
    if reading.value <= config.trigger_value:
        return ModeEvent(EventType.PROXIMITY_EVENT, reading.t, "altitude_sensor",
                          {"value": reading.value, "trigger_value": config.trigger_value})
    return None


@dataclass
class MotionDetector:
    """Stateful threshold/persistence/hysteresis (+ optional rate-of-change)
    detector over an abstract acceleration/motion-like signal.

    Persistence: the signal must sit at or above threshold for
    `persistence_samples` consecutive VALID, non-stale readings before the
    detector fires. Hysteresis: once armed (signal has crossed `threshold`),
    the detector stays armed down to `threshold - hysteresis`, so noise
    dithering near the threshold does not reset the persistence count on
    every sample. A dropout, invalid, or stale reading resets the count --
    a gap in coverage cannot silently satisfy persistence.
    """

    config: MotionEventConfig
    _consecutive: int = field(default=0, init=False)
    _armed_above: bool = field(default=False, init=False)
    _last_value: Optional[float] = field(default=None, init=False)
    _last_t: Optional[float] = field(default=None, init=False)

    def reset(self) -> None:
        self._consecutive = 0
        self._armed_above = False
        self._last_value = None
        self._last_t = None

    def update(self, reading: Optional[SensorReading]) -> Optional[ModeEvent]:
        if not self.config.enabled:
            return None
        if reading is None or not reading.valid or reading.stale or reading.value is None:
            self.reset()
            return None

        value = reading.value
        level_ok = (value >= self.config.threshold - self.config.hysteresis
                    if self._armed_above else value >= self.config.threshold)

        roc_ok = True
        if self.config.rate_of_change_threshold is not None:
            if self._last_value is not None and self._last_t is not None:
                dt = reading.t - self._last_t
                roc = abs(value - self._last_value) / dt if dt > 0.0 else float("inf")
                roc_ok = roc >= self.config.rate_of_change_threshold
            else:
                roc_ok = False

        self._last_value, self._last_t = value, reading.t

        if level_ok and roc_ok:
            self._consecutive += 1
            self._armed_above = True
        else:
            self._consecutive = 0
            self._armed_above = False

        if self._consecutive >= self.config.persistence_samples:
            return ModeEvent(EventType.IMPACT_EVENT, reading.t, "motion_sensor",
                              {"value": value, "consecutive": self._consecutive})
        return None
