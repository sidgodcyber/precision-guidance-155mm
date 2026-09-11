"""
Configuration schema for the generic simulation event engine.

These are software state-machine and detector parameters, not real-world
arming, impact, or proximity thresholds -- every default here is a
simulation fixture chosen to exercise the state machine, not a value derived
from any operational specification. Ranges are enforced by pydantic at
construction; `fuze.validation` wraps construction errors into
`FuzeConfigError` and adds cross-object checks.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "EVENT_KINDS",
    "ArmingConfig",
    "TimeEventConfig",
    "MotionEventConfig",
    "ProximityEventConfig",
    "FuzeEngineConfig",
]

#: The three abstract event kinds arbitration chooses among. Names echo the
#: engagement's terminology (impact/proximity/time-like) but are abstract
#: software event labels -- see fuze/arbitration.py.
EVENT_KINDS = ("IMPACT_EVENT", "PROXIMITY_EVENT", "TIME_EVENT")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArmingConfig(_Strict):
    """Software arming-state timing. `arming_delay_s` is a simulation
    fixture, not a fielded arming-delay specification."""

    arming_delay_s: float = Field(2.0, ge=0.0, le=120.0)
    require_startup_ok: bool = True


class TimeEventConfig(_Strict):
    enabled: bool = True
    event_time_s: float = Field(30.0, ge=0.0, le=600.0)
    min_time_s: float = Field(0.0, ge=0.0, le=600.0)
    max_time_s: float = Field(120.0, ge=0.0, le=600.0)

    @model_validator(mode="after")
    def _check_range(self) -> "TimeEventConfig":
        if self.min_time_s > self.max_time_s:
            raise ValueError("min_time_s must be <= max_time_s")
        if not (self.min_time_s <= self.event_time_s <= self.max_time_s):
            raise ValueError("event_time_s must lie within [min_time_s, max_time_s]")
        return self


class MotionEventConfig(_Strict):
    """Abstract acceleration/motion-like signal detector: threshold with
    persistence and hysteresis, and an optional rate-of-change gate."""

    enabled: bool = True
    threshold: float = Field(50.0, gt=0.0)
    persistence_samples: int = Field(3, ge=1, le=50)
    hysteresis: float = Field(5.0, ge=0.0)
    rate_of_change_threshold: Optional[float] = Field(None, gt=0.0)

    @model_validator(mode="after")
    def _check_hysteresis(self) -> "MotionEventConfig":
        if self.hysteresis >= self.threshold:
            raise ValueError("hysteresis must be smaller than threshold")
        return self


class ProximityEventConfig(_Strict):
    """Abstract altitude-like signal detector."""

    enabled: bool = True
    trigger_value: float = Field(5.0, ge=0.0)


class FuzeEngineConfig(_Strict):
    """The full configuration consumed by the state machine + detectors for
    one simulated round."""

    arming: ArmingConfig = Field(default_factory=ArmingConfig)
    time_event: TimeEventConfig = Field(default_factory=TimeEventConfig)
    motion_event: MotionEventConfig = Field(default_factory=MotionEventConfig)
    proximity_event: ProximityEventConfig = Field(default_factory=ProximityEventConfig)
    precedence: tuple = Field(default=EVENT_KINDS)

    @model_validator(mode="after")
    def _check_precedence(self) -> "FuzeEngineConfig":
        if set(self.precedence) - set(EVENT_KINDS):
            raise ValueError(f"precedence entries must be a subset of {EVENT_KINDS}")
        if len(set(self.precedence)) != len(self.precedence):
            raise ValueError("precedence must not repeat an event kind")
        return self
