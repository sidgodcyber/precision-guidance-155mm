"""
Simulation sensor abstractions for the generic event engine.

`TimeSensor`, `MotionSensor`, and `AltitudeSensor` turn trajectory truth
(`fuze.trajectory_adapter.TrajectorySample`) into simulated measurements with
configurable noise, sampling period, dropout, and latency. Sensor generation
is deliberately kept separate from state-transition logic: a sensor here
knows nothing about arming, modes, or events -- it produces a
`SensorReading` and nothing else. `fuze.events` and `fuze.state_machine`
consume readings; they never generate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["SensorReading", "TimeSensor", "MotionSensor", "AltitudeSensor"]


@dataclass(frozen=True)
class SensorReading:
    """One simulated measurement.

    `valid=False` means the sensor produced nothing this tick (dropout, or no
    sample has arrived yet under latency). `stale=True` means a value is
    present but older than `stale_after_s` -- still returned, so a consumer
    can decide whether a stale reading is usable, rather than the sensor
    deciding for it.
    """

    t: float
    value: Optional[float]
    valid: bool
    stale: bool = False
    age_s: float = 0.0


class TimeSensor:
    """The simulation clock, optionally jittered or dropped out.

    There is no real-world timing hardware modelled here -- this is a
    software clock reading used to drive the abstract TIME_EVENT detector.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None,
                 jitter_s: float = 0.0, dropout_prob: float = 0.0):
        if jitter_s < 0.0:
            raise ValueError("jitter_s must be >= 0")
        if not (0.0 <= dropout_prob <= 1.0):
            raise ValueError("dropout_prob must be in [0, 1]")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.jitter_s = float(jitter_s)
        self.dropout_prob = float(dropout_prob)

    def read(self, t_true: float) -> SensorReading:
        if self.dropout_prob and self.rng.random() < self.dropout_prob:
            return SensorReading(t=t_true, value=None, valid=False)
        jitter = self.rng.normal(0.0, self.jitter_s) if self.jitter_s else 0.0
        return SensorReading(t=t_true, value=t_true + jitter, valid=True)


class _SampledNoisySensor:
    """Shared sample/noise/dropout/latency machinery for a scalar sensor
    reading a scalar truth signal (acceleration-like or altitude-like).

    Sampling: a new measurement is only drawn every `sample_period_s`; in
    between, the sensor holds (sample-and-hold) its last delivered value.
    Latency: a measurement drawn at `t_true` is not delivered until
    `t_true + latency_s`, modelling a processing/transport delay.
    Dropout: a drawn measurement is discarded with probability
    `dropout_prob` and never delivered.
    Staleness: once delivered, a held value older than `stale_after_s` is
    flagged `stale` but still returned.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None,
                 noise_std: float = 0.0, sample_period_s: float = 0.0,
                 dropout_prob: float = 0.0, latency_s: float = 0.0,
                 stale_after_s: Optional[float] = None):
        if noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        if sample_period_s < 0.0:
            raise ValueError("sample_period_s must be >= 0")
        if not (0.0 <= dropout_prob <= 1.0):
            raise ValueError("dropout_prob must be in [0, 1]")
        if latency_s < 0.0:
            raise ValueError("latency_s must be >= 0")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.noise_std = float(noise_std)
        self.sample_period_s = float(sample_period_s)
        self.dropout_prob = float(dropout_prob)
        self.latency_s = float(latency_s)
        self.stale_after_s = stale_after_s
        self._last_sample_t: Optional[float] = None
        self._pending: list = []
        self._held: Optional[SensorReading] = None

    def _measure(self, t_true: float, true_value: float) -> Optional[SensorReading]:
        if self.dropout_prob and self.rng.random() < self.dropout_prob:
            return None
        noise = self.rng.normal(0.0, self.noise_std) if self.noise_std else 0.0
        return SensorReading(t=t_true, value=float(true_value) + noise, valid=True)

    def read(self, t_true: float, true_value: float) -> SensorReading:
        take_new = (self.sample_period_s <= 0.0 or self._last_sample_t is None
                    or (t_true - self._last_sample_t) >= self.sample_period_s)
        if take_new:
            self._last_sample_t = t_true
            measured = self._measure(t_true, true_value)
            if measured is not None:
                self._pending.append((t_true + self.latency_s, measured))

        ready = [r for (ready_t, r) in self._pending if ready_t <= t_true]
        self._pending = [(ready_t, r) for (ready_t, r) in self._pending
                          if ready_t > t_true]
        if ready:
            self._held = ready[-1]

        if self._held is None:
            return SensorReading(t=t_true, value=None, valid=False)

        age = t_true - self._held.t
        stale = self.stale_after_s is not None and age > self.stale_after_s
        return SensorReading(t=t_true, value=self._held.value, valid=True,
                              stale=stale, age_s=age)


class MotionSensor(_SampledNoisySensor):
    """Reads an abstract acceleration/motion-like truth signal."""


class AltitudeSensor(_SampledNoisySensor):
    """Reads an abstract altitude-like truth signal."""
