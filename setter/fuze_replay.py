"""
Deterministic replay of the fuze event engine (`fuze/`) against a REAL
fired round's state trajectory.

"Rendered state machine, not a simulation" (Control Room spec, Flight Deck):
no sensor noise, no dropout, no new randomness. `fuze.sensors.SensorReading`
objects are built directly from the fired round's own truth arrays -- a
zero-noise passthrough -- because this is a replay of an ACTUAL round, not
a fresh Monte Carlo draw with its own dispersion. `fuze.state_machine`'s
`EventStateMachine`, `fuze.events`' three detectors, and
`fuze.arbitration.arbitrate` are reused exactly as `fuze/` defines them --
nothing about arming or detection logic is reimplemented here.

ARMING START. The fuze's arming sequence is taken to begin at deployment
(`t_dep_actual`): `state_trajectory` has no earlier data (see
`fire_worker.py` -- the pre-deployment leg isn't captured), so there is no
earlier point to start it from. `arming_delay_s` is
`fuze.config.ArmingConfig`'s own fixture default (2.0 s) -- per the spec's
own out-of-scope note, "No threshold values for the event engine -- those
stay fixtures."

MOTION SIGNAL. `fuze.events.MotionDetector` wants an acceleration-like
truth signal, which the real 6-DOF trajectory does not log directly. It is
derived here by finite-differencing |velocity| -- the same approach
`fuze.trajectory_adapter.generate_trajectory` already uses for its own
(reduced-order) trajectory; not a new derivation invented for this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from fuze.arbitration import arbitrate
from fuze.config import ArmingConfig, MotionEventConfig, ProximityEventConfig, TimeEventConfig
from fuze.events import MotionDetector, detect_proximity_event, detect_time_event
from fuze.sensors import SensorReading
from fuze.state_machine import EventStateMachine, SimState

__all__ = ["FuzeReplayResult", "replay_fuze_state"]

#: The fixture default from fuze.config.ArmingConfig, named here so the UI
#: can display it without constructing a throwaway ArmingConfig just to
#: read one field.
ARMING_DELAY_S = ArmingConfig().arming_delay_s


@dataclass(frozen=True)
class FuzeReplayResult:
    state: str  # SimState value ("SAFE"/"ARMING"/"ARMED"/"FUNCTION_EVENT")
    reason: str
    terminal_event_kind: Optional[str]
    terminal_event_t: Optional[float]


def _configs_for_mode(mode: str, event_time_s: float, motion_threshold: float,
                      proximity_trigger_m: float) -> Tuple[TimeEventConfig, MotionEventConfig, ProximityEventConfig]:
    """Only the selected mode's detector is enabled -- "combined" enables
    all three, matching setter.schemas.EventConfiguration's own mode
    vocabulary. Hysteresis is a fixed FRACTION of the threshold (not a
    fixed absolute value) so an operator-chosen small threshold can never
    make MotionEventConfig's own hysteresis-must-be-smaller-than-threshold
    check fail."""
    return (
        TimeEventConfig(enabled=mode in ("time", "combined"), event_time_s=event_time_s,
                        min_time_s=0.0, max_time_s=600.0),
        MotionEventConfig(enabled=mode in ("motion", "combined"), threshold=motion_threshold,
                          persistence_samples=3, hysteresis=motion_threshold * 0.1),
        ProximityEventConfig(enabled=mode in ("proximity", "combined"), trigger_value=proximity_trigger_m),
    )


def replay_fuze_state(state_trajectory: dict, t_dep_actual: float, mode: str,
                      event_time_s: float, motion_threshold: float,
                      proximity_trigger_m: float, scrubber_t: float) -> FuzeReplayResult:
    """The fuze's SimState at `scrubber_t`, replayed deterministically from
    `state_trajectory` (a fired round's `state_trajectory` dict -- see
    `fire_worker.py`) up to that time. Cheap: one pass over the guided-phase
    samples (hundreds, not thousands), pure arithmetic, no simulation."""
    t_cfg, m_cfg, p_cfg = _configs_for_mode(mode, event_time_s, motion_threshold, proximity_trigger_m)
    arming_cfg = ArmingConfig()
    sm = EventStateMachine(arming_cfg)
    sm.confirm_startup()  # the only "startup ok" this replay can assert is "yes"
    sm.begin_arming(t_dep_actual)

    ts = state_trajectory["t"]
    positions = state_trajectory["position"]
    velocities = state_trajectory["velocity"]
    speed = [float(np.linalg.norm(v)) for v in velocities]
    accel = list(np.gradient(speed, ts)) if len(ts) > 1 else [0.0] * len(ts)

    motion_det = MotionDetector(m_cfg)
    terminal = None

    for i, t in enumerate(ts):
        if t > scrubber_t:
            break
        sm.update(t)
        if sm.state == SimState.ARMED and terminal is None:
            altitude = -positions[i][2]
            candidates = [
                detect_time_event(t, t_cfg),
                detect_proximity_event(SensorReading(t=t, value=altitude, valid=True), p_cfg),
                motion_det.update(SensorReading(t=t, value=accel[i], valid=True)),
            ]
            event = arbitrate(candidates)
            if event is not None:
                sm.function(t, event)
                terminal = event

    if sm.state == SimState.SAFE:
        reason = "Has not begun arming."
    elif sm.state == SimState.ARMING:
        elapsed = max(0.0, scrubber_t - t_dep_actual)
        reason = f"{elapsed:.1f} of {arming_cfg.arming_delay_s:.1f} s arming delay elapsed."
    elif sm.state == SimState.ARMED:
        reason = f"Armed; waiting for a {mode} event ({t_cfg.event_time_s:.0f} s / " \
                 f"{m_cfg.threshold:.0f} threshold / {p_cfg.trigger_value:.0f} m, as configured)."
    else:
        reason = (f"Functioned via {terminal.kind.value} at t={terminal.t:.2f} s."
                  if terminal else "Functioned.")

    return FuzeReplayResult(
        state=sm.state.value, reason=reason,
        terminal_event_kind=(terminal.kind.value if terminal else None),
        terminal_event_t=(terminal.t if terminal else None),
    )
