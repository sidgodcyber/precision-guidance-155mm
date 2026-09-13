"""
Generic multi-mode trajectory-event state machine.

States are software states, not physical fuze states:

    SAFE -> ARMING -> ARMED -> FUNCTION_EVENT

`FUNCTION_EVENT` is a terminal software state whose only meaning is "the
event engine produced its abstract terminal event." Nothing downstream of it
is modelled here.

Invariants enforced:
  * only the four listed transitions are legal; anything else raises
    `InvalidTransitionError`.
  * `SAFE -> ARMING` requires the configured startup prerequisite
    (`confirm_startup()`) to have been satisfied first, when
    `require_startup_ok` is set.
  * `ARMING -> ARMED` requires `config.arming_delay_s` of simulated time to
    have elapsed since `ARMING` was entered; `update()` performs this
    transition only once the delay has been met and is a no-op otherwise.
  * `ARMED -> FUNCTION_EVENT` requires an arbitrated `fuze.events.ModeEvent`.
  * `FUNCTION_EVENT` is a fixed point: a repeated `function()` call once
    terminal is a deterministic no-op (returns False), never an exception
    and never a second transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from fuze.config import ArmingConfig
from fuze.events import ModeEvent

__all__ = ["SimState", "InvalidTransitionError", "EventStateMachine"]


class SimState(str, Enum):
    SAFE = "SAFE"
    ARMING = "ARMING"
    ARMED = "ARMED"
    FUNCTION_EVENT = "FUNCTION_EVENT"


class InvalidTransitionError(Exception):
    """A transition was attempted that the state machine does not allow."""


@dataclass
class EventStateMachine:
    config: ArmingConfig
    state: SimState = field(default=SimState.SAFE, init=False)
    terminal_event: Optional[ModeEvent] = field(default=None, init=False)
    history: List[Tuple[float, SimState, SimState]] = field(default_factory=list, init=False)
    _startup_ok: bool = field(default=False, init=False)
    _arming_entered_t: Optional[float] = field(default=None, init=False)

    def confirm_startup(self) -> None:
        """Satisfies the SAFE -> ARMING startup prerequisite."""
        self._startup_ok = True

    def begin_arming(self, t: float) -> None:
        if self.state is not SimState.SAFE:
            raise InvalidTransitionError(
                f"begin_arming is only valid from SAFE, not {self.state.value}")
        if self.config.require_startup_ok and not self._startup_ok:
            raise InvalidTransitionError("startup prerequisite not confirmed")
        self._transition(t, SimState.ARMING)
        self._arming_entered_t = t

    def update(self, t: float) -> None:
        """Advance time-driven transitions. Only ARMING -> ARMED is
        time-driven; every other transition is event-driven and goes through
        `begin_arming()` or `function()`."""
        if self.state is SimState.ARMING:
            assert self._arming_entered_t is not None
            if t - self._arming_entered_t >= self.config.arming_delay_s:
                self._transition(t, SimState.ARMED)

    def function(self, t: float, event: ModeEvent) -> bool:
        """Attempt the terminal transition given an arbitrated event.

        Returns True if this call performed the transition, False if the
        machine was already terminal (a deterministic no-op). Raises
        `InvalidTransitionError` if called from any state other than ARMED
        or FUNCTION_EVENT.
        """
        if self.state is SimState.FUNCTION_EVENT:
            return False
        if self.state is not SimState.ARMED:
            raise InvalidTransitionError(
                f"function() is only valid from ARMED, not {self.state.value}")
        self._transition(t, SimState.FUNCTION_EVENT)
        self.terminal_event = event
        return True

    def _transition(self, t: float, to: SimState) -> None:
        self.history.append((t, self.state, to))
        self.state = to
