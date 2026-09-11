"""
Deterministic arbitration among candidate abstract events.

Precedence order (highest first), per `fuze.config.EVENT_KINDS`:

    IMPACT_EVENT > PROXIMITY_EVENT > TIME_EVENT

Rationale (an engineering decision, not a physical one): the motion/impact
detector fires latest and closest to the terminal condition it is
demonstrating, so when it and a slower-forming candidate (proximity, time)
are both present in the same tick, the impact-like candidate is taken. Ties
within the same kind resolve by earliest `t`. This module only chooses among
`ModeEvent`s that detectors already produced -- it never talks to sensors and
never invents a candidate. A disabled mode or a failed/stale sensor has
already been filtered out upstream (see `fuze.events`); this module simply
never sees a candidate for it.
"""

from __future__ import annotations

from typing import Iterable, Optional

from fuze.config import EVENT_KINDS
from fuze.events import ModeEvent

__all__ = ["arbitrate"]


def arbitrate(candidates: Iterable[Optional[ModeEvent]],
              precedence: tuple = EVENT_KINDS) -> Optional[ModeEvent]:
    """Pick one `ModeEvent` from `candidates` (None entries are ignored).

    Returns None when there is no candidate -- "no event occurred" is a
    normal, deterministic outcome and not an error.
    """
    valid = [c for c in candidates if c is not None]
    if not valid:
        return None
    rank = {kind: i for i, kind in enumerate(precedence)}
    ranked = [c for c in valid if c.kind in rank]
    if not ranked:
        return None
    ranked.sort(key=lambda c: (rank[c.kind], c.t))
    return ranked[0]
