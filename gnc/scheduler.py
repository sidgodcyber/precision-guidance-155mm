"""
Authority schedulers: how much of a finite, expiring correction budget to
spend, and when.

Step 3, Task C. docs/GUIDANCE-DESIGN.md.

THE PROBLEM
-----------
Correction effectiveness scales roughly as the square of the flight time
remaining, so a metre of correction bought at deployment costs a small
fraction of what the same metre costs near impact. That argues for spending
everything immediately. Three things argue against it:

  1. For the first 2.74 s +/- 0.19 after deployment there is no steering force
     in any direction at all (docs/STAGED-DEPLOYMENT.md section 9), so the
     early part of the budget does not exist to be spent.
  2. The impact-point prediction is worst early, because the reduced-order
     model must extrapolate the whole remaining flight. Committing the
     direction on the earliest estimate points the correction at the wrong
     place.
  3. The budget is finite and the actuator switches ONCE. Magnitude control is
     a single contiguous hold of variable length
     (docs/CONTROL-CHARACTERISATION.md section 8.4), so "spend a bit more
     later" means "do not stop yet", and stopping is irreversible on the
     timescale of a re-acquisition.

So the scheduler's real decision, once per cycle, is binary -- keep holding or
release -- plus the angle to hold at. Every law here has that shape. What
differs is when they commit the direction, and what they do with the
difference between the correction still required and the correction still
available.

THE ANISOTROPY IS NOT AVERAGED OVER
-----------------------------------
The reachable set is an ellipse of axis ratio 1.17 on the adopted
configuration, so a metre of range correction and a metre of deflection
correction cost different amounts of hold time. Every law here asks the map
for the correction available ALONG THE MISS DIRECTION
(`AuthorityMap.reach_along`, equation 4 of `gnc.inverse_map`) rather than for
a scalar "authority". `IsotropicScheduler` deliberately does not, and exists
to measure what that costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .inverse_map import AuthorityMap

__all__ = [
    "Context", "Decision", "Scheduler", "ProportionalScheduler",
    "DeadbandScheduler", "BudgetScheduler", "SingleShotScheduler",
    "IsotropicScheduler", "SCHEDULERS", "make_scheduler",
]


@dataclass
class Context:
    """Everything one scheduler cycle is allowed to see."""

    t: float                     # s, absolute
    t_go: float                  # s to predicted impact
    miss: np.ndarray             # m, target minus predicted impact
    amap: AuthorityMap
    armed: bool                  # a steering force is believed to exist
    holding: bool                # the loop is currently holding
    phi: Optional[float]         # rad, the command in force
    cycles: int                  # cycles since deployment
    authority_ok: bool = True    # the monitor has not declared a failure


@dataclass
class Decision:
    hold: bool
    phi: float
    note: str = ""
    #: Diagnostics the log carries; no law reads them back.
    required_m: float = float("nan")
    available_m: float = float("nan")
    planned_t_go_end: float = float("nan")


class Scheduler:
    """
    Common machinery. Subclasses implement `_decide`.

    The contiguity constraint lives here and not in the laws: once `released`
    is set nothing re-holds, because a release is followed by the nose running
    away backwards at tens of rad/s and holds shorter than about 0.5 s deliver
    nothing (docs/CONTROL-CHARACTERISATION.md section 8.3).
    """

    name = "base"

    def __init__(self, **kw):
        self.released = False
        self.committed_phi: Optional[float] = None
        self.commit_time: Optional[float] = None
        self.release_time: Optional[float] = None
        self.plan: Optional[dict] = None
        self.opts = dict(kw)

    # -- the public entry point ------------------------------------------
    def __call__(self, ctx: Context) -> Decision:
        if self.released:
            return Decision(False, self.committed_phi or 0.0, "released")
        if not ctx.authority_ok:
            self.released = True
            self.release_time = ctx.t
            return Decision(False, self.committed_phi or 0.0, "no_authority")
        d = self._decide(ctx)
        if d.hold:
            self.committed_phi = d.phi
            # The pre-arm cycles are a hold, but they are not a COMMITMENT:
            # no steering force exists to spend, and the servo is being given
            # something to acquire because the acquisition is what opens the
            # stage-2 gate. Only an armed cycle starts the clock.
            if self.commit_time is None and ctx.armed:
                self.commit_time = ctx.t
        elif ctx.armed and self.commit_time is not None:
            self.released = True
            self.release_time = ctx.t
        return d

    def _decide(self, ctx: Context) -> Decision:
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------
    @staticmethod
    def _hold_now(ctx: Context, phi: Optional[float], note: str,
                  **kw) -> Decision:
        """
        Hold at `phi`. If the map could not be inverted -- a singular
        increment, which is what zero authority looks like from inside the law
        -- keep the previous command rather than inventing one.
        """
        if phi is None:
            phi = ctx.phi if ctx.phi is not None else 0.0
            note = note + "+singular"
        return Decision(True, float(phi), note, **kw)

    def summary(self) -> dict:
        return {
            "scheduler": self.name,
            "commit_time": self.commit_time,
            "release_time": self.release_time,
            "committed_phi_deg": (math.degrees(self.committed_phi)
                                  if self.committed_phi is not None else None),
            "plan": self.plan,
        }


# ===========================================================================
# 1. Proportional
# ===========================================================================
class ProportionalScheduler(Scheduler):
    """
    Point at the miss, hold while there is a miss.

    Each cycle: invert the map for the correction still required, command that
    angle, and keep holding until the required correction has been delivered
    -- which the map reports as "the release time that delivers it is now".

    This is the null-seeking law and it is the obvious one. Its weakness is
    that it re-points on every cycle including the earliest, when the
    prediction is worst, and it has no notion of a budget: it will spend the
    whole of the remaining authority chasing a miss it cannot reach, and it
    will keep holding to remove the last metre of a residual that is smaller
    than the prediction error producing it.
    """

    name = "proportional"

    def _decide(self, ctx: Context) -> Decision:
        need = float(np.hypot(ctx.miss[0], ctx.miss[1]))
        A, _ = ctx.amap.remaining(ctx.t_go)
        avail = ctx.amap.reach_along(A, ctx.miss) if need > 0 else 0.0
        if not ctx.armed:
            phi = ctx.amap.command_for(ctx.miss, ctx.t_go, 0.0) if need > 0 else ctx.phi
            return self._hold_now(ctx, phi, "pre_arm",
                                  required_m=need, available_m=avail)
        tge, phi, got, sat = ctx.amap.hold_end_for(ctx.miss, ctx.t_go)
        if not sat and tge >= ctx.t_go - 1e-9:
            return Decision(False, self.committed_phi or 0.0, "nulled",
                            required_m=need, available_m=avail,
                            planned_t_go_end=tge)
        return self._hold_now(ctx, phi, "saturated" if sat else "tracking",
                              required_m=need, available_m=avail,
                              planned_t_go_end=tge)


# ===========================================================================
# 2. Deadband
# ===========================================================================
class DeadbandScheduler(Scheduler):
    """
    As proportional, but stop when the residual falls inside a band sized on
    the PREDICTION ERROR rather than on the CEP target.

    The point of the band is not to save authority. It is that below the
    prediction error the law is steering against noise: the residual it is
    trying to remove is not known to exist. `deadband_m` is therefore set from
    the measured impact-point-prediction error at the time the decision is
    made (docs/GUIDANCE-DESIGN.md section 3), and freezing the direction
    inside the band is the same argument applied to pointing.
    """

    name = "deadband"

    def __init__(self, deadband_m: float = 15.0, freeze_deg: float = 5.0, **kw):
        super().__init__(deadband_m=deadband_m, freeze_deg=freeze_deg, **kw)
        self.deadband_m = float(deadband_m)
        self.freeze = math.radians(float(freeze_deg))

    def _decide(self, ctx: Context) -> Decision:
        need = float(np.hypot(ctx.miss[0], ctx.miss[1]))
        A, _ = ctx.amap.remaining(ctx.t_go)
        avail = ctx.amap.reach_along(A, ctx.miss) if need > 0 else 0.0
        if not ctx.armed:
            phi = ctx.amap.command_for(ctx.miss, ctx.t_go, 0.0) if need > 0 else ctx.phi
            return self._hold_now(ctx, phi, "pre_arm",
                                  required_m=need, available_m=avail)
        if need <= self.deadband_m:
            return Decision(False, self.committed_phi or 0.0, "deadband",
                            required_m=need, available_m=avail)
        tge, phi, got, sat = ctx.amap.hold_end_for(ctx.miss, ctx.t_go)
        # Re-point only if the new angle differs by more than the band: a
        # sub-degree correction costs a slew and buys nothing.
        if (self.committed_phi is not None and phi is not None
                and abs(_wrap(phi - self.committed_phi)) < self.freeze):
            phi = self.committed_phi
        return self._hold_now(ctx, phi, "saturated" if sat else "tracking",
                              required_m=need, available_m=avail,
                              planned_t_go_end=tge)


# ===========================================================================
# 3. Explicit budget allocation with a held reserve
# ===========================================================================
class BudgetScheduler(Scheduler):
    """
    Plan the whole hold against the budget, hold back a reserve, then spend it.

    Each cycle:
      * ask the map for the correction still AVAILABLE along the miss
        direction, `avail`, and for the correction still REQUIRED, `need`;
      * if `need > avail` the target is outside the reachable set: commit to
        the best available direction and hold to impact. There is nothing to
        allocate and no reason to reserve;
      * otherwise plan a release that delivers `need` but spends no more than
        `1 - reserve` of `avail`, so that a later correction of up to
        `reserve * avail` remains possible;
      * inside `endgame_t_go` the reserve has no future use, so release it and
        plan for the whole of `need`.

    The reserve is not idle capacity. It is the answer to a specific failure of
    the null-seeking laws: they converge the prediction to zero, release, and
    then have nothing left when the prediction moves -- which it does, because
    the prediction error is largest early and the law that releases earliest is
    the one that trusted it most.

    Because the hold is contiguous, "reserving" authority means declining to
    extend the hold, not saving it for a second burst. The reserve is spent by
    holding longer than the plan, which is available for as long as the round
    has not released.
    """

    name = "budget"

    def __init__(self, reserve: float = 0.25, endgame_t_go: float = 8.0,
                 commit_t_go: Optional[float] = None, **kw):
        super().__init__(reserve=reserve, endgame_t_go=endgame_t_go,
                         commit_t_go=commit_t_go, **kw)
        self.reserve = float(reserve)
        self.endgame = float(endgame_t_go)

    def _decide(self, ctx: Context) -> Decision:
        need = float(np.hypot(ctx.miss[0], ctx.miss[1]))
        A, _ = ctx.amap.remaining(ctx.t_go)
        avail = ctx.amap.reach_along(A, ctx.miss) if need > 0 else 0.0
        if not ctx.armed:
            phi = ctx.amap.command_for(ctx.miss, ctx.t_go, 0.0) if need > 0 else ctx.phi
            return self._hold_now(ctx, phi, "pre_arm",
                                  required_m=need, available_m=avail)

        if need <= 0.0:
            return Decision(False, self.committed_phi or 0.0, "no_miss",
                            required_m=need, available_m=avail)

        if need >= avail:
            phi = ctx.amap.command_for(ctx.miss, ctx.t_go, 0.0)
            self.plan = {"mode": "saturated", "need_m": need, "avail_m": avail,
                         "t_go": ctx.t_go}
            return self._hold_now(ctx, phi, "saturated",
                                  required_m=need, available_m=avail,
                                  planned_t_go_end=0.0)

        endgame = ctx.t_go <= self.endgame
        cap = need if endgame else min(need, (1.0 - self.reserve) * avail)
        u = ctx.miss / need
        tge, phi, got, sat = ctx.amap.hold_end_for(u * cap, ctx.t_go)
        self.plan = {"mode": "endgame" if endgame else "planned",
                     "need_m": need, "avail_m": avail, "spend_m": cap,
                     "t_go_end": tge, "t_go": ctx.t_go}
        if tge >= ctx.t_go - 1e-9:
            return Decision(False, self.committed_phi or 0.0,
                            "delivered", required_m=need, available_m=avail,
                            planned_t_go_end=tge)
        return self._hold_now(ctx, phi, "endgame" if endgame else "planned",
                              required_m=need, available_m=avail,
                              planned_t_go_end=tge)


# ===========================================================================
# 4. Single shot -- the literal reading of "decided once"
# ===========================================================================
class SingleShotScheduler(Scheduler):
    """
    Decide the direction and the duration once, at the first armed cycle, and
    fly them open loop.

    This is the law docs/CONTROL-CHARACTERISATION.md section 8.4's
    recommendation describes most literally, and it is the cheapest thing that
    could work: one propagation, one inversion, one timer. It is included as a
    measured baseline, because the case for closing the loop has to be made
    against it rather than assumed.
    """

    name = "single_shot"

    def _decide(self, ctx: Context) -> Decision:
        need = float(np.hypot(ctx.miss[0], ctx.miss[1]))
        A, _ = ctx.amap.remaining(ctx.t_go)
        avail = ctx.amap.reach_along(A, ctx.miss) if need > 0 else 0.0
        if not ctx.armed:
            phi = ctx.amap.command_for(ctx.miss, ctx.t_go, 0.0) if need > 0 else ctx.phi
            return self._hold_now(ctx, phi, "pre_arm",
                                  required_m=need, available_m=avail)
        if self.plan is None:
            tge, phi, got, sat = ctx.amap.hold_end_for(ctx.miss, ctx.t_go)
            self.plan = {"t_go_end": tge, "phi_deg": math.degrees(phi),
                         "need_m": need, "avail_m": avail,
                         "delivered_m": got, "saturated": sat,
                         "decided_at_t_go": ctx.t_go}
            return self._hold_now(ctx, phi, "committed",
                                  required_m=need, available_m=avail,
                                  planned_t_go_end=tge)
        tge = self.plan["t_go_end"]
        if ctx.t_go <= tge:
            return Decision(False, self.committed_phi or 0.0, "plan_complete",
                            required_m=need, available_m=avail,
                            planned_t_go_end=tge)
        return Decision(True, self.committed_phi or 0.0, "executing",
                        required_m=need, available_m=avail,
                        planned_t_go_end=tge)


# ===========================================================================
# 5. Isotropic -- the control experiment
# ===========================================================================
class IsotropicScheduler(Scheduler):
    """
    A budget law that has been told the reachable set is a circle.

    It uses the rms amplitude of the remaining set as "the authority" in every
    direction, and it points by applying one fixed rotation -- the map's mean
    rotation over the guided phase -- rather than inverting the map. That is
    exactly the modelling shortcut a law would take if it treated the miss
    vector as isotropic, and it is here to price it rather than to assert that
    it is wrong.
    """

    name = "isotropic"

    def __init__(self, rotation_deg: float = 0.0, **kw):
        super().__init__(rotation_deg=rotation_deg, **kw)
        self.rotation = math.radians(float(rotation_deg))

    def _phi(self, ctx: Context) -> float:
        return _wrap(math.atan2(ctx.miss[1], ctx.miss[0]) - self.rotation)

    def _decide(self, ctx: Context) -> Decision:
        need = float(np.hypot(ctx.miss[0], ctx.miss[1]))
        A, _ = ctx.amap.remaining(ctx.t_go)
        s = np.linalg.svd(A, compute_uv=False)
        avail = float(math.sqrt(0.5 * (s[0] ** 2 + s[1] ** 2)))   # rms, not directional
        if not ctx.armed:
            return self._hold_now(ctx, self._phi(ctx), "pre_arm",
                                  required_m=need, available_m=avail)
        if need <= 0.0:
            return Decision(False, self.committed_phi or 0.0, "no_miss",
                            required_m=need, available_m=avail)
        if need >= avail:
            return self._hold_now(ctx, self._phi(ctx), "saturated",
                                  required_m=need, available_m=avail)
        # Solve for the release time on the ISOTROPIC magnitude model.
        lo, hi = 0.0, ctx.t_go
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            Am, _ = ctx.amap.increment(ctx.t_go, mid)
            sm = np.linalg.svd(Am, compute_uv=False)
            if math.sqrt(0.5 * (sm[0] ** 2 + sm[1] ** 2)) >= need:
                lo = mid
            else:
                hi = mid
        tge = 0.5 * (lo + hi)
        if tge >= ctx.t_go - 1e-9:
            return Decision(False, self.committed_phi or 0.0, "delivered",
                            required_m=need, available_m=avail,
                            planned_t_go_end=tge)
        return self._hold_now(ctx, self._phi(ctx), "tracking",
                              required_m=need, available_m=avail,
                              planned_t_go_end=tge)


# ===========================================================================
def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


SCHEDULERS = {
    "proportional": ProportionalScheduler,
    "deadband": DeadbandScheduler,
    "budget": BudgetScheduler,
    "single_shot": SingleShotScheduler,
    "isotropic": IsotropicScheduler,
}


def make_scheduler(name: str, **kw) -> Scheduler:
    if name not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {name!r}; have {sorted(SCHEDULERS)}")
    return SCHEDULERS[name](**kw)
