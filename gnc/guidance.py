"""
Step 3 guidance: predict the impact point, invert the reachable set, schedule
the authority, and degrade when any of that stops being possible.

docs/GUIDANCE-DESIGN.md, docs/CEP-CLOSED-LOOP.md.

WHAT THIS MODULE IS
-------------------
Three pieces, deliberately separable:

  `ImpactPredictor`   propagates the step-2 reduced-order model (MPMM) from
                      the current state to ground impact.
  `gnc.inverse_map`   turns a desired ground-plane correction into a commanded
                      roll angle, and reports what is still reachable.
  `gnc.scheduler`     decides, once per cycle, whether to keep holding and at
                      what angle.

`GuidanceLaw` wires them to the 6-DOF through the same two seams step 4 used,
and adds no third one: it is a `step_hook` that latches a command, and the
command it latches is read by `gnc.roll_control.BrakeLaw` exactly as a
`DutyCycleCommander`'s would be. No equation in `sim/` changes.

THE STATE IS TRUE, ON PURPOSE
-----------------------------
`sample()` reads the true 6-DOF state. Step 5 replaces that read with a
navigation filter, and keeping the two separable is what will let step 6 price
the navigation contribution on its own. Every CEP in this step therefore
EXCLUDES navigation error, and says so.

WHY 1 Hz AND NOT 10 Hz
----------------------
docs/MPMM-COMPUTE.md prices one propagation at 1052 derivative evaluations and
concludes 10 Hz is affordable. Affordable is not the same as useful. The servo
has 1.35 Hz of closed-loop bandwidth and takes 0.33-0.48 s to slew 90 degrees
(docs/CONTROL-CHARACTERISATION.md), so a command updated at 10 Hz is a command
the actuator averages: nine of every ten updates are overwritten before the
nose has moved. The scheduler's output is in any case a decision -- keep
holding or release -- whose consequences accrue over seconds. 1 Hz is set by
the actuator, and the propagation budget is spent on `iterate_yaw=True` and
dt = 0.1 s instead. Section 3 of docs/GUIDANCE-DESIGN.md measures what 2 Hz
and 0.5 Hz do instead of asserting it.

THE PREDICTOR KNOWS THE KIT IS OUT
----------------------------------
The MPMM is a bare-shell model. The deployed canards cost 227 m of range on
the adopted engagement, so a bare-shell propagation from the deployment state
over-predicts range by that whole amount. `kit_drag_table` adds the panel
axial force -- profile plus deflection-induced -- to the shell's C_X0, from
the SAME estimated panel aerodynamics `sim.canards` uses for the truth. No
coefficient is fitted to the 6-DOF: the residual error that leaves is measured
and reported (docs/GUIDANCE-DESIGN.md section 3) rather than tuned away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from sim import aerodata, canards as cn, projectile as pr
from models import mpmm

from .inverse_map import AuthorityMap
from .scheduler import Context, Scheduler, make_scheduler

__all__ = [
    "kit_drag_table", "ImpactPredictor", "GuidanceConfig", "GuidanceLaw",
    "ScheduledHold", "MODES",
]

#: The degradation ladder of Task E. `MODES` is the authority on the names.
MODES = ("full", "degraded", "reversionary", "inhibit", "stage2_fail")


# ===========================================================================
# The onboard drag model for the deployed kit
# ===========================================================================
def kit_drag_table(geometry: cn.CanardGeometry, projectile: pr.Projectile,
                   base: Optional[aerodata.AeroTable] = None,
                   n_panels: int = 4) -> aerodata.AeroTable:
    """
    The shell's aerodynamic table with the deployed panels' axial force folded
    into C_X0.

    Per panel the truth model (`sim.canards.CanardModel.__call__`) applies

        F_x = -qbar S_p (C_D0(M) + C_L^2 / (pi AR e))

    so referred to the shell reference area the increment is

        dC_X0 = n S_p / S_ref * (C_D0(M) + C_La(M)^2 <delta^2> / (pi AR e))

    with <delta^2> the mean square panel deflection over the deployed set. The
    incidence-induced part is omitted because the onboard model has no angle
    of attack to evaluate it at; what that leaves is measured, not assumed.
    """
    tab = base if base is not None else aerodata.make_m107_table()
    machs = tab.mach.copy()
    vals = tab.values.copy()
    ratio = n_panels * float(geometry.panel_area) / float(projectile.reference_area)
    pi_ar_e = (math.pi * float(geometry.aspect_ratio_effective)
               * float(geometry.induced_drag_efficiency))
    if n_panels >= 4:
        d2 = 0.5 * (float(geometry.steering_deflection) ** 2
                    + float(geometry.cant_angle) ** 2)
    else:
        d2 = float(geometry.cant_angle) ** 2
    for j, m in enumerate(machs):
        C_D0 = cn.canard_zero_lift_drag(float(m))
        C_La = cn.canard_lift_curve_slope(float(m),
                                          float(geometry.aspect_ratio_effective))
        vals[j, 0] += ratio * (C_D0 + (C_La ** 2) * d2 / pi_ar_e)
    return aerodata.AeroTable(np.column_stack([machs, vals]),
                              name=f"{tab.name}+kit{n_panels}",
                              source="sim.canards panel axial force, "
                                     "referred to the shell reference area")


class ImpactPredictor:
    """
    One MPMM propagation from a 6-DOF state to ground impact.

    `dt` and `iterate_yaw` are the recommendations of docs/MPMM-COMPUTE.md:
    0.1 s costs 1052 derivative evaluations for a 26 s propagation and 1 cm of
    range, and the yaw-of-repose iteration is on because this predictor runs
    at 1 Hz and the compute freed by not running at 10 Hz pays for it.
    """

    def __init__(self, projectile: pr.Projectile, environment,
                 geometry: Optional[cn.CanardGeometry] = None,
                 dt: float = 0.1, iterate_yaw: bool = True,
                 n_panels: int = 4, aero: Optional[aerodata.AeroTable] = None,
                 wind=None, atmosphere=None):
        base = aero if aero is not None else aerodata.make_m107_table()
        table = (kit_drag_table(geometry, projectile, base, n_panels)
                 if geometry is not None else base)
        # `wind` and `atmosphere` are THE MET MESSAGE the fuze was set with.
        # Both default to none-and-standard, which is what every step before 6
        # ran and what a round fired without a met message flies on. Step 6
        # gives them a `sim.atmosphere.MetProfile` of a stated age and prices
        # the difference -- see docs/ATMOSPHERIC-ERROR.md.
        kw = {}
        if wind is not None:
            kw["wind"] = wind
        if atmosphere is not None:
            kw["atmosphere"] = atmosphere
        self.model = mpmm.MpmmModel(projectile=projectile, aero=table,
                                    environment=environment,
                                    iterate_yaw=iterate_yaw, **kw)
        self.dt = float(dt)
        self.calls = 0
        self.derivative_calls = 0

    def predict(self, t: float, y6: np.ndarray) -> tuple:
        """(range_m, deflection_m, time_of_flight_s, terminated)."""
        y7 = mpmm.state_from_sixdof(np.asarray(y6, dtype=float))
        r = mpmm.propagate_to_impact(y7, self.model, dt=self.dt, t0=float(t),
                                     t_max=float(t) + 300.0)
        self.calls += 1
        self.derivative_calls += 4 * r.steps
        return r.range_m, r.drift_m, r.impact_time, r.terminated


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class GuidanceConfig:
    """
    Everything that sets the guidance loop, and the reason for each number.

    As with `gnc.roll_control.ControllerConfig`, no number here was chosen to
    hit a CEP. `rate_hz` is set by the servo, `arm_delay_s` by the measured
    stage-2 dead time, and the monitor thresholds by the measured prediction
    error.
    """

    #: Guidance cycle rate. Set by the 1.35 Hz servo bandwidth, not by the
    #: compute budget. See the module docstring.
    rate_hz: float = 1.0

    #: No steering force exists in any direction until 2.74 s +/- 0.19 after
    #: deployment (docs/STAGED-DEPLOYMENT.md section 9). Before `arm_delay_s`
    #: the law still commands an angle -- the servo has to be given something
    #: to acquire, and the acquisition is what opens the stage-2 gate -- but it
    #: does not treat the correction as begun, and the scheduler is told it is
    #: not armed.
    arm_delay_s: float = 3.0

    predictor_dt: float = 0.1
    iterate_yaw: bool = True

    # -- the ladder -------------------------------------------------------
    mode: str = "full"
    #: `degraded`: seconds after deployment at which navigation stops being
    #: valid. The law then holds the last good command and the last plan.
    nav_fail_after_s: Optional[float] = None
    #: `inhibit`: release and stop commanding if the predicted miss exceeds
    #: this. Sized as a multiple of the reachable set, not as a CEP.
    inhibit_threshold_m: Optional[float] = None

    # -- the authority monitor -------------------------------------------
    #: Detect an actuator that cannot act. Over `monitor_window_s` of holding,
    #: the map predicts a definite movement of the predicted impact point; if
    #: less than `monitor_fraction` of it is observed, the law declares that it
    #: has no authority and stops commanding. This is what makes the stage-2
    #: mechanical failure of docs/STAGED-DEPLOYMENT.md section 7.2 a detected
    #: fault rather than a loop that keeps issuing commands to a dead
    #: actuator.
    authority_monitor: bool = True
    monitor_window_s: float = 6.0
    monitor_fraction: float = 0.30
    #: Absolute floor: below this much predicted movement the test is not
    #: resolvable against the prediction error and is not applied.
    monitor_floor_m: float = 20.0
    #: EXPERIMENT KNOB, DEFAULT OFF. First-order low pass on the PREDICTED
    #: IMPACT POINT, time constant in seconds, applied across guidance cycles.
    #:
    #: Not a design change and not adopted: it exists so that
    #: `analysis.nav_ablation` Task E can test whether the navigation
    #: contribution is estimate ACCURACY or command CHATTER, by feeding the
    #: scheduler a deliberately smoothed prediction and accepting the lag.
    #: `None` leaves every number in steps 3 to 5 bit-identical, and
    #: `test_guidance_prediction_filter_off_is_identity` asserts it.
    #:
    #: The prediction is filtered rather than the state because the state is
    #: moving at 500 m/s and a 5 s lag on POSITION is 2.5 km, while the same
    #: lag on the predicted impact point is the few metres it moves per cycle.
    pred_filter_tau: Optional[float] = None

    #: No monitor window may OPEN before this many seconds after deployment.
    #:
    #: The floor above guards against the prediction error's SIZE; this guards
    #: against its RATE OF CHANGE. Over the first few seconds the predictor's
    #: own bias moves by tens of metres as the reduced model absorbs the
    #: deployment transient, and a window opened there measures that movement
    #: rather than the actuator's. Measured cost of getting it wrong: the
    #: stage-2 failure of docs/DEGRADATION-LADDER.md section 5 was detected on
    #: 69 % of rounds with the first window opening at arming, because on the
    #: rest the predictor's drift masqueraded as a working actuator.
    monitor_start_s: float = 0.0

    #: Step 6. Run the monitor's arithmetic and RECORD it, but never act on
    #: it. `GuidanceLaw.monitor_log` then holds the test statistic of every
    #: window of a round that was allowed to fly to impact, which is what
    #: makes the false-alarm and missed-detection rates of
    #: docs/MONITOR-RETUNE.md a sweep over one campaign instead of one
    #: campaign per threshold. It changes nothing about the flight -- with
    #: `authority_monitor` False and this True, the round is the unmonitored
    #: round -- and `analysis.monte_carlo.task_d` re-flies the operating
    #: points it selects rather than trusting the sweep for the CEP.
    monitor_shadow: bool = False

    @property
    def period(self) -> float:
        return 1.0 / self.rate_hz


# ===========================================================================
# The law
# ===========================================================================
class GuidanceLaw:
    """
    The guidance cycle, and the seam it reaches the 6-DOF through.

    Usage mirrors `gnc.roll_control.BrakeLaw`:

        law = BrakeLaw(controller, guidance.command, deploy_time=t_dep, ...)
        integrate(..., step_hook=guidance.chain(law.sample))

    `command` is a function of TIME ALONE and returns the latched
    (angle, holding) pair, so the restriction step 2.5 placed on the control
    seam is not weakened. `sample` runs on its own 1 Hz grid inside the same
    per-step callback the servo already uses.
    """

    def __init__(self, config: GuidanceConfig, amap: AuthorityMap,
                 predictor: ImpactPredictor, scheduler: Scheduler,
                 target: tuple, deploy_time: float, nav=None):
        self.config = config
        self.amap = amap
        self.predictor = predictor
        self.scheduler = scheduler
        self.target = np.array([float(target[0]), float(target[1])])
        self.deploy_time = float(deploy_time)
        #: Step 5. `None` keeps the promise this module's docstring makes --
        #: `sample()` reads the TRUE state and every CEP excludes navigation
        #: error. A `gnc.navigation.NavigationSystem` here replaces that one
        #: read with the filter's estimate, and its `valid` flag drives the
        #: same `nav_valid` path the degradation ladder already had. The one
        #: read this module makes of the 6-DOF is the only thing that changes.
        self.nav = nav
        #: Whether guidance resumes when the filter becomes valid again.
        #: True is the measured behaviour of a GNSS outage; False reproduces
        #: step 3's scripted, permanent loss.
        self.nav_recovers = True
        #: [start, end] of each interval the filter declared itself invalid.
        self.nav_outages: list = []

        self._phi = 0.0
        self._holding = True
        self._next = -math.inf
        self.samples = 0
        self.mode = config.mode
        self.reverted = config.mode == "reversionary"
        self.nav_valid = True
        self.inhibited = False
        self.inhibit_time: Optional[float] = None
        self.authority_ok = True
        self.authority_fault_time: Optional[float] = None
        self.nav_lost_time: Optional[float] = None
        #: (t, predicted impact) at the start of the current monitor window,
        #: with the map's prediction of where it should have moved to.
        self._mon: Optional[tuple] = None
        #: The last cycle at which navigation WAS valid. `None` until the
        #: first prediction, so a navigation failure before any prediction has
        #: been made leaves the law with no plan to run out and it simply
        #: holds -- which is the right answer, because the alternative is to
        #: release an actuator that has not yet done anything.
        self._last_t: Optional[float] = None
        self._last_t_go: Optional[float] = None
        #: State of the Task E prediction low pass. `None` when it is off.
        self._pred_lp: Optional[np.ndarray] = None
        #: Step 6. One entry per closed monitor window, always recorded and
        #: never read by the law. See `_check_authority`.
        self.monitor_log: list = []

        self.log: dict = {k: [] for k in (
            "t", "t_go", "pred_range", "pred_defl", "miss_range", "miss_defl",
            "miss_m", "phi_deg", "holding", "note", "required_m",
            "available_m", "planned_t_go_end", "armed", "mode")}

        if self.reverted:
            # Navigation was invalid at deployment. The canards are left
            # neutral: all four panels out, the nose brake released, so the
            # steering force rotates with the free nose and averages away. The
            # round flies its ballistic trajectory less the kit's drag, which
            # is what the 224 m aim-off was laid for.
            self._holding = False

    # -- the seam ---------------------------------------------------------
    def command(self, t: float) -> tuple:
        """Held between samples. Deliberately ignores `t`."""
        return self._phi, self._holding

    def chain(self, *hooks) -> Callable:
        """Return a step_hook that runs this law and then `hooks`, in order."""
        def hook(t, y, model):
            self.sample(t, y, model)
            for h in hooks:
                h(t, y, model)
        return hook

    # -- the cycle --------------------------------------------------------
    def sample(self, t: float, y: np.ndarray, model) -> None:
        cfg = self.config
        if t < self.deploy_time:
            self._next = self.deploy_time
            return
        if t < self._next:
            return
        self._next = (self.deploy_time if self._next == -math.inf
                      else self._next) + cfg.period
        if self._next <= t:
            self._next = t + cfg.period
        self.samples += 1

        elapsed = t - self.deploy_time

        if self.reverted:
            self._holding = False
            self._record(t, None, None, "reversionary", armed=False)
            return

        # -- navigation validity -----------------------------------------
        # Two routes to the same state. `nav_fail_after_s` is step 3's
        # SCRIPTED loss, which the ladder was measured with; `self.nav.valid`
        # is step 5's MEASURED one, the filter's own report on itself. They
        # drive the identical branch on purpose, so the ladder's numbers still
        # mean what they meant.
        if self.nav is not None:
            ok = bool(self.nav.valid)
            if self.nav_valid and not ok:
                self.nav_valid = False
                self.nav_lost_time = t
                self.nav_outages.append([float(t), None])
            elif ok and not self.nav_valid and self.nav_recovers:
                # A GNSS outage ends. Step 3's ladder had no way for
                # navigation to come back, because its loss was scripted and
                # permanent; a filter's is not, and refusing to resume would
                # be measuring a design decision rather than the outage.
                self.nav_valid = True
                if self.nav_outages and self.nav_outages[-1][1] is None:
                    self.nav_outages[-1][1] = float(t)
        if (cfg.nav_fail_after_s is not None and self.nav_valid
                and elapsed >= cfg.nav_fail_after_s):
            self.nav_valid = False
            self.nav_lost_time = t
        if not self.nav_valid:
            # Hold the last good command and the last plan. The scheduler is
            # not consulted -- it has nothing to consult it with -- so the
            # hold runs to the release the last valid cycle planned, or to
            # impact if none was planned.
            plan = self.scheduler.plan
            tge = (plan or {}).get("t_go_end")
            if (tge is not None and self._holding
                    and self._last_t_go is not None):
                # Time to go is no longer measurable, so run the plan on the
                # clock: the last valid cycle's t_go, less the time since.
                if self._last_t_go - (t - self._last_t) <= tge:
                    self._holding = False
            self._record(t, None, None, "nav_lost", armed=True)
            return

        # -- predict ------------------------------------------------------
        # Once the scheduler has released there is nothing left to decide --
        # the hold is contiguous and cannot be resumed -- so the propagation
        # is not run. That is not an optimisation for the report's benefit: a
        # fuze that keeps propagating after its actuator is finished is
        # burning the power budget step 7 has to size.
        if self.scheduler.released:
            self._holding = False
            self._record(t, None, None,
                         "inhibited" if self.inhibited else "released",
                         armed=True)
            return
        y_est = y if self.nav is None else self.nav.state()
        rng, defl, tof, term = self.predictor.predict(t, y_est)
        t_go = max(tof - t, 0.0)
        pred = np.array([rng, defl])
        if cfg.pred_filter_tau is not None:
            # Task E's smoothing test. Exponential, on the guidance grid.
            a = math.exp(-cfg.period / max(float(cfg.pred_filter_tau), 1e-9))
            if self._pred_lp is None:
                self._pred_lp = pred.copy()
            else:
                self._pred_lp = a * self._pred_lp + (1.0 - a) * pred
            pred = self._pred_lp.copy()
            rng, defl = float(pred[0]), float(pred[1])
        miss = self.target - pred
        need = float(np.hypot(miss[0], miss[1]))
        self._last_t, self._last_t_go = t, t_go

        armed = elapsed >= cfg.arm_delay_s

        # -- inhibit ------------------------------------------------------
        if (cfg.inhibit_threshold_m is not None and not self.inhibited
                and armed and need > cfg.inhibit_threshold_m):
            self.inhibited = True
            self.inhibit_time = t
            self.scheduler.released = True
            self._holding = False
            self._record(t, pred, miss, "inhibited", armed=armed)
            return
        if self.inhibited:
            self._holding = False
            self._record(t, pred, miss, "inhibited", armed=armed)
            return

        # -- the authority monitor ---------------------------------------
        if (cfg.authority_monitor or cfg.monitor_shadow) and armed                 and self.authority_ok:
            self._check_authority(t, t_go, pred, miss)

        # -- schedule -----------------------------------------------------
        ctx = Context(t=t, t_go=t_go, miss=miss, amap=self.amap, armed=armed,
                      holding=self._holding, phi=self._phi,
                      cycles=self.samples, authority_ok=self.authority_ok)
        d = self.scheduler(ctx)
        self._phi = float(d.phi)
        self._holding = bool(d.hold)
        self._record(t, pred, miss, d.note, armed=armed, decision=d)

    # -- the monitor ------------------------------------------------------
    def _check_authority(self, t: float, t_go: float, pred: np.ndarray,
                         miss: np.ndarray) -> None:
        """
        Has holding moved the predicted impact point by as much as the map
        said it would?

        Opened when the law starts holding after arming and closed
        `monitor_window_s` later. The comparison is between the OBSERVED
        movement of the predicted impact point and the movement expected over
        the same interval.

        THE EXPECTATION IS NOT THE MAP'S PROMISE. It is the smaller of the
        map's promise and the miss the law was trying to remove, and getting
        that wrong was a measured false-alarm source rather than a
        hypothetical one: a first pass used the map's promise alone and
        faulted 4 healthy rounds in 64 at the adopted engagement -- every one
        of them a round whose miss was SMALLER than the promise, which
        converged, stopped moving because there was nothing left to remove,
        and was declared to have no actuator. A monitor that punishes a law
        for succeeding is worse than no monitor, because it releases the brake
        for the rest of the flight.

        The test is skipped entirely when that expectation is below
        `monitor_floor_m`, which is sized from the MEASURED impact-point
        prediction error: below it the thing being tested is smaller than the
        noise on the measurement of it.
        """
        cfg = self.config
        if not self._holding or t - self.deploy_time < cfg.monitor_start_s:
            self._mon = None
            return
        if self._mon is None:
            self._mon = (t, t_go, pred.copy(), self._phi, miss.copy())
            return
        t0, t_go0, pred0, phi0, miss0 = self._mon
        if t - t0 < cfg.monitor_window_s:
            return
        A, b = self.amap.increment(t_go0, t_go)
        w = np.array([math.cos(phi0), math.sin(phi0)])
        expect = A @ w + b
        promise = float(np.hypot(expect[0], expect[1]))
        wanted = float(np.hypot(miss0[0], miss0[1]))
        e = min(promise, wanted)
        along = float("nan")
        if promise > 0.0:
            moved = pred - pred0
            along = float(moved @ expect) / promise
        # THE WINDOW'S TEST STATISTIC, RECORDED WHATEVER THE THRESHOLD DOES
        # WITH IT. `along` and `e` are the two numbers the test compares, so
        # logging them lets any (monitor_fraction, monitor_floor_m) pair be
        # evaluated afterwards WITHOUT re-flying: the monitor would have
        # faulted at the first window with e >= floor and along < fraction*e.
        #
        # That is what makes the false-alarm and missed-detection rates of
        # docs/MONITOR-RETUNE.md a sweep over one campaign rather than one
        # campaign per threshold -- but it is only exact on a round that never
        # actually faulted, because a fault changes the flight and the later
        # windows then do not exist. `analysis.monte_carlo.task_d` therefore
        # flies the sweep with the monitor DISABLED and re-flies only the
        # operating points it selects.
        self.monitor_log.append({
            "t": float(t), "t_since_deploy": float(t - self.deploy_time),
            "t_go": float(t_go), "promise_m": promise, "wanted_m": wanted,
            "expected_m": e, "along_m": along,
            "ratio": (along / e) if e > 0.0 else float("nan"),
        })
        if promise > 0.0 and e >= cfg.monitor_floor_m:
            if along < cfg.monitor_fraction * e:
                if cfg.authority_monitor:
                    self.authority_ok = False
                    self.authority_fault_time = t
        self._mon = (t, t_go, pred.copy(), self._phi, miss.copy())

    # -- logging ----------------------------------------------------------
    def _record(self, t, pred, miss, note, armed=True, decision=None) -> None:
        L = self.log
        L["t"].append(float(t))
        L["t_go"].append(float(self._last_t_go)
                         if self._last_t_go is not None else float("nan"))
        L["pred_range"].append(float(pred[0]) if pred is not None else float("nan"))
        L["pred_defl"].append(float(pred[1]) if pred is not None else float("nan"))
        L["miss_range"].append(float(miss[0]) if miss is not None else float("nan"))
        L["miss_defl"].append(float(miss[1]) if miss is not None else float("nan"))
        L["miss_m"].append(float(np.hypot(miss[0], miss[1]))
                           if miss is not None else float("nan"))
        L["phi_deg"].append(math.degrees(self._phi))
        L["holding"].append(bool(self._holding))
        L["note"].append(note)
        L["armed"].append(bool(armed))
        L["mode"].append(self.mode)
        L["required_m"].append(float(decision.required_m) if decision else float("nan"))
        L["available_m"].append(float(decision.available_m) if decision else float("nan"))
        L["planned_t_go_end"].append(float(decision.planned_t_go_end)
                                     if decision else float("nan"))

    def command_activity(self, threshold_deg: float = 1.0) -> dict:
        """
        How much the commanded direction MOVED over the flight.

        The servo does not track a number, it slews to it, and
        docs/CONTROL-CHARACTERISATION.md prices every hold-to-hold transition
        at 0.4-0.75 s of reacquisition. So the count of times the command
        changed is a cost, not a diagnostic, and it is measured here rather
        than inferred from the impact point.

        Counted only over cycles where the law was HOLDING, because a released
        actuator cannot be asked to slew, and reported both over the whole
        guided phase and after arming, because the pre-arm cycles move the
        command freely at no cost -- there is no steering force yet.
        """
        thr = math.radians(threshold_deg)
        L = self.log
        phi = [math.radians(v) for v in L["phi_deg"]]
        out = {}
        for tag, mask in (("", [True] * len(phi)), ("_armed", L["armed"])):
            steps = []
            prev = None
            for i, p in enumerate(phi):
                if not (L["holding"][i] and mask[i]):
                    prev = None
                    continue
                if prev is not None:
                    d = abs(math.atan2(math.sin(p - prev), math.cos(p - prev)))
                    steps.append(d)
                prev = p
            s = np.array(steps) if steps else np.zeros(0)
            out[f"cmd_cycles{tag}"] = int(s.size)
            out[f"cmd_changes{tag}"] = int((s > thr).sum())
            out[f"cmd_total_variation_deg{tag}"] = float(np.degrees(s.sum()))
            out[f"cmd_max_step_deg{tag}"] = float(np.degrees(s.max())) if s.size else 0.0
            out[f"cmd_rms_step_deg{tag}"] = (float(np.degrees(math.sqrt(
                float((s ** 2).mean())))) if s.size else 0.0)
        return out

    def saturation(self) -> dict:
        """
        How much of the guided phase the law spent unable to deliver the miss.

        A cycle is SATURATED when holding the commanded angle all the way to
        impact still would not remove the predicted miss. The scheduler
        already decides that and records it as its own per-cycle note, so
        this counts notes rather than re-deriving the condition -- every law
        that can be saturated writes the same note, and a law that cannot be
        (`single_shot` after it has committed, which flies its plan whatever
        the miss does) correctly reports nothing.

        Counted over ARMED cycles only. Before arming there is no steering
        force to be short of, so a pre-arm cycle cannot be authority-limited
        however large the miss is.

        THIS EXISTS BECAUSE THE PREVIOUS MEASUREMENT WAS AN ARTEFACT.
        `analysis.guidance_cep.cep_of` read saturation off the scheduler's
        `plan` dictionary, which only `SingleShotScheduler` and
        `BudgetScheduler` write and which the latter overwrites every cycle,
        so every proportional, deadband and isotropic run reported 0.0
        whatever it did. See docs/MONTE-CARLO-DESIGN.md section 1.1.
        """
        note = self.log["note"]
        armed = self.log["armed"]
        n = tot = 0
        first = None
        for i, (v, a) in enumerate(zip(note, armed)):
            if not a:
                continue
            tot += 1
            if v == "saturated":
                n += 1
                if first is None:
                    first = float(self.log["t"][i])
        return {
            "armed_cycles": tot,
            "saturated_cycles": n,
            "saturated_fraction": (float(n) / tot) if tot else float("nan"),
            "ever_saturated": bool(n),
            # The last armed cycle is the one that decides where the round
            # lands: a round saturated at the end held to impact, which is the
            # regime docs/NAV-ERROR-DECOMPOSITION.md section 2 shows has a
            # completely different sensitivity to state error.
            "terminal_saturated": bool(
                tot and next(v for v, a in zip(reversed(note), reversed(armed))
                             if a) == "saturated"),
            "first_saturated_t": first,
        }

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "samples": self.samples,
            **self.command_activity(),
            **self.saturation(),
            "predictor_calls": self.predictor.calls,
            "predictor_derivative_calls": self.predictor.derivative_calls,
            "nav_lost_time": self.nav_lost_time,
            "inhibited": self.inhibited,
            "inhibit_time": self.inhibit_time,
            "authority_ok": self.authority_ok,
            "authority_fault_time": self.authority_fault_time,
            "nav_outages": [list(o) for o in self.nav_outages],
            "nav_invalid_cycles": sum(
                1 for v in self.log["note"] if v == "nav_lost"),
            **self.scheduler.summary(),
        }


# ===========================================================================
# The calibration commander
# ===========================================================================
class ScheduledHold:
    """
    Hold `angle` from deployment until `t_end`, then release. No guidance.

    This is the primitive the authority map is MEASURED with, and it is also
    exactly what a guidance law executes: one contiguous hold of variable
    length at one commanded direction. Keeping it in this module rather than
    in the analysis script means the thing measured and the thing flown are
    the same object.

    `predictor`, when given, is run on the same 1 Hz grid the guidance law
    would use and its output logged, so one calibration run yields both a
    point of the map and a sample of the impact-point-prediction error.
    """

    def __init__(self, angle: float, t_end: float, deploy_time: float,
                 predictor: Optional[ImpactPredictor] = None,
                 rate_hz: float = 1.0):
        self.angle = float(angle)
        self.t_end = float(t_end)
        self.deploy_time = float(deploy_time)
        self.predictor = predictor
        self.period = 1.0 / rate_hz
        self._next = -math.inf
        self.log = {k: [] for k in ("t", "pred_range", "pred_defl", "tof")}

    def command(self, t: float) -> tuple:
        return self.angle, (t < self.t_end)

    def sample(self, t: float, y: np.ndarray, model) -> None:
        if self.predictor is None or t < self.deploy_time:
            self._next = self.deploy_time
            return
        if t < self._next:
            return
        self._next = (self.deploy_time if self._next == -math.inf
                      else self._next) + self.period
        if self._next <= t:
            self._next = t + self.period
        rng, defl, tof, _ = self.predictor.predict(t, y)
        self.log["t"].append(float(t))
        self.log["pred_range"].append(float(rng))
        self.log["pred_defl"].append(float(defl))
        self.log["tof"].append(float(tof))

    def chain(self, *hooks) -> Callable:
        def hook(t, y, model):
            self.sample(t, y, model)
            for h in hooks:
                h(t, y, model)
        return hook
