"""
Staged canard deployment: does releasing the cant pair first buy anything?

Step 4.5. docs/STEP4-CLOSEOUT.md limitation 5 named a staged deployment as
"the obvious way to remove both the phase sensitivity and most of the
angle-of-attack excursion". This module tries it and prices it.

THE HYPOTHESIS
--------------
Three of step 4's worst results trace to one cause: the nose despins from
1308 rad/s with the STEERING canards already deployed, so the steering force
sweeps through several revolutions before the loop captures it, at a body roll
phase nobody controls. Deploy in two stages and the sweep never happens:

    stage 1   the CANT pair only. It carries the whole despin torque and
              produces no commandable transverse force, so there is nothing
              to sweep.
    stage 2   the STEERING pair, released once the nose has stopped turning.

WHY IT MIGHT NOT WORK, AND THE THING TO COMPUTE FIRST
-----------------------------------------------------
`T_cant` sums over the cant pair; `c_aero` sums over all four. Releasing two
panels therefore leaves the despin torque unchanged and HALVES the damping
that resists it, so the nose despins on a time constant twice as long to an
equilibrium rate twice as far from zero. Task B computes that before any
trajectory is flown, because if the halved damping left the nose at a rate the
brake cannot bracket, the staging would fail there.

WHAT IS MEASURED, AND AT WHAT BRAKE
------------------------------------
Everything here is at the 0.85 N m brake of docs/CONTROL-ROBUSTNESS.md
section 4.3, which supersedes the 0.75 N m every step-4 number was measured
at. The single-stage 0.85 N m baseline is reported as its own result (Task A)
so that the brake change and the staging change are not confounded.

SAMPLE SIZE
-----------
Step 4's phase ensemble was five samples and docs/STEP4-CLOSEOUT.md section 2.2
explicitly declines to claim a variance difference on that basis. This is a
comparison of two configurations on a random quantity, so five cannot resolve
it. `N_PHASES` = 24 deployment states, 15 degrees of body roll apart, spanning
one full body roll period in a 4.8 ms window over which dynamic pressure moves
by 0.06 %. Same flight condition, every phase the fuze might deploy at.

Run:  python -m analysis.staged_deployment              full, ~50 min on 8 cores
      python -m analysis.staged_deployment --tasks b    the algebra only, seconds
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from sim import canards as cn
from sim import integrate as ig, projectile as pr
from gnc import roll_control as rc
from analysis import roll_servo as rs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#: The adopted brake capacity. docs/CONTROL-ROBUSTNESS.md section 4.3 sizes it
#: against the UPPER bound of the +/-30 % canard aerodynamic estimate rather
#: than its nominal, and it supersedes the 0.75 N m of step 4.
BRAKE = rs.BRAKE_MAX_ADOPTED

#: Deployment phases per configuration. Task D. Twenty-four is chosen so the
#: body roll period divides evenly into 15-degree steps; the requirement is at
#: least twenty.
N_PHASES = 24

#: The eight commanded roll angles the reachable-set ellipse is fitted through.
ANGLES = tuple(float(a) for a in range(0, 360, 45))
#: Four cardinal angles are enough to determine the first-harmonic ellipse
#: (three parameters, four equations). The delay sweep uses them to buy phases.
ANGLES_CARDINAL = (0.0, 90.0, 180.0, 270.0)

#: Phases used for the sweeps that do not need the full ensemble.
N_PHASES_SWEEP = 6

#: Stage-2 delays for the Task C trade, s after stage 1. 0.0 is single-stage
#: to within one controller sample and is the sweep's own anchor.
DELAYS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)

#: Backstop for the triggered release, s after stage 1. Sized in Task B.3:
#: the nominal plant releases at 2.4 s and the weakest plant inside the
#: +/-30 % canard aerodynamic band at 3.3 s, so 4.0 s is never reached by a
#: plant the servo can control, and is 9.3 % of the guided phase for one that
#: cannot.
MAX_DELAY = 4.0

STAGED_CONFIG = dict(enabled=True, max_delay=MAX_DELAY)


# ===========================================================================
# The deployment-phase ensemble
# ===========================================================================
def phase_ensemble(base: dict, n: int = N_PHASES) -> dict:
    """
    `n` deployment states one body roll period apart divided into `n`, all at
    the same flight condition.

    Step 4 sampled the phase by restarting from successive LOGGED samples of
    the uncorrected leg, 0.05 s apart, which is ten and a half revolutions of
    the body: unrelated phases, but only as many of them as the log cadence
    allowed, and spread over 0.2 s of changing dynamic pressure.

    This instead re-integrates the uncorrected leg from the deployment state
    at a step of exactly one `n`-th of the body roll period and keeps every
    step. The states are then 360/n degrees of body roll apart by construction,
    they cover the circle exactly once, and the whole ensemble lives inside one
    roll period -- 4.8 ms, over which the dynamic pressure moves by 0.06 %.
    Phase is varied; nothing else is.
    """
    t_dep = base["deploy_time"]
    y_dep = np.asarray(base["y_deploy"])
    spin = base["spin_at_deploy"]
    period = 2.0 * math.pi / spin
    dt = period / n

    res = ig.integrate(y_dep, rs.base_model(), dt=dt, log_every=1,
                       t_max=t_dep + period * (1.0 - 0.5 / n),
                       t_start=t_dep, stop_on_impact=False)
    tr = res.trajectory
    ys = rs._states_from(tr)[:n]
    ts = [float(x) for x in tr.t[:n]]
    phis = [math.degrees(cn.CanardModel.body_roll_angle(tr.quaternion[i]))
            for i in range(n)]
    return {
        "n": n,
        "roll_period_s": period,
        "phase_dt_s": dt,
        "phase_t": ts,
        "phase_y": [row.tolist() for row in ys],
        "phase_body_roll_deg": phis,
        "qbar_spread_relative": float(
            (tr.dynamic_pressure[:n].max() - tr.dynamic_pressure[:n].min())
            / tr.dynamic_pressure[:n].mean()),
    }


# ===========================================================================
# One 6-DOF trajectory
# ===========================================================================
def run_case(args) -> dict:
    """
    One guided trajectory, single-stage or staged, restarted at one of the
    ensemble's deployment phases.

    A dict rather than a callable so the case can be shipped to a worker.
    """
    base = args["baseline"]
    ens = args["ensemble"]
    k = int(args.get("phase", 0))
    t_dep = ens["phase_t"][k]
    y_dep = np.asarray(ens["phase_y"][k])
    dt = args.get("dt", base["dt"])
    brake_max = float(args.get("brake_max", BRAKE))
    spec = args["command"]

    cfg = rc.ControllerConfig(**args.get("config", {}))
    model = rs.make_plant(brake_max=brake_max, **args.get("nose_overrides", {}))
    truth_scale = float(args.get("truth_aero_scale", 1.0))

    controller = rs._controller(model, cfg)
    stage_cfg = rc.StagedDeploymentConfig(**args.get("staged", {}))
    deployment = rc.StagedDeployment(stage_cfg, t_dep, controller=controller,
                                     plant=model)
    command = rs._build_command(spec, t_dep)
    law = rc.BrakeLaw(controller, command, deploy_time=t_dep,
                      deployment=deployment if stage_cfg.enabled else None)

    nose = replace(model.nose, deploy_time=t_dep,
                   brake_command=law.brake_command,
                   steering_gate=deployment.gate if stage_cfg.enabled else None)
    guided = cn.guided_model(rs.base_model(), rs.GEOMETRY, nose)
    if truth_scale != 1.0:
        # The TRUTH is weaker than what the controller was told, which is how
        # a canard aerodynamic error actually presents: one lift-curve slope
        # driving the transverse force, the cant torque and the roll damping
        # together. The controller is not told.
        guided = replace(guided,
                         control=ScaledCanards(guided.control, truth_scale))

    res = ig.integrate(y_dep, guided, dt=dt, log_every=rs.LOG_EVERY, t_max=200.0,
                       t_start=t_dep, step_hook=law.sample)
    if law.samples == 0:
        raise RuntimeError("the controller never sampled: step_hook not wired")

    hist = law.history()
    err = np.abs(hist["error"])
    acq = rs._acquisition_time(hist["t"], err, hist["holding"], t_dep)
    release = deployment.delay if stage_cfg.enabled else 0.0
    # The dead time that costs metres is not when the loop holds an angle but
    # when a correctly directed steering force first exists. For a single-stage
    # kit those are the same instant; for a staged one they are not.
    if acq is None:
        steering_ready = None
    else:
        steering_ready = max(acq, release if release is not None else 0.0)

    settled = hist["holding"] & (hist["t"] > t_dep + (acq or 0.0) + 0.5)
    out = {
        "label": spec.get("label", ""),
        "angle_deg": spec.get("angle_deg"),
        "phase": k,
        "phase_body_roll_deg": ens["phase_body_roll_deg"][k],
        "deploy_time": t_dep,
        "brake_max_Nm": brake_max,
        "staged": stage_cfg.enabled,
        "stage2_delay_s": release,
        "stage2_reason": deployment.release_reason,
        "engage_s": (controller.state.engage_time - t_dep)
        if controller.state.engage_time is not None else None,
        "range_m": res.range_m,
        "drift_m": res.drift_m,
        "tof_s": res.impact_time,
        "d_range_m": res.range_m - base["uncorrected_range"],
        "d_deflection_m": res.drift_m - base["uncorrected_drift"],
        "max_total_aoa_deg": math.degrees(res.max_total_aoa),
        "acquisition_s": acq,
        "steering_ready_s": steering_ready,
        "mean_abs_error_deg_settled": float(np.degrees(err[settled].mean()))
        if settled.any() else None,
        "rms_error_deg_settled": float(
            np.degrees(np.sqrt((err[settled] ** 2).mean()))) if settled.any() else None,
        "samples": law.samples,
    }
    if args.get("keep_history"):
        out["history"] = {kk: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for kk, v in hist.items()}
        tr = res.trajectory
        out["traj_t"] = tr.t.tolist()
        out["traj_nose_spin"] = tr.nose_spin.tolist()
        out["traj_total_aoa_deg"] = np.degrees(tr.total_aoa).tolist()
    return out


class ScaledCanards:
    """
    A canard model whose panel loads are multiplied by `scale`.

    Module level, and holding only picklable state, because the cases run in
    worker processes. Used only for the deadlock demonstration, where the
    truth must be weaker than the controller's belief. Scaling the three
    returned loads together is exactly a lift-curve-slope error to first
    order; the panel induced drag is quadratic in it and is therefore scaled
    linearly rather than exactly, which moves the axial force by parts in ten
    thousand and nothing else.
    """

    def __init__(self, inner, scale: float):
        self.inner = inner
        self.scale = float(scale)

    def __call__(self, t, y, st):
        f, m, roll = self.inner(t, y, st)
        return f * self.scale, m * self.scale, roll * self.scale


def run_ideal(args) -> dict:
    """
    The step-2.5 idealisation at one deployment phase: `phi_nose` pinned
    kinematically from deployment, infinite bandwidth, unlimited torque, zero
    error, all four panels out. The denominator.
    """
    base = args["baseline"]
    ens = args["ensemble"]
    k = int(args.get("phase", 0))
    t_dep = ens["phase_t"][k]
    y_dep = np.asarray(ens["phase_y"][k])
    angle = math.radians(args["angle_deg"])
    nose = replace(rs.NOSE, deploy_time=t_dep, hold_angle=angle)
    guided = cn.guided_model(rs.base_model(), rs.GEOMETRY, nose)
    res = ig.integrate(y_dep, guided, dt=base["dt"], log_every=rs.LOG_EVERY,
                       t_max=200.0, t_start=t_dep)
    return {
        "label": f"ideal_p{k}_{args['angle_deg']:.0f}",
        "angle_deg": args["angle_deg"],
        "phase": k,
        "phase_body_roll_deg": ens["phase_body_roll_deg"][k],
        "d_range_m": res.range_m - base["uncorrected_range"],
        "d_deflection_m": res.drift_m - base["uncorrected_drift"],
        "max_total_aoa_deg": math.degrees(res.max_total_aoa),
    }


# ===========================================================================
# TASK B -- the algebra, before any trajectory
# ===========================================================================
def task_b(base: dict) -> dict:
    """
    Despin torque, roll damping, equilibrium rates and time constants with two
    panels against four, and the three questions that decide whether staging
    is even possible.
    """
    sched = rs.schedule_from(base)
    t_dep = base["deploy_time"]
    offsets = (0.0, 0.5, 1.0, 3.0, 8.0, 17.0, 27.0, 37.0, 42.4)

    def rows(panels: int) -> list:
        plant = replace(rs.make_plant(brake_max=BRAKE), deployed_panels=panels)
        out = []
        for o in offsets:
            c = sched.at(t_dep + o)
            env = plant.envelope(c)
            out.append({
                "t_after_deploy_s": o,
                "qbar_kPa": c.qbar / 1e3,
                "mach": c.mach,
                "T_cant_Nm": plant.cant_torque(c),
                "c_aero_Nms": plant.aero_damping(c),
                "c_tot_Nms": plant.total_damping(c),
                "u_hold_Nm": plant.hold_command(c),
                "saturation_margin": BRAKE / plant.hold_command(c),
                "p_free_rads": plant.equilibrium_rate(c, 0.0),
                "p_full_rads": plant.equilibrium_rate(c, BRAKE),
                "time_constant_s": plant.time_constant(c),
                "can_hold": bool(env.can_hold),
            })
        return out

    four, two = rows(4), rows(2)

    # Question 2: does the nose still despin usefully on the cant pair alone?
    # It despins if the free-brake equilibrium is negative, and it is holdable
    # if the achievable interval brackets zero. Both are scanned exactly.
    despins = all(r["p_free_rads"] < 0.0 for r in two)
    holdable = all(r["can_hold"] for r in two)

    # Time from 1308 rad/s to within 5 rad/s of the equilibrium, first order.
    c0 = sched.at(t_dep)
    p0 = base["spin_at_deploy"]
    settle = {}
    for panels, tag in ((4, "four_panel"), (2, "two_panel")):
        plant = replace(rs.make_plant(brake_max=BRAKE), deployed_panels=panels)
        tau = plant.time_constant(c0)
        for u, name in ((0.0, "free"), (BRAKE, "full_brake")):
            pinf = plant.equilibrium_rate(c0, u)
            settle[f"{tag}_{name}"] = {
                "tau_s": tau,
                "p_inf_rads": pinf,
                "t_to_within_5_rads": tau * math.log(abs(p0 - pinf) / 5.0),
            }

    # Question 3: does the section 4.3 brake sizing rule still hold in stage 1?
    # u_hold = T_cant - c_v p_body - T_coulomb contains NO damping term, so it
    # is identical for two panels and four. Asserted numerically rather than
    # argued.
    hold_identical = max(abs(a["u_hold_Nm"] - b_["u_hold_Nm"])
                         for a, b_ in zip(four, two))
    cant_identical = max(abs(a["T_cant_Nm"] - b_["T_cant_Nm"])
                         for a, b_ in zip(four, two))
    damping_ratio = [b_["c_aero_Nms"] / a["c_aero_Nms"] for a, b_ in zip(four, two)]

    return {
        "brake_max_Nm": BRAKE,
        "four_panel": four,
        "two_panel": two,
        "despin_settling": settle,
        "stage1_still_despins": despins,
        "stage1_still_holdable": holdable,
        "max_u_hold_difference_Nm": hold_identical,
        "max_T_cant_difference_Nm": cant_identical,
        "damping_ratio_two_over_four": damping_ratio,
        "sizing_rule_unchanged": hold_identical < 1e-12,
    }


# ===========================================================================
# Reduction
# ===========================================================================
def _ellipse_for(rows: list) -> dict:
    pts = [(r["angle_deg"], r["d_range_m"], r["d_deflection_m"]) for r in rows]
    return rs._ellipse(pts)


def _harmonic_residual(rows: list) -> float:
    """
    How well the FIRST harmonic describes these impact shifts, as an rms
    residual divided by the rms amplitude of the fitted ellipse.

    The reachable set is quoted as a first-harmonic ellipse because step 2.5
    measured the response to be linear in the commanded angle to better than a
    per cent. Stage 1 puts a term modulated at TWICE the nose angle into the
    flow (section 2.3), so that assumption has to be re-checked for the staged
    configuration rather than inherited. A residual comparable to the
    amplitude would mean the ellipse is the wrong description and the
    comparison in this document is being made on a bad summary.
    """
    ang = np.radians([r["angle_deg"] for r in rows])
    xy = np.array([[r["d_range_m"], r["d_deflection_m"]] for r in rows])
    basis = np.column_stack([np.cos(ang), np.sin(ang), np.ones_like(ang)])
    coef, *_ = np.linalg.lstsq(basis, xy, rcond=None)
    resid = xy - basis @ coef
    amp = math.sqrt(0.5 * float((coef[:2, :] ** 2).sum()))
    rms = math.sqrt(float((resid ** 2).sum() / resid.shape[0]))
    return rms / amp if amp > 0 else math.inf


def _group(rows: list, key="phase") -> dict:
    out = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r["angle_deg"])
    return out


def _stats(values: list) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "sem": float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "median": float(np.median(v)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "values": [float(x) for x in v],
    }


def reduce_ensemble(rows: list, ideal: dict) -> dict:
    """
    Per-phase reachable set for one configuration, plus the distribution over
    phases. `ideal` is the denominator ellipse.
    """
    per_phase = []
    for k, group in sorted(_group(rows).items()):
        e = _ellipse_for(group)
        per_phase.append({
            "phase": k,
            "body_roll_deg": group[0]["phase_body_roll_deg"],
            "semi_major_m": e["semi_major_m"],
            "semi_minor_m": e["semi_minor_m"],
            "rms_amplitude_m": e["rms_amplitude_m"],
            "axis_ratio": e["axis_ratio"],
            "harmonic_residual": _harmonic_residual(group) if len(group) > 3
            else None,
            "retained_semi_major": e["semi_major_m"] / ideal["semi_major_m"],
            "retained_rms": e["rms_amplitude_m"] / ideal["rms_amplitude_m"],
            "max_aoa_deg": max(r["max_total_aoa_deg"] for r in group),
            "max_acquisition_s": max((r["acquisition_s"] or math.nan)
                                     for r in group),
            "max_steering_ready_s": max((r["steering_ready_s"] or math.nan)
                                        for r in group),
            # The release condition depends on the commanded angle through
            # `capture`, so the eight runs at one phase need not release at the
            # same instant. The mean is shown here; the full distribution over
            # every run is `stage2_delay_per_run_s`.
            "stage2_delay_s": float(np.mean([r["stage2_delay_s"] for r in group
                                             if r["stage2_delay_s"] is not None]))
            if any(r["stage2_delay_s"] is not None for r in group) else None,
            "stage2_reason": "/".join(sorted({str(r["stage2_reason"])
                                              for r in group})),
            "n_angles": len(group),
        })
    return {
        "per_phase": per_phase,
        "retained_rms": _stats([p["retained_rms"] for p in per_phase]),
        "retained_semi_major": _stats([p["retained_semi_major"] for p in per_phase]),
        "rms_amplitude_m": _stats([p["rms_amplitude_m"] for p in per_phase]),
        "semi_major_m": _stats([p["semi_major_m"] for p in per_phase]),
        "axis_ratio": _stats([p["axis_ratio"] for p in per_phase]),
        "harmonic_residual": _stats([p["harmonic_residual"] for p in per_phase]),
        "max_aoa_deg": _stats([p["max_aoa_deg"] for p in per_phase]),
        # Per-RUN, not per-phase: angle of attack and acquisition are properties
        # of a trajectory, and there are eight per phase.
        "aoa_per_run_deg": _stats([r["max_total_aoa_deg"] for r in rows]),
        "acquisition_per_run_s": _stats([r["acquisition_s"] for r in rows]),
        "steering_ready_per_run_s": _stats([r["steering_ready_s"] for r in rows]),
        "engage_per_run_s": _stats([r["engage_s"] for r in rows]),
        "stage2_delay_per_run_s": _stats([r["stage2_delay_s"] for r in rows]),
        "tracking_rms_deg": _stats([r["rms_error_deg_settled"] for r in rows]),
        "n_runs": len(rows),
    }


def welch(a: dict, b: dict) -> dict:
    """
    Welch's t on two independent samples, and the difference against its own
    sampling noise. Unequal variances are the point: one hypothesis of this
    session is that staging narrows the spread.

    No p-value is quoted from a table. What is reported is the difference, the
    standard error of the difference, and their ratio, which is what "larger
    than the sampling noise" means. |t| below about 2 is not resolved by these
    samples.
    """
    if a.get("n", 0) < 2 or b.get("n", 0) < 2:
        return {"resolved": False, "reason": "fewer than two samples"}
    diff = b["mean"] - a["mean"]
    se = math.sqrt(a["sem"] ** 2 + b["sem"] ** 2)
    t = diff / se if se > 0 else math.inf
    # Welch-Satterthwaite degrees of freedom, reported for honesty about the
    # approximation rather than used to look anything up.
    va, vb = a["sem"] ** 2, b["sem"] ** 2
    dof = ((va + vb) ** 2 / (va ** 2 / (a["n"] - 1) + vb ** 2 / (b["n"] - 1))
           if (va + vb) > 0 else 0.0)
    return {
        "difference": diff,
        "se_difference": se,
        "t": t,
        "dof": dof,
        "ci95_low": diff - 2.0 * se,
        "ci95_high": diff + 2.0 * se,
        "resolved": abs(t) >= 2.0,
        "variance_ratio": (b["sd"] ** 2 / a["sd"] ** 2) if a["sd"] > 0 else math.inf,
    }


# ===========================================================================
# Case builders
# ===========================================================================
def ensemble_cases(base, ens, staged: bool, phases, angles=ANGLES,
                   staged_overrides=None, tag="") -> list:
    cases = []
    scfg = dict(STAGED_CONFIG) if staged else {"enabled": False}
    if staged_overrides:
        scfg.update(staged_overrides)
    for k in phases:
        for a in angles:
            cases.append({
                "baseline": base, "ensemble": ens, "phase": k,
                "brake_max": BRAKE, "staged": scfg,
                "command": {"kind": "hold", "angle_deg": a,
                            "label": f"{tag}p{k}_{a:.0f}"},
            })
    return cases


def ideal_cases(base, ens, phases, angles=ANGLES) -> list:
    return [{"baseline": base, "ensemble": ens, "phase": k, "angle_deg": a}
            for k in phases for a in angles]


def delay_sweep_cases(base, ens, phases, angles=ANGLES_CARDINAL) -> list:
    cases = []
    for d in DELAYS:
        for k in phases:
            for a in angles:
                cases.append({
                    "baseline": base, "ensemble": ens, "phase": k,
                    "brake_max": BRAKE,
                    "staged": {"enabled": True, "fixed_delay": d,
                               "max_delay": max(MAX_DELAY, d + 1.0)},
                    "command": {"kind": "hold", "angle_deg": a,
                                "label": f"d{d:g}_p{k}_{a:.0f}"},
                })
    return cases


def deadlock_cases(base, ens) -> list:
    """
    The 6-DOF demonstration that the stage-2 backstop earns its keep.

    The controller is told the nominal canard aerodynamics and flown against a
    truth 15 % weaker -- the perturbation that deadlocked the engagement gate
    in docs/CONTROL-ROBUSTNESS.md section 6 -- with the engagement gate's own
    model-free criterion disabled, so the loop never engages at all. A round
    in that state has no servo. The question is whether it at least ends up
    with a steering surface.
    """
    out = []
    for backstop in (True, False):
        scfg = {"enabled": True, "max_delay": MAX_DELAY if backstop else 1e9}
        out.append({
            "baseline": base, "ensemble": ens, "phase": 0,
            "brake_max": BRAKE, "staged": scfg,
            "truth_aero_scale": 0.85,
            "config": {"engage_settled_accel": 0.0},
            "command": {"kind": "hold", "angle_deg": 180.0,
                        "label": f"deadlock_{'backstop' if backstop else 'none'}"},
        })
    return out


# ===========================================================================
# Driver
# ===========================================================================
def _imap(pool, fn, cases, label):
    t0 = time.time()
    out = []
    it = pool.imap_unordered(fn, cases) if pool else map(fn, cases)
    for i, r in enumerate(it, 1):
        out.append(r)
        if i % 25 == 0 or i == len(cases):
            el = time.time() - t0
            print(f"  {label}: {i}/{len(cases)}  {el:6.1f}s  "
                  f"eta {el / i * (len(cases) - i):6.1f}s", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="bcde",
                    help="subset of b c d e to run")
    ap.add_argument("--processes", type=int, default=os.cpu_count())
    ap.add_argument("--phases", type=int, default=N_PHASES)
    ap.add_argument("--out", default="docs/staged_deployment.json")
    ap.add_argument("--cache", default=None,
                    help="directory to cache the baseline in")
    args = ap.parse_args(argv)
    tasks = set(args.tasks.lower())

    t_start = time.time()
    cache = args.cache and os.path.join(args.cache, "baseline.json")
    if cache and os.path.exists(cache):
        print(f"baseline from cache {cache}", flush=True)
        base = json.load(open(cache))
    else:
        print("flying the baseline ...", flush=True)
        base = rs.baseline()
        if cache:
            os.makedirs(args.cache, exist_ok=True)
            json.dump(base, open(cache, "w"))

    ens = phase_ensemble(base, args.phases)
    print(f"phase ensemble: {ens['n']} states, "
          f"{360.0 / ens['n']:.1f} deg apart, qbar spread "
          f"{ens['qbar_spread_relative'] * 100:.3f} %", flush=True)

    out = {
        "configuration": {
            "brake_max_Nm": BRAKE,
            "brake_max_step4_Nm": rs.BRAKE_MAX_SIZED,
            "n_phases": ens["n"],
            "angles_deg": list(ANGLES),
            "delays_s": list(DELAYS),
            "max_delay_s": MAX_DELAY,
            "guided_phase_s": base["uncorrected_tof"] - base["deploy_time"],
        },
        "baseline": {k: v for k, v in base.items()
                     if not k.startswith(("schedule_", "y_", "free_nose"))},
        "ensemble": {k: v for k, v in ens.items() if k != "phase_y"},
    }

    if "b" in tasks:
        print("task B: the algebra", flush=True)
        out["task_b"] = task_b(base)
        b = out["task_b"]
        print(f"  stage-1 despins: {b['stage1_still_despins']}, "
              f"holdable: {b['stage1_still_holdable']}, "
              f"sizing rule unchanged: {b['sizing_rule_unchanged']}", flush=True)
        if not (b["stage1_still_despins"] and b["stage1_still_holdable"]):
            print("  STAGING FAILS AT TASK B -- stopping", flush=True)
            _write(args.out, out)
            return 1

    pool = Pool(args.processes) if args.processes and args.processes > 1 else None
    try:
        phases_all = list(range(ens["n"]))
        phases_sweep = list(range(0, ens["n"],
                                 max(1, ens["n"] // N_PHASES_SWEEP)))[:N_PHASES_SWEEP]

        if {"c", "d"} & tasks:
            print(f"ideal hold: {len(phases_sweep)} phases x {len(ANGLES)} angles",
                  flush=True)
            ideal_rows = _imap(pool, run_ideal,
                               ideal_cases(base, ens, phases_sweep), "ideal")
            out["ideal"] = _reduce_ideal(ideal_rows)
            ideal = out["ideal"]["pooled_ellipse"]

        if "d" in tasks:
            for tag, staged in (("single", False), ("staged", True)):
                cases = ensemble_cases(base, ens, staged, phases_all, tag=tag[0])
                print(f"task D: {tag}, {len(cases)} runs", flush=True)
                rows = _imap(pool, run_case, cases, tag)
                out[f"ensemble_{tag}"] = reduce_ensemble(rows, ideal)
                out[f"ensemble_{tag}"]["rows"] = rows
            a = out["ensemble_single"]
            b = out["ensemble_staged"]
            out["comparison"] = {
                key: welch(a[key], b[key])
                for key in ("retained_rms", "retained_semi_major",
                            "max_aoa_deg", "aoa_per_run_deg",
                            "acquisition_per_run_s", "steering_ready_per_run_s",
                            "axis_ratio")
            }

        if "c" in tasks:
            cases = delay_sweep_cases(base, ens, phases_sweep)
            print(f"task C: delay sweep, {len(cases)} runs", flush=True)
            rows = _imap(pool, run_case, cases, "delay")
            out["delay_sweep"] = _reduce_delays(rows, ideal, base)

        if "e" in tasks:
            cases = deadlock_cases(base, ens)
            print(f"deadlock demonstration, {len(cases)} runs", flush=True)
            out["deadlock"] = _imap(pool, run_case, cases, "deadlock")
    finally:
        if pool:
            pool.close()
            pool.join()

    out["runtime_s"] = time.time() - t_start
    _write(args.out, out)
    print(f"wrote {args.out} in {out['runtime_s']:.0f} s", flush=True)
    return 0


def _reduce_ideal(rows: list) -> dict:
    """
    The ideal kinematic hold, per phase and pooled.

    The ideal has no acquisition transient, so it should be phase-insensitive
    and the pooled ellipse should be a fair denominator. That is checked here
    rather than assumed.
    """
    per_phase = []
    for k, group in sorted(_group(rows).items()):
        e = _ellipse_for(group)
        per_phase.append({"phase": k, "body_roll_deg": group[0]["phase_body_roll_deg"],
                          **e,
                          "max_aoa_deg": max(r["max_total_aoa_deg"] for r in group)})
    # Pooled: fit one ellipse through every (angle, shift) pair from every
    # phase, which is the least-squares first harmonic of the whole set.
    pooled = rs._ellipse([(r["angle_deg"], r["d_range_m"], r["d_deflection_m"])
                          for r in rows])
    return {
        "per_phase": per_phase,
        "pooled_ellipse": pooled,
        "semi_major_m": _stats([p["semi_major_m"] for p in per_phase]),
        "rms_amplitude_m": _stats([p["rms_amplitude_m"] for p in per_phase]),
        "max_aoa_deg": _stats([p["max_aoa_deg"] for p in per_phase]),
        "aoa_per_run_deg": _stats([r["max_total_aoa_deg"] for r in rows]),
        "rows": rows,
    }


def _reduce_delays(rows: list, ideal: dict, base: dict) -> dict:
    """Reachable set, angle of attack and dead time against the stage-2 delay."""
    guided = base["uncorrected_tof"] - base["deploy_time"]
    by_delay = {}
    for r in rows:
        by_delay.setdefault(round(r["stage2_delay_s"] or 0.0, 3), []).append(r)
    out = []
    for d in sorted(by_delay):
        group = by_delay[d]
        red = reduce_ensemble(group, ideal)
        out.append({
            "delay_s": d,
            "fraction_of_guided_phase": d / guided,
            "n_phases": len(red["per_phase"]),
            "retained_rms": red["retained_rms"],
            "retained_semi_major": red["retained_semi_major"],
            "axis_ratio": red["axis_ratio"],
            "max_aoa_deg": red["max_aoa_deg"],
            "aoa_per_run_deg": red["aoa_per_run_deg"],
            "acquisition_per_run_s": red["acquisition_per_run_s"],
            "steering_ready_per_run_s": red["steering_ready_per_run_s"],
        })
    return {"guided_phase_s": guided, "by_delay": out, "rows": rows}


def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)


if __name__ == "__main__":
    raise SystemExit(main())
