"""
Task E: does the servo still meet its numbers when the plant is not what the
controller was told it was?

Produces docs/CONTROL-ROBUSTNESS.md and docs/roll_robustness.json.

WHAT IS VARIED, AND ON WHAT BASIS
---------------------------------
Two quantities, both of which the project already grades as estimates:

BEARING FRICTION. docs/CANARD-MODEL.md section 8 gives the viscous
coefficient +-60 % and the Coulomb term +-70 %, and section 5 records that
the linearisation is wrong in FORM as well as in value -- real churning
torque grows as n^(2/3), so a coefficient fitted at 1100 rad/s over-predicts
above that rate and under-predicts below. There is a second, one-sided
argument for the low end: the bearing dissipates 80-130 W for the whole
guided phase, and grease that hot thins, so the true drag late in flight is
more likely below the estimate than above it.

The task asks for at least +-50 %. This sweeps the viscous term over
0.25x to 2.5x and the Coulomb term over 0.3x to 3x, which covers the
register's own bands with room to spare, and reports where the loop stops
meeting Task C rather than where it was asked to stop.

CANARD AERODYNAMICS. The despin torque and the aerodynamic roll damping are
both proportional to the same qbar * S_panel * C_Lalpha, so a C_Lalpha
uncertainty scales them TOGETHER; scaling only the despin torque would
flatter the loop by leaving its damping intact. The panel lift-curve slope is
graded +-20 % supersonic and +-40 % transonic in
docs/CANARD-MODEL.md section 8, and the session brief carries +-30 %. This
sweeps 0.6x to 1.5x.

THE CONTROLLER IS NOT TOLD
--------------------------
Every case is run with the TRUTH plant perturbed and the controller still
carrying the nominal estimate, because that is the situation a fielded round
is in: the onboard feed-forward has the design value, not the value the
bearing actually has that day. The `informed` variant, where the controller
is given the true value, is reported alongside so that the two failure modes
-- the hardware cannot do it, and the controller does not know -- can be told
apart.

WHICH MODEL
-----------
The reduced model, which reproduces the 6-DOF nose rate to better than
5 rad/s and its roll angle to a fraction of a degree
(docs/CONTROL-CHARACTERISATION.md section 5). The one Task C number it cannot
produce is the steady-state tracking error, which is dominated by a kinematic
term at the spin frequency that has nothing to do with bearing friction or
canard aerodynamics; that term is carried across from the 6-DOF as a constant
and stated as such.

Run:  python -m analysis.roll_robustness        ~3 min, single core
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace

import numpy as np

from sim import canards as cn, projectile as pr
from gnc import roll_control as rc
from analysis.roll_servo import (BRAKE_MAX_SIZED, GEOMETRY, NOSE,
                                 baseline, make_plant, schedule_from,
                                 task_c, task_c_acquisition, _controller,
                                 _settle, _describe, _minus3db,
                                 BANDWIDTH_HZ, BANDWIDTH_CYCLES)

#: Bearing viscous coefficient, as a multiple of the 6.5e-5 N m s/rad of
#: record. The register grades it +-60 %; this goes wider both ways.
VISCOUS_SCALES = (0.25, 0.4, 0.5, 0.75, 1.0, 1.5, 1.6, 2.0, 2.5)

#: Bearing Coulomb term, as a multiple of 2.0e-3 N m. Graded +-70 %.
COULOMB_SCALES = (0.3, 0.5, 1.0, 1.7, 2.0, 3.0)

#: Panel lift-curve slope, scaling the cant torque and the roll damping
#: together. Graded +-20 to 40 %; the brief carries +-30 %.
AERO_SCALES = (0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)

#: Where the step metrics are measured, seconds after deployment.
#:
#: The earliest is 2.0 s, not 0.5, so that every case gets a full two seconds
#: of settling before the step. A perturbed plant starts out of equilibrium --
#: the controller pre-charges the brake coil to the torque IT thinks holds the
#: nose, which is not the torque that does -- and stepping into the middle of
#: that transient measures the transient, not the step. The transient itself
#: is not thrown away: it is what `acquisition` measures, from the far harsher
#: initial condition of a nose still locked to the body.
PROBE_OFFSETS = (2.0, 5.0, 17.0, 37.0)

#: Task C's answers at the sized brake, to be met or missed. Filled from
#: docs/roll_servo.json when it is present, so that the pass/fail line is the
#: measured nominal rather than a number typed in here.
FALLBACK_TARGETS = {
    "rise_forward_s": 0.50,
    "rise_return_s": 0.55,
    "settle_s": 1.25,
    "overshoot_pct": 1.0,
    "acquisition_s": 1.5,
    "bandwidth_hz": 1.4,
}


def truth_plant(brake_max: float, viscous_scale: float = 1.0,
                coulomb_scale: float = 1.0, aero_scale: float = 1.0):
    nose = replace(NOSE, brake_max=brake_max,
                   viscous=NOSE.viscous * viscous_scale,
                   coulomb=NOSE.coulomb * coulomb_scale)
    return rc.plant_from(GEOMETRY, nose, pr.M107, aero_scale=aero_scale)


# ===========================================================================
# One perturbed case
# ===========================================================================
def probe(base: dict, truth: rc.NoseRollPlant, model: rc.NoseRollPlant,
          cfg: rc.ControllerConfig = None,
          offsets=PROBE_OFFSETS) -> dict:
    """
    Measure the Task C metrics for one (truth, controller-model) pair.

    `model` is what the controller believes; `truth` is what it flies against.
    Passing the same object for both is the informed case.
    """
    cfg = cfg or rc.ControllerConfig()
    sched = schedule_from(base)
    t_dep = base["deploy_time"]
    t_imp = base["free_tof"]
    rows = []

    for off in offsets:
        t_step = t_dep + off
        t0 = max(t_dep, t_step - 2.0)
        t1 = min(t_imp, t_step + 4.0)
        cond = sched.at(t_step)
        env = truth.envelope(cond)
        # Pre-charge to what the CONTROLLER believes, not to the truth: a
        # fielded round energises the coil from its own estimate.
        u_hold0 = min(model.hold_command(sched.at(t0)), model.nose.brake_max)
        row = {
            "since_deploy_s": off,
            "qbar_kPa": 1e-3 * cond.qbar,
            "can_hold": env.can_hold,
            "saturation_margin": env.saturation_margin,
            "rate_free_degs": env.rate_free_degs,
            "rate_full_degs": env.rate_full_degs,
            "accel_forward_rads2": env.accel_forward,
            "accel_return_rads2": env.accel_return,
        }
        for size_deg, key in ((90.0, "forward"), (-90.0, "return")):
            ctl = _controller(model, cfg)
            ctl.actuator.reset(max(u_hold0, 0.0))
            run = rc.simulate_reduced(model, ctl, sched,
                                      rc.step_command(0.0, math.radians(size_deg),
                                                      t_step),
                                      t0, t1, phi0=0.0, p_nose0=0.0, truth=truth)
            met = _settle(run, t_step, math.radians(size_deg))
            row[f"rise_{key}_s"] = met["rise_s"]
            row[f"settle_{key}_s"] = met["settle_s"]
            row[f"settled_{key}"] = met["settled"]
            row[f"overshoot_{key}_pct"] = met["overshoot_pct"]

        # Steady-state hold.
        ctl = _controller(model, cfg)
        ctl.actuator.reset(max(u_hold0, 0.0))
        t1h = min(t_imp, t_step + 6.0)
        run = rc.simulate_reduced(model, ctl, sched, rc.constant_command(0.0),
                                  t0, t1h, phi0=0.0, p_nose0=0.0, truth=truth)
        tail = run.t >= t1h - 1.0
        err = np.abs(run.error[tail])
        row["hold_mean_error_deg"] = float(np.degrees(err.mean()))
        row["hold_max_error_deg"] = float(np.degrees(err.max()))
        row["hold_mean_brake_Nm"] = float(run.brake[tail].mean())
        row["hold_saturated_fraction"] = float(run.saturated[tail].mean())
        rows.append(row)

    return {"rows": rows}


def bandwidth_at(base: dict, truth, model, offset: float = 17.0,
                 cfg: rc.ControllerConfig = None,
                 amplitude_deg: float = 10.0) -> float:
    cfg = cfg or rc.ControllerConfig()
    sched = schedule_from(base)
    t_step = base["deploy_time"] + offset
    cond = sched.at(t_step)
    pts = []
    for f in BANDWIDTH_HZ:
        t1 = t_step + max(1.5, BANDWIDTH_CYCLES / f)
        if t1 > base["free_tof"]:
            continue
        ctl = _controller(model, cfg)
        ctl.actuator.reset(min(model.hold_command(cond), model.nose.brake_max))
        run = rc.simulate_reduced(model, ctl, sched,
                                  rc.sine_command(0.0, math.radians(amplitude_deg),
                                                  f, t_step),
                                  max(base["deploy_time"], t_step - 1.0), t1,
                                  phi0=0.0, p_nose0=0.0, truth=truth)
        m = run.t >= t1 - 2.0 / f
        if m.sum() < 8:
            continue
        g, ph = _describe(run.t[m], run.phi_command[m], run.phi_nose[m], f)
        pts.append({"frequency_hz": f, "gain": g,
                    "gain_dB": 20.0 * math.log10(max(g, 1e-9)), "phase_deg": ph})
    return _minus3db(pts)


def acquisition(base: dict, truth, model, cfg=None, n_angles=8) -> dict:
    """Deployment acquisition with a perturbed truth plant."""
    cfg = cfg or rc.ControllerConfig()
    sched = schedule_from(base)
    t_dep = base["deploy_time"]
    p0 = sched.at(t_dep).body_spin
    got = []
    for k in range(n_angles):
        angle = 2.0 * math.pi * k / n_angles
        ctl = _controller(model, cfg)
        run = rc.simulate_reduced(model, ctl, sched, rc.constant_command(angle),
                                  t_dep, min(t_dep + 15.0, base["free_tof"]),
                                  phi0=0.0, p_nose0=p0, truth=truth)
        err = np.abs(run.error)
        t = _first(run.t, err, math.radians(5.0), 0.5)
        got.append((t - t_dep) if t is not None else None)
    ok = [g for g in got if g is not None]
    return {"acquired": len(ok), "of": n_angles,
            "median_s": float(np.median(ok)) if ok else None,
            "max_s": float(max(ok)) if ok else None}


def _first(t, err, tol, dwell):
    inside = err <= tol
    for i in range(t.size):
        if not inside[i]:
            continue
        j = int(np.searchsorted(t, t[i] + dwell))
        if j >= t.size:
            return None
        if inside[i:j].all():
            return float(t[i])
    return None


# ===========================================================================
# Verdicts
# ===========================================================================
def verdict(case: dict, targets: dict) -> dict:
    """
    Does this case still meet the nominal numbers? Reported per criterion, so
    a case that fails one and passes the rest says so rather than collapsing
    to a single pass or fail.
    """
    rows = case["probe"]["rows"]
    checks = {
        "holds_everywhere": all(r["can_hold"] for r in rows),
        "all_steps_settle": all(r["settled_forward"] and r["settled_return"]
                                for r in rows),
        "rise_forward_within_25pct": _within(
            [r["rise_forward_s"] for r in rows], targets["rise_forward_s"], 1.25),
        "rise_return_within_25pct": _within(
            [r["rise_return_s"] for r in rows], targets["rise_return_s"], 1.25),
        "settle_within_25pct": _within(
            [r["settle_forward_s"] for r in rows] +
            [r["settle_return_s"] for r in rows], targets["settle_s"], 1.25),
        "overshoot_under_1pct": all(
            (r["overshoot_forward_pct"] or 0.0) < 1.0 and
            (r["overshoot_return_pct"] or 0.0) < 1.0 for r in rows),
        # Task C.3. One degree of roll error loses 1 - cos(1 deg) = 1.5e-4 of
        # the commanded correction, so this is a generous line; it is here to
        # catch a loop that has stopped tracking, not to police tenths.
        "tracking_under_1deg": all(r["hold_mean_error_deg"] < 1.0 for r in rows),
        "acquisition_within_25pct":
            case["acquisition"]["median_s"] is not None and
            case["acquisition"]["median_s"] <= 1.25 * targets["acquisition_s"],
        "bandwidth_within_25pct":
            case["bandwidth_hz"] is not None and
            case["bandwidth_hz"] >= 0.75 * targets["bandwidth_hz"],
    }
    return {"checks": checks, "passes": all(checks.values()),
            "failed": [k for k, v in checks.items() if not v]}


def _within(values, target, factor):
    vals = [v for v in values if v is not None]
    if len(vals) < len([v for v in values]):
        return False
    return bool(vals) and max(vals) <= factor * target


def targets_from(path: str) -> dict:
    """The pass line, taken from the nominal characterisation if it is there."""
    if not os.path.exists(path):
        return dict(FALLBACK_TARGETS)
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    tc = d["task_c"]["sized_brake"]
    fwd = [s["rise_s"] for s in tc["steps"]
           if s["direction"] == "forward" and s["rise_s"] is not None
           and abs(s["step_deg"]) == 90.0]
    ret = [s["rise_s"] for s in tc["steps"]
           if s["direction"] == "return" and s["rise_s"] is not None
           and abs(s["step_deg"]) == 90.0]
    settles = [s["settle_s"] for s in tc["steps"] if s["settle_s"] is not None]
    bws = [b["bandwidth_hz"] for b in tc["bandwidth"]
           if b["amplitude_deg"] == 10.0 and b["bandwidth_hz"]]
    return {
        "rise_forward_s": max(fwd) if fwd else FALLBACK_TARGETS["rise_forward_s"],
        "rise_return_s": max(ret) if ret else FALLBACK_TARGETS["rise_return_s"],
        "settle_s": max(settles) if settles else FALLBACK_TARGETS["settle_s"],
        "overshoot_pct": 1.0,
        "acquisition_s": d["acquisition"]["sized_brake"]["median_s"],
        "bandwidth_hz": min(bws) if bws else FALLBACK_TARGETS["bandwidth_hz"],
    }


# ===========================================================================
# Driver
# ===========================================================================
def sweep(base: dict, brake_max: float, targets: dict, informed: bool) -> list:
    nominal_model = make_plant(brake_max=brake_max)
    cases = []

    families = (
        ("bearing_viscous", VISCOUS_SCALES,
         lambda s: dict(viscous_scale=s)),
        ("bearing_coulomb", COULOMB_SCALES,
         lambda s: dict(coulomb_scale=s)),
        ("canard_aero", AERO_SCALES,
         lambda s: dict(aero_scale=s)),
        # Both bearing terms together, which is what a single wrong grease or
        # preload assumption would actually do.
        ("bearing_both", VISCOUS_SCALES,
         lambda s: dict(viscous_scale=s, coulomb_scale=s)),
    )
    for name, scales, kw in families:
        for s in scales:
            truth = truth_plant(brake_max, **kw(s))
            model = truth if informed else nominal_model
            case = {
                "family": name, "scale": s, "informed": informed,
                "brake_max_Nm": brake_max,
                "probe": probe(base, truth, model),
                "acquisition": acquisition(base, truth, model),
                "bandwidth_hz": bandwidth_at(base, truth, model),
            }
            case["verdict"] = verdict(case, targets)
            cases.append(case)
    return cases


def worst_case(base: dict, brake_max: float, targets: dict) -> dict:
    """
    The corner that stacks every uncertainty the adverse way at once.

    Adverse for HOLDING is less bearing drag and more cant torque: the brake
    then has the most to fight and the least help. Adverse for the passive
    RETURN is the opposite. Both corners are run, because the servo has to do
    both jobs.
    """
    nominal = make_plant(brake_max=brake_max)
    corners = {
        "hardest_to_hold": dict(viscous_scale=0.4, coulomb_scale=0.3,
                                aero_scale=1.3),
        "slowest_return": dict(viscous_scale=1.6, coulomb_scale=1.7,
                               aero_scale=0.7),
    }
    out = {}
    for name, kw in corners.items():
        truth = truth_plant(brake_max, **kw)
        case = {"family": f"corner_{name}", "scale": None, "informed": False,
                "brake_max_Nm": brake_max, "perturbation": kw,
                "probe": probe(base, truth, nominal),
                "acquisition": acquisition(base, truth, nominal),
                "bandwidth_hz": bandwidth_at(base, truth, nominal)}
        case["verdict"] = verdict(case, targets)
        out[name] = case
    return out


def failure_boundary(base: dict, brake_max: float, family: str,
                     lo: float, hi: float, tol: float = 0.01) -> float:
    """
    Bisect for the scale factor at which the loop stops being able to hold the
    angle everywhere. Reported instead of a retune, per the task.

    Returns the scale at the boundary, or None if the whole bracket holds.
    """
    def holds(s):
        kw = ({"viscous_scale": s, "coulomb_scale": s} if family == "bearing_both"
              else {"aero_scale": s} if family == "canard_aero"
              else {"viscous_scale": s})
        truth = truth_plant(brake_max, **kw)
        sched = schedule_from(base)
        times = np.asarray(base["schedule_t"])
        times = times[(times >= base["deploy_time"]) & (times <= base["free_tof"])]
        return all(truth.envelope(sched.at(float(t))).can_hold for t in times)

    if holds(hi):
        return None
    if not holds(lo):
        return float(lo)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if holds(mid):
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def sign_inversion_boundary(base: dict) -> dict:
    """
    The bearing scale at which the drag stops losing to the cant.

    Closed form, no simulation: the nose can be held only while the cant
    torque exceeds the bearing drag, so with the bearing scaled by `s`

        s · c_v · p_body + s · T_coulomb  <  T_cant(q̄, V, M)

    and the binding point is wherever `T_cant / (c_v·p + T_coulomb)` is
    smallest over the guided phase. Above that scale the nose creeps FORWARD
    with the brake released and nothing can slow it, because the brake pushes
    the same way. docs/CANARD-MODEL.md section 5 puts the nominal margin at
    2.2x at the lowest dynamic pressure in the whole firing-table envelope;
    this is the same statement, at the engagement being flown.
    """
    sched = schedule_from(base)
    plant = make_plant(brake_max=BRAKE_MAX_SIZED)
    times = np.asarray(base["schedule_t"])
    times = times[(times >= base["deploy_time"]) & (times <= base["free_tof"])]
    worst, worst_t, worst_q = math.inf, None, None
    for t in times:
        c = sched.at(float(t))
        drag = plant.nose.viscous * c.body_spin + plant.nose.coulomb
        ratio = plant.cant_torque(c) / drag if drag > 0 else math.inf
        if ratio < worst:
            worst, worst_t, worst_q = ratio, float(t), c.qbar
    return {
        "bearing_scale_at_inversion": worst,
        "at_time_s": worst_t,
        "since_deploy_s": worst_t - base["deploy_time"],
        "at_qbar_kPa": 1e-3 * worst_q,
        "note": "above this multiple of the bearing drag the nose cannot be "
                "held at any brake command, at any capacity",
    }


def holding_band(base: dict, brake_max: float, family: str,
                 lo: float = 0.05, hi: float = 6.0, n: int = 1200) -> dict:
    """
    The interval of scale factors over which the angle can be held at EVERY
    point of the guided phase.

    A bisection is the wrong tool here, because the hold fails at both ends
    and for different reasons:

        too much cant, or too little brake   full brake cannot stop the nose
        too much bearing drag, or too little cant
                                             the bearing beats the cant, the
                                             nose creeps forward and nothing
                                             can slow it

    so a bracket whose two ends both fail returns an edge and says nothing.
    This scans instead and returns the contiguous run containing the nominal
    scale of 1.0, or reports that the nominal itself fails.

    Closed form throughout -- `can_hold` is algebra, not a trajectory -- so
    the scan is free.
    """
    sched = schedule_from(base)
    times = np.asarray(base["schedule_t"])
    times = times[(times >= base["deploy_time"]) & (times <= base["free_tof"])]
    conds = [sched.at(float(t)) for t in times]

    def holds(s: float) -> bool:
        kw = ({"viscous_scale": s, "coulomb_scale": s} if family == "bearing_both"
              else {"aero_scale": s} if family == "canard_aero"
              else {"viscous_scale": s})
        truth = truth_plant(brake_max, **kw)
        return all(truth.envelope(c).can_hold for c in conds)

    scales = np.linspace(lo, hi, n)
    ok = np.array([holds(float(s)) for s in scales])
    i_nom = int(np.argmin(np.abs(scales - 1.0)))
    if not ok[i_nom]:
        return {"family": family, "brake_max_Nm": brake_max,
                "nominal_holds": False, "lower": None, "upper": None}
    a = i_nom
    while a > 0 and ok[a - 1]:
        a -= 1
    b = i_nom
    while b < n - 1 and ok[b + 1]:
        b += 1
    return {
        "family": family, "brake_max_Nm": brake_max, "nominal_holds": True,
        "lower": None if a == 0 else float(scales[a]),
        "upper": None if b == n - 1 else float(scales[b]),
        "lower_reason": "the bearing beats the cant" if family != "canard_aero"
        else "the cant falls below the bearing drag",
        "upper_reason": "full brake cannot out-torque the cant"
        if family == "canard_aero" else "the bearing beats the cant",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brake", type=float, default=BRAKE_MAX_SIZED)
    ap.add_argument("--nominal-json", default=os.path.join("docs", "roll_servo.json"))
    ap.add_argument("--out", default=os.path.join("docs", "roll_robustness.json"))
    args = ap.parse_args(argv)

    t0 = time.time()
    print("flying the baseline legs ...", flush=True)
    base = baseline()
    targets = targets_from(args.nominal_json)
    print("pass line taken from the nominal characterisation:")
    for k, v in targets.items():
        print(f"  {k:24s} {v}")

    results = {"targets": targets, "brake_max_Nm": args.brake,
               "scales": {"viscous": list(VISCOUS_SCALES),
                          "coulomb": list(COULOMB_SCALES),
                          "aero": list(AERO_SCALES)},
               "probe_offsets_s": list(PROBE_OFFSETS)}

    print("sweeping with the controller UNINFORMED ...", flush=True)
    results["uninformed"] = sweep(base, args.brake, targets, informed=False)
    print("sweeping with the controller INFORMED ...", flush=True)
    results["informed"] = sweep(base, args.brake, targets, informed=True)
    print("stacked corners ...", flush=True)
    results["corners"] = worst_case(base, args.brake, targets)

    print("where it stops working ...", flush=True)
    results["sign_inversion"] = sign_inversion_boundary(base)
    si = results["sign_inversion"]
    print(f"  bearing drag beats the cant at x{si['bearing_scale_at_inversion']:.2f}, "
          f"at t+{si['since_deploy_s']:.1f} s and "
          f"{si['at_qbar_kPa']:.1f} kPa")
    results["holding_bands"] = {
        f"{family}@{brake:.2f}": holding_band(base, brake, family)
        for brake in (NOSE.brake_max, args.brake, 0.85)
        for family in ("canard_aero", "bearing_both", "bearing_viscous")
    }
    for k, b in results["holding_bands"].items():
        if not b["nominal_holds"]:
            print(f"  {k}: the NOMINAL plant cannot be held")
        else:
            print(f"  {k}: holds for x{b['lower'] or 0:.2f} to "
                  f"x{b['upper'] if b['upper'] else float('inf'):.2f}")

    for name in ("uninformed", "informed"):
        bad = [c for c in results[name] if not c["verdict"]["passes"]]
        print(f"  {name}: {len(results[name]) - len(bad)}/{len(results[name])} "
              f"cases meet every Task C number")
        for c in bad:
            print(f"    {c['family']} x{c['scale']}: fails "
                  f"{', '.join(c['verdict']['failed'])}")
    for name, c in results["corners"].items():
        print(f"  corner {name}: "
              f"{'passes' if c['verdict']['passes'] else 'fails ' + ', '.join(c['verdict']['failed'])}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"wrote {args.out} in {time.time()-t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
