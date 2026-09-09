"""
Calibrate the analytic inverse map, and measure what the onboard predictor
costs.

Step 3, Tasks A and B. docs/INVERSE-MAP.md.

WHAT IS MEASURED
----------------
One primitive, flown many times: hold the despun nose at earth-referenced roll
angle `phi` from deployment until `t_end`, then release the brake. That is
exactly what a guidance law executes (docs/CONTROL-CHARACTERISATION.md section
8.4: one contiguous hold of variable length), and it is also the primitive the
map is built from, so the thing measured and the thing flown are the same
object -- `gnc.guidance.ScheduledHold`.

For each `t_end` the eight commanded angles are fitted to the first harmonic

    d(phi) = G(t_end) [cos phi, sin phi] + c(t_end)

against the SAME engagement's `t_end = t_deploy` run, which releases
immediately and is therefore the "do nothing" reference. Because the guidance
law only ever consults DIFFERENCES of G and c, the choice of reference cancels
out of everything downstream; it is fixed here so the table has a definite
zero.

Eight angles rather than four. Four determine the ellipse exactly (three
parameters, four equations) and leave one degree of freedom, which is not
enough to say whether the first harmonic describes the set. Step 4.5 quoted a
1.6 % residual from eight, and this module has to reproduce that before it is
entitled to invert anything.

THE PREDICTOR COMES FREE
------------------------
Every calibration run also carries `gnc.guidance.ImpactPredictor` on the 1 Hz
grid the guidance law uses. Its output at time `t` is a prediction of where
this round lands if it releases now; the run whose `t_end` IS that `t` measures
where it actually lands. Differencing the two over the grid is the
impact-point-prediction error of Task B, measured against the trajectory the
guidance law actually flies rather than against a bare ballistic one.

Run:  python -m analysis.guidance_authority                 all five engagements
      python -m analysis.guidance_authority --engagements long
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

from sim import aerodata, canards as cn
from sim import dynamics as dyn, integrate as ig, projectile as pr
from gnc import roll_control as rc
from gnc import guidance as gd
from gnc.inverse_map import AuthorityMap, MapNode, fit_harmonic, ellipse_of
from analysis import roll_servo as rs

# ---------------------------------------------------------------------------
# The firing-table envelope
# ---------------------------------------------------------------------------
#: Five engagements spanning the M107 firing table, all of them points
#: `run_ballistic.py` validates against FT 155-AM-2 (via Lim, NPS 2016). Step 4
#: and step 4.5 measured one of them; docs/STAGED-DEPLOYMENT.md limitation 1
#: names that as the largest untested risk in the adopted configuration.
#:
#: `tof_s` is the firing-table time of flight, carried here to show why the
#: short engagement is the stress case: 2.74 s of dead time is 43 % of a 6.4 s
#: flight and 5.6 % of a 48.9 s one.
ENGAGEMENTS = [
    {"label": "short",  "charge": 4, "qe_mils": 97.2,  "ft_range": 2000.0,  "ft_tof": 6.4},
    {"label": "short2", "charge": 4, "qe_mils": 211.6, "ft_range": 4000.0,  "ft_tof": 13.5},
    {"label": "middle", "charge": 6, "qe_mils": 378.6, "ft_range": 9000.0,  "ft_tof": 28.9},
    {"label": "mid2",   "charge": 7, "qe_mils": 520.7, "ft_range": 13000.0, "ft_tof": 42.6},
    {"label": "long",   "charge": 8, "qe_mils": 525.3, "ft_range": 16000.0, "ft_tof": 48.9},
]
ENGAGEMENT_BY_LABEL = {e["label"]: e for e in ENGAGEMENTS}

#: The adopted configuration, unchanged from step 4.5.
BRAKE = rs.BRAKE_MAX_ADOPTED
STAGED = dict(enabled=True, max_delay=4.0)

#: Eight commanded angles. See the module docstring.
ANGLES = tuple(float(a) for a in range(0, 360, 45))

#: Hold end times, as seconds after deployment. The absolute entries resolve
#: the dead time and the acquisition, which happen on a fixed timescale set by
#: the servo; the fractional ones spread the rest over the guided phase,
#: whatever its length.
NODE_ABS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.5)
NODE_FRAC = (0.10, 0.15, 0.22, 0.30, 0.40, 0.52, 0.65, 0.80, 0.90, 1.0)


def node_offsets(guided_s: float) -> list:
    """Hold-end offsets after deployment for a guided phase of `guided_s`."""
    xs = [x for x in NODE_ABS if x < guided_s * 0.95]
    xs += [f * guided_s for f in NODE_FRAC]
    xs = sorted(set(round(x, 4) for x in xs))
    # Drop nodes closer together than 0.25 s: below that the runs differ by
    # less than the correction quantum of docs/CONTROL-CHARACTERISATION.md 8.3.
    out = [xs[0]]
    for x in xs[1:]:
        if x - out[-1] >= 0.25:
            out.append(x)
    return out


# ===========================================================================
# Baseline, per engagement
# ===========================================================================
def baseline(eng: dict, dt: float = rs.SWEEP_DT) -> dict:
    """
    Generalisation of `analysis.roll_servo.baseline` to any charge and QE.

    Deployment stays at a quarter of the apogee time, which is the schedule
    docs/ARCHITECTURE-DECISION.md section 1 adopted and section 4.4 flagged as
    the hardest requirement it places on steps 3 and 5.
    """
    mv = pr.CHARGE_TABLE[eng["charge"]]
    launch = pr.LaunchConditions.from_mils(mv, eng["qe_mils"])
    y0 = dyn.initial_state(pr.M107, launch)
    unc = ig.integrate(y0, rs.base_model(), dt=dt, log_every=rs.LOG_EVERY,
                       t_max=300.0)
    tr = unc.trajectory
    alt = -tr.position[:, 2]
    t_apogee = float(tr.t[int(np.argmax(alt))])
    i = int(np.argmin(np.abs(tr.t - rs.DEPLOY_FRACTION_OF_APOGEE * t_apogee)))
    t_dep = float(tr.t[i])
    y_dep = rs._states_from(tr)[i]

    free_nose = replace(rs.NOSE, deploy_time=t_dep, brake_max=BRAKE)
    guided = cn.guided_model(rs.base_model(), rs.GEOMETRY, free_nose)
    gr = ig.integrate(y_dep, guided, dt=dt, log_every=rs.LOG_EVERY, t_max=300.0,
                      t_start=t_dep)
    gt = gr.trajectory
    return {
        "label": eng["label"], "charge": eng["charge"],
        "qe_mils": eng["qe_mils"], "muzzle_velocity": mv, "dt": dt,
        "ft_range": eng["ft_range"], "ft_tof": eng["ft_tof"],
        "uncorrected_range": unc.range_m, "uncorrected_drift": unc.drift_m,
        "uncorrected_tof": unc.impact_time,
        "apogee_time": t_apogee, "deploy_time": t_dep,
        "mach_at_deploy": float(tr.mach[i]),
        "qbar_at_deploy": float(tr.dynamic_pressure[i]),
        "spin_at_deploy": float(tr.omega[i, 0]),
        "y_deploy": y_dep.tolist(),
        "free_range": gr.range_m, "free_drift": gr.drift_m,
        "free_tof": gr.impact_time,
        "guided_phase_s": gr.impact_time - t_dep,
        "schedule_t": gt.t.tolist(),
        "schedule_mach": gt.mach.tolist(),
    }


def _predictor(base: dict, n_panels: int = 4) -> gd.ImpactPredictor:
    return gd.ImpactPredictor(pr.M107, rs.base_model().environment,
                              geometry=rs.GEOMETRY, dt=0.1, iterate_yaw=True,
                              n_panels=n_panels)


# ===========================================================================
# One calibration trajectory
# ===========================================================================
def run_hold(args) -> dict:
    """
    Hold `angle_deg` from deployment to `t_end_offset`, then release.

    A dict rather than a callable so the case can be shipped to a worker.
    """
    base = args["baseline"]
    t_dep = base["deploy_time"]
    y_dep = np.asarray(base["y_deploy"])
    dt = args.get("dt", base["dt"])
    angle = math.radians(args["angle_deg"])
    t_end = t_dep + float(args["t_end_offset"])

    model = rs.make_plant(brake_max=BRAKE)
    controller = rs._controller(model, rc.ControllerConfig())
    stage_cfg = rc.StagedDeploymentConfig(**STAGED)
    deployment = rc.StagedDeployment(stage_cfg, t_dep, controller=controller,
                                     plant=model)
    predictor = _predictor(base) if args.get("predict") else None
    commander = gd.ScheduledHold(angle, t_end, t_dep, predictor=predictor)
    law = rc.BrakeLaw(controller, commander.command, deploy_time=t_dep,
                      deployment=deployment)

    nose = replace(model.nose, deploy_time=t_dep,
                   brake_command=law.brake_command,
                   steering_gate=deployment.gate)
    guided = cn.guided_model(rs.base_model(), rs.GEOMETRY, nose)
    res = ig.integrate(y_dep, guided, dt=dt, log_every=rs.LOG_EVERY,
                       t_max=300.0, t_start=t_dep,
                       step_hook=commander.chain(law.sample))
    if law.samples == 0:
        raise RuntimeError("the controller never sampled: step_hook not wired")

    hist = law.history()
    err = np.abs(hist["error"])
    acq = rs._acquisition_time(hist["t"], err, hist["holding"], t_dep)
    out = {
        "engagement": base["label"],
        "angle_deg": args["angle_deg"],
        "t_end_offset": float(args["t_end_offset"]),
        "range_m": res.range_m, "drift_m": res.drift_m, "tof_s": res.impact_time,
        "d_range_m": res.range_m - base["uncorrected_range"],
        "d_deflection_m": res.drift_m - base["uncorrected_drift"],
        "max_total_aoa_deg": math.degrees(res.max_total_aoa),
        "stage2_delay_s": deployment.delay,
        "stage2_reason": deployment.release_reason,
        "acquisition_s": acq,
        "engage_s": ((controller.state.engage_time - t_dep)
                     if controller.state.engage_time is not None else None),
        "samples": law.samples,
    }
    if predictor is not None:
        out["predictions"] = {k: list(v) for k, v in commander.log.items()}
    if "validation" in args:
        out["validation"] = args["validation"]
    return out


def run_ideal(args) -> dict:
    """The step-2.5 kinematic idealisation: `phi_nose` pinned, no servo."""
    base = args["baseline"]
    t_dep = base["deploy_time"]
    y_dep = np.asarray(base["y_deploy"])
    nose = replace(rs.NOSE, deploy_time=t_dep, brake_max=BRAKE,
                   hold_angle=math.radians(args["angle_deg"]))
    guided = cn.guided_model(rs.base_model(), rs.GEOMETRY, nose)
    res = ig.integrate(y_dep, guided, dt=base["dt"], log_every=rs.LOG_EVERY,
                       t_max=300.0, t_start=t_dep)
    return {
        "engagement": base["label"], "angle_deg": args["angle_deg"],
        "d_range_m": res.range_m - base["uncorrected_range"],
        "d_deflection_m": res.drift_m - base["uncorrected_drift"],
        "max_total_aoa_deg": math.degrees(res.max_total_aoa),
    }


# ===========================================================================
# Reduction: rows -> AuthorityMap
# ===========================================================================
def build_map(base: dict, rows: list) -> dict:
    """
    Fit G and c at every node, and assemble the table plus its diagnostics.
    """
    by_node = {}
    for r in rows:
        by_node.setdefault(round(r["t_end_offset"], 4), []).append(r)
    ref = by_node[min(by_node)]
    ref_r = float(np.mean([r["d_range_m"] for r in ref]))
    ref_d = float(np.mean([r["d_deflection_m"] for r in ref]))

    t_imp = base["free_tof"]
    t_dep = base["deploy_time"]
    sched_t = np.asarray(base["schedule_t"])
    sched_m = np.asarray(base["schedule_mach"])

    nodes, diag = [], []
    for off in sorted(by_node):
        rs_ = sorted(by_node[off], key=lambda r: r["angle_deg"])
        angs = [r["angle_deg"] for r in rs_]
        shifts = [[r["d_range_m"] - ref_r, r["d_deflection_m"] - ref_d]
                  for r in rs_]
        G, c, resid = fit_harmonic(angs, shifts)
        # Impact time is itself a (small) function of the hold, so index the
        # node on the MEAN time to go of the runs that built it.
        tof = float(np.mean([r["tof_s"] for r in rs_]))
        t_end = t_dep + off
        t_go = max(tof - t_end, 0.0)
        mach = float(np.interp(t_end, sched_t, sched_m))
        nodes.append(MapNode(t_go=t_go, G=G, c=c, mach=mach, residual=resid))
        e = ellipse_of(G)
        diag.append({
            "t_end_offset": off, "t_go": t_go, "mach": mach,
            "residual": resid, "n_angles": len(rs_),
            "tof_s": tof,
            "stage2_delay_s": float(np.mean([r["stage2_delay_s"] for r in rs_
                                             if r["stage2_delay_s"] is not None]))
            if any(r["stage2_delay_s"] is not None for r in rs_) else None,
            "acquisition_s": float(np.mean([r["acquisition_s"] for r in rs_
                                            if r["acquisition_s"] is not None]))
            if any(r["acquisition_s"] is not None for r in rs_) else None,
            "max_total_aoa_deg": max(r["max_total_aoa_deg"] for r in rs_),
            "centre_range_m": float(c[0]), "centre_deflection_m": float(c[1]),
            **e,
        })
    amap = AuthorityMap(nodes, impact_time=t_imp, deploy_time=t_dep,
                        label=base["label"])
    return {"map": amap.as_dict(), "nodes": diag,
            "reference": {"d_range_m": ref_r, "d_deflection_m": ref_d}}


def pointing_table(amap: AuthorityMap, t_go_now: float,
                   n_dirs: int = 24) -> list:
    """
    The map's own view of itself: for `n_dirs` desired directions, the
    commanded angle, the rotation between them, and the reach.

    This is the analytic content of Task A. The 6-DOF validation of it lives
    in `main`, which flies the commanded angles and measures where the impact
    actually goes.
    """
    A, _ = amap.remaining(t_go_now)
    out = []
    for k in range(n_dirs):
        th = 2.0 * math.pi * k / n_dirs
        u = np.array([math.cos(th), math.sin(th)])
        phi = AuthorityMap.invert(A, u)
        if phi is None:
            continue
        out.append({
            "desired_deg": math.degrees(th),
            "phi_deg": math.degrees(phi),
            "rotation_deg": math.degrees(_wrap(th - phi)),
            "reach_m": AuthorityMap.reach_along(A, u),
        })
    return out


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def prediction_error(base: dict, rows: list) -> dict:
    """
    Task B. The predictor's output at time `t` against the impact of the run
    that released at `t`.

    The predictor's answer is "where this round lands if it releases now", so
    the truth it must be scored against is the impact of the run that DID
    release then, at the same commanded angle. That is not a bare ballistic
    trajectory: it still carries the kit's drag and the residual force of a
    free-spinning nose, and scoring against a ballistic reference would credit
    the predictor with an error it does not make.

    Predictions are taken from the longest-hold run at each angle. Up to its
    own release time every run at that angle is flying the identical
    trajectory, so its prediction series covers every node.

    Grouped by commanded angle as well as by time, because the residual comes
    from the canard normal force the reduced model has nowhere to put, and
    that force points where the nose is held.
    """
    t_dep = base["deploy_time"]
    by_angle_truth = {}
    longest = {}
    for r in rows:
        by_angle_truth.setdefault(r["angle_deg"], []).append(
            (round(r["t_end_offset"], 4), r["range_m"], r["drift_m"]))
        if "predictions" in r:
            cur = longest.get(r["angle_deg"])
            if cur is None or r["t_end_offset"] > cur["t_end_offset"]:
                longest[r["angle_deg"]] = r

    per_t, per_angle = {}, {}
    for ang, run in longest.items():
        P = run["predictions"]
        tp = np.asarray(P["t"], dtype=float)
        if tp.size == 0:
            continue
        pr_ = np.asarray(P["pred_range"], dtype=float)
        pd_ = np.asarray(P["pred_defl"], dtype=float)
        for off, tr_, td_ in sorted(by_angle_truth[ang]):
            t = t_dep + off
            if t < tp[0] - 1e-9 or t > tp[-1] + 1e-9:
                continue
            er = float(np.interp(t, tp, pr_)) - tr_
            ed = float(np.interp(t, tp, pd_)) - td_
            per_t.setdefault(off, []).append((er, ed))
            per_angle.setdefault(ang, []).append((off, er, ed))

    rows_out = []
    for k in sorted(per_t):
        v = np.array(per_t[k])
        rows_out.append({
            "t_end_offset": k,
            "t_go": base["free_tof"] - (t_dep + k),
            "n": int(v.shape[0]),
            "bias_range_m": float(v[:, 0].mean()),
            "bias_deflection_m": float(v[:, 1].mean()),
            "sd_range_m": float(v[:, 0].std(ddof=1)) if v.shape[0] > 1 else 0.0,
            "sd_deflection_m": float(v[:, 1].std(ddof=1)) if v.shape[0] > 1 else 0.0,
            "rms_m": float(np.sqrt((v ** 2).sum(axis=1).mean())),
            "max_m": float(np.sqrt((v ** 2).sum(axis=1)).max()),
        })
    return {"rows": rows_out,
            "per_angle": {str(a): [[float(x) for x in e] for e in v]
                          for a, v in per_angle.items()}}


# ===========================================================================
# Cases
# ===========================================================================
def calibration_cases(base: dict, predict: bool = True) -> list:
    offs = node_offsets(base["guided_phase_s"])
    return [{"baseline": base, "t_end_offset": o, "angle_deg": a,
             "predict": predict}
            for o in offs for a in ANGLES]


def ideal_cases(base: dict) -> list:
    return [{"baseline": base, "angle_deg": a} for a in ANGLES]


# ===========================================================================
# Task A validation -- fly the map's own answers
# ===========================================================================
#: Desired ground-plane correction directions for the validation, degrees
#: measured from downrange. Twelve, at 30-degree spacing, deliberately NOT the
#: eight calibration angles: the map is fitted through commanded angles 45
#: degrees apart, and the commanded angles these twelve invert to fall between
#: them, so this is an out-of-sample test of the fit and not a restatement of
#: it.
VALIDATION_DIRECTIONS = tuple(float(a) for a in range(0, 360, 30))

#: Release fractions of the guided phase at which the inversion is also
#: tested. 1.0 is the full hold and exercises `remaining`; the others exercise
#: `increment`, which is what the guidance law actually calls once it has
#: started holding.
VALIDATION_HOLD_FRACTIONS = (1.0, 0.55)


def pointing_cases(base: dict, amap: AuthorityMap) -> list:
    """
    For each desired direction, the commanded angle the map says delivers it.
    """
    t_dep = base["deploy_time"]
    guided = base["guided_phase_s"]
    t_go_dep = amap.nodes[0].t_go
    out = []
    for frac in VALIDATION_HOLD_FRACTIONS:
        off = frac * guided
        t_go_end = max(t_go_dep - off, 0.0)
        A, b = amap.increment(t_go_dep, t_go_end)
        for d in VALIDATION_DIRECTIONS:
            u = np.array([math.cos(math.radians(d)), math.sin(math.radians(d))])
            phi = AuthorityMap.invert(A, u)
            if phi is None:
                continue
            w = np.array([math.cos(phi), math.sin(phi)])
            out.append({
                "baseline": base, "t_end_offset": off,
                "angle_deg": math.degrees(phi), "predict": False,
                "validation": {
                    "desired_deg": d, "hold_fraction": frac,
                    "phi_deg": math.degrees(phi),
                    "predicted_steer": (A @ w).tolist(),
                    "predicted_offset": b.tolist(),
                    "reach_m": AuthorityMap.reach_along(A, u),
                },
            })
    return out


def reduce_pointing(base: dict, rows: list, ref: dict) -> dict:
    """
    Residual pointing error: the angle between the correction the map was
    asked for and the correction the 6-DOF delivered.

    Measured on the STEERABLE part. The delivered displacement is
    `A w + b`, and only `A w` is commandable -- `b` is what holding any angle
    at all costs, and it arrives whatever direction is asked for. The map
    predicts both, so both are differenced, but the pointing residual is the
    angle of the steerable part and `b` is reported beside it so its size is
    visible rather than hidden.
    """
    ref_r, ref_d = ref["d_range_m"], ref["d_deflection_m"]
    out = []
    for r in rows:
        v = r["validation"]
        got = np.array([r["d_range_m"] - ref_r, r["d_deflection_m"] - ref_d])
        steer = got - np.asarray(v["predicted_offset"])
        d = math.radians(v["desired_deg"])
        u = np.array([math.cos(d), math.sin(d)])
        mag = float(np.hypot(steer[0], steer[1]))
        err = _wrap(math.atan2(steer[1], steer[0]) - d)
        out.append({
            "desired_deg": v["desired_deg"], "hold_fraction": v["hold_fraction"],
            "phi_deg": v["phi_deg"],
            "predicted_m": v["reach_m"],
            "delivered_m": mag,
            "along_m": float(steer @ u),
            "cross_m": float(steer[0] * -u[1] + steer[1] * u[0]),
            "pointing_error_deg": math.degrees(err),
            "magnitude_error_frac": (mag / v["reach_m"] - 1.0)
            if v["reach_m"] > 0 else float("nan"),
            "offset_m": float(np.hypot(*v["predicted_offset"])),
        })
    per_frac = {}
    for frac in VALIDATION_HOLD_FRACTIONS:
        sel = [o for o in out if abs(o["hold_fraction"] - frac) < 1e-9]
        if not sel:
            continue
        pe = np.array([o["pointing_error_deg"] for o in sel])
        me = np.array([o["magnitude_error_frac"] for o in sel])
        cr = np.array([o["cross_m"] for o in sel])
        per_frac[str(frac)] = {
            "n": int(pe.size),
            "pointing_rms_deg": float(np.sqrt((pe ** 2).mean())),
            "pointing_max_deg": float(np.abs(pe).max()),
            "pointing_mean_deg": float(pe.mean()),
            "magnitude_rms_frac": float(np.sqrt((me ** 2).mean())),
            "magnitude_max_frac": float(np.abs(me).max()),
            "cross_rms_m": float(np.sqrt((cr ** 2).mean())),
            "cross_max_m": float(np.abs(cr).max()),
        }
    return {"rows": out, "summary": per_frac}


def _imap(pool, fn, cases, label):
    t0 = time.time()
    out = []
    for i, r in enumerate(pool.imap_unordered(fn, cases, chunksize=1), 1):
        out.append(r)
        if i % 16 == 0 or i == len(cases):
            print(f"  {label}: {i}/{len(cases)}  {time.time() - t0:6.1f} s",
                  flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engagements", default="all")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="docs/guidance_map.json")
    ap.add_argument("--force", action="store_true",
                    help="recompute engagements already present in --out")
    ap.add_argument("--tasks", default="map,pointing",
                    help="map = the calibration; pointing = the Task A "
                         "out-of-sample validation of it")
    args = ap.parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    labels = ([e["label"] for e in ENGAGEMENTS] if args.engagements == "all"
              else args.engagements.split(","))
    t_start = time.time()
    out = {"engagements": {}, "config": {
        "brake_max_Nm": BRAKE, "staged": STAGED, "angles_deg": list(ANGLES),
        "dt": rs.SWEEP_DT, "predictor_dt": 0.1, "predictor_iterate_yaw": True,
    }}
    # Resume. Each engagement costs a quarter of an hour of 6-DOF trajectories
    # and they are independent of each other, so the file is written after
    # every one and a re-run picks up where it stopped rather than starting
    # again.
    if os.path.exists(args.out):
        try:
            with open(args.out) as fh:
                prev = json.load(fh)
            out["engagements"].update(prev.get("engagements", {}))
            print(f"resuming from {args.out}: have "
                  f"{sorted(out['engagements'])}", flush=True)
        except (ValueError, OSError) as exc:
            print(f"could not read {args.out} ({exc}); starting fresh")

    with Pool(args.workers) as pool:
        for lbl in ([] if "map" not in tasks else labels):
            if lbl in out["engagements"] and not args.force:
                print(f"\n=== {lbl}: already present, skipping ===", flush=True)
                continue
            eng = ENGAGEMENT_BY_LABEL[lbl]
            print(f"\n=== {lbl}: charge {eng['charge']}, QE {eng['qe_mils']} mils "
                  f"===", flush=True)
            base = baseline(eng)
            print(f"  uncorrected {base['uncorrected_range']:.1f} m, "
                  f"tof {base['uncorrected_tof']:.2f} s (FT {eng['ft_range']:.0f} m, "
                  f"{eng['ft_tof']:.1f} s)")
            print(f"  deploy at {base['deploy_time']:.3f} s, M "
                  f"{base['mach_at_deploy']:.3f}, qbar "
                  f"{base['qbar_at_deploy'] / 1e3:.1f} kPa; guided phase "
                  f"{base['guided_phase_s']:.2f} s", flush=True)

            cases = calibration_cases(base)
            rows = _imap(pool, run_hold, cases, f"{lbl} hold")
            ideal = _imap(pool, run_ideal, ideal_cases(base), f"{lbl} ideal")

            built = build_map(base, rows)
            amap = AuthorityMap.from_dict(built["map"])
            Gi, ci, ri = fit_harmonic([r["angle_deg"] for r in ideal],
                                      [[r["d_range_m"], r["d_deflection_m"]]
                                       for r in ideal])
            full = amap.nodes[-1]
            e_full = ellipse_of(full.G)
            e_ideal = ellipse_of(Gi)
            print(f"  full hold: semi-major {e_full['semi_major_m']:.1f} m, "
                  f"semi-minor {e_full['semi_minor_m']:.1f} m, ratio "
                  f"{e_full['axis_ratio']:.2f}, residual {full.residual:.3f}")
            print(f"  ideal    : semi-major {e_ideal['semi_major_m']:.1f} m, "
                  f"retained {e_full['rms_amplitude_m'] / e_ideal['rms_amplitude_m'] * 100:.1f} %",
                  flush=True)

            out["engagements"][lbl] = {
                "baseline": {k: v for k, v in base.items()
                             if k not in ("schedule_t", "schedule_mach",
                                          "y_deploy")},
                "y_deploy": base["y_deploy"],
                "schedule_t": base["schedule_t"],
                "schedule_mach": base["schedule_mach"],
                **built,
                "ideal": {"ellipse": e_ideal, "residual": ri,
                          "G": Gi.tolist(), "c": ci.tolist(),
                          "rows": ideal},
                "full_hold_ellipse": e_full,
                "retained_rms": (e_full["rms_amplitude_m"]
                                 / e_ideal["rms_amplitude_m"]),
                "retained_semi_major": (e_full["semi_major_m"]
                                        / e_ideal["semi_major_m"]),
                "pointing": pointing_table(amap, amap.nodes[0].t_go),
                "prediction_error": prediction_error(base, rows),
                "rows": [{k: v for k, v in r.items() if k != "predictions"}
                         for r in rows],
            }
            _write(args.out, out)
            print(f"  wrote {args.out} ({time.time() - t_start:.1f} s elapsed)",
                  flush=True)

        # -- Task A validation, out of sample ----------------------------
        if "pointing" in tasks:
            for lbl in labels:
                e = out["engagements"].get(lbl)
                if e is None or ("pointing_validation" in e and not args.force):
                    continue
                base = dict(e["baseline"])
                base["y_deploy"] = e["y_deploy"]
                amap = AuthorityMap.from_dict(e["map"])
                cases = pointing_cases(base, amap)
                print(f"\n=== {lbl}: pointing validation, {len(cases)} cases "
                      f"===", flush=True)
                rows = _imap(pool, run_hold, cases, f"{lbl} pointing")
                e["pointing_validation"] = reduce_pointing(
                    base, rows, e["reference"])
                for frac, sm in e["pointing_validation"]["summary"].items():
                    print(f"  hold {frac}: pointing rms "
                          f"{sm['pointing_rms_deg']:.2f} deg, worst "
                          f"{sm['pointing_max_deg']:.2f}; magnitude rms "
                          f"{100 * sm['magnitude_rms_frac']:.1f} %; cross-track "
                          f"rms {sm['cross_rms_m']:.2f} m", flush=True)
                _write(args.out, out)

    _write(args.out, out)
    print(f"\nwrote {args.out}  ({time.time() - t_start:.1f} s)")
    return 0


def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


if __name__ == "__main__":
    raise SystemExit(main())
