"""
Step 5, Tasks D, F and G: the warm start, the navigation contribution to CEP,
and what degradation costs.

Produces docs/NAV-CEP.md, docs/NAV-DEGRADATION.md and docs/nav_cep.json.

HOW THE NAVIGATION CONTRIBUTION IS ISOLATED
-------------------------------------------
Exactly the way step 2 isolated model error: fly the SAME round twice and
difference it.

  1. `use_nav=False` -- guidance and control read the true 6-DOF state. This
     is `analysis.guidance_cep.run_guided` in all but name and it reproduces
     step 3's CEP.
  2. `use_nav=True`  -- the identical case, with a `NavigationSystem` in the
     loop driven by simulated sensors. The perturbation, the aim-off, the
     scheduler, the staged deployment and the authority monitor are the same
     objects with the same settings.
  3. The difference in impact point IS the navigation contribution, per round,
     with the common dispersion cancelled.

PAIRING IS NOT A CONVENIENCE HERE, IT IS THE MEASUREMENT. The uncorrected
dispersion is 200 m CEP and the navigation contribution is expected to be tens
of metres; comparing two 64-round CEPs would resolve it about as well as
weighing a letter by weighing the postman. Differencing the same draw removes
the dispersion exactly.

SEEDS, AND STEP 4.5's LESSON
----------------------------
docs/STAGED-DEPLOYMENT.md found that five samples understate a spread by a
factor of five. So the navigation contribution is measured over
`N_SEEDS` sensor realisations per draw, not one: a single seed gives one draw
from the navigation error distribution and reports it as if it were the
distribution.

Run:  python -m analysis.nav_cep --tasks d,f,g
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np

from sim import projectile as pr
from gnc import navigation as nv, sensors as sn
from analysis import guidance_cep as gc, nav_common as nc

#: Draws per engagement for the paired Task F comparison.
N_DRAWS = 24
#: Sensor realisations per draw. Three, so the navigation contribution is a
#: distribution rather than a draw; 24 x 3 = 72 navigation-fed rounds per
#: engagement against 24 truth-fed ones.
N_SEEDS = 3
#: Engagements. The whole envelope step 3 characterised.
ENGAGEMENTS = ("short2", "middle", "mid2", "long")
MAIN = "long"


# ===========================================================================
def _cases(ctx: dict, draws: list, **kw) -> list:
    mon = {k: v for k, v in ctx.get("monitor", {}).items()
           if not k.startswith("_")}
    return [{"baseline": ctx["base"], "map": ctx["map"], **mon,
             "dmv": d["dmv"], "daz": d["daz"], "draw": d["draw"],
             "scheduler": "proportional",
             "scheduler_opts": kw.pop("scheduler_opts", {}) if False else
             kw.get("scheduler_opts", {}),
             **{k: v for k, v in kw.items() if k != "scheduler_opts"}}
            for d in draws]


def _imap(pool, fn, cases, label):
    t0 = time.time()
    out = []
    for i, r in enumerate(pool.imap_unordered(fn, cases), 1):
        out.append(r)
        if i % 10 == 0 or i == len(cases):
            el = time.time() - t0
            print(f"    {label}: {i}/{len(cases)}  {el:6.1f} s  "
                  f"(eta {el / i * (len(cases) - i):5.1f} s)", flush=True)
    return sorted(out, key=lambda r: (r.get("draw") or 0, r.get("seed") or 0))


def paired_navigation_delta(truth_rows: list, nav_rows: list) -> dict:
    """
    The navigation contribution, per axis, from paired rounds.

    Range and deflection separately, because step 3 found the semi-minor axis
    is what predicts closed-loop performance and a circular figure would hide
    it. Reported as bias and 1 sigma; the CEP budget wants the sigma but the
    bias is what tells you whether something is systematically wrong.
    """
    by_draw = {}
    for r in truth_rows:
        by_draw[r["draw"]] = r
    dr, dd, dm = [], [], []
    pairs = []
    for r in nav_rows:
        t = by_draw.get(r["draw"])
        if t is None:
            continue
        a = r["range_m"] - t["range_m"]
        b = r["drift_m"] - t["drift_m"]
        dr.append(a); dd.append(b); dm.append(math.hypot(a, b))
        pairs.append({"draw": r["draw"], "seed": r.get("seed"),
                      "d_range_m": a, "d_defl_m": b,
                      "truth_miss_m": t["miss_m"], "nav_miss_m": r["miss_m"]})
    dr, dd, dm = np.array(dr), np.array(dd), np.array(dm)
    if dr.size == 0:
        return {"n": 0}
    return {
        "n": int(dr.size),
        "range": {"bias_m": float(dr.mean()), "sigma_m": float(dr.std(ddof=1)),
                  "rms_m": float(math.sqrt((dr ** 2).mean())),
                  "p95_abs_m": float(np.percentile(np.abs(dr), 95)),
                  "max_abs_m": float(np.abs(dr).max())},
        "deflection": {"bias_m": float(dd.mean()), "sigma_m": float(dd.std(ddof=1)),
                       "rms_m": float(math.sqrt((dd ** 2).mean())),
                       "p95_abs_m": float(np.percentile(np.abs(dd), 95)),
                       "max_abs_m": float(np.abs(dd).max())},
        "radial": {"median_m": float(np.median(dm)), "mean_m": float(dm.mean()),
                   "p90_m": float(np.percentile(dm, 90)),
                   "max_m": float(dm.max())},
        "pairs": pairs,
    }


def _cep(rows: list) -> dict:
    return gc.cep_of(rows)


def reacquisition_correlation(nav_rows: list, truth_rows: list,
                              arm_time: float) -> dict:
    """
    Does the navigation contribution track WHEN the receiver came back?

    Each navigation-fed round carries its own drawn reacquisition time
    (`n_gnss_reacquire_s`). If the contribution is driven by guidance arming
    before the filter has a GNSS-corrected solution, then the rounds whose
    receiver returned after `arm_time` should be displaced further -- and that
    is a prediction this function tests rather than a story told after the
    fact.
    """
    by_draw = {r["draw"]: r for r in truth_rows}
    late, early = [], []
    x, y = [], []
    for r in nav_rows:
        t = by_draw.get(r["draw"])
        if t is None or r.get("n_gnss_reacquire_s") is None:
            continue
        d = math.hypot(r["range_m"] - t["range_m"], r["drift_m"] - t["drift_m"])
        x.append(float(r["n_gnss_reacquire_s"]))
        y.append(d)
        (late if r["n_gnss_reacquire_s"] > arm_time else early).append(d)
    if len(x) < 4:
        return {"n": len(x)}
    x, y = np.array(x), np.array(y)
    out = {"n": int(x.size), "arm_time_s": float(arm_time),
           "spearman": float(_spearman(x, y)),
           "fraction_late": float(np.mean(x > arm_time))}
    for name, v in (("reacquired_before_arming", early),
                    ("reacquired_after_arming", late)):
        v = np.array(v)
        out[name] = ({"n": int(v.size), "median_m": float(np.median(v)),
                      "mean_m": float(v.mean())} if v.size else {"n": 0})
    return out


def _spearman(a, b) -> float:
    """Rank correlation, so one outlier cannot make the case on its own."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


# ===========================================================================
# Task D -- the warm start
# ===========================================================================
def task_d(pool, mapdata: dict, label: str = MAIN, n: int = 32) -> dict:
    """
    What the gun-data upload is worth, measured as TIME TO A USABLE SOLUTION.

    The kit cannot command a direction before t_dep + 3 s, so a warm start is
    only a feature if it moves the solution inside that window. This measures
    whether it does; it does not assume it.

    Cheap, because it needs no 6-DOF: the question is when the FILTER has a
    solution, and the filter's own a-priori propagation plus the receiver's
    reacquisition time answer it. What it does need is the reacquisition
    distribution, which is the LOW-confidence number the whole task turns on.
    """
    ctx = gc.engagement_context(mapdata, label)
    base = ctx["base"]
    t_dep = base["deploy_time"]
    deadline = t_dep + 3.0
    cases = [(label, s, warm) for warm in (True, False) for s in range(n)]
    rows = pool.map(_warm_case, [(base, s, warm, deadline) for _, s, warm in cases])
    out = {"engagement": label, "n": n, "deploy_time_s": t_dep,
           "deadline_s": deadline, "modes": {}}
    for warm in (True, False):
        sub = [r for r in rows if r["warm_start"] == warm]
        usable = np.array([r["first_usable_s"] if r["first_usable_s"] is not None
                           else np.inf for r in sub])
        fix = np.array([r["gnss_reacquire_s"] for r in sub])
        err = np.array([r["position_error_at_deadline_m"] for r in sub
                        if r["position_error_at_deadline_m"] is not None])
        out["modes"]["warm" if warm else "cold"] = {
            "first_usable_s": {
                "median": float(np.median(usable[np.isfinite(usable)]))
                if np.isfinite(usable).any() else None,
                "p90": float(np.percentile(usable[np.isfinite(usable)], 90))
                if np.isfinite(usable).any() else None,
                "max": float(usable[np.isfinite(usable)].max())
                if np.isfinite(usable).any() else None},
            "fraction_usable_by_deadline": float(np.mean(usable <= deadline)),
            "gnss_reacquire_s": {"median": float(np.median(fix)),
                                 "min": float(fix.min()), "max": float(fix.max())},
            "position_error_at_deadline_m": {
                "median": float(np.median(err)) if err.size else None,
                "p90": float(np.percentile(err, 90)) if err.size else None,
                "max": float(err.max()) if err.size else None},
        }
    out["rows"] = rows
    return out


def _warm_case(a) -> dict:
    """
    One warm/cold start, run WITHOUT the 6-DOF.

    The filter is stepped on the a-priori model and the receiver's fixes only,
    which is exactly what it has before the IMU wakes up. Truth for the
    comparison is the same reduced-order propagation the guidance predictor
    trusts to 0.65 m over a whole flight, so using it here as truth costs
    nothing this measurement can resolve.
    """
    base, seed, warm, deadline = a
    from models import mpmm
    from sim import aerodata
    from analysis import roll_servo as rs

    env = rs.base_model().environment
    gun = nc.gun_data(base)
    suite = sn.SuiteConfig()
    nav = nv.NavigationSystem(nv.NavConfig(), suite, seed=seed,
                              projectile=pr.M107, environment=env, gun=gun,
                              muzzle_time=0.0, log=False, warm_start=warm)
    launch = pr.LaunchConditions(muzzle_velocity=base["muzzle_velocity"],
                                 quadrant_elevation=base["qe_mils"] * pr.MIL_TO_RAD)
    y7 = mpmm.initial_state(pr.M107, launch)
    model = mpmm.MpmmModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                           environment=env, iterate_yaw=True)
    dt = 1.0 / nav.config.imu_rate
    # The step-1 ballistic model, purely so `sensors.truth_at` has a
    # real derivative to evaluate. Its output barely matters here: before
    # deployment the accelerometers are pinned at 4 361 g by the body
    # spin, so the IMU contributes NOTHING to this phase and the answer
    # is set by the a-priori model and the receiver alone -- which is the
    # whole point of Task D.
    flight_model = rs.base_model()
    from sim import dynamics as dyn, frames
    y = np.zeros(dyn.STATE_SIZE)
    t = 0.0
    err_at_deadline = None
    n_steps = int(math.ceil(deadline / dt)) + 4
    for k in range(n_steps):
        r, v, p = y7[0:3], y7[3:6], y7[6]
        # A body attitude consistent with the reduced model: axis along the
        # velocity, rolling at the axial rate. Enough to drive the sensors,
        # and the roll it implies is what the magnetometer will report.
        sp = float(np.hypot(v[0], v[1]))
        q = frames.quat_from_euler(math.atan2(float(v[1]), float(v[0])),
                                   math.atan2(-float(v[2]), sp), p * t)
        y[0:3] = r; y[3:6] = v; y[6:10] = q
        y[10:13] = (p, 0.0, 0.0); y[13] = 0.0; y[14] = 0.0
        nav.sample(t, y, flight_model)
        if err_at_deadline is None and t >= deadline and nav.filter.initialised:
            err_at_deadline = float(np.linalg.norm(nav.filter.r - r))
        k1 = np.array(mpmm.derivative(t, y7, model))
        k2 = np.array(mpmm.derivative(t + 0.5 * dt, y7 + 0.5 * dt * k1, model))
        k3 = np.array(mpmm.derivative(t + 0.5 * dt, y7 + 0.5 * dt * k2, model))
        k4 = np.array(mpmm.derivative(t + dt, y7 + dt * k3, model))
        y7 = y7 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return {"seed": seed, "warm_start": warm,
            "first_usable_s": nav.first_usable_t,
            "first_fix_s": nav.first_fix_t,
            "gnss_reacquire_s": nav.sensors.gnss.reacquire_time,
            "position_error_at_deadline_m": err_at_deadline}


# ===========================================================================
# Task F -- the navigation contribution to CEP
# ===========================================================================
def task_f(pool, mapdata: dict, n: int = N_DRAWS, seeds: int = N_SEEDS,
           engagements=ENGAGEMENTS) -> dict:
    out = {"n_draws": n, "n_seeds": seeds, "engagements": {}}
    for label in engagements:
        ctx = gc.engagement_context(mapdata, label)
        opts = gc.scheduler_options(mapdata, label)
        draws = gc.make_draws(ctx, n)
        so = opts["proportional"]
        truth = _imap(pool, nc.run_guided_nav,
                      _cases(ctx, draws, use_nav=False, scheduler_opts=so),
                      f"F {label} truth")
        navc = []
        for s in range(seeds):
            navc += _cases(ctx, draws, use_nav=True, seed=s, scheduler_opts=so)
        nav = _imap(pool, nc.run_guided_nav, navc, f"F {label} nav")
        delta = paired_navigation_delta(truth, nav)
        out["engagements"][label] = {
            "truth_cep": _cep(truth), "nav_cep": _cep(nav),
            "delta": delta,
            "truth_rows": truth, "nav_rows": nav,
        }
        d = delta
        print(f"  {label:8s} truth CEP {out['engagements'][label]['truth_cep']['cep_m']:7.2f} m"
              f"   nav CEP {out['engagements'][label]['nav_cep']['cep_m']:7.2f} m"
              f"   contribution 1 sigma range {d['range']['sigma_m']:6.2f} m,"
              f" deflection {d['deflection']['sigma_m']:6.2f} m", flush=True)
    return out


# ===========================================================================
# Task G -- degradation
# ===========================================================================
#: Outage durations, s, and where in the guided phase they start as a fraction
#: of it. Step 3 measured that losing navigation at 25 % of the guided phase
#: costs 23 m of CEP and at 60 % costs nothing at the median, so the ladder is
#: placed to straddle that.
OUTAGE_DURATIONS = (2.0, 5.0, 10.0)
OUTAGE_STARTS = (0.15, 0.40, 0.70)


def task_g(pool, mapdata: dict, label: str = MAIN, n: int = N_DRAWS,
           seeds: int = 2) -> dict:
    ctx = gc.engagement_context(mapdata, label)
    opts = gc.scheduler_options(mapdata, label)
    so = opts["proportional"]
    base = ctx["base"]
    t_dep = base["deploy_time"]
    guided = base["uncorrected_tof"] - t_dep
    draws = gc.make_draws(ctx, n)

    def cases(seedset, **kw):
        c = []
        for s in seedset:
            c += _cases(ctx, draws, use_nav=True, seed=s, scheduler_opts=so, **kw)
        return c

    out = {"engagement": label, "n_draws": n, "seeds": seeds,
           "guided_phase_s": guided, "cases": {}}
    seedset = list(range(seeds))

    nominal = _imap(pool, nc.run_guided_nav, cases(seedset), "G nominal")
    out["cases"]["nominal"] = {"cep": _cep(nominal), "rows": nominal}
    print(f"  nominal          CEP {_cep(nominal)['cep_m']:7.2f} m", flush=True)

    for dur in OUTAGE_DURATIONS:
        for frac in OUTAGE_STARTS:
            t0 = t_dep + frac * guided
            key = f"outage_{dur:.0f}s_at_{int(frac * 100)}pct"
            rows = _imap(pool, nc.run_guided_nav,
                         cases(seedset, gnss_outages=((t0, t0 + dur),)), f"G {key}")
            out["cases"][key] = {
                "cep": _cep(rows), "rows": rows, "duration_s": dur,
                "start_fraction": frac, "start_s": t0,
                "paired_vs_nominal": gc.paired_delta(rows, nominal)}
            print(f"  {key:24s} CEP {out['cases'][key]['cep']['cep_m']:7.2f} m",
                  flush=True)

    rows = _imap(pool, nc.run_guided_nav, cases(seedset, gnss_denied=True),
                 "G denied")
    out["cases"]["gnss_denied"] = {"cep": _cep(rows), "rows": rows}
    print(f"  gnss_denied      CEP {_cep(rows)['cep_m']:7.2f} m", flush=True)

    rows = _imap(pool, nc.run_guided_nav, cases(seedset, mag_calibrated=False),
                 "G mag fault")
    out["cases"]["mag_uncalibrated"] = {"cep": _cep(rows), "rows": rows}
    print(f"  mag_uncalibrated CEP {_cep(rows)['cep_m']:7.2f} m", flush=True)

    rows = _imap(pool, nc.run_guided_nav, cases(seedset, warm_start=False),
                 "G cold start")
    out["cases"]["cold_start"] = {"cep": _cep(rows), "rows": rows,
                                  "paired_vs_nominal": gc.paired_delta(rows, nominal)}
    print(f"  cold_start       CEP {_cep(rows)['cep_m']:7.2f} m", flush=True)
    return out


def task_c(mapdata: dict, label: str = MAIN, seeds: int = 4) -> dict:
    """
    How far does the solution DRIFT during a GNSS outage?

    A different question from Task G's CEP, and the two answers disagree: a
    10 s outage drifts the solution 48.8 m and moves the impact point by a
    metre and a half, because the drift is recovered when the receiver returns
    and the law has the rest of the flight to correct with. Reporting only the
    CEP would hide the drift; reporting only the drift would overstate it.

    Runs on REPLAY, so it costs seconds rather than the hour a closed-loop
    campaign would: the question is what the filter does, and the filter's
    behaviour during an outage does not depend on what the round does about it.
    """
    from dataclasses import replace as dc_replace
    traj = nc.fly_truth(label, mapdata)
    t_dep = float(traj["t"][0])
    guided = float(traj["t"][-1]) - t_dep
    out = {"engagement": label, "seeds": seeds, "rows": []}
    for dur in (2.0, 5.0, 10.0, 30.0):
        for frac in OUTAGE_STARTS:
            t0 = t_dep + frac * guided
            if t0 + dur > float(traj["t"][-1]):
                continue
            marks = [k for k in (1, 2, 5, 10, 20, 30) if k <= dur]
            pe = {k: [] for k in marks}
            ve = {k: [] for k in marks}
            for seed in range(seeds):
                suite = dc_replace(sn.SuiteConfig(),
                                   gnss_outages=((t0, t0 + dur),))
                r = nc.replay(traj, seed=seed, suite=suite, log_nis=False)
                t = r["t"]
                p = np.linalg.norm(r["position_error"], axis=1)
                v = np.linalg.norm(r["velocity_error"], axis=1)
                for k in marks:
                    i = int(np.argmin(np.abs(t - (t0 + k))))
                    pe[k].append(float(p[i])); ve[k].append(float(v[i]))
            out["rows"].append({
                "duration_s": dur, "start_fraction": frac, "start_s": t0,
                "position_error_m": {str(k): float(np.mean(pe[k])) for k in marks},
                "velocity_error_ms": {str(k): float(np.mean(ve[k])) for k in marks}})
            print(f"  outage {dur:4.0f} s at {100 * frac:3.0f} %: " +
                  "  ".join(f"{k}s {np.mean(pe[k]):6.1f} m /"
                            f" {np.mean(ve[k]):5.2f} m/s" for k in marks),
                  flush=True)
    return out


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="d,f,g")
    ap.add_argument("--out", default="docs/nav_cep.json")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--g-draws", type=int, default=16)
    ap.add_argument("--g-seeds", type=int, default=2)
    ap.add_argument("--engagements", default=",".join(ENGAGEMENTS))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args(argv)

    sn.warn_low_confidence()
    tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]
    md = gc.load_maps()
    out = {"tasks": tasks, "draws": args.draws, "seeds": args.seeds}
    t0 = time.time()
    with Pool(args.workers) as pool:
        if "c" in tasks:
            print("Task G addendum -- coast drift during an outage", flush=True)
            out["task_c"] = task_c(md)
        if "d" in tasks:
            print("Task D -- the warm start", flush=True)
            out["task_d"] = task_d(pool, md)
        if "f" in tasks:
            print("Task F -- the navigation contribution to CEP", flush=True)
            out["task_f"] = task_f(
                pool, md, n=args.draws, seeds=args.seeds,
                engagements=tuple(s for s in args.engagements.split(",") if s))
        if "g" in tasks:
            print("Task G -- degradation", flush=True)
            out["task_g"] = task_g(pool, md, n=args.g_draws,
                                   seeds=args.g_seeds)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
