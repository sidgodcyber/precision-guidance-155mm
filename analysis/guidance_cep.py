"""
Closed-loop CEP: schedulers, aim-off, the degradation ladder, and the firing
table.

Step 3, Tasks C to G. docs/GUIDANCE-DESIGN.md, docs/AIM-OFF.md,
docs/DEGRADATION-LADDER.md, docs/CEP-CLOSED-LOOP.md.

HOW A MISS IS REALISED
----------------------
Not by moving the target and leaving the round alone. A drawn miss is flown:
the muzzle velocity is perturbed until the round's own ballistic range moves by
the drawn range error, and the firing azimuth is rotated until the deflection
moves by the drawn deflection error. That costs one bisection on the
reduced-order model per draw and it buys two things a target-shifting scheme
cannot have -- the deployment state, the deployment Mach and therefore the
reachable set vary with the draw the way they really would, and the map the
flight computer carries is the NOMINAL engagement's map, so every run is also a
test of using one table off-nominal.

Muzzle velocity rather than quadrant elevation because it is the dominant
physical cause of range dispersion in a real gun, and because it moves the
whole trajectory rather than just its shape.

THE FUZE IS SET BEFORE LAUNCH
-----------------------------
Deployment time is the NOMINAL engagement's, not a quarter of the perturbed
round's own apogee time. A fuze is programmed from the fire-control solution
before firing; it cannot know that this particular round is 4 m/s fast.

WHAT IS AND IS NOT IN THESE NUMBERS
-----------------------------------
IN:   the servo, the staged deployment and its dead time, the reachable set as
      the 6-DOF produces it, impact-point-prediction error from the
      reduced-order model, the guidance law's own decisions, and the
      off-nominal use of a nominal map.
OUT:  navigation error (step 5 -- guidance reads the true state), and
      atmospheric-knowledge, aerodynamic-uncertainty and deployment-time error
      (step 6). Every CEP here is therefore an underestimate, and
      docs/CEP-CLOSED-LOOP.md says by roughly how much.

Run:  python -m analysis.guidance_cep --tasks c,d,e,f,g
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
from sim import dynamics as dyn, integrate as ig, projectile as pr
from gnc import roll_control as rc
from gnc import guidance as gd
from gnc.inverse_map import AuthorityMap, ellipse_of
from gnc import scheduler as sch
from analysis import roll_servo as rs
from analysis import guidance_authority as ga
from analysis import cep_projection as cp

MAP_PATH = "docs/guidance_map.json"

#: Draws per case.
#:
#: Sixty-four, not more, and the reason is measured rather than chosen: the
#: development machine is a four-core 15 W laptop and one closed-loop 6-DOF
#: trajectory at the long engagement costs about 6.7 s of wall clock at
#: eight-way parallelism, so the campaigns below are a little under three
#: hours as they stand.
#:
#: What that costs in precision is stated rather than hidden. The CEP is a
#: median over a stratified sample (see `draws`); at n = 64 its bootstrap
#: standard error is of order a tenth of the CEP, which is NOT enough to
#: separate two laws differing by 10 %. The comparison between laws is
#: therefore made PAIRWISE on the same draws (`paired_delta`), where the
#: common dispersion cancels and the difference is resolved an order of
#: magnitude better than either absolute value.
N_DRAWS = 64
N_DRAWS_SUB = 24

#: Uncorrected dispersion. The brief gives an uncorrected miss of 200-300 m at
#: the long engagement; 200 m is the value docs/ARCHITECTURE-DECISION.md
#: section 2 states its authority requirement against, so it is the one
#: carried here.
#:
#: Two shapes. `circular` is what the published criterion assumed. Artillery
#: dispersion is range-dominated, so `range_dominated` at 3:1 is the physical
#: one and is the default; Task G turns on which of the two the reachable
#: ellipse contains, and that depends on exactly this ratio.
DISPERSIONS = {
    "range_dominated": {"cep_m": 200.0, "ratio": 3.0},
    "circular": {"cep_m": 200.0, "ratio": 1.0},
    # A dispersion the kit can actually cover, for the scheduler comparison.
    # At an uncorrected CEP of 200 m against 184 m of semi-major authority
    # almost every round is authority-limited: the law holds to impact because
    # the miss never falls below what is left, so no scheduler has a decision
    # to make and the comparison measures pointing alone. Halving the
    # dispersion puts most rounds inside the reachable set, which is the
    # regime the scheduling question actually lives in. Both are reported.
    "range_dominated_reachable": {"cep_m": 100.0, "ratio": 3.0},
}
DEFAULT_DISPERSION = "range_dominated"

#: Dispersion scales with range: a CEP of 200 m at 15.8 km is 1.26 % of range,
#: and the same fraction is applied at the shorter engagements. Quoting 200 m
#: at a 2 km shot would be 10 % of range and would not be artillery.
DISPERSION_REFERENCE_RANGE = 15839.5

#: The aim-off adopted before guidance existed
#: (docs/ARCHITECTURE-DECISION.md section 4.3). Task D tests it.
AIM_OFF_NOMINAL = 224.0
AIM_OFFS = (0.0, 100.0, 175.0, 224.0, 275.0, 350.0)


def default_aim_off(base: dict) -> float:
    """
    The aim-off the fire-control solution would actually use: this
    engagement's own kit drag bias, measured as the difference between the
    uncorrected round and the same round with the kit deployed and the nose
    free.

    224 m is that number for the long engagement and ONLY for the long
    engagement -- it is 1.4 % of 15.8 km, and applying it to a 2 km shot whose
    kit costs 10 m of range would lay the gun 214 m long for no reason. Every
    campaign except the Task D sweep, which is testing the published constant,
    uses this.
    """
    return base["uncorrected_range"] - base["free_range"]


def staged_aim_off(mapdata: dict, label: str) -> float:
    """
    The aim-off the ADOPTED configuration actually wants, measured.

    `default_aim_off` is the free-nose drag bias, which is the number
    docs/ARCHITECTURE-DECISION.md section 4.3 published and it is measured on a
    kit with all four panels out from deployment. The adopted kit does not fly
    that way: it holds the steering pair stowed for 2.4 s and then holds an
    angle for the rest of the flight, and both of those move the mean impact
    point. Read off the map:

        bias = (impact of a round that releases at deployment)
             + (the angle-independent part of holding, `c` at full hold)

    At the long engagement that is 136.4 + 50.4 = 186.7 m against the
    published 224 m — the steering pair's first 4 s of drag is worth 90 m of
    range on its own, and staging does not pay it.

    Every campaign in this module is flown at `default_aim_off`, which is the
    published number to within 3 m, so that the design is measured as
    published. Task D sweeps the aim-off and reports what this one is worth.
    """
    e = mapdata["engagements"][label]
    return -(e["reference"]["d_range_m"] + e["nodes"][-1]["centre_range_m"])

#: Fractions of the guided phase at which navigation is lost, for the
#: `degraded` rung of the ladder.
NAV_FAIL_FRACTIONS = (0.25, 0.60)


# ===========================================================================
# The miss distribution
# ===========================================================================
def sigmas_for(cep_m: float, ratio: float) -> tuple:
    """
    (sigma_range, sigma_deflection) with sigma_range / sigma_deflection =
    `ratio` and a median radial miss of `cep_m`, by the same quadrature
    `analysis.cep_projection` uses. Bisection on the common scale.
    """
    lo, hi = 1.0, 5000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        got = cp.uncorrected_cep(mid * ratio, mid, n=301)
        if got < cep_m:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return s * ratio, s


def _halton(i: int, b: int) -> float:
    f, r, k = 1.0, 0.0, i
    while k > 0:
        f /= b
        r += f * (k % b)
        k //= b
    return r


def draws(n: int, sigma_r: float, sigma_d: float) -> list:
    """
    `n` (range, deflection) miss vectors from the bivariate normal.

    A stratified Halton sequence through the inverse normal, not pseudo-random
    numbers. The quantity being estimated is a MEDIAN over a small sample, and
    a low-discrepancy sequence removes most of the sampling noise from it --
    which matters because these campaigns compare CEPs that differ by tens of
    per cent, and pseudo-random draws at n = 96 would put several per cent of
    noise on each. It is also exactly reproducible without carrying a seed.
    """
    from math import erf

    def inv_norm(p: float) -> float:
        # Acklam's rational approximation; |error| < 1.15e-9.
        a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
        b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01)
        c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
        d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00)
        pl, ph = 0.02425, 1 - 0.02425
        if p < pl:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > ph:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

    out = []
    for i in range(1, n + 1):
        p1 = (_halton(i, 2) + 0.5 / n) % 1.0
        p2 = (_halton(i, 3) + 0.5 / n) % 1.0
        out.append((sigma_r * inv_norm(min(max(p1, 1e-6), 1 - 1e-6)),
                    sigma_d * inv_norm(min(max(p2, 1e-6), 1 - 1e-6))))
    return out


def dispersion_for(base: dict, name: str = DEFAULT_DISPERSION) -> dict:
    d = DISPERSIONS[name]
    scale = base["uncorrected_range"] / DISPERSION_REFERENCE_RANGE
    sr, sd = sigmas_for(d["cep_m"] * scale, d["ratio"])
    return {"name": name, "cep_m": d["cep_m"] * scale, "ratio": d["ratio"],
            "sigma_range": sr, "sigma_deflection": sd, "range_scale": scale}


# ===========================================================================
# Realising a draw
# ===========================================================================
_MPMM_CACHE: dict = {}


def _mpmm_model():
    from models import mpmm
    from sim import aerodata
    m = _MPMM_CACHE.get("model")
    if m is None:
        m = mpmm.MpmmModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                           environment=rs.base_model().environment,
                           iterate_yaw=True)
        _MPMM_CACHE["model"] = m
    return m


def _mpmm_range(base: dict, dmv: float, dt: float = 0.2) -> float:
    """
    Ballistic range of a perturbed launch, from the reduced-order model.

    dt = 0.2 s: docs/MPMM-COMPUTE.md prices that at 5 cm of range on this
    engagement, four orders of magnitude below the miss being realised.
    """
    from models import mpmm
    launch = pr.LaunchConditions.from_mils(base["muzzle_velocity"] + dmv,
                                           base["qe_mils"])
    y0 = mpmm.initial_state(pr.M107, launch)
    return mpmm.propagate_to_impact(y0, _mpmm_model(), dt=dt).range_m


def perturbation_for(base: dict, d_range: float, d_defl: float,
                     tol: float = 0.25) -> dict:
    """
    Muzzle-velocity and azimuth offsets realising a drawn (range, deflection)
    miss.

    Range by a secant solve on the reduced-order model, which reproduces the
    6-DOF impact to 0.7 m on this engagement (docs/GUIDANCE-DESIGN.md section
    3), so the realised miss is right to well inside the CEP being measured.
    Secant rather than bisection because the range is smooth and nearly linear
    in muzzle velocity: it converges in three or four propagations against
    forty, and this runs once per draw per campaign.

    Deflection by rotating the firing azimuth, which is exact to the extent
    that a rotation of the launch rotates the impact.
    """
    key = ("r0", base["label"])
    r0 = _MPMM_CACHE.get(key)
    if r0 is None:
        r0 = _mpmm_range(base, 0.0)
        _MPMM_CACHE[key] = r0
    want = r0 + d_range
    if abs(d_range) < 1e-9:
        dmv, rr = 0.0, r0
    else:
        # Seed from the local sensitivity, then secant.
        key_s = ("dr_dmv", base["label"])
        s = _MPMM_CACHE.get(key_s)
        if s is None:
            s = (_mpmm_range(base, 2.0) - r0) / 2.0
            _MPMM_CACHE[key_s] = s
        x0, f0 = 0.0, r0 - want
        x1 = d_range / s
        f1 = _mpmm_range(base, x1) - want
        for _ in range(8):
            if abs(f1) < tol or f1 == f0:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            x0, f0 = x1, f1
            x1 = x2
            f1 = _mpmm_range(base, x1) - want
        dmv, rr = x1, want + f1
    daz = d_defl / max(rr, 1.0)
    return {"dmv": float(dmv), "daz": float(daz), "mpmm_range": float(rr),
            "range_residual_m": float(rr - want)}


# ===========================================================================
# One closed-loop trajectory
# ===========================================================================
def run_guided(args) -> dict:
    """
    Fly one round: perturbed launch, kit deployed at the nominal fuze time,
    guidance closed on the true state.
    """
    base = args["baseline"]
    amap = AuthorityMap.from_dict(args["map"])
    dt = args.get("dt", base["dt"])
    t_dep = base["deploy_time"]
    mode = args.get("mode", "full")

    # -- the round -------------------------------------------------------
    launch = pr.LaunchConditions.from_mils(
        base["muzzle_velocity"] + args.get("dmv", 0.0), base["qe_mils"],
        azimuth=args.get("daz", 0.0))
    y0 = dyn.initial_state(pr.M107, launch)
    # `impact_state` and `impact_time` are the state and time the run STOPPED
    # at, whatever stopped it, so for a run capped by `t_max` they are the
    # deployment state. Nothing is logged: this leg is flown, not inspected.
    pre = ig.integrate(y0, rs.base_model(), dt=dt, log_every=10 ** 9,
                       t_max=t_dep, stop_on_impact=False)
    y_dep = pre.impact_state.copy()
    t_dep_actual = float(pre.impact_time)

    # -- the kit ---------------------------------------------------------
    model = rs.make_plant(brake_max=ga.BRAKE)
    controller = rs._controller(model, rc.ControllerConfig())
    stage_cfg = rc.StagedDeploymentConfig(**ga.STAGED)
    deployment = rc.StagedDeployment(stage_cfg, t_dep_actual,
                                     controller=controller, plant=model)

    # -- the target ------------------------------------------------------
    aim_off = float(args.get("aim_off", default_aim_off(base)))
    aim_off_d = float(args.get("aim_off_deflection", 0.0))
    target = (base["uncorrected_range"] - aim_off,
              base["uncorrected_drift"] - aim_off_d)

    # -- the law ---------------------------------------------------------
    gcfg = gd.GuidanceConfig(
        rate_hz=args.get("rate_hz", 1.0),
        arm_delay_s=args.get("arm_delay_s", 3.0),
        mode=mode,
        nav_fail_after_s=args.get("nav_fail_after_s"),
        inhibit_threshold_m=args.get("inhibit_threshold_m"),
        authority_monitor=args.get("authority_monitor", True),
        monitor_window_s=args.get("monitor_window_s", 6.0),
        monitor_fraction=args.get("monitor_fraction", 0.30),
        monitor_floor_m=args.get("monitor_floor_m", 20.0),
        monitor_start_s=args.get("monitor_start_s", 0.0),
    )
    predictor = gd.ImpactPredictor(pr.M107, rs.base_model().environment,
                                   geometry=rs.GEOMETRY, dt=gcfg.predictor_dt,
                                   iterate_yaw=gcfg.iterate_yaw)
    scheduler = sch.make_scheduler(args.get("scheduler", "budget"),
                                   **args.get("scheduler_opts", {}))
    law_g = gd.GuidanceLaw(gcfg, amap, predictor, scheduler, target,
                           t_dep_actual)
    law_b = rc.BrakeLaw(controller, law_g.command, deploy_time=t_dep_actual,
                        deployment=deployment)

    # -- the mechanical failure ------------------------------------------
    # Stage 2 is commanded and does not happen. The release LAW still runs --
    # it fires, and the controller swaps to its four-panel plant belief,
    # exactly as it would if the electronics were healthy and the mechanism
    # were not -- but the panels never enter the flow.
    if mode == "stage2_fail":
        gate = _never
    else:
        gate = deployment.gate

    nose = replace(model.nose, deploy_time=t_dep_actual,
                   brake_command=law_b.brake_command, steering_gate=gate)
    guided = cn.guided_model(rs.base_model(), rs.GEOMETRY, nose)
    res = ig.integrate(y_dep, guided, dt=dt, log_every=rs.LOG_EVERY,
                       t_max=300.0, t_start=t_dep_actual,
                       step_hook=law_g.chain(law_b.sample))
    if law_b.samples == 0:
        raise RuntimeError("the controller never sampled: step_hook not wired")

    miss_r = res.range_m - target[0]
    miss_d = res.drift_m - target[1]
    hist = law_b.history()
    err = np.abs(hist["error"])
    out = {
        "engagement": base["label"], "mode": mode,
        "scheduler": args.get("scheduler", "budget"),
        "draw": args.get("draw"), "aim_off": aim_off,
        "aim_off_deflection": aim_off_d,
        "dmv": args.get("dmv", 0.0), "daz": args.get("daz", 0.0),
        "drawn_range_miss": args.get("drawn_range_miss"),
        "drawn_defl_miss": args.get("drawn_defl_miss"),
        "target_range": target[0], "target_defl": target[1],
        "range_m": res.range_m, "drift_m": res.drift_m, "tof_s": res.impact_time,
        "miss_range_m": miss_r, "miss_defl_m": miss_d,
        "miss_m": math.hypot(miss_r, miss_d),
        "max_total_aoa_deg": math.degrees(res.max_total_aoa),
        "stage2_delay_s": deployment.delay,
        "stage2_reason": deployment.release_reason,
        "acquisition_s": rs._acquisition_time(hist["t"], err, hist["holding"],
                                              t_dep_actual),
        "hold_start_s": None, "hold_end_s": None,
        **{f"g_{k}": v for k, v in law_g.summary().items()},
    }
    ht = np.asarray(law_g.log["t"])
    hh = np.asarray(law_g.log["holding"], dtype=bool)
    if hh.any():
        out["hold_start_s"] = float(ht[hh][0] - t_dep_actual)
        out["hold_end_s"] = float(ht[hh][-1] - t_dep_actual)
        out["hold_length_s"] = out["hold_end_s"] - out["hold_start_s"]
    if args.get("keep_log"):
        out["guidance_log"] = {k: list(v) for k, v in law_g.log.items()}
    return out


def _never(t: float) -> bool:
    """Module level so it survives pickling to a worker."""
    return False


# ===========================================================================
# Reduction
# ===========================================================================
def _bootstrap_median_se(v: np.ndarray, n_boot: int = 2000) -> float:
    """
    Standard error of the median, by resampling.

    Reported with every CEP in this module. The sample is 64 draws and the
    estimator is a median, so quoting a CEP to a tenth of a metre without its
    own uncertainty would be claiming a precision the campaign does not have.
    """
    if v.size < 3:
        return float("nan")
    rng = np.random.default_rng(20260905)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    return float(np.median(v[idx], axis=1).std(ddof=1))


def paired_delta(rows_a: list, rows_b: list, key: str = "miss_m") -> dict:
    """
    The PAIRED difference in residual miss between two laws flown on the same
    draws.

    Two CEPs measured on 64 draws each carry a standard error of order a tenth
    of the CEP, which cannot separate laws differing by less than that. But
    the draws are common, so the per-round difference removes the dispersion
    that both laws faced and resolves the comparison far better than the two
    absolute numbers do. Positive means `a` missed by more than `b`.
    """
    by_b = {r["draw"]: r for r in rows_b}
    d = [r[key] - by_b[r["draw"]][key] for r in rows_a if r["draw"] in by_b]
    if not d:
        return {"n": 0}
    v = np.array(d, dtype=float)
    return {
        "n": int(v.size),
        "median_delta_m": float(np.median(v)),
        "mean_delta_m": float(v.mean()),
        "se_median_m": _bootstrap_median_se(v),
        "fraction_worse": float((v > 0).mean()),
    }


def cep_of(rows: list, key: str = "miss_m") -> dict:
    v = np.array([r[key] for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    r = np.array([r["miss_range_m"] for r in rows], dtype=float)
    d = np.array([r["miss_defl_m"] for r in rows], dtype=float)
    # A round is authority-limited when the law never got to choose a release:
    # it held until impact because the miss was always larger than what was
    # left to spend. That is the regime most of these engagements sit in, and a
    # scheduler comparison that did not report it would be comparing laws on
    # rounds where no law has a decision to make.
    #
    # READ FROM THE SCHEDULER'S OWN PER-CYCLE NOTE, via
    # `gnc.guidance.GuidanceLaw.saturation`. Until step 6 this tested
    # `g_plan["mode"]`, a field only `SingleShotScheduler` and
    # `BudgetScheduler` write -- so every proportional, deadband and isotropic
    # run reported 0.0 whatever it did, and that zero was quoted. The rows
    # carry `g_ever_saturated` and `g_terminal_saturated` since step 6; older
    # JSON does not, and is reported as nan rather than as zero so that a
    # missing measurement cannot be mistaken for a measured absence.
    have = [r for r in rows if r.get("g_armed_cycles") is not None]
    if have:
        ever = float(np.mean([bool(r["g_ever_saturated"]) for r in have]))
        term = float(np.mean([bool(r["g_terminal_saturated"]) for r in have]))
        cyc = float(np.sum([r["g_saturated_cycles"] for r in have])
                    / max(np.sum([r["g_armed_cycles"] for r in have]), 1))
    else:
        ever = term = cyc = float("nan")
    return {
        "n": int(v.size),
        "cep_m": float(np.median(v)),
        "cep_se_m": _bootstrap_median_se(v),
        # Rounds that were saturated on their LAST armed cycle, i.e. held to
        # impact. This is the successor to the field that used to be called
        # `authority_limited`; the old name is not reused, because it was
        # quoted with a wrong value and a reader meeting it again would have
        # no way to tell which measurement they were looking at.
        "authority_limited_terminal": term,
        "authority_limited_ever": ever,
        "saturated_cycle_fraction": cyc,
        "mean_miss_m": float(v.mean()),
        "p90_m": float(np.percentile(v, 90)),
        "max_m": float(v.max()),
        "within_30m": float((v <= 30.0).mean()),
        "within_50m": float((v <= 50.0).mean()),
        "within_100m": float((v <= 100.0).mean()),
        "bias_range_m": float(r.mean()),
        "bias_defl_m": float(d.mean()),
        "sd_range_m": float(r.std(ddof=1)) if v.size > 1 else 0.0,
        "sd_defl_m": float(d.std(ddof=1)) if v.size > 1 else 0.0,
    }


def _imap(pool, fn, cases, label):
    t0 = time.time()
    out = []
    for i, r in enumerate(pool.imap_unordered(fn, cases, chunksize=1), 1):
        out.append(r)
        if i % 24 == 0 or i == len(cases):
            print(f"  {label}: {i}/{len(cases)}  {time.time() - t0:7.1f} s",
                  flush=True)
    return out


def load_maps(path: str = MAP_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


def engagement_context(mapdata: dict, label: str,
                       dispersion: str = DEFAULT_DISPERSION) -> dict:
    e = mapdata["engagements"][label]
    base = dict(e["baseline"])
    base["y_deploy"] = e["y_deploy"]
    disp = dispersion_for(base, dispersion)
    return {"base": base, "map": e["map"], "dispersion": disp,
            "full_hold_ellipse": e["full_hold_ellipse"],
            "monitor": monitor_options(mapdata, label)}


def make_draws(ctx: dict, n: int) -> list:
    d = ctx["dispersion"]
    out = []
    for i, (dr, dd) in enumerate(draws(n, d["sigma_range"],
                                       d["sigma_deflection"])):
        p = perturbation_for(ctx["base"], dr, dd)
        out.append({"draw": i, "drawn_range_miss": dr, "drawn_defl_miss": dd,
                    **p})
    return out


def cases_for(ctx: dict, dr: list, **kw) -> list:
    mon = {k: v for k, v in ctx.get("monitor", {}).items()
           if not k.startswith("_")}
    return [{"baseline": ctx["base"], "map": ctx["map"], **mon,
             "dmv": d["dmv"], "daz": d["daz"], "draw": d["draw"],
             "drawn_range_miss": d["drawn_range_miss"],
             "drawn_defl_miss": d["drawn_defl_miss"], **kw}
            for d in dr]


# ===========================================================================
# Sizing the two thresholds that are not free parameters
# ===========================================================================
def mean_rotation_deg(mapdata: dict, label: str) -> float:
    """
    The map's mean rotation between commanded angle and delivered direction.

    `IsotropicScheduler` needs one number to stand in for the whole map, and
    this is the fairest one to give it: the average of the rotation it would
    have looked up. Giving it a worse number would make the comparison a straw
    man.
    """
    rows = mapdata["engagements"][label]["pointing"]
    s = np.array([math.sin(math.radians(r["rotation_deg"])) for r in rows])
    c = np.array([math.cos(math.radians(r["rotation_deg"])) for r in rows])
    return math.degrees(math.atan2(s.mean(), c.mean()))


def deadband_from_prediction(mapdata: dict, label: str,
                             t_go_window=(0.35, 0.80)) -> float:
    """
    Size the deadband on the MEASURED impact-point-prediction error, not on a
    CEP target.

    The band exists so the law does not steer against a residual it cannot
    resolve. The relevant error is the one carried over the middle of the
    guided phase, where the release decision is made; `t_go_window` is that
    span as a fraction of the guided phase.
    """
    e = mapdata["engagements"][label]
    rows = e["prediction_error"]["rows"]
    guided = e["baseline"]["guided_phase_s"]
    sel = [r["rms_m"] for r in rows
           if t_go_window[0] * guided <= r["t_end_offset"] <= t_go_window[1] * guided]
    if not sel:
        sel = [r["rms_m"] for r in rows]
    return float(np.median(sel))


def scheduler_options(mapdata: dict, label: str) -> dict:
    return {
        "proportional": {},
        "deadband": {"deadband_m": deadband_from_prediction(mapdata, label)},
        "budget": {"reserve": 0.25, "endgame_t_go":
                   0.20 * mapdata["engagements"][label]["baseline"]["guided_phase_s"]},
        "single_shot": {},
        "isotropic": {"rotation_deg": mean_rotation_deg(mapdata, label)},
    }


def monitor_options(mapdata: dict, label: str, window_s: float = 6.0,
                    sigmas: float = 4.0) -> dict:
    """
    Size the authority monitor from the MEASURED prediction error.

    The monitor compares how far the predicted impact point moved over a
    window against how far the map said it would. The noise on that
    comparison is the change in the prediction error over the window, of order
    sqrt(2) times its rms. Declaring a fault when the map expected less
    movement than a few times that noise would be declaring a fault on the
    predictor, so the test is simply not run below `sigmas` times it.

    `monitor_fraction` stays at 0.30: a hold delivering less than 30 % of what
    the map promised is not a working actuator on any reading, and the number
    is not swept against CEP.
    """
    e = mapdata["engagements"][label]
    guided = e["baseline"]["guided_phase_s"]
    rows = e["prediction_error"]["rows"]
    sel = [r["rms_m"] for r in rows if 0.2 * guided <= r["t_end_offset"] <= 0.9 * guided]
    rms = float(np.median(sel)) if sel else 10.0
    floor = sigmas * math.sqrt(2.0) * rms
    # And the start: the first time the MEASURED prediction rms has fallen to
    # half the floor, which is where the predictor's own drift over a window
    # stops being comparable to the movement the window is testing for. Capped
    # at half the guided phase so a short engagement still gets a window.
    start = 0.0
    for r in rows:
        if r["rms_m"] <= 0.5 * floor:
            start = r["t_end_offset"]
            break
    start = min(start, 0.5 * guided)
    return {"monitor_window_s": window_s, "monitor_fraction": 0.30,
            "monitor_floor_m": floor, "monitor_start_s": start,
            "_prediction_rms_m": rms}


# ===========================================================================
# Tasks
# ===========================================================================
def task_c(pool, mapdata: dict, label: str = "long", n: int = N_DRAWS,
           dispersion: str = DEFAULT_DISPERSION,
           rate_sweep: bool = True) -> dict:
    """The scheduler comparison, and the guidance-rate justification."""
    ctx = engagement_context(mapdata, label, dispersion)
    dr = make_draws(ctx, n)
    opts = scheduler_options(mapdata, label)
    out = {"engagement": label, "dispersion": ctx["dispersion"],
           "n_draws": n, "scheduler_options": opts, "schedulers": {},
           "aim_off_m": default_aim_off(ctx["base"]),
           "monitor": ctx["monitor"], "rate_sweep": {}}
    for name in ("proportional", "deadband", "budget", "single_shot",
                 "isotropic"):
        rows = _imap(pool, run_guided,
                     cases_for(ctx, dr, scheduler=name,
                               scheduler_opts=opts[name]),
                     f"C {name}")
        out["schedulers"][name] = {"cep": cep_of(rows), "rows": rows}
        print(f"    {name:14s} CEP {out['schedulers'][name]['cep']['cep_m']:7.1f} m "
              f"  P(<=30 m) {100 * out['schedulers'][name]['cep']['within_30m']:5.1f} %",
              flush=True)

    best = min(out["schedulers"], key=lambda k: out["schedulers"][k]["cep"]["cep_m"])
    out["best"] = best
    out["never_released"] = sum(
        1 for r in out["schedulers"][best]["rows"]
        if r.get("g_release_time") is None) / max(n, 1)
    print(f"    rounds that never released: "
          f"{100 * out['never_released']:.0f} %", flush=True)
    out["paired_vs_best"] = {
        name: paired_delta(r["rows"], out["schedulers"][best]["rows"])
        for name, r in out["schedulers"].items() if name != best}
    for name, d in out["paired_vs_best"].items():
        print(f"    {name:14s} vs {best}: paired median "
              f"{d['median_delta_m']:+7.1f} +/- {d['se_median_m']:.1f} m, "
              f"worse on {100 * d['fraction_worse']:.0f} % of rounds", flush=True)
    # -- the rate justification ------------------------------------------
    # 1.0 Hz is the main campaign above and is not re-flown. `task_r` is the
    # resolved version of this sweep; this one is kept only so `task_c` is
    # self-contained.
    sub = dr[:N_DRAWS_SUB] if rate_sweep else []
    for hz in ((0.5, 2.0, 5.0) if rate_sweep else ()):
        rows = _imap(pool, run_guided,
                     cases_for(ctx, sub, scheduler=best,
                               scheduler_opts=opts[best], rate_hz=hz),
                     f"C rate {hz} Hz")
        out["rate_sweep"][str(hz)] = {
            "cep": cep_of(rows), "rows": rows,
            "paired_vs_1hz": paired_delta(
                rows, [r for r in out["schedulers"][best]["rows"]
                       if r["draw"] < len(sub)]),
            "predictor_calls": float(np.mean(
                [r["g_predictor_calls"] for r in rows])),
            "derivative_calls": float(np.mean(
                [r["g_predictor_derivative_calls"] for r in rows]))}
        print(f"    {hz:4.1f} Hz     CEP {out['rate_sweep'][str(hz)]['cep']['cep_m']:7.1f} m",
              flush=True)
    return out


def task_r(pool, mapdata: dict, label: str, scheduler: str, opts: dict,
           n: int = 32, rates=(0.5, 2.0, 5.0)) -> dict:
    """
    The guidance-rate justification, paired.

    Split out of Task C because the comparison it has to support is between
    rates that differ by a few metres of CEP, and two medians over 24 draws
    each carry more than twenty metres of standard error. The draws are common
    to every rate, so the paired difference resolves it; the absolute CEPs are
    reported beside it and are NOT the evidence.
    """
    ctx = engagement_context(mapdata, label)
    dr = make_draws(ctx, n)
    out = {"engagement": label, "scheduler": scheduler, "n_draws": n,
           "rates": {}}
    base = _imap(pool, run_guided,
                 cases_for(ctx, dr, scheduler=scheduler, scheduler_opts=opts,
                           rate_hz=1.0), "R 1.0 Hz")
    out["rates"]["1.0"] = {"cep": cep_of(base), "rows": base,
                           "predictor_calls": float(np.mean(
                               [r["g_predictor_calls"] for r in base])),
                           "derivative_calls": float(np.mean(
                               [r["g_predictor_derivative_calls"]
                                for r in base]))}
    print(f"     1.0 Hz  CEP {out['rates']['1.0']['cep']['cep_m']:7.1f} m",
          flush=True)
    for hz in rates:
        rows = _imap(pool, run_guided,
                     cases_for(ctx, dr, scheduler=scheduler,
                               scheduler_opts=opts, rate_hz=hz),
                     f"R {hz} Hz")
        d = paired_delta(rows, base)
        out["rates"][str(hz)] = {
            "cep": cep_of(rows), "rows": rows, "paired_vs_1hz": d,
            "predictor_calls": float(np.mean(
                [r["g_predictor_calls"] for r in rows])),
            "derivative_calls": float(np.mean(
                [r["g_predictor_derivative_calls"] for r in rows]))}
        print(f"    {hz:5.1f} Hz  CEP {out['rates'][str(hz)]['cep']['cep_m']:7.1f} m"
              f"  paired {d['median_delta_m']:+6.2f} +/- {d['se_median_m']:.2f} m"
              f"  worse on {100 * d['fraction_worse']:.0f} %", flush=True)
    return out


def task_d(pool, mapdata: dict, label: str, scheduler: str, opts: dict,
           n: int = N_DRAWS) -> dict:
    """The aim-off sweep, with the loop closed."""
    ctx = engagement_context(mapdata, label)
    dr = make_draws(ctx, n)
    out = {"engagement": label, "scheduler": scheduler, "n_draws": n,
           "dispersion": ctx["dispersion"], "offsets": {}}
    for a in AIM_OFFS:
        rows = _imap(pool, run_guided,
                     cases_for(ctx, dr, scheduler=scheduler,
                               scheduler_opts=opts, aim_off=a),
                     f"D aim-off {a:.0f} m")
        out["offsets"][str(a)] = {"cep": cep_of(rows), "rows": rows}
        print(f"    aim-off {a:6.1f} m  CEP {out['offsets'][str(a)]['cep']['cep_m']:7.1f} m",
              flush=True)

    best = min(out["offsets"], key=lambda k: out["offsets"][k]["cep"]["cep_m"])
    out["best_offset_m"] = float(best)

    # -- the deflection half of the aim-off ------------------------------
    # The published aim-off is range only, because the drag bias it was set
    # against is range only. The kit also moves the impact SIDEWAYS -- it costs
    # deflection as well as range -- so there is a systematic cross-range
    # residual the range aim-off cannot touch. It is read off the runs rather
    # than modelled, and then flown.
    out["two_d"] = two_d_aimoff(pool, ctx, dr, scheduler, opts, out)
    return out


def two_d_aimoff(pool, ctx: dict, dr: list, scheduler: str, opts: dict,
                 d: dict) -> dict:
    """
    The deflection half of the aim-off.

    The published aim-off is range only, because the drag bias it was set
    against is range only. The kit also moves the impact SIDEWAYS -- it costs
    deflection as well as range -- so there is a systematic cross-range
    residual the range aim-off cannot touch. It is read off the runs rather
    than modelled, and then flown.

    SIGN. `bias_defl_m` is the mean of (impact - target). If the rounds land
    LEFT of the target the target must move LEFT with them, and the target is
    `uncorrected_drift - aim_off_deflection`, so the offset is MINUS the bias.
    Getting this backwards doubles the residual instead of removing it, which
    is what the first pass did: 41.3 m against 31.2 m for the range-only
    offset it was supposed to improve on.
    """
    best = min(d["offsets"], key=lambda k: d["offsets"][k]["cep"]["cep_m"])
    dbias = d["offsets"][best]["cep"]["bias_defl_m"]
    aod = -dbias
    rows = _imap(pool, run_guided,
                 cases_for(ctx, dr, scheduler=scheduler, scheduler_opts=opts,
                           aim_off=float(best), aim_off_deflection=aod),
                 f"D 2-D aim-off ({aod:+.1f} m)")
    out = {"aim_off_range_m": float(best), "aim_off_deflection_m": aod,
           "measured_defl_bias_m": dbias, "cep": cep_of(rows), "rows": rows}
    print(f"    2-D aim-off ({best} m, {aod:+.1f} m)  CEP "
          f"{out['cep']['cep_m']:7.1f} m", flush=True)
    return out


def task_e(pool, mapdata: dict, label: str, scheduler: str, opts: dict,
           n: int = N_DRAWS, reuse_full: list = None) -> dict:
    """The degradation ladder."""
    ctx = engagement_context(mapdata, label)
    dr = make_draws(ctx, n)
    guided = ctx["base"]["guided_phase_s"]
    a = ctx["full_hold_ellipse"]["semi_major_m"]
    out = {"engagement": label, "scheduler": scheduler, "n_draws": n,
           "dispersion": ctx["dispersion"], "rungs": {},
           "aim_off_m": default_aim_off(ctx["base"]),
           "monitor": ctx["monitor"], "inhibit_threshold_m": 1.5 * a}

    def run(tag, reuse=None, **kw):
        # `full` is the same campaign Task C already flew for the winning law.
        # Re-flying 64 identical trajectories to fill in a row of a table
        # would be an hour of this machine for no information.
        rows = reuse if reuse is not None else _imap(
            pool, run_guided,
            cases_for(ctx, dr, scheduler=scheduler, scheduler_opts=opts, **kw),
            f"E {tag}")
        # The monitor's false-alarm rate is not optional information: it fires
        # by releasing the actuator for the rest of the flight, so a false
        # positive on a healthy round costs the whole correction.
        faults = sum(1 for r in rows if r["g_authority_fault_time"] is not None)
        out["rungs"][tag] = {"cep": cep_of(rows), "rows": rows,
                             "authority_faults": faults,
                             "authority_fault_rate": faults / max(len(rows), 1),
                             "kw": {k: v for k, v in kw.items()}}
        c = out["rungs"][tag]["cep"]
        print(f"    {tag:24s} CEP {c['cep_m']:7.1f} m  P(<=30 m) "
              f"{100 * c['within_30m']:5.1f} %  monitor fired {faults}/{len(rows)}",
              flush=True)
        return rows

    run("full", reuse=reuse_full)
    for f in NAV_FAIL_FRACTIONS:
        run(f"degraded_{int(100 * f):02d}pc", mode="degraded",
            nav_fail_after_s=f * guided)
    run("reversionary", mode="reversionary")
    run("inhibit", inhibit_threshold_m=1.5 * a)
    rows = run("stage2_fail", mode="stage2_fail")
    det = [r for r in rows if r["g_authority_fault_time"] is not None]
    out["stage2_detection"] = {
        "detected": len(det), "of": len(rows),
        "rate": len(det) / max(len(rows), 1),
        "median_detect_s_after_deploy": float(np.median(
            [r["g_authority_fault_time"] - ctx["base"]["deploy_time"]
             for r in det])) if det else None,
        "median_hold_length_s": float(np.median(
            [r["hold_length_s"] for r in rows if r.get("hold_length_s")]))
        if any(r.get("hold_length_s") for r in rows) else None,
    }
    print(f"    stage-2 fault detected in {len(det)}/{len(rows)} rounds",
          flush=True)
    # The same failure with the monitor switched off, to price the monitor.
    rows_off = _imap(pool, run_guided,
                     cases_for(ctx, dr, scheduler=scheduler,
                               scheduler_opts=opts, mode="stage2_fail",
                               authority_monitor=False),
                     "E stage2_fail, monitor off")
    out["rungs"]["stage2_fail_monitor_off"] = {"cep": cep_of(rows_off),
                                               "rows": rows_off}
    return out


def task_e2(pool, mapdata: dict, label: str, scheduler: str, opts: dict,
            n: int = N_DRAWS) -> dict:
    """
    The two rungs the authority monitor decides, re-flown with the monitor's
    start time sized from the measured prediction error.

    Task E's first pass opened the monitor's first window as soon as the law
    armed, and detected the stage-2 mechanical failure on 69 % of rounds: on
    the rest the predictor's own transient moved the predicted impact point far
    enough, in the right direction, to look like a working actuator. The floor
    guards against the prediction error's SIZE; `monitor_start_s` guards
    against its RATE OF CHANGE, and this measures what that is worth. Both
    rungs are re-flown -- the failure to see whether detection improves, and
    the healthy one to confirm the change does not buy it with false alarms.
    """
    ctx = engagement_context(mapdata, label)
    dr = make_draws(ctx, n)
    out = {"engagement": label, "n_draws": n, "monitor": ctx["monitor"],
           "rungs": {}}
    for tag, kw in (("full", {}), ("stage2_fail", {"mode": "stage2_fail"})):
        rows = _imap(pool, run_guided,
                     cases_for(ctx, dr, scheduler=scheduler,
                               scheduler_opts=opts, **kw), f"E2 {tag}")
        det = [r for r in rows if r["g_authority_fault_time"] is not None]
        out["rungs"][tag] = {
            "cep": cep_of(rows), "rows": rows,
            "authority_faults": len(det),
            "authority_fault_rate": len(det) / max(len(rows), 1),
            "median_detect_s_after_deploy": float(np.median(
                [r["g_authority_fault_time"] - ctx["base"]["deploy_time"]
                 for r in det])) if det else None}
        print(f"    {tag:16s} CEP {out['rungs'][tag]['cep']['cep_m']:7.1f} m  "
              f"monitor fired {len(det)}/{len(rows)}", flush=True)
    return out


def task_f(pool, mapdata: dict, scheduler: str, n: int = N_DRAWS,
           aim_off: str = "published") -> dict:
    """
    CEP across the firing table.

    `aim_off`: "published" lays the gun by each engagement's free-nose drag
    bias, which is what docs/ARCHITECTURE-DECISION.md section 4.3 specified;
    "staged" lays it by the adopted configuration's measured mean impact
    (`staged_aim_off`). Both are run, because the difference is 40 m at the
    long engagement and pretending the published number is the right one would
    understate the design.
    """
    out = {"scheduler": scheduler, "n_draws": n, "aim_off_rule": aim_off,
           "engagements": {}}
    for lbl in mapdata["engagements"]:
        ctx = engagement_context(mapdata, lbl)
        opts = scheduler_options(mapdata, lbl)[scheduler]
        dr = make_draws(ctx, n)
        extra = ({} if aim_off == "published"
                 else {"aim_off": staged_aim_off(mapdata, lbl)})
        rows = _imap(pool, run_guided,
                     cases_for(ctx, dr, scheduler=scheduler,
                               scheduler_opts=opts, **extra), f"F {lbl}")
        # The unguided reference is a ballistic round with the kit fitted and
        # neutral: no guidance decisions, so its spread needs fewer draws than
        # the guided case to be resolved.
        base_rows = _imap(pool, run_guided,
                          cases_for(ctx, dr[:max(n // 2, 8)],
                                    scheduler=scheduler, scheduler_opts=opts,
                                    mode="reversionary", **extra),
                          f"F {lbl} unguided")
        out["engagements"][lbl] = {
            "aim_off_used_m": (extra.get("aim_off")
                               if extra else default_aim_off(ctx["base"])),
            "cep": cep_of(rows), "unguided_cep": cep_of(base_rows),
            "dispersion": ctx["dispersion"],
            "aim_off_m": default_aim_off(ctx["base"]),
            "monitor": ctx["monitor"],
            "ellipse": ctx["full_hold_ellipse"],
            "baseline": {k: ctx["base"][k] for k in
                         ("uncorrected_range", "uncorrected_tof",
                          "deploy_time", "guided_phase_s", "mach_at_deploy",
                          "qbar_at_deploy", "free_range")},
            "rows": rows, "unguided_rows": base_rows,
        }
        c = out["engagements"][lbl]
        print(f"    {lbl:8s} CEP {c['cep']['cep_m']:7.1f} m against unguided "
              f"{c['unguided_cep']['cep_m']:7.1f} m", flush=True)
    return out


def task_g(pool, mapdata: dict, label: str, scheduler: str, opts: dict,
           n: int = N_DRAWS) -> dict:
    """
    Containment, and the closed loop against the published criterion.

    The analytic half is free; the flown half is the same campaign as Task C
    run against the CIRCULAR dispersion the published criterion assumed, so
    the two shapes can be compared at equal CEP.
    """
    from gnc.inverse_map import ellipse_contains_fraction, containment_margin
    out = {"engagement": label, "containment": [], "circular": None}
    e = mapdata["engagements"][label]
    base = dict(e["baseline"])
    base["y_deploy"] = e["y_deploy"]
    ell = e["full_hold_ellipse"]
    ideal = e["ideal"]["ellipse"]
    # Reachable-set centre relative to the uncorrected impact: the map's own
    # `c` at full hold, plus the reference run's offset.
    ref = e["reference"]
    node = sorted(e["map"]["nodes"], key=lambda nd: nd["t_go"])[0]
    bias = (node["c"][0] + ref["d_range_m"], node["c"][1] + ref["d_deflection_m"])
    out["reachable_centre_bias"] = {"range_m": bias[0], "deflection_m": bias[1]}

    for dname in DISPERSIONS:
        disp = dispersion_for(base, dname)
        for tag, E in (("staged_closed_loop", ell), ("ideal_hold", ideal)):
            p = ellipse_contains_fraction(
                E["semi_major_m"], E["semi_minor_m"], E["major_axis_tilt_deg"],
                disp["sigma_range"], disp["sigma_deflection"], bias=bias,
                aim_offset=(-bias[0], -bias[1]))
            m = containment_margin(
                E["semi_major_m"], E["semi_minor_m"], E["major_axis_tilt_deg"],
                disp["sigma_range"], disp["sigma_deflection"], bias=bias,
                aim_offset=(-bias[0], -bias[1]))
            # The scalar test the project published, for comparison.
            need = cp.required_authority(disp["sigma_range"],
                                         disp["sigma_deflection"], 30.0,
                                         axis_ratio=E["axis_ratio"],
                                         bias_range=bias[0])
            out["containment"].append({
                "dispersion": dname, "set": tag,
                "sigma_range": disp["sigma_range"],
                "sigma_deflection": disp["sigma_deflection"],
                "semi_major_m": E["semi_major_m"],
                "semi_minor_m": E["semi_minor_m"],
                "axis_ratio": E["axis_ratio"],
                "p_contains": p, "scale_for_50pc": m,
                "scalar_required_a_m": need,
                "scalar_passes": E["semi_major_m"] >= need,
            })

    ctx = engagement_context(mapdata, label, dispersion="circular")
    dr = make_draws(ctx, n)
    rows = _imap(pool, run_guided,
                 cases_for(ctx, dr, scheduler=scheduler, scheduler_opts=opts),
                 "G circular")
    out["circular"] = {"cep": cep_of(rows), "dispersion": ctx["dispersion"],
                       "rows": rows}
    print(f"    circular dispersion CEP {out['circular']['cep']['cep_m']:.1f} m",
          flush=True)
    return out


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="c,d,e,f,g")
    ap.add_argument("--map", default=MAP_PATH)
    ap.add_argument("--engagement", default="long")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="docs/guidance_cep.json")
    ap.add_argument("--force", action="store_true",
                    help="recompute tasks already present in --out")
    args = ap.parse_args(argv)
    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]

    mapdata = load_maps(args.map)
    lbl = args.engagement
    t0 = time.time()
    out = {"map": args.map, "engagement": lbl, "n_draws": args.draws,
           "aim_off_nominal": AIM_OFF_NOMINAL}
    # Each task is tens of minutes of 6-DOF trajectories. The file is written
    # after every one and a re-run skips what is already in it, so an
    # interrupted campaign resumes rather than restarting. `--force` recomputes.
    if os.path.exists(args.out) and not args.force:
        try:
            with open(args.out) as fh:
                out.update(json.load(fh))
            done = [k[-1] for k in out if k.startswith("task_")]
            print(f"resuming from {args.out}: have tasks {sorted(done)}",
                  flush=True)
            tasks = [t for t in tasks if f"task_{t}" not in out]
        except (ValueError, OSError) as exc:
            print(f"could not read {args.out} ({exc}); starting fresh")

    def save():
        _write(args.out, out)
        print(f"  wrote {args.out} ({time.time() - t0:.1f} s elapsed)",
              flush=True)

    with Pool(args.workers) as pool:
        best = out.get("best_scheduler", "budget")
        opts_best = out.get("best_scheduler_options",
                            scheduler_options(mapdata, lbl)["budget"])
        if "c" in tasks:
            print("\n=== TASK C: schedulers, against a dispersion the kit "
                  "cannot cover ===", flush=True)
            out["task_c"] = task_c(pool, mapdata, lbl, args.draws,
                                   rate_sweep=False)
            save()
        if "c2" in tasks:
            print("\n=== TASK C.2: schedulers, against a dispersion the kit "
                  "CAN cover ===", flush=True)
            out["task_c2"] = task_c(pool, mapdata, lbl, 48,
                                    dispersion="range_dominated_reachable",
                                    rate_sweep=False)
            save()
        if "task_c" in out:
            best = out["task_c"]["best"]
            opts_best = out["task_c"]["scheduler_options"][best]
            print(f"  best by measured CEP: {best}", flush=True)
        out["best_scheduler"] = best
        out["best_scheduler_options"] = opts_best
        if "d" in tasks:
            print("\n=== TASK D: aim-off ===", flush=True)
            out["task_d"] = task_d(pool, mapdata, lbl, best, opts_best,
                                   args.draws)
            save()
        if "d2" in tasks and "task_d" in out:
            print("\n=== TASK D.2: the deflection half of the aim-off ===",
                  flush=True)
            ctx = engagement_context(mapdata, lbl)
            out["task_d"]["two_d"] = two_d_aimoff(
                pool, ctx, make_draws(ctx, args.draws), best, opts_best,
                out["task_d"])
            save()
        if "e" in tasks:
            print("\n=== TASK E: degradation ladder ===", flush=True)
            reuse = None
            if "task_c" in out and args.draws == N_DRAWS:
                reuse = out["task_c"]["schedulers"][best]["rows"]
            out["task_e"] = task_e(pool, mapdata, lbl, best, opts_best,
                                   args.draws, reuse_full=reuse)
            save()
        if "e2" in tasks:
            print("\n=== TASK E.2: the monitor, with its start time sized ===",
                  flush=True)
            out["task_e2"] = task_e2(pool, mapdata, lbl, best, opts_best,
                                     args.draws)
            save()
        if "f" in tasks:
            print("\n=== TASK F: the firing table, published aim-off ===",
                  flush=True)
            out["task_f"] = task_f(pool, mapdata, best, args.draws)
            save()
        if "f2" in tasks:
            print("\n=== TASK F.2: the firing table, staged aim-off ===",
                  flush=True)
            out["task_f2"] = task_f(pool, mapdata, best, args.draws,
                                    aim_off="staged")
            save()
        if "r" in tasks:
            print("\n=== TASK C.3: guidance rate, paired ===", flush=True)
            out["task_r"] = task_r(pool, mapdata, lbl, best, opts_best)
            save()
        if "g" in tasks:
            print("\n=== TASK G: containment ===", flush=True)
            out["task_g"] = task_g(pool, mapdata, lbl, best, opts_best,
                                   args.draws)
            save()

    _write(args.out, out)
    print(f"\nwrote {args.out}  ({time.time() - t0:.1f} s)")
    return 0


def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


if __name__ == "__main__":
    raise SystemExit(main())
