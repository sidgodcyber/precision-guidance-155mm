"""
Step 5.5: WHERE the 30.4 m navigation contribution lives.

docs/NAV-ERROR-DECOMPOSITION.md (Tasks A-D), docs/NAV-CHATTER.md (Task E).

WHAT THIS STEP EXISTS FOR
-------------------------
[NAV-CEP.md](../docs/NAV-CEP.md) measured the navigation contribution to the
impact point at 30.4 m 1 sigma in range at the adopted engagement, tested the
hypothesis that it was velocity error amplified over the time to go, and found
that hypothesis worth about a third of it. Twenty-four metres of the thirty-two
survived both identified mitigations and was attributed to nothing.

That is the project's largest single error term and it has no explanation.
This module decomposes it. REDUCING IT IS NOT THE OBJECTIVE and no mitigation
found here is adopted here.

TWO DECOMPOSITIONS, INDEPENDENTLY DERIVED, MADE TO AGREE
--------------------------------------------------------
The centre-of-pressure test of step 2.6 worked because a third, independently
derived quantity constrained the answer. The same structure is used here.

  Task A  the SENSITIVITY of the impact point to each state element, from the
          onboard predictor and the authority map, by central difference.
  Task B  the MEASURED 1 sigma error in each of those elements, from the
          step-5 filter in the closed loop, bias separated from noise.
  Task C  A x B, in quadrature where independent and with the covariance where
          not -- the predicted decomposition.
  Task D  each sensor error source zeroed in turn, end to end -- the measured
          decomposition, which shares no line of code with Task C.
  Task E  the hypothesis neither of them can see: that the error is not an
          estimate at all but the cost of an actuator chasing a moving command.
  Task F  the term neither decomposition was looking for, found by asking
          step 5's own saved rows how often the guidance law's authority
          monitor declared a healthy actuator dead. It flies nothing.

Where C and D disagree, one of them is wrong, and finding which is the point.
Here they disagreed completely, a third quantity -- the error in the PREDICTED
impact point, which neither of them measures -- decided between them, and what
survived the reconciliation was Task F.

THE TRANSFER FUNCTION IS NOT 1:1, AND WHICH ONE APPLIES IS A MEASUREMENT
-----------------------------------------------------------------------
There are two regimes and they have completely different sensitivities:

  RELEASING     the law nulls the predicted miss and then stops. The true
                impact lands where the prediction said the target was, so an
                impact-point PREDICTION error of `d` becomes a miss of `-d`.
                Task A's `predictor` Jacobian is the whole answer.
  AUTHORITY-    the law holds to impact whatever it predicts. The magnitude of
  LIMITED       the correction is then fixed by the physics and NOT by the
                estimate, and the only thing the estimate still sets is the
                DIRECTION the correction is applied in. A state error reaches
                the impact point through the commanded roll angle and through
                nothing else, and the predictor Jacobian badly overstates it.

`docs/NAV-CEP.md` reports the adopted engagement as authority-limited, so the
second is expected to dominate -- but `guidance_cep.cep_of` measures that from
a `g_plan` field only `SingleShotScheduler` populates, so the reported 0 % is
an artefact and not a measurement. `saturation_fraction` below counts the
scheduler's own per-cycle note instead.

Run:  python -m analysis.nav_ablation --tasks a,b,c,d,e,f

`--tasks f` alone flies nothing, takes a second, and produces the
largest single finding in the step: it asks step 5's own saved rows how
often the authority monitor declared a healthy actuator dead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np

from sim import frames
from gnc import guidance as gd, sensors as sn
from gnc.inverse_map import AuthorityMap
from analysis import guidance_cep as gc, nav_cep as ncep, nav_common as nc
from analysis import roll_servo as rs

ENGAGEMENT = "long"
#: Draws and sensor seeds per ablation variant. 24 x 2 = 48 paired rounds, the
#: same as `analysis.nav_drivers`, so the baseline here is directly comparable
#: with the 31.98 m that module measured.
N_DRAWS = 24
N_SEEDS = 2


# ===========================================================================
# Task A -- the sensitivity of the impact point to the state
# ===========================================================================
#
# ELEMENTS is deliberately larger than the set the reduced-order model reads.
# `models.mpmm.state_from_sixdof` takes position, velocity and the axial spin
# and DISCARDS the quaternion and the transverse rates, so the attitude
# sensitivity of the predictor should be identically zero. That is a
# prediction, and it is cheaper to measure it than to argue it -- so the
# attitude and the resolver states are perturbed too, and a zero in the table
# is a result rather than an omission.
#
#: (name, kind, index, step, unit)
ELEMENTS = (
    ("r_x", "state", 0, 2.0, "m"),
    ("r_y", "state", 1, 2.0, "m"),
    ("r_z", "state", 2, 2.0, "m"),
    ("v_x", "state", 3, 0.5, "m/s"),
    ("v_y", "state", 4, 0.5, "m/s"),
    ("v_z", "state", 5, 0.5, "m/s"),
    ("psi_n", "attitude", 0, 0.02, "rad"),
    ("psi_e", "attitude", 1, 0.02, "rad"),
    ("psi_d", "attitude", 2, 0.02, "rad"),
    ("p_body", "state", 10, 5.0, "rad/s"),
    ("phi_rel", "state", 13, 0.05, "rad"),
)

#: Which elements the filter actually estimates, in the order Task B reports
#: and Task C combines. The attitude triad collapses to one number here
#: because the only attitude error that reaches the impact point is the ROLL,
#: and it reaches it through the command rather than through the predictor.
CHANNELS = ("r_x", "r_y", "r_z", "v_x", "v_y", "v_z", "p_body", "roll")


def _quat_small(dpsi: np.ndarray) -> np.ndarray:
    """The quaternion of a small EARTH-frame rotation vector."""
    h = 0.5 * np.asarray(dpsi, dtype=float)
    return frames.quat_normalize(np.array([1.0, h[0], h[1], h[2]]))


def perturb(y: np.ndarray, spec: tuple, delta: float) -> np.ndarray:
    """
    One element of a 6-DOF-shaped state moved by `delta`.

    The attitude perturbation is applied on the LEFT, because the filter's own
    error parametrisation is `C_true = (I + [dpsi]x) C_est` -- an earth-frame
    rotation -- and Task B measures the error in exactly that parametrisation.
    Perturbing on the right would be differentiating with respect to a
    different quantity from the one being measured.
    """
    y2 = np.array(y, dtype=float).copy()
    kind, idx = spec[1], spec[2]
    if kind == "state":
        y2[idx] += delta
    elif kind == "attitude":
        d = np.zeros(3)
        d[idx] = delta
        y2[6:10] = frames.quat_normalize(
            frames.quat_multiply(_quat_small(d), np.asarray(y[6:10], float)))
    else:
        raise ValueError(kind)
    return y2


def predictor_jacobian(predictor: gd.ImpactPredictor, t: float,
                       y: np.ndarray, scale: float = 1.0) -> dict:
    """
    d(predicted impact) / d(state), by central difference.

    Metres of predicted range and deflection per unit of each state element,
    plus d(time of flight)/d(element), which is what sets the authority the
    map reports as still available and is therefore an error channel of its
    own.
    """
    out = {}
    for spec in ELEMENTS:
        name, h = spec[0], spec[3] * scale
        rp, dp, tp, _ = predictor.predict(t, perturb(y, spec, +h))
        rm, dm, tm, _ = predictor.predict(t, perturb(y, spec, -h))
        out[name] = {"d_range": (rp - rm) / (2 * h),
                     "d_defl": (dp - dm) / (2 * h),
                     "d_tof": (tp - tm) / (2 * h),
                     "step": h, "unit": spec[4]}
    return out


def _decide(predictor, amap: AuthorityMap, t: float, y: np.ndarray,
            target: np.ndarray) -> tuple:
    """
    Exactly what `ProportionalScheduler._decide` decides, given a state.

    Reproduced here rather than imported because the sensitivity wanted is of
    the DECISION to the state, and running the scheduler object would carry
    its latched history between perturbations.
    """
    rng, defl, tof, _ = predictor.predict(t, y)
    t_go = max(tof - t, 0.0)
    miss = target - np.array([rng, defl])
    tge, phi, got, sat = amap.hold_end_for(miss, t_go)
    return float(phi), float(tge), float(t_go), bool(sat), float(got), miss


def _delivered(amap: AuthorityMap, t_go_true: float, phi: float,
               t_go_end: float) -> np.ndarray:
    """
    Where the TRUE impact point moves to, given a command decided on an
    estimate.

    The map is evaluated at the TRUE time to go -- the round flies the
    trajectory it flies, not the one the filter believes -- while the angle
    and the release come from the estimate. That split is the whole content of
    the closed-loop sensitivity: the estimate sets the command, the truth sets
    what the command buys.
    """
    A, b = amap.increment(t_go_true, min(t_go_end, t_go_true))
    return A @ np.array([math.cos(phi), math.sin(phi)]) + b


def loop_jacobian(predictor, amap: AuthorityMap, t: float, y: np.ndarray,
                  target: np.ndarray, scale: float = 1.0) -> dict:
    """
    d(impact) / d(state) THROUGH THE COMMAND, which is the transfer that
    applies to an authority-limited round.

    The nominal decision is made on the true state; each perturbed decision is
    made on a perturbed state; both are then flown against the TRUE authority.
    The difference is what the estimate error costs, and unlike the predictor
    Jacobian it saturates: once the law is holding to impact, moving the
    predicted impact point further away buys no more correction and the
    sensitivity collapses onto the direction term alone.
    """
    _, tge0, t_go_true, _, _, _ = _decide(predictor, amap, t, y, target)
    out = {}
    for spec in ELEMENTS:
        name, h = spec[0], spec[3] * scale
        pp, tp, _, _, _, _ = _decide(predictor, amap, t, perturb(y, spec, +h),
                                     target)
        pm, tm, _, _, _, _ = _decide(predictor, amap, t, perturb(y, spec, -h),
                                     target)
        d = (_delivered(amap, t_go_true, pp, tp)
             - _delivered(amap, t_go_true, pm, tm)) / (2 * h)
        out[name] = {"d_range": float(d[0]), "d_defl": float(d[1]),
                     "step": h, "unit": spec[4]}
    # THE ROLL CHANNEL, WHICH NEITHER JACOBIAN ABOVE CONTAINS.
    #
    # The predictor cannot see attitude at all. The servo can: it drives the
    # nose to `phi_nose` as the filter reports it, so a roll estimate error
    # is a roll COMMAND error of the same size, and the correction is
    # delivered along the wrong bearing by exactly that angle. This is the
    # only path by which an attitude error reaches the impact point on this
    # kit, and it is an authority-map term rather than a predictor term.
    phi0, tge0b, t_go_true, _, _, _ = _decide(predictor, amap, t, y, target)
    h = 0.02
    d = (_delivered(amap, t_go_true, phi0 + h, tge0b)
         - _delivered(amap, t_go_true, phi0 - h, tge0b)) / (2 * h)
    out["roll"] = {"d_range": float(d[0]), "d_defl": float(d[1]),
                   "step": h, "unit": "rad"}
    return out


def task_a(md: dict, cycles: int, refine: bool, draws: list = None) -> dict:
    """
    Both Jacobians, at the states guidance actually runs at.

    The trajectory is a truth-fed round at the adopted engagement, and the
    sample times are its own guidance cycles, so "the states guidance actually
    runs at" is meant literally rather than as a nominal.

    TWO TRAJECTORIES, AND THE SECOND IS NOT A REFINEMENT. The nominal round
    is laid on its own aim-off and has 44 m of miss to remove out of a 250 m
    reachable set, so it never saturates and the loop simply nulls its
    prediction. Two thirds of ARMED CYCLES across the dispersion do saturate
    (`saturation_fraction`), and a saturated cycle has a different transfer
    function entirely: the magnitude of the correction is fixed by the
    physics, the estimate sets only its DIRECTION, and the sensitivity to a
    roll error grows with the size of the correction being pointed. Measuring
    the Jacobian on the nominal alone would report the sensitivity of the
    third of cycles that is not the problem.
    """
    ctx = gc.engagement_context(md, ENGAGEMENT)
    base = ctx["base"]
    amap = AuthorityMap.from_dict(ctx["map"])
    aim = gc.default_aim_off(base)
    target = np.array([base["uncorrected_range"] - aim, base["uncorrected_drift"]])

    predictor = gd.ImpactPredictor(nc.pr.M107, rs.base_model().environment,
                                   geometry=rs.GEOMETRY, dt=0.1,
                                   iterate_yaw=True)

    def sweep(label: str, dmv: float, daz: float) -> tuple:
        print(f"  flying the {label} trajectory (dmv {dmv:+.2f} m/s, "
              f"daz {daz:+.5f} rad) ...", flush=True)
        traj = nc.fly_truth(ENGAGEMENT, md, dmv=dmv, daz=daz)
        t_dep = traj["t_dep"]
        # The guidance grid: 1 Hz from deployment. Sampled `cycles` times
        # across the guided phase, because the whole point is that the
        # velocity sensitivity must decay as the time to go shrinks.
        tof = float(traj["t"][-1])
        times = np.linspace(t_dep + 3.0, tof - 4.0, cycles)
        y_at = _state_interpolator(traj)
        rows = []
        t0 = time.time()
        for i, t in enumerate(times):
            y = y_at(float(t))
            rng, defl, pred_tof, _ = predictor.predict(float(t), y)
            phi, tge, t_go, sat, got, miss = _decide(predictor, amap, float(t),
                                                     y, target)
            rows.append({
                "t": float(t), "t_since_deploy": float(t - t_dep),
                "t_go": float(t_go),
                "pred_range": float(rng), "pred_defl": float(defl),
                "miss_m": float(np.hypot(miss[0], miss[1])),
                "phi_deg": math.degrees(phi), "t_go_end": float(tge),
                "saturated": bool(sat), "reach_m": float(got),
                "predictor": predictor_jacobian(predictor, float(t), y),
                "loop": loop_jacobian(predictor, amap, float(t), y, target),
            })
            print(f"    {label} cycle {i + 1}/{cycles}  t_go {t_go:5.1f} s  "
                  f"miss {rows[-1]['miss_m']:6.1f} m  sat {sat}  "
                  f"{time.time() - t0:6.1f} s", flush=True)
        return rows, traj, t_dep, tof, y_at, times

    rows, traj, t_dep, tof, y_at, times = sweep("nominal", 0.0, 0.0)
    out = {"engagement": ENGAGEMENT, "t_dep": t_dep, "tof": tof,
           "target": [float(v) for v in target], "cycles": rows}

    # The dispersed case: the draw with the largest laid-in range miss, which
    # is the one the law has to spend its whole budget on.
    if draws:
        d = max(draws, key=lambda x: abs(x.get("drawn_range_miss", 0.0)))
        rows_d, *_ = sweep("dispersed", d.get("dmv", 0.0), d.get("daz", 0.0))
        out["dispersed"] = {"draw": d.get("draw"),
                            "drawn_range_miss": d.get("drawn_range_miss"),
                            "dmv": d.get("dmv"), "daz": d.get("daz"),
                            "cycles": rows_d}

    if refine:
        # THE STEP-SIZE CHECK. A central difference of a propagation whose
        # impact is found by interpolating one RK4 step is smooth, but that is
        # an argument and this is a measurement: halve every step and report
        # what moves. Anything that moves by more than a few per cent is a
        # differencing artefact and not a sensitivity.
        t = float(times[len(times) // 2])
        y = y_at(t)
        a = predictor_jacobian(predictor, t, y, scale=1.0)
        b = predictor_jacobian(predictor, t, y, scale=0.5)
        out["step_refinement"] = {
            "t": t,
            "elements": {k: {"h": a[k]["d_range"], "h_half": b[k]["d_range"],
                             "rel_change": _rel(a[k]["d_range"], b[k]["d_range"])}
                         for k in a}}
    return out


def _rel(a: float, b: float) -> float:
    d = max(abs(a), abs(b))
    return float(abs(a - b) / d) if d > 1e-12 else 0.0


def _state_interpolator(traj: dict):
    """A 6-DOF-shaped state at any time along a recorded trajectory."""
    t = traj["t"]

    def at(tq: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(t, tq), 1, len(t) - 1))
        y = np.zeros(15)
        y[0:3] = traj["position"][i]
        y[3:6] = traj["velocity"][i]
        y[6:10] = traj["quaternion"][i]
        y[10:13] = traj["omega"][i]
        y[13] = traj["nose_angle"][i]
        y[14] = traj["nose_rate"][i]
        return y
    return at


# ===========================================================================
# Task B -- the measured state error, per element, bias separated from noise
# ===========================================================================
def _split_bias_noise(series: np.ndarray) -> tuple:
    """
    One flight's error time series into a constant and a fluctuation.

    THEY PROPAGATE DIFFERENTLY AND THAT IS THE REASON TO SEPARATE THEM. The
    guidance law commands once a second for forty seconds. A BIAS is present
    at every one of those cycles, so it moves the command the same way every
    time and the loop cannot average it out -- it converges to a miss it has
    no direction left to remove. NOISE is redrawn each cycle, and a loop with
    an actuator slower than the cycle rate averages most of it. Reporting a
    single 1 sigma would hand Task C one number where the transfer function
    needs two.

    Step 5 made exactly this split for the roll error and found the bias 0.54
    deg against 2.68 deg of noise; this generalises it to every element.
    """
    s = np.asarray(series, dtype=float)
    if s.size == 0:
        return 0.0, 0.0
    b = float(s.mean())
    return b, float(s.std(ddof=1)) if s.size > 1 else 0.0


def task_b(md: dict, pool, draws: list, seeds: int) -> dict:
    """
    The 1 sigma error in each element Task A differentiated, in the CLOSED
    LOOP and not in a replay.

    A replay is cheaper and it is the wrong measurement here: the estimate
    feeds back into what the round does, so the states the filter is asked
    about in flight are states its own errors helped produce. Task B is a
    number Task C multiplies by a closed-loop sensitivity, so it has to be a
    closed-loop error.
    """
    ctx = gc.engagement_context(md, ENGAGEMENT)
    opts = gc.scheduler_options(md, ENGAGEMENT)["proportional"]
    cases = []
    for s in range(seeds):
        cases += ncep._cases(ctx, draws, use_nav=True, seed=s,
                             scheduler_opts=opts, keep_nav_log=True,
                             keep_error_log=True)
    rows = ncep._imap(pool, nc.run_guided_nav, cases, "task-b")

    t_dep = ctx["base"]["deploy_time"]
    per_flight = {c: {"bias": [], "noise": []} for c in CHANNELS}
    stacked = {c: [] for c in CHANNELS}
    n_used = 0
    for r in rows:
        e = r.get("e_log")
        if not e:
            continue
        t = np.array(e["t"])
        # Only the ARMED guided phase. Before arming there is no steering
        # force, so an estimate error there costs nothing and including it
        # would inflate every channel with the despin transient.
        m = t >= t_dep + 3.0
        if m.sum() < 4:
            continue
        n_used += 1
        pe = np.array(e["position_error"])[m]
        ve = np.array(e["velocity_error"])[m]
        series = {"r_x": pe[:, 0], "r_y": pe[:, 1], "r_z": pe[:, 2],
                  "v_x": ve[:, 0], "v_y": ve[:, 1], "v_z": ve[:, 2],
                  "p_body": np.array(e["spin_error"])[m],
                  "roll": np.array(e["roll_error"])[m]}
        for c in CHANNELS:
            b, n = _split_bias_noise(series[c])
            per_flight[c]["bias"].append(b)
            per_flight[c]["noise"].append(n)
            stacked[c].append(series[c])

    out = {"n_flights": n_used, "channels": {}}
    for c in CHANNELS:
        b = np.array(per_flight[c]["bias"])
        n = np.array(per_flight[c]["noise"])
        allv = np.concatenate(stacked[c]) if stacked[c] else np.zeros(0)
        out["channels"][c] = {
            "bias_mean": float(b.mean()) if b.size else 0.0,
            "bias_sigma": float(b.std(ddof=1)) if b.size > 1 else 0.0,
            "noise_sigma": float(np.sqrt((n ** 2).mean())) if n.size else 0.0,
            "total_sigma": float(allv.std(ddof=1)) if allv.size > 1 else 0.0,
            "total_rms": float(np.sqrt((allv ** 2).mean())) if allv.size else 0.0,
        }

    # THE CORRELATION MATRIX, because Task C is not allowed to assume the
    # channels are independent. A GNSS position error and the velocity error
    # the same filter derives from it are not, and summing them in quadrature
    # would understate or overstate the total depending on the sign.
    lens = [min(len(s) for s in stacked[c]) for c in CHANNELS if stacked[c]]
    if lens:
        L = min(lens)
        M = np.array([np.concatenate([s[:L] for s in stacked[c]])
                      for c in CHANNELS])
        out["correlation"] = np.corrcoef(M).tolist()
        out["covariance"] = np.cov(M).tolist()
        out["channel_order"] = list(CHANNELS)
    return out, rows


# ===========================================================================
# Task C -- the predicted decomposition
# ===========================================================================
def task_c(a: dict, b: dict) -> dict:
    """
    Sensitivity x error, per element, under both transfer functions.

    Reported under BOTH the predictor Jacobian and the loop Jacobian, because
    which one applies is a property of the round and the two differ by an
    order of magnitude. The saturation fraction measured in Task D says which
    of the two the reader should be looking at.
    """
    ch = b["channels"]
    order = b.get("channel_order", list(CHANNELS))
    C = np.array(b["covariance"]) if "covariance" in b else None

    out = {"per_cycle": _decompose(a["cycles"], ch, order, C),
           "channel_order": order}
    if "dispersed" in a:
        # The same multiplication on the saturated trajectory. Task B's state
        # errors are unchanged -- the filter does not know which round it is
        # on -- so everything that moves between the two tables is
        # sensitivity, which is the point of reporting both.
        out["per_cycle_dispersed"] = _decompose(a["dispersed"]["cycles"],
                                                ch, order, C)
    return out


def _decompose(cycles: list, ch: dict, order: list, C) -> list:
    rows = []
    for row in cycles:
        entry = {"t_since_deploy": row["t_since_deploy"], "t_go": row["t_go"],
                 "saturated": row["saturated"], "miss_m": row["miss_m"]}
        for kind in ("predictor", "loop"):
            J = row[kind]
            # The predictor has no roll channel at all; the loop has one and
            # it is the only place attitude appears. Missing entries are zero,
            # which is a statement about the model rather than a gap.
            terms = {}
            jr = np.zeros(len(order))
            jd = np.zeros(len(order))
            for i, c in enumerate(order):
                j = J.get(c)
                if j is None:
                    continue
                # The predictor Jacobian's SIGN convention: a prediction that
                # moves +x metres downrange, when the loop nulls it, moves the
                # true impact -x. Magnitudes are what the budget adds, so the
                # sign is dropped here and carried in the text.
                jr[i], jd[i] = j["d_range"], j["d_defl"]
                s = ch[c]
                terms[c] = {
                    "d_range_per_unit": jr[i], "d_defl_per_unit": jd[i],
                    "bias_range_m": abs(jr[i]) * abs(s["bias_mean"]),
                    "bias_sigma_range_m": abs(jr[i]) * s["bias_sigma"],
                    "noise_range_m": abs(jr[i]) * s["noise_sigma"],
                    "total_range_m": abs(jr[i]) * s["total_sigma"],
                    "total_defl_m": abs(jd[i]) * s["total_sigma"],
                }
            q_r = math.sqrt(sum(v["total_range_m"] ** 2 for v in terms.values()))
            q_d = math.sqrt(sum(v["total_defl_m"] ** 2 for v in terms.values()))
            entry[kind] = {"terms": terms,
                           "quadrature_range_m": q_r,
                           "quadrature_defl_m": q_d}
            if C is not None:
                # WITH the covariance. `J C J^T` is the variance of the
                # impact-point error under the measured joint distribution of
                # the state errors, and it differs from the quadrature sum by
                # exactly the amount the channels are correlated.
                entry[kind]["covariance_range_m"] = float(
                    math.sqrt(max(0.0, jr @ C @ jr)))
                entry[kind]["covariance_defl_m"] = float(
                    math.sqrt(max(0.0, jd @ C @ jd)))
        rows.append(entry)
    return rows


# ===========================================================================
# Task D -- the ablation, which shares no line of code with Task C
# ===========================================================================
#
# ONE SOURCE PER VARIANT, and the override mechanism replaces a FIELD on a
# frozen part spec rather than rebuilding the suite. `SensorSuite` draws one
# child generator per sensor from a `SeedSequence`, so zeroing the gyro's
# angle random walk does not reshuffle the GNSS error: the ablation is a
# change to one number, and the difference it makes is that number's
# contribution rather than a different realisation of everything else.
#
#: Zeroing a GNSS sigma is legitimate for the SENSOR -- it is an error draw --
#: but the same field also scales the filter's Schmidt consider floor, so the
#: near-zero used here goes through `truth_spec_overrides` and the filter's
#: floor is left where it was. 1 cm is three hundred times better than the
#: part and is not numerically degenerate.
PERFECT_M = 0.01
PERFECT_MS = 0.001

VARIANTS = {
    "baseline": {},
    # THE CONTROL, and the most informative single run in the module. Every
    # sensor error term removed at once, with the filter's tuning untouched.
    # If this does not return a contribution of about zero, then whatever
    # remains is NOT a sensor error, and no amount of decomposing the sensors
    # will find it.
    "all_perfect": {"truth_spec_overrides": {
        "gyro": {"arw": 0.0, "bias_stability": 0.0, "bias_repeatability": 0.0,
                 "scale_factor": 0.0, "misalignment": 0.0, "quantisation": 0.0},
        "accel": {"vrw": 0.0, "bias_stability": 0.0, "bias_repeatability": 0.0,
                  "scale_factor": 0.0, "misalignment": 0.0, "quantisation": 0.0},
        "gnss": {"sigma_horizontal": PERFECT_M, "sigma_vertical": PERFECT_M,
                 "sigma_velocity": PERFECT_MS, "latency": 0.0},
        "resolver": {"accuracy": 0.0, "quantisation": 0.0, "rate_noise": 0.0},
    }, "mag_residual_fraction": 1e-5, "antenna_transverse": 0.0},

    # -- the gyro ---------------------------------------------------------
    "gyro_bias": {"truth_spec_overrides": {
        "gyro": {"bias_stability": 0.0, "bias_repeatability": 0.0}}},
    "gyro_arw": {"truth_spec_overrides": {"gyro": {"arw": 0.0}}},
    "gyro_scale_misalign": {"truth_spec_overrides": {
        "gyro": {"scale_factor": 0.0, "misalignment": 0.0}}},

    # -- the accelerometer ------------------------------------------------
    "accel_bias": {"truth_spec_overrides": {
        "accel": {"bias_stability": 0.0, "bias_repeatability": 0.0}}},
    "accel_vrw": {"truth_spec_overrides": {"accel": {"vrw": 0.0}}},
    "accel_scale_misalign": {"truth_spec_overrides": {
        "accel": {"scale_factor": 0.0, "misalignment": 0.0}}},

    # -- GNSS -------------------------------------------------------------
    "gnss_position": {"truth_spec_overrides": {
        "gnss": {"sigma_horizontal": PERFECT_M, "sigma_vertical": PERFECT_M}}},
    "gnss_velocity": {"truth_spec_overrides": {
        "gnss": {"sigma_velocity": PERFECT_MS}}},
    "gnss_latency": {"truth_spec_overrides": {"gnss": {"latency": 0.0}}},
    #: A GEOMETRY change, not an error removal, so it moves the filter's
    #: belief with it: a receiver that reports at 20 Hz reports at 20 Hz.
    "gnss_rate_20hz": {"spec_overrides": {"gnss": {"rate_hz": 20.0}}},
    #: Likewise: the phase centre's position is layout the filter is entitled
    #: to know, and `update_gnss` uses it to compensate the lever arm.
    "gnss_antenna": {"antenna_transverse": 0.0},

    # -- the roll references ----------------------------------------------
    #: Already the truth-only knob: `mag_residual_fraction` drives the
    #: `Magnetometer`'s distortion draw and nothing in the filter. Its
    #: filter-side twin is `mag_floor` below, and step 5 found they are not
    #: the same quantity even though they share a symbol.
    "mag_residual": {"mag_residual_fraction": 0.001},
    #: The filter's DECLARED magnetometer accuracy, which step 5 found is what
    #: limits the roll estimate rather than the sensor. Separated from the
    #: sensor's own residual because they are different objects that share a
    #: symbol, and NAV-CONSISTENCY section 3 raised exactly that.
    "mag_floor": {"nav_config": {"mag_sigma_floor": 0.001}},
    "resolver": {"truth_spec_overrides": {
        "resolver": {"accuracy": 0.0, "quantisation": 0.0, "rate_noise": 0.0}}},
}

#: The barometer is absent from VARIANTS on purpose. `gnc.navigation` has no
#: barometer update -- `update_gnss`, `update_magnetometer` and
#: `update_velocity_attitude` are the only three -- because
#: docs/SENSOR-MODELS.md section 6 measured the static-port pressure
#: coefficient at 600 m of pressure altitude and disqualified it as an
#: altitude source. Its ablation is therefore a null BY CONSTRUCTION and
#: running 48 rounds to confirm that a disconnected sensor contributes nothing
#: would be spending 20 minutes to measure a wiring diagram.
BAROMETER_IS_NOT_AN_AIDING_SENSOR = True


def saturation_fraction(rows: list) -> float:
    """
    The fraction of ARMED guidance cycles at which the law could not deliver
    the miss even by holding to impact.

    Read from the scheduler's own per-cycle note rather than from
    `guidance_cep.cep_of`'s `authority_limited`, which tests a `g_plan` field
    only `SingleShotScheduler` writes and therefore reports 0 for every
    proportional run whatever the round did.
    """
    n = tot = 0
    for r in rows:
        L = r.get("g_log")
        if not L:
            continue
        for note, armed in zip(L["note"], L["armed"]):
            if not armed:
                continue
            tot += 1
            n += note == "saturated"
    return float(n / tot) if tot else float("nan")


def paired_sigma_change(base_pairs: list, abl_pairs: list,
                        n_boot: int = 2000, seed: int = 12345) -> dict:
    """
    How much an ablation moved the 1 sigma, WITH the standard error of that
    movement.

    Every ablation flies the same draws with the same sensor seeds as the
    baseline -- common random numbers -- so the two sigmas are strongly
    correlated and their DIFFERENCE is resolved far better than either is on
    its own. Differencing two independently-quoted sigmas throws that away:
    at 24 draws each sigma carries about 15 % of standard error, which is
    +-5 m on a 32 m number, and every ablation step 5 found was smaller than
    that.

    So the difference is bootstrapped over DRAWS, resampling the baseline and
    the ablation together. `delta_sigma_se_m` is what says whether a 2 m
    ablation is a measurement or a coin toss, and several of them are not.
    """
    bb = {(p["draw"], p["seed"]): p["d_range_m"] for p in base_pairs}
    ab = {(p["draw"], p["seed"]): p["d_range_m"] for p in abl_pairs}
    keys = sorted(set(bb) & set(ab))
    if len(keys) < 4:
        return {"n": len(keys)}
    x = np.array([bb[k] for k in keys])
    y = np.array([ab[k] for k in keys])
    draws_ = sorted({k[0] for k in keys})
    idx = {d: [i for i, k in enumerate(keys) if k[0] == d] for d in draws_}
    rng = np.random.default_rng(seed)
    d0 = float(y.std(ddof=1) - x.std(ddof=1))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(draws_), len(draws_))
        sel = [j for p in pick for j in idx[draws_[p]]]
        boot[i] = y[sel].std(ddof=1) - x[sel].std(ddof=1)
    return {
        "n": len(keys),
        "delta_sigma_m": d0,
        "delta_sigma_se_m": float(boot.std(ddof=1)),
        "paired_shift_mean_m": float((y - x).mean()),
        "paired_shift_sd_m": float((y - x).std(ddof=1)),
        "correlation_with_baseline": float(np.corrcoef(x, y)[0, 1]),
    }


def task_d(md: dict, pool, draws: list, seeds: int, want: list,
           truth: list, out: dict) -> list:
    """
    One ablation per variant. Returns the baseline rows so Task E can use them
    as its navigation-fed reference instead of flying them twice.
    """
    ctx = gc.engagement_context(md, ENGAGEMENT)
    opts = gc.scheduler_options(md, ENGAGEMENT)["proportional"]
    out.setdefault("variants", {})
    base_rows = out.get("_baseline_rows")
    for name in want:
        kw = dict(VARIANTS[name])
        cases = []
        for s in range(seeds):
            cases += ncep._cases(ctx, draws, use_nav=True, seed=s,
                                 scheduler_opts=opts, keep_nav_log=True,
                                 keep_guidance_log=(name == "baseline"), **kw)
        rows = ncep._imap(pool, nc.run_guided_nav, cases, f"D:{name}")
        d = ncep.paired_navigation_delta(truth, rows)
        rec = {
            "overrides": kw, "seeds": seeds,
            "range_sigma_m": d["range"]["sigma_m"],
            "range_bias_m": d["range"]["bias_m"],
            "defl_sigma_m": d["deflection"]["sigma_m"],
            "defl_bias_m": d["deflection"]["bias_m"],
            "radial_median_m": d["radial"]["median_m"],
            "n": d["n"],
            **_row_stats(rows),
        }
        if name == "baseline":
            out["saturation_fraction"] = saturation_fraction(rows)
            out["baseline_pairs"] = d["pairs"]
            out["prediction_transfer"] = prediction_transfer(truth, rows)
            out["_baseline_rows"] = base_rows = rows
        elif out.get("baseline_pairs"):
            rec["vs_baseline"] = paired_sigma_change(out["baseline_pairs"],
                                                     d["pairs"])
        out["variants"][name] = rec
        vb = rec.get("vs_baseline", {})
        print(f"  {name:22s} range 1 sigma {rec['range_sigma_m']:6.2f} m  "
              f"delta {vb.get('delta_sigma_m', float('nan')):+6.2f} "
              f"+- {vb.get('delta_sigma_se_m', float('nan')):4.2f} m  "
              f"defl {rec['defl_sigma_m']:5.2f} m  "
              f"cmd {rec['cmd_changes_armed']:.1f}", flush=True)
    return base_rows


def _row_stats(rows: list) -> dict:
    def m(key, default=float("nan")):
        v = [r[key] for r in rows if r.get(key) is not None
             and np.isfinite(r.get(key, np.nan))]
        return float(np.mean(v)) if v else default
    out = {"cmd_changes_armed": m("g_cmd_changes_armed"),
           "cmd_total_variation_deg_armed": m("g_cmd_total_variation_deg_armed"),
           "cmd_rms_step_deg_armed": m("g_cmd_rms_step_deg_armed"),
           "slew_fraction": m("slew_fraction"),
           "slew_time_s": m("slew_time_s"),
           "servo_error_deg": m("servo_error_deg")}
    vel = [r["n_vel_rms_ms"] for r in rows if "n_vel_rms_ms" in r]
    pos = [r["n_pos_rms_m"] for r in rows if "n_pos_rms_m" in r]
    if vel:
        out["velocity_rms_ms"] = [float(v) for v in
                                  np.sqrt((np.array(vel) ** 2).mean(axis=0))]
    if pos:
        out["position_rms_m"] = [float(v) for v in
                                 np.sqrt((np.array(pos) ** 2).mean(axis=0))]
    rb = [abs(r["n_roll_bias_deg"]) for r in rows
          if r.get("n_roll_bias_deg") is not None]
    rsd = [r["n_roll_sd_deg"] for r in rows if r.get("n_roll_sd_deg") is not None]
    out["roll_bias_deg"] = float(np.mean(rb)) if rb else None
    out["roll_sd_deg"] = float(np.mean(rsd)) if rsd else None
    return out


def prediction_transfer(truth: list, nav_rows: list) -> dict:
    """
    THE THIRD, INDEPENDENTLY DERIVED QUANTITY.

    Task C assumes a transfer function from state error to impact error. Task
    D measures the impact error. Neither measures the thing in between: the
    error in the PREDICTED impact point, which is what the scheduler actually
    decides on.

    This differences the navigation-fed round's own guidance log against the
    truth-fed round's, cycle by cycle, and regresses the final miss difference
    on it. If the loop nulls its prediction, the slope is -1. If the round is
    authority-limited and only the command DIRECTION matters, the slope is
    much smaller and the correlation is weak. The slope is therefore a direct
    measurement of which of Task C's two transfer functions applies, made
    without using either of them.
    """
    by_draw = {r["draw"]: r for r in truth if r.get("g_log")}
    x_last, x_mean, y = [], [], []
    for r in nav_rows:
        t = by_draw.get(r.get("draw"))
        L, LT = r.get("g_log"), (t or {}).get("g_log")
        if t is None or not L or not LT:
            continue
        tn = np.array(L["t"])
        tt = np.array(LT["t"])
        pr_n = np.array(L["pred_range"])
        pr_t = np.interp(tn, tt, np.array(LT["pred_range"]))
        good = np.isfinite(pr_n) & np.isfinite(pr_t) & (tn >= tn[0] + 3.0)
        if good.sum() < 4:
            continue
        e = pr_n[good] - pr_t[good]
        x_last.append(float(e[-1]))
        x_mean.append(float(e.mean()))
        y.append(float(r["range_m"] - t["range_m"]))
    if len(y) < 4:
        return {"n": len(y)}
    out = {"n": len(y)}
    for tag, x in (("last_cycle", x_last), ("mean_over_flight", x_mean)):
        x = np.array(x)
        yy = np.array(y)
        A = np.column_stack([x, np.ones_like(x)])
        slope, icept = np.linalg.lstsq(A, yy, rcond=None)[0]
        out[tag] = {
            "prediction_error_sigma_m": float(x.std(ddof=1)),
            "slope": float(slope), "intercept_m": float(icept),
            "correlation": float(np.corrcoef(x, yy)[0, 1]),
        }
    return out


# ===========================================================================
# Task F -- the term neither decomposition was looking for
# ===========================================================================
def authority_monitor_false_alarms(path: str = "docs/nav_cep.json") -> dict:
    """
    How often the step-3 authority monitor declares a healthy actuator dead,
    truth-fed against navigation-fed, and what those rounds cost.

    NOTHING IS FLOWN FOR THIS. `analysis.nav_cep` already saved `g_authority_ok`
    on every row of the Task F campaign, so this is a query against step 5's
    own data -- which is the point worth making about it: the largest single
    term in the navigation contribution was sitting in a file the project had
    already written and nobody had asked the file this question.

    The monitor compares the OBSERVED movement of the predicted impact point
    against the movement the authority map promised, and declares a fault when
    too little of it happened. `monitor_fraction` and `monitor_floor_m` were
    sized in docs/GUIDANCE-DESIGN.md against the prediction error measured with
    the guidance law reading TRUTH. With navigation error in the loop the
    predicted impact point also moves for reasons that have nothing to do with
    the actuator, so the test statistic acquires a noise term its threshold was
    never sized against -- and the fault it raises is a LATCHING, permanent
    release of the brake for the rest of the flight.
    """
    with open(path, encoding="utf-8") as fh:
        cep = json.load(fh)
    out = {"source": path, "engagements": {}}
    for eng, L in cep["task_f"]["engagements"].items():
        truth = {r["draw"]: r for r in L["truth_rows"]}
        dr, fault = [], []
        for r in L["nav_rows"]:
            t = truth.get(r["draw"])
            if t is None:
                continue
            dr.append(r["range_m"] - t["range_m"])
            fault.append(not r.get("g_authority_ok", True))
        dr, fault = np.array(dr), np.array(fault, dtype=bool)
        if dr.size < 4:
            continue
        m = dr.mean()
        total = float(((dr - m) ** 2).sum())
        clean = dr[~fault]
        out["engagements"][eng] = {
            "n_nav": int(dr.size),
            "n_truth": len(truth),
            "truth_fault_rate": float(np.mean(
                [not r.get("g_authority_ok", True) for r in L["truth_rows"]])),
            "nav_fault_rate": float(fault.mean()),
            "variance_share_of_faulted": (
                float(((dr[fault] - m) ** 2).sum() / total) if total > 0 else 0.0),
            "range_sigma_all_m": float(dr.std(ddof=1)),
            "range_sigma_clean_m": (float(clean.std(ddof=1))
                                    if clean.size > 1 else float("nan")),
            "range_bias_all_m": float(m),
            "range_bias_faulted_m": (float(dr[fault].mean())
                                     if fault.any() else float("nan")),
            "range_bias_clean_m": (float(clean.mean())
                                   if clean.size else float("nan")),
        }
    return out


# ===========================================================================
# Task E -- the hypothesis no open-loop ablation can find
# ===========================================================================
#
# The mechanism, if it exists: navigation noise moves the predicted impact
# point between guidance cycles, guidance re-commands, and the servo slews to
# the new angle. Step 4 priced every hold-to-hold transition at 0.4-0.75 s of
# reacquisition plus one draw from the capture distribution, and step 3
# concluded that magnitude control should be ONE contiguous hold precisely
# because switching is expensive. Authority spent slewing is authority not
# spent correcting, and no open-loop measurement of the filter can see it.
#
# The unmodified truth-fed and navigation-fed rounds are NOT re-flown here:
# they are Task D's own reference and baseline, handed in, so the counts E1
# compares are the counts D measured rather than a second campaign that might
# differ.
E_VARIANTS = {
    #: The prediction low pass. Its time constant is well above the 1 Hz
    #: guidance rate and above the servo's own 0.4-0.75 s reacquisition, so if
    #: chatter is the mechanism this must remove most of it.
    #:
    #: BOTH HALVES ARE FLOWN. A smoothed navigation-fed round on its own
    #: cannot separate "less chatter" from "a different loop"; the smoothed
    #: TRUTH-fed round carries no navigation error at all, so whatever it
    #: costs against the unsmoothed truth-fed reference is the pure price of
    #: the lag, and the difference of the two differences is the chatter.
    "truth_smoothed_8s": {"use_nav": False, "pred_filter_tau": 8.0},
    "nav_smoothed_8s": {"use_nav": True, "pred_filter_tau": 8.0},
    #: Command hysteresis, which is what the fix would actually be if the
    #: hypothesis holds. `DeadbandScheduler.freeze_deg` already implements it:
    #: a recomputed angle within `freeze_deg` of the committed one is
    #: discarded and the command is left alone.
    "truth_hysteresis": {"use_nav": False, "scheduler": "deadband"},
    "nav_hysteresis": {"use_nav": True, "scheduler": "deadband"},
}


#: Bin edges on the miss still outstanding, metres. The reachable set's
#: semi-minor axis at this engagement is about 200 m and the CEP criterion is
#: 30 m, so the bins straddle both.
MISS_BINS = (0.0, 5.0, 25.0, 100.0, float("inf"))


def _activity_by_miss(rows: list, threshold_deg: float = 1.0) -> list:
    """
    Command changes per cycle, binned by how much miss was still outstanding.

    A guidance law that has already nulled its miss has no direction left to
    point in -- the commanded angle is the bearing of a residual of a few
    metres and it wanders freely -- so counting those cycles alongside the
    ones that matter would make a converged loop look like a chattering one.
    """
    counts = [{"lo": MISS_BINS[i], "hi": MISS_BINS[i + 1], "cycles": 0,
               "changes": 0, "total_variation_deg": 0.0}
              for i in range(len(MISS_BINS) - 1)]
    for r in rows:
        L = r.get("g_log")
        if not L:
            continue
        prev = None
        for phi, miss, hold, armed in zip(L["phi_deg"], L["miss_m"],
                                          L["holding"], L["armed"]):
            if not (hold and armed) or not np.isfinite(miss):
                prev = None
                continue
            if prev is not None:
                d = abs((phi - prev + 180.0) % 360.0 - 180.0)
                k = int(np.searchsorted(MISS_BINS, miss, side="right") - 1)
                k = min(max(k, 0), len(counts) - 1)
                counts[k]["cycles"] += 1
                counts[k]["changes"] += int(d > threshold_deg)
                counts[k]["total_variation_deg"] += float(d)
            prev = phi
    for c in counts:
        c["changes_per_cycle"] = (c["changes"] / c["cycles"]
                                  if c["cycles"] else float("nan"))
        c["mean_step_deg"] = (c["total_variation_deg"] / c["cycles"]
                              if c["cycles"] else float("nan"))
    return counts


def task_e(md: dict, pool, draws: list, seeds: int, truth: list,
           base_rows: list = None) -> dict:
    ctx = gc.engagement_context(md, ENGAGEMENT)
    sopts = gc.scheduler_options(md, ENGAGEMENT)
    out = {"variants": {}}
    # E1: the count comparison, on the rounds Task D already flew. If the
    # navigation-fed loop is not switching MORE than the truth-fed one, the
    # hypothesis is dead and no smoothing result can revive it.
    if truth:
        out["variants"]["truth"] = {
            "overrides": {"use_nav": False, "scheduler": "proportional"},
            **_row_stats(truth), "cep": ncep._cep(truth),
            "saturation_fraction": saturation_fraction(truth),
            "activity_by_miss": _activity_by_miss(truth),
            "lag_cost_range_sigma_m": 0.0, "lag_cost_range_bias_m": 0.0,
            "note": "Task D's own truth-fed reference, not re-flown"}
    if base_rows:
        d = ncep.paired_navigation_delta(truth, base_rows)
        out["variants"]["nav"] = {
            "overrides": {"use_nav": True, "scheduler": "proportional"},
            **_row_stats(base_rows), "cep": ncep._cep(base_rows),
            "saturation_fraction": saturation_fraction(base_rows),
            "activity_by_miss": _activity_by_miss(base_rows),
            "range_sigma_m": d["range"]["sigma_m"],
            "range_bias_m": d["range"]["bias_m"],
            "defl_sigma_m": d["deflection"]["sigma_m"],
            "defl_bias_m": d["deflection"]["bias_m"], "n": d["n"],
            "pairs": d["pairs"],
            "note": "Task D's own baseline, not re-flown"}
    for name, kw in E_VARIANTS.items():
        kw = dict(kw)
        use_nav = kw.pop("use_nav")
        sched = kw.pop("scheduler", "proportional")
        opts = sopts[sched]
        # The deadband scheduler's freeze angle is the hysteresis being
        # tested, and it is not one of `scheduler_options`' defaults, so it is
        # set here where the reader can see what was tried.
        if sched == "deadband":
            opts = dict(opts, freeze_deg=10.0)
        cases = []
        for s in (range(seeds) if use_nav else [None]):
            cases += ncep._cases(ctx, draws, use_nav=use_nav,
                                 **({"seed": s} if s is not None else {}),
                                 scheduler=sched, scheduler_opts=opts,
                                 keep_nav_log=use_nav, keep_guidance_log=True,
                                 **kw)
        rows = ncep._imap(pool, nc.run_guided_nav, cases, f"E:{name}")
        rec = {"overrides": dict(kw, scheduler=sched, use_nav=use_nav),
               **_row_stats(rows), "cep": ncep._cep(rows),
               "saturation_fraction": saturation_fraction(rows),
               # WHEN the command moves, not just how often. A command that
               # swings late in the flight, with the miss already nulled and
               # nothing left to correct, costs nothing; one that swings while
               # there is still 40 m to remove costs the whole slew. Binned by
               # the miss still outstanding so the two cannot be confused.
               "activity_by_miss": _activity_by_miss(rows)}
        if use_nav:
            # Always paired against the UNSMOOTHED truth-fed reference. The
            # question is what the round's impact point does, and a smoothed
            # truth-fed reference would hide the lag the smoothing costs.
            d = ncep.paired_navigation_delta(truth, rows)
            rec.update({"range_sigma_m": d["range"]["sigma_m"],
                        "range_bias_m": d["range"]["bias_m"],
                        "defl_sigma_m": d["deflection"]["sigma_m"],
                        "defl_bias_m": d["deflection"]["bias_m"], "n": d["n"],
                        "pairs": d["pairs"]})
            nb = out["variants"].get("nav", {}).get("pairs")
            if nb:
                rec["vs_nav_baseline"] = paired_sigma_change(nb, d["pairs"])
        else:
            d = ncep.paired_navigation_delta(truth, rows)
            # For a truth-fed variant the "contribution" is the cost of the
            # variant ITSELF against the unmodified truth-fed round: pure lag,
            # with no navigation error anywhere in it.
            rec.update({"lag_cost_range_sigma_m": d["range"]["sigma_m"],
                        "lag_cost_range_bias_m": d["range"]["bias_m"],
                        "n": d["n"]})
        out["variants"][name] = rec
        print(f"  {name:20s} cmd changes {rec['cmd_changes_armed']:5.1f}  "
              f"slew {rec['slew_fraction']:5.3f}  "
              f"tv {rec['cmd_total_variation_deg_armed']:7.1f} deg  "
              f"range 1 sigma "
              f"{rec.get('range_sigma_m', rec.get('lag_cost_range_sigma_m')):6.2f} m",
              flush=True)
    return out


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="a,b,c,d,e,f")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--ablation-seeds", type=int, default=1,
                    help="sensor seeds for the individual ablations. One is "
                         "enough because `paired_sigma_change` differences "
                         "them against the baseline under common random "
                         "numbers; the controls keep --seeds.")
    ap.add_argument("--cycles", type=int, default=9,
                    help="Task A sample points across the guided phase")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--out", default="docs/nav_ablation.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args(argv)
    tasks = {c.strip() for c in args.tasks.split(",") if c.strip()}

    sn.warn_low_confidence()
    md = gc.load_maps()
    ctx = gc.engagement_context(md, ENGAGEMENT)
    draws = gc.make_draws(ctx, args.draws)
    opts = gc.scheduler_options(md, ENGAGEMENT)["proportional"]
    t0 = time.time()
    out = {"engagement": ENGAGEMENT, "n_draws": args.draws,
           "n_seeds": args.seeds, "tasks": sorted(tasks)}

    if "f" in tasks:
        # Cheap, flies nothing, and it is the headline of the whole step --
        # so it runs first and unconditionally where it can.
        print("Task F -- the authority monitor's false-alarm rate", flush=True)
        try:
            out["task_f"] = authority_monitor_false_alarms()
            for eng, v in out["task_f"]["engagements"].items():
                print(f"  {eng:8s} truth {v['truth_fault_rate']:5.1%}  "
                      f"nav {v['nav_fault_rate']:5.1%}  carrying "
                      f"{v['variance_share_of_faulted']:5.1%} of the variance  "
                      f"| range 1 sigma {v['range_sigma_all_m']:6.2f} m -> "
                      f"{v['range_sigma_clean_m']:6.2f} m without them",
                      flush=True)
        except (OSError, KeyError) as exc:
            print(f"  skipped: {exc}", flush=True)

    if "a" in tasks:
        print("Task A -- impact-point sensitivity to the state", flush=True)
        out["task_a"] = task_a(md, args.cycles, not args.no_refine,
                               draws=draws)

    need_pool = tasks & {"b", "d", "e"}
    if need_pool:
        with Pool(args.workers) as pool:
            truth = None
            if tasks & {"d", "e"}:
                print("Truth-fed reference rounds", flush=True)
                truth = ncep._imap(pool, nc.run_guided_nav,
                                   ncep._cases(ctx, draws, use_nav=False,
                                               scheduler_opts=opts,
                                               keep_guidance_log=True),
                                   "truth")
                out["truth_cep"] = ncep._cep(truth)
            if "b" in tasks:
                print("Task B -- the measured state error", flush=True)
                out["task_b"], _ = task_b(md, pool, draws, args.seeds)
            if "c" in tasks:
                print("Task C -- the predicted decomposition", flush=True)
                if "task_a" in out and "task_b" in out:
                    out["task_c"] = task_c(out["task_a"], out["task_b"])
                else:
                    print("  skipped: needs a and b in the same run", flush=True)
            # ORDER MATTERS AND IT IS NOT ALPHABETICAL. The control variants
            # and Task E answer the question; the fourteen individual
            # ablations refine the answer and cost more than everything else
            # put together. Running them last means a session that runs out of
            # time still has the finding, and `--variants` can drop them.
            base_rows = None
            want = [v.strip() for v in args.variants.split(",") if v.strip()]
            controls = [v for v in want if v in ("baseline", "all_perfect")]
            rest = [v for v in want if v not in controls]
            if "d" in tasks:
                print("Task D -- controls", flush=True)
                out["task_d"] = {}
                base_rows = task_d(md, pool, draws, args.seeds, controls,
                                   truth, out["task_d"])
            if "e" in tasks:
                print("Task E -- command chatter", flush=True)
                out["task_e"] = task_e(md, pool, draws, args.seeds, truth,
                                       base_rows)
                _save(args.out, out)
            if "d" in tasks and rest:
                print(f"Task D -- {len(rest)} individual ablations at "
                      f"{args.ablation_seeds} seed(s)", flush=True)
                task_d(md, pool, draws, args.ablation_seeds, rest, truth,
                       out["task_d"])
    elif "c" in tasks and "task_a" in out and "task_b" in out:
        out["task_c"] = task_c(out["task_a"], out["task_b"])

    _save(args.out, out)
    print(f"wrote {args.out} in {time.time() - t0:.1f} s")
    return 0


def _save(path: str, out: dict) -> None:
    """
    Written once after Task E and again at the end, so an interrupted run
    still leaves the finding on disk.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    slim = {k: v for k, v in out.items()}
    if "task_d" in slim:
        slim["task_d"] = {k: v for k, v in slim["task_d"].items()
                          if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(slim, fh, indent=1, default=float)


if __name__ == "__main__":
    raise SystemExit(main())
