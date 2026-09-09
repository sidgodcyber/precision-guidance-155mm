"""
Step 5, Task E: is the filter CONSISTENT, not merely accurate?

Produces docs/NAV-CONSISTENCY.md's numbers and docs/nav_consistency.json.

WHY THIS RUNS BEFORE THE CEP
----------------------------
A filter that is accurate on one trajectory has told you nothing. The test
that means something is whether its claimed covariance matches its actual
error, because that is the property everything downstream relies on: the
guidance law's validity flag, the degradation ladder's decision to keep
steering, and step 6's right to add a navigation term to a budget in
quadrature all assume the filter knows how wrong it is.

    NEES   normalised estimation error squared, e' P^-1 e, against truth.
           Expectation is the state dimension. Above it, the filter is
           OVERCONFIDENT -- its error is larger than it claims.
    NIS    normalised innovation squared, per measurement stream. Expectation
           is the measurement dimension. It tests the same thing WITHOUT
           truth, which is what a flight test can compute.
    WHITENESS  the autocorrelation of the innovations. A consistent filter has
           extracted everything predictable from its measurements, so what is
           left must be white. Innovations that are correlated at lag 1 mean
           there is structure the filter is not modelling, even if NIS passes.

TUNING HAPPENS HERE AND NOWHERE ELSE
------------------------------------
`--tune` sweeps the three consider-inflations and reports the NEES each
produces. The values adopted are the ones that make the filter consistent, and
docs/NAV-CEP.md then reports whatever CEP that filter gives. No number in this
project has ever been chosen by looking at a CEP and this one is not going to
be the first.

WHAT THE THREE INFLATIONS ARE FOR, PHYSICALLY
---------------------------------------------
All three cover errors a 15-state filter has no state for, and all three are
CORRELATED rather than white, which is precisely why declaring the raw
per-sample sigma makes the filter overconfident:

  gnss_position_inflation   80 % of the receiver's error is a Gauss-Markov
                            process at a 100 s correlation time. Two hundred
                            fixes over a 43 s flight are not two hundred
                            independent looks at it.
  mag_sigma_floor           the hard/soft-iron calibration residual is a FIXED
                            distortion in the nose frame. It never averages
                            down, because it is not noise.
  alignment_sigma           the angle of attack swings at the epicyclic
                            frequency, which is slow in the nose frame -- so
                            consecutive alignment updates see nearly the same
                            error.

Run:  python -m analysis.nav_consistency            the campaign
      python -m analysis.nav_consistency --tune     the sweep that set it
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from gnc import navigation as nv, sensors as sn
from analysis import guidance_cep as gc, nav_common as nc

#: Engagements the consistency campaign runs over. FOUR, not one, because the
#: whole point of NEES is that a filter can be accurate on one trajectory and
#: inconsistent on another.
ENGAGEMENTS = ("short2", "middle", "mid2", "long")

#: Sensor seeds per trajectory. 4 x 16 = 64 runs, comfortably over the 50 the
#: brief asks for, and spread over the envelope rather than stacked on one
#: trajectory where they would only re-sample the sensor noise.
SEEDS_PER = 16

#: The tuning grid. Coarse on purpose: these are consider factors, not fitted
#: parameters, and quoting one to three significant figures would be claiming a
#: precision the chi-square test does not have.
TUNE_GRID = {
    "gnss_position_inflation": (1.0, 10.0, 50.0, 200.0),
    "gnss_velocity_inflation": (1.0, 50.0, 500.0),
    "mag_sigma_floor": (0.015, 0.05),
    "alignment_sigma_deg": (1.78, 4.0),
}

_TRAJ: dict = {}


def _init(trajectories):
    global _TRAJ
    _TRAJ = trajectories


# ===========================================================================
# Chi-square bounds
# ===========================================================================
def chi2_bounds(dof: int, n: int, alpha: float = 0.05) -> tuple:
    """
    Two-sided (1-alpha) acceptance interval for the AVERAGE NEES over `n`
    independent samples, expressed per degree of freedom.

    The average of n chi-square(dof) variables scaled by n is chi-square(n*dof),
    so the bound on the average is that quantile over n. Uses scipy when it is
    available and a Wilson-Hilferty normal approximation when it is not, so
    the module does not acquire a hard dependency for one number.
    """
    m = dof * n
    try:
        from scipy.stats import chi2
        lo, hi = chi2.ppf(alpha / 2.0, m), chi2.ppf(1.0 - alpha / 2.0, m)
    except Exception:
        z = 1.959963984540054
        lo = m * (1.0 - 2.0 / (9.0 * m) - z * math.sqrt(2.0 / (9.0 * m))) ** 3
        hi = m * (1.0 - 2.0 / (9.0 * m) + z * math.sqrt(2.0 / (9.0 * m))) ** 3
    return lo / n, hi / n


def _whiteness(x: np.ndarray, max_lag: int = 10) -> dict:
    """
    Normalised autocorrelation of a scalar innovation sequence, and the
    +-1.96/sqrt(N) band a white sequence stays inside.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 20:
        return {"n": int(n), "acf": [], "band": None, "fraction_outside": None}
    x = x - x.mean()
    denom = float((x * x).sum())
    acf = []
    for k in range(1, max_lag + 1):
        acf.append(float((x[:-k] * x[k:]).sum() / denom) if denom > 0 else 0.0)
    band = 1.959963984540054 / math.sqrt(n)
    return {"n": int(n), "acf": acf, "band": float(band),
            "fraction_outside": float(np.mean(np.abs(acf) > band))}


# ===========================================================================
# One run
# ===========================================================================
def _run(a) -> dict:
    label, seed, cfg_kw = a
    cfg = _config(cfg_kw)
    r = nc.replay(_TRAJ[label], seed=seed, config=cfg, log_nis=True)
    nav = r["nav"]
    f = nav.filter
    out = {"engagement": label, "seed": seed,
           "t": r["t"].tolist(),
           "nees": r["nees"].tolist(), "nees_nav": r["nees_nav"].tolist(),
           "nees_pv": r["nees_pv"].tolist(),
           "position_error": r["position_error"].tolist(),
           "velocity_error": r["velocity_error"].tolist(),
           "attitude_error": r["attitude_error"].tolist(),
           "nis": {k: [v for v in vals if np.isfinite(v)]
                   for k, vals in f.nis.items()},
           "updates": dict(f.updates),
           "first_usable_s": nav.first_usable_t,
           "gnss_reacquire_s": nav.sensors.gnss.reacquire_time}
    # Innovation whiteness, on the first component of each stream.
    out["whiteness"] = {}
    for k, vals in f.innovations.items():
        if not vals:
            continue
        arr = np.array([v[0] if np.ndim(v) else v for v in vals], dtype=float)
        out["whiteness"][k] = _whiteness(arr)
    return out


def _config(kw: dict) -> nv.NavConfig:
    kw = dict(kw)
    align = kw.pop("alignment_sigma_deg", None)
    cfg = nv.NavConfig(**kw)
    if align is not None:
        cfg = replace(cfg, alignment_sigma=math.radians(align))
    return cfg


def _reduce(rows: list, dof_key: str, dof: int) -> dict:
    v = np.concatenate([np.asarray(r[dof_key], dtype=float) for r in rows])
    v = v[np.isfinite(v)]
    n_eff = len(rows)
    lo, hi = chi2_bounds(dof, max(n_eff, 1))
    return {"dof": dof, "mean": float(v.mean()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "samples": int(v.size),
            "runs": n_eff, "bound_lo": lo, "bound_hi": hi,
            "consistent": bool(lo <= v.mean() <= hi),
            "ratio_to_dof": float(v.mean() / dof)}


# ===========================================================================
def tune(pool, trajectories: dict, seeds: int = 6) -> dict:
    """
    Sweep the consider-inflations and report the NEES each gives.

    Reported in full, not just the winner: a reader has to be able to see that
    the adopted values sit on a plateau rather than at a fitted point.
    """
    grid = [dict(gnss_position_inflation=a, gnss_velocity_inflation=b,
                 mag_sigma_floor=c, alignment_sigma_deg=d)
            for a, b, c, d in itertools.product(
                TUNE_GRID["gnss_position_inflation"],
                TUNE_GRID["gnss_velocity_inflation"],
                TUNE_GRID["mag_sigma_floor"],
                TUNE_GRID["alignment_sigma_deg"])]
    cases = [("long", s, g) for g in grid for s in range(seeds)]
    print(f"  tuning: {len(grid)} combinations x {seeds} seeds "
          f"= {len(cases)} replays", flush=True)
    out = []
    results = pool.map(_run, cases)
    for i, g in enumerate(grid):
        rows = results[i * seeds:(i + 1) * seeds]
        row = dict(g)
        for key, dof in (("nees", 15), ("nees_nav", 9), ("nees_pv", 6)):
            row[key] = _reduce(rows, key, dof)["mean"]
            row[f"{key}_ratio"] = row[key] / dof
        pe = np.concatenate([np.asarray(r["position_error"]) for r in rows])
        ve = np.concatenate([np.asarray(r["velocity_error"]) for r in rows])
        ae = np.concatenate([np.asarray(r["attitude_error"]) for r in rows])
        row["position_rms_m"] = [float(v) for v in np.sqrt((pe ** 2).mean(axis=0))]
        row["velocity_rms_ms"] = [float(v) for v in np.sqrt((ve ** 2).mean(axis=0))]
        row["attitude_rms_deg"] = [float(v) for v in
                                   np.degrees(np.sqrt((ae ** 2).mean(axis=0)))]
        out.append(row)
    out.sort(key=lambda r: abs(math.log(max(r["nees_nav_ratio"], 1e-9))))
    return {"grid": TUNE_GRID, "seeds": seeds, "rows": out}


def campaign(pool, trajectories: dict, cfg_kw: dict,
             seeds: int = SEEDS_PER) -> dict:
    cases = [(lab, s, cfg_kw) for lab in trajectories for s in range(seeds)]
    print(f"  campaign: {len(trajectories)} trajectories x {seeds} seeds "
          f"= {len(cases)} runs", flush=True)
    rows = pool.map(_run, cases)
    out = {"config": cfg_kw, "engagements": list(trajectories),
           "seeds": seeds, "n_runs": len(rows), "per_engagement": {}}
    for key, dof in (("nees", 15), ("nees_nav", 9), ("nees_pv", 6)):
        out[key] = _reduce(rows, key, dof)
    for lab in trajectories:
        sub = [r for r in rows if r["engagement"] == lab]
        out["per_engagement"][lab] = {
            k: _reduce(sub, k, d) for k, d in
            (("nees", 15), ("nees_nav", 9), ("nees_pv", 6))}

    # -- NIS, per stream --------------------------------------------------
    dofs = {"gnss_pos": 3, "gnss_vel": 3, "mag": 3, "cn0": 1, "align": 3}
    out["nis"] = {}
    for stream, dof in dofs.items():
        vals = np.concatenate([np.asarray(r["nis"].get(stream, []), dtype=float)
                               for r in rows]) if rows else np.zeros(0)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        lo, hi = chi2_bounds(dof, len(rows))
        out["nis"][stream] = {
            "dof": dof, "mean": float(vals.mean()),
            "median": float(np.median(vals)), "samples": int(vals.size),
            "bound_lo": lo, "bound_hi": hi,
            "consistent": bool(lo <= vals.mean() <= hi),
            "ratio_to_dof": float(vals.mean() / dof)}

    # -- whiteness --------------------------------------------------------
    out["whiteness"] = {}
    for stream in dofs:
        acfs = [r["whiteness"][stream]["acf"] for r in rows
                if stream in r.get("whiteness", {}) and r["whiteness"][stream]["acf"]]
        if not acfs:
            continue
        a = np.array(acfs)
        bands = [r["whiteness"][stream]["band"] for r in rows
                 if stream in r.get("whiteness", {}) and r["whiteness"][stream]["acf"]]
        band = float(np.mean(bands))
        mean_acf = a.mean(axis=0)
        out["whiteness"][stream] = {
            "acf_mean": [float(v) for v in mean_acf],
            "band": band,
            "lag1": float(mean_acf[0]),
            "white": bool(abs(mean_acf[0]) <= band),
            "runs": int(a.shape[0])}

    # -- accuracy, for the record ----------------------------------------
    pe = np.concatenate([np.asarray(r["position_error"]) for r in rows])
    ve = np.concatenate([np.asarray(r["velocity_error"]) for r in rows])
    ae = np.concatenate([np.asarray(r["attitude_error"]) for r in rows])
    out["accuracy"] = {
        "position_rms_m": [float(v) for v in np.sqrt((pe ** 2).mean(axis=0))],
        "position_bias_m": [float(v) for v in pe.mean(axis=0)],
        "velocity_rms_ms": [float(v) for v in np.sqrt((ve ** 2).mean(axis=0))],
        "attitude_rms_deg": [float(v) for v in
                             np.degrees(np.sqrt((ae ** 2).mean(axis=0)))],
        "attitude_bias_deg": [float(v) for v in np.degrees(ae.mean(axis=0))],
    }
    out["rows"] = rows
    return out


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/nav_consistency.json")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS_PER)
    ap.add_argument("--tune-seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args(argv)

    sn.warn_low_confidence()
    t0 = time.time()
    md = gc.load_maps()
    labels = ("long",) if args.tune else ENGAGEMENTS
    print(f"flying {len(labels)} reference trajectories ...", flush=True)
    trajectories = {lab: nc.fly_truth(lab, md) for lab in labels}
    print(f"  {time.time() - t0:.1f} s", flush=True)

    adopted = {"gnss_position_inflation": nv.NavConfig.gnss_position_inflation,
               "gnss_velocity_inflation": nv.NavConfig.gnss_velocity_inflation,
               "mag_sigma_floor": nv.NavConfig.mag_sigma_floor,
               "alignment_sigma_deg": math.degrees(nv.NavConfig.alignment_sigma)}

    out = {"adopted": adopted}
    with Pool(args.workers, initializer=_init, initargs=(trajectories,)) as pool:
        if args.tune:
            out["tuning"] = tune(pool, trajectories, seeds=args.tune_seeds)
            print(f"\n{'kp':>7} {'kv':>7} {'mag':>6} {'align':>6} | "
                  f"{'NEES15/15':>10} {'NEES9/9':>9} {'NEES6/6':>9} | pos rms m")
            for r in out["tuning"]["rows"][:20]:
                print(f"{r['gnss_position_inflation']:7.1f} "
                      f"{r['gnss_velocity_inflation']:7.1f} "
                      f"{r['mag_sigma_floor']:6.3f} "
                      f"{r['alignment_sigma_deg']:6.2f} | "
                      f"{r['nees_ratio']:10.2f} {r['nees_nav_ratio']:9.2f} "
                      f"{r['nees_pv_ratio']:9.2f} | "
                      f"{np.round(r['position_rms_m'], 2)}")
        else:
            out["campaign"] = campaign(pool, trajectories, adopted,
                                       seeds=args.seeds)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out} in {time.time() - t0:.1f} s")

    if "campaign" in out:
        c = out["campaign"]
        for key, name in (("nees", "NEES, 15 states"),
                          ("nees_nav", "NEES, 9 navigation states"),
                          ("nees_pv", "NEES, position and velocity")):
            r = c[key]
            print(f"  {name:32s} {r['mean']:9.2f}  "
                  f"bounds [{r['bound_lo']:.2f}, {r['bound_hi']:.2f}]  "
                  f"{'PASS' if r['consistent'] else 'FAIL'}")
        for stream, r in c["nis"].items():
            print(f"  NIS {stream:12s} dof {r['dof']}  {r['mean']:9.2f}  "
                  f"{'PASS' if r['consistent'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
