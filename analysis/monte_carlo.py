"""
Step 6 -- the dispersion campaign and the headline CEP.

Every prior step measured ONE contributor with everything else held still.
This module flies the whole loop against realistic dispersion and produces the
project's accuracy figure.

docs/MONTE-CARLO-DESIGN.md (the sampling design and the epistemic/aleatoric
split), docs/ATMOSPHERIC-ERROR.md, docs/MONITOR-RETUNE.md,
docs/STAGING-DECISION-FINAL.md, docs/CEP-FINAL.md.

THE ONE RULE THIS MODULE EXISTS TO ENFORCE
------------------------------------------
A quantity that varies round to round in reality must be drawn INDEPENDENTLY
FOR EVERY ROUND. Two prior steps got this wrong in the same way and both
found it themselves:

  step 4.5   five deployment phases understated a spread by a factor of five;
  step 5.5   the per-flight sensor bias vector was drawn once per SEED and
             reused across 24 draws, so the navigation sigma is resolved only
             to about +-10 m -- two seeds of the same draws gave 19.8 m and
             40.2 m.

`sample_round` therefore draws everything from one generator keyed on the
round index, and the sensor seed is the round index. Nothing is reused across
rounds, including the deployment phase, which the fuze cannot choose.

AND THE OPPOSITE RULE, WHICH IS CORRECT FOR THE OPPOSITE PURPOSE
---------------------------------------------------------------
A COMPARISON between two configurations pairs on COMMON RANDOM NUMBERS: the
same round index gives the same draw whatever configuration is flying it,
because the generator is keyed on the index and not on a stream position. The
dispersion then cancels in the difference and the comparison resolves an order
of magnitude better than either absolute number. `paired_delta` is that
comparison and every result in this module states which mode it is in.

TWO SAMPLING MODES, AND WHY THE STRATIFIED ONE IS NOT USED HERE
--------------------------------------------------------------
`analysis.guidance_cep.draws` is a stratified Halton sequence, and for a
MEDIAN over a small sample it is better than pseudo-random: it removes most of
the sampling noise. Step 6 does not use it for the absolute figure, for one
reason -- a bootstrap confidence interval on a low-discrepancy sample is not
valid, and Task A of this session requires the convergence to be MEASURED
rather than asserted. Independent draws cost precision at fixed n and buy a
confidence interval that means what it says. The comparisons keep their
precision through pairing instead.

THE TASKS, AND THE ORDER THEY DEPEND ON EACH OTHER IN
-----------------------------------------------------
  u   the uncorrected dispersion, MEASURED rather than assumed. Sizes the
      top-up every later task uses, so it runs first.
  c   the atmospheric knowledge term against met message age.
  cs  the same term against the ASSUMED atmospheric variability, so a reader
      who disagrees with the sigmas can rescale the answer.
  d   the authority monitor re-derived with navigation in the loop, as a
      false-alarm/missed-detection trade curve.
  a   the headline, with its convergence evidence. Flies the rounds that
      `n`, `e`, `g` and `b` reuse as their baseline arm, so it runs before
      them.
  n   the navigation contribution, paired truth-fed.
  e   staged against single stage, over the distributions.
  g   the two "cheap levers" of step 5, re-measured across many sensor seeds.
  b   the epistemic band.

Run, in this order -- these are the exact invocations step 6 used:

  python -m analysis.monte_carlo --tasks u --n 96  --engagement long
  python -m analysis.monte_carlo --tasks c --n 128 --engagement long
  python -m analysis.monte_carlo --tasks d --n 96  --engagement long
  python -m analysis.monte_carlo --tasks a --n 192 --engagement long
  python -m analysis.monte_carlo --tasks n --n 192 --n-e 96 --engagement long
  python -m analysis.monte_carlo --tasks e --n 192 --n-e 96 --engagement long
  python -m analysis.monte_carlo --tasks g --n 192 --n-b 96 --engagement long
  python -m analysis.monte_carlo --tasks b --n 192 --n-b 32 --engagement long
  python -m analysis.monte_carlo --tasks a --n 96 --met-age 0.0 --tag fresh_met       --engagement long
  python -m analysis.monte_carlo --tasks a --n 64 --no-inflate --tag physical       --engagement long
  python -m analysis.monte_carlo --tasks cs --n-c 32 --engagement long
  for E in mid2 middle short2 short; do
      python -m analysis.monte_carlo --tasks u --n 64 --u-quick --engagement $E
      python -m analysis.monte_carlo --tasks a --n 96 --engagement $E
  done
  python -m analysis.monte_carlo_report
  python -m analysis.monte_carlo_figures

THE SAMPLE COUNTS ARE SET BY THE MACHINE AND NOT BY THE STATISTICS, and that
is stated rather than hidden. The development machine is a four-core 15 W
laptop with 8 GB. A navigation-in-the-loop round at the long engagement costs
10.5 s of wall clock at eight-way parallelism on an otherwise idle machine and
14-19 s with an editor, a browser and a WSL virtual machine resident -- the
workers are then partly paged out, and the difference is memory pressure
rather than CPU. The whole set above is nine to twelve hours.

`keep_awake` below asks Windows not to suspend while a campaign runs, because
this one suspended for 4.8 hours in the middle of Task D. That costs no
correctness -- the pool simply resumes -- but it makes a long run
unschedulable and it made one progress line read 547 s per round.

docs/CEP-FINAL.md reports the convergence evidence and the residual
uncertainty these counts leave, rather than asserting that they are enough.
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

from sim import atmosphere as atm
from sim import canards as cn, dynamics as dyn, integrate as ig, projectile as pr
from gnc import guidance as gd
from gnc.inverse_map import AuthorityMap
from analysis import guidance_authority as ga
from analysis import guidance_cep as gc
from analysis import nav_common as nc
from analysis import roll_servo as rs

OUT = "docs/monte_carlo.json"

# ===========================================================================
# 1. THE PER-ROUND RANDOM VARIABLES -- the aleatoric set
# ===========================================================================
# Every number below is a 1 sigma, every one of them is drawn afresh for every
# round, and every one carries its source and a confidence grade. Where the
# project already carried a value it is REUSED rather than re-invented, and
# the place it came from is named.
#
# GRADES.  HIGH   measured in this project or taken from a primary source.
#          MEDIUM a standard engineering value for this class of hardware.
#          LOW    an estimate. Section 8 of docs/MONTE-CARLO-DESIGN.md reports
#                 what the answer does when each LOW term is moved.

#: Muzzle velocity, m/s. MEDIUM -- `gnc.navigation.GunData` has carried this
#: since step 5 as "the gun's round-to-round dispersion", where it sets the
#: warm start's a-priori covariance. It is 0.35 % of the charge-8 muzzle
#: velocity.
SIGMA_MV = 2.4

#: Quadrant elevation and azimuth, rad. MEDIUM, same source: the laying error
#: of a surveyed, calibrated piece. 0.5 mrad is about 0.9 mil.
SIGMA_QE = 0.5e-3
SIGMA_AZ = 0.5e-3

#: Projectile mass, as a FRACTION of nominal. LOW. `sim/projectile.py` records
#: a 0.85 % difference between the BRL mass of 43.454 kg and the M107 nominal
#: service weight of 43.09 kg and attributes it to "fuze and lot variation";
#: 0.3 % within a weight zone is the estimate this takes from that.
SIGMA_MASS_FRAC = 0.003

#: Axial and transverse inertia, as a fraction, ON TOP of the mass scaling.
#: A shell whose mass moves also moves its inertia; what is drawn here is the
#: residual variation in the DISTRIBUTION of that mass. LOW.
#: The axial term matters most: it sets the spin, and the spin sets the
#: Magnus force, the drift and the gyroscopic stability at deployment.
SIGMA_IAXIAL_FRAC = 0.010
SIGMA_ITRANS_FRAC = 0.005

#: Fuze setting, s, drawn UNIFORM on +-half a setting increment. MEDIUM on the
#: structure, LOW on the increment.
#:
#: This is the term that samples THE DEPLOYMENT PHASE, and the phase is the
#: reason it is here. The nose is turning at about 1308 rad/s at deployment,
#: a roll period of 4.8 ms, so a setting increment of 0.1 s is twenty periods
#: and the roll angle the kit deploys at is uniform on the circle whatever
#: else happens. docs/STAGED-DEPLOYMENT.md section 5 measured a 35-point
#: spread of retained authority across that phase for a single-stage kit; step
#: 4.5 found five samples of it understated the spread five-fold. It is not
#: fixed here and it is not sampled on a grid.
FUZE_INCREMENT_S = 0.1

#: Scale on every atmospheric sigma in `sim.atmosphere`. 1.0 is the nominal
#: model; docs/ATMOSPHERIC-ERROR.md sweeps it, so a reader who disagrees with
#: the assumed variability can rescale the answer without re-running.
MET_SCALE = 1.0

#: Age of the met message at fuze setting, hours. None means NO met message --
#: the fuze flies the standard atmosphere, which is what every step before 6
#: assumed and which is the bound on all the others.
MET_AGE_HOURS = 2.0

#: Uncorrected dispersion the project has quoted since
#: docs/ARCHITECTURE-DECISION.md section 2: CEP 200 m, range-dominated 3:1.
#: Task U measures what the physical draws above actually produce and this is
#: what the shortfall is made up to, as an explicit, named, flown term. See
#: docs/MONTE-CARLO-DESIGN.md section 2.3 -- the shortfall is large and it is
#: a finding in its own right.
TARGET_UNCORRECTED_CEP = 200.0
TARGET_UNCORRECTED_RATIO = 3.0


# ===========================================================================
# 2. THE EPISTEMIC SET -- a band around the CEP, never a term inside it
# ===========================================================================
# These are the same for every round of a lot and their value is not known.
# Folding a +-30 % canard uncertainty into a CEP as though it were per-round
# scatter would be wrong: every round of a lot flies with the same canards, so
# the CEP is a different number for each possible lot rather than a wider one
# for all of them. The correct treatment is to compute the CEP at the nominal
# and again at the edges and report the interval.
#
# Each entry is (label, kwargs to `run_guided_nav`).
EPISTEMIC = {
    "nominal": {},
    # docs/CANARD-MODEL.md section 8 and sim/canards.py ESTIMATES: +-20 %
    # subsonic and supersonic, +-40 % transonic. +-30 % is the band the
    # session brief and docs/CONTROL-ROBUSTNESS.md section 4.3 carry.
    "cla_low": {"cla_scale": 0.70},
    "cla_high": {"cla_scale": 1.30},
    # docs/COEFFICIENTS.md grades C_Ypalpha LOW: a single source, and BRL
    # could not extract it from full-scale swerve. The factor of 3.3 is the
    # spread between the sources considered.
    "magnus_low": {"aero_scales": {"C_Ypalpha": 1.0 / 3.3}},
    "magnus_high": {"aero_scales": {"C_Ypalpha": 3.3}},
    # docs/CONTROL-ROBUSTNESS.md sweeps the bearing over 0.25x-2.5x viscous
    # and 0.3x-3x Coulomb. The corners below are that document's own
    # "hardest to hold" and "slowest return".
    "bearing_light": {"nose_overrides": {"viscous": rs.NOSE.viscous * 0.4,
                                         "coulomb": rs.NOSE.coulomb * 0.3}},
    "bearing_heavy": {"nose_overrides": {"viscous": rs.NOSE.viscous * 1.6,
                                         "coulomb": rs.NOSE.coulomb * 1.7}},
    # gnc/roll_control.py BrakeActuator.tau, ESTIMATED at 10 ms.
    "coil_slow": {"brake_tau": 0.030},
}

#: The two corners a lot could actually sit in, taken TOGETHER rather than one
#: term at a time: the combination that helps and the combination that hurts.
#: A band built from single-term excursions understates its own width whenever
#: the terms do not cancel, and there is no reason they should.
EPISTEMIC_CORNERS = {
    "corner_favourable": {"cla_scale": 1.30,
                          "nose_overrides": {"viscous": rs.NOSE.viscous * 0.4,
                                             "coulomb": rs.NOSE.coulomb * 0.3}},
    "corner_adverse": {"cla_scale": 0.70,
                       "brake_tau": 0.030,
                       "nose_overrides": {"viscous": rs.NOSE.viscous * 1.6,
                                          "coulomb": rs.NOSE.coulomb * 1.7}},
}


# ===========================================================================
# 3. Drawing one round
# ===========================================================================
def round_rng(index: int, campaign_seed: int = 20260906) -> np.random.Generator:
    """
    The generator for round `index`.

    Keyed on the index rather than advanced from a stream, so round 37 is the
    same round whatever else the campaign flies and however many
    configurations have run before it. That is what makes common random
    numbers work across configurations without carrying state, and it is also
    what makes a campaign resumable.
    """
    return np.random.default_rng([campaign_seed, int(index)])


def sample_round(index: int, base: dict, met_scale: float = MET_SCALE,
                 met_age_hours=MET_AGE_HOURS, inflate: dict = None,
                 campaign_seed: int = 20260906, lay: bool = True) -> dict:
    """
    Every per-round random variable, drawn independently, and the fire-control
    solution that goes with it.

    ORDER MATTERS AND IT IS THE PHYSICAL ORDER.
      1. the atmosphere this round will fly through is realised;
      2. the met message fire control holds is a measurement of it, taken
         `met_age_hours` ago;
      3. the gun is LAID on that message -- quadrant elevation, azimuth and
         the fuze setting all come from the message and not from the air;
      4. the laying error, the muzzle-velocity error and the shell are drawn
         on top of the ordered solution;
      5. the round flies through the air from (1).

    `inflate` is the calibrated residual gun-and-lot dispersion of Task U,
    `{"sigma_mv": x, "sigma_az": y}`, an extra muzzle-velocity and azimuth
    offset that makes the total uncorrected dispersion match the 200 m CEP the
    project has quoted since step 2.5. It is drawn here, from this round's own
    generator, like everything else.
    """
    rng = round_rng(index, campaign_seed)

    # -- 1 and 2: the air, and what fire control was told about it ---------
    met = atm.MetProfile.sample(rng, scale=met_scale)
    # A "perfect" message is the air itself: a fuze that knows the atmosphere
    # exactly. It is not achievable and it is not proposed; it is the
    # REFERENCE Task C pairs every real message against, so that what is
    # measured is the message's error and not the atmosphere's anomaly.
    #
    # THE MESSAGE IS DRAWN FROM ITS OWN GENERATOR, and that is not a detail.
    # A "perfect" message consumes nothing and a stale one consumes four
    # correlated profiles, so drawing them from the shared stream would make
    # the muzzle velocity, the shell and the fuze offset of round 37 DIFFERENT
    # AT EVERY MESSAGE AGE -- and Task C's whole method is that they are the
    # same. The pairing would have quietly become an unpaired comparison of
    # two independent samples, which is exactly the class of error step 5.5
    # found and this module exists to avoid.
    msg_rng = np.random.default_rng([campaign_seed, int(index), 0xE7])
    msg = (met if met_age_hours == "perfect"
           else met.message(msg_rng, met_age_hours, scale=met_scale))

    # -- 3: the fire-control solution -------------------------------------
    sol = lay_gun(base, msg) if lay else {"dqe_mils": 0.0, "daz": 0.0,
                                          "lay_residual_m": 0.0,
                                          "lay_iterations": 0}
    t_set = fuze_setting(base, msg, sol["dqe_mils"]) if lay else base["deploy_time"]

    # -- 4: what the gun does that fire control did not order --------------
    dmv = float(rng.normal(0.0, SIGMA_MV))
    dqe_err = float(rng.normal(0.0, SIGMA_QE))
    daz_err = float(rng.normal(0.0, SIGMA_AZ))

    mass_f = 1.0 + float(rng.normal(0.0, SIGMA_MASS_FRAC))
    # Inertia scales with the mass it is made of, plus a residual variation in
    # how that mass is distributed.
    ia_f = mass_f * (1.0 + float(rng.normal(0.0, SIGMA_IAXIAL_FRAC)))
    it_f = mass_f * (1.0 + float(rng.normal(0.0, SIGMA_ITRANS_FRAC)))

    # The fuze is set to the nearest increment, and the increment is twenty
    # roll periods, so THE DEPLOYMENT PHASE IS UNIFORM ON THE CIRCLE and is
    # drawn afresh for every round. It is not a swept parameter here.
    d_fuze = float(rng.uniform(-0.5, 0.5) * FUZE_INCREMENT_S)

    if inflate:
        dmv += float(rng.normal(0.0, inflate.get("sigma_mv", 0.0)))
        daz_err += float(rng.normal(0.0, inflate.get("sigma_az", 0.0)))

    return {
        "index": int(index),
        "dmv": dmv,
        "dqe_mils": sol["dqe_mils"] + dqe_err / pr.MIL_TO_RAD,
        "daz": sol["daz"] + daz_err,
        "projectile_overrides": {
            "mass": pr.M107.mass * mass_f,
            "I_axial": pr.M107.I_axial * ia_f,
            "I_transverse": pr.M107.I_transverse * it_f,
        },
        "deploy_time": float(t_set) + d_fuze,
        "met": met,
        "met_message": msg,
        # THE SENSOR SEED IS THE ROUND INDEX. Never reused, never shared
        # between rounds, and the same for the same round under every
        # configuration -- which is the pairing.
        "seed": int(index),
        "_draw_scalars": {
            "dmv": dmv, "dqe_err_rad": dqe_err, "daz_err": daz_err,
            "lay_dqe_mils": sol["dqe_mils"], "lay_daz": sol["daz"],
            "lay_residual_m": sol["lay_residual_m"],
            "fuze_setting_s": float(t_set), "fuze_offset_s": d_fuze,
            "mass_frac": mass_f, "i_axial_frac": ia_f,
            "i_transverse_frac": it_f,
            **{f"met_{k}": v for k, v in met.summary().items() if k != "label"},
            **{f"msg_{k}": v for k, v in msg.summary().items() if k != "label"},
        },
    }


#: Draws are memoised because common random numbers means every configuration
#: flies the SAME ones, and re-deriving them costs a fire-control solve per
#: round. Keyed on everything that changes them.
_DRAWS: dict = {}


def draw_set(n: int, base: dict, met_scale: float = MET_SCALE,
             met_age_hours=MET_AGE_HOURS, inflate: dict = None,
             campaign_seed: int = 20260906) -> list:
    key = (n, base["label"], met_scale, met_age_hours, campaign_seed,
           None if not inflate else tuple(sorted(inflate.items())))
    got = _DRAWS.get(key)
    if got is None:
        t0 = time.time()
        got = [sample_round(i, base, met_scale, met_age_hours, inflate,
                            campaign_seed) for i in range(n)]
        _DRAWS[key] = got
        print(f"  drew {n} rounds ({time.time() - t0:.1f} s: a fire-control "
              f"solution and a fuze setting each)", flush=True)
    return got


def case_for(ctx: dict, draw: dict, **kw) -> dict:
    """One `nav_common.run_guided_nav` argument dict, from a drawn round."""
    mon = {k: v for k, v in ctx.get("monitor", {}).items()
           if not k.startswith("_")}
    case = {"baseline": ctx["base"], "map": ctx["map"], **mon,
            "draw": draw["index"], "scheduler": "proportional",
            "scheduler_opts": ctx["scheduler_opts"]}
    case.update({k: v for k, v in draw.items()
                 if not k.startswith("_") and k != "index"})
    # Epistemic overrides may carry nested dicts that must MERGE rather than
    # replace: a corner setting `nose_overrides` must not silently discard a
    # `projectile_overrides` the draw put there.
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(case.get(k), dict):
            case[k] = {**case[k], **v}
        else:
            case[k] = v
    return case


def context(mapdata: dict, label: str) -> dict:
    ctx = gc.engagement_context(mapdata, label)
    ctx["scheduler_opts"] = gc.scheduler_options(mapdata, label)["proportional"]
    ctx["label"] = label
    return ctx


# ===========================================================================
# 4. Statistics -- absolute, and paired
# ===========================================================================
def _cep(v: np.ndarray) -> float:
    return float(np.median(v))


def bootstrap_cep(v: np.ndarray, n_boot: int = 4000, seed: int = 5150,
                  alpha: float = 0.05) -> dict:
    """
    The CEP with a percentile bootstrap confidence interval.

    Valid here and not in the step-3 campaigns for one reason: these draws are
    independent. A bootstrap over a stratified low-discrepancy sample
    resamples a design, not a population, and its interval does not cover.
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 4:
        return {"n": int(v.size), "cep_m": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    b = np.median(v[idx], axis=1)
    return {
        "n": int(v.size),
        "cep_m": _cep(v),
        "cep_se_m": float(b.std(ddof=1)),
        "cep_lo_m": float(np.percentile(b, 100 * alpha / 2)),
        "cep_hi_m": float(np.percentile(b, 100 * (1 - alpha / 2))),
    }


def summarise(rows: list, key: str = "miss_m") -> dict:
    """Everything reported about one configuration's rounds."""
    ok = [r for r in rows if np.isfinite(r.get(key, float("nan")))]
    v = np.array([r[key] for r in ok], dtype=float)
    if v.size == 0:
        return {"n": 0}
    r = np.array([x["miss_range_m"] for x in ok], dtype=float)
    d = np.array([x["miss_defl_m"] for x in ok], dtype=float)
    out = dict(bootstrap_cep(v))
    out.update({
        "n_flown": len(rows),
        "mean_miss_m": float(v.mean()),
        "p90_m": float(np.percentile(v, 90)),
        "p95_m": float(np.percentile(v, 95)),
        "max_m": float(v.max()),
        "within_30m": float((v <= 30.0).mean()),
        "within_50m": float((v <= 50.0).mean()),
        "within_100m": float((v <= 100.0).mean()),
        "bias_range_m": float(r.mean()),
        "bias_defl_m": float(d.mean()),
        "sd_range_m": float(r.std(ddof=1)) if v.size > 1 else 0.0,
        "sd_defl_m": float(d.std(ddof=1)) if v.size > 1 else 0.0,
        # Excess kurtosis, because step 5.5 measured 5.94 on the navigation
        # distribution with one round in 48 carrying 42 % of the variance. A
        # CEP quoted without it can be a statement about 90 % of the rounds.
        "kurtosis_range": _excess_kurtosis(r),
        "max_variance_share": _max_variance_share(r),
    })
    for f in ("g_authority_ok", "g_ever_saturated", "g_terminal_saturated"):
        vals = [x.get(f) for x in ok if x.get(f) is not None]
        if vals:
            out[f.replace("g_", "") + "_fraction"] = float(np.mean(
                [bool(z) for z in vals]))
    return out


def _excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return float("nan")
    m = x - x.mean()
    s2 = float((m ** 2).mean())
    return float((m ** 4).mean() / (s2 * s2) - 3.0) if s2 > 0 else float("nan")


def _max_variance_share(x: np.ndarray) -> float:
    """Share of the total squared deviation carried by the single worst round."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float("nan")
    d = (x - x.mean()) ** 2
    return float(d.max() / d.sum()) if d.sum() > 0 else float("nan")


def paired_delta(rows_a: list, rows_b: list, key: str = "miss_m",
                 n_boot: int = 4000, seed: int = 9711) -> dict:
    """
    a minus b, on the rounds both flew, with a bootstrap over ROUNDS.

    The two configurations saw the same muzzle velocity, the same shell, the
    same atmosphere, the same fuze setting and the same sensor seed, because
    `round_rng` is keyed on the round index. Everything common therefore
    cancels in the difference and what is left is the configuration.
    """
    by_b = {r["draw"]: r for r in rows_b}
    pairs = [(r[key], by_b[r["draw"]][key]) for r in rows_a
             if r["draw"] in by_b
             and np.isfinite(r[key]) and np.isfinite(by_b[r["draw"]][key])]
    if len(pairs) < 4:
        return {"n": len(pairs)}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    dmed = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return {
        "n": int(a.size),
        "cep_a_m": _cep(a), "cep_b_m": _cep(b),
        "delta_cep_m": _cep(a) - _cep(b),
        "delta_cep_se_m": float(dmed.std(ddof=1)),
        "delta_cep_lo_m": float(np.percentile(dmed, 2.5)),
        "delta_cep_hi_m": float(np.percentile(dmed, 97.5)),
        "median_paired_delta_m": float(np.median(a - b)),
        "fraction_a_worse": float((a > b).mean()),
    }


def convergence(rows: list, key: str = "miss_m",
                steps=(16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                       768, 1024)) -> list:
    """
    The CEP against sample count, in the order the rounds were drawn.

    Reported rather than assumed. `stable_at` in the campaign output is the
    smallest count beyond which every later estimate stays inside the final
    estimate's own 95 % interval; if there is no such count the campaign says
    so and quotes the residual uncertainty instead of claiming convergence.
    """
    v = np.array([r[key] for r in sorted(rows, key=lambda x: x["draw"])],
                 dtype=float)
    v = v[np.isfinite(v)]
    out = []
    for n in steps:
        if n > v.size:
            break
        out.append({"n": n, **bootstrap_cep(v[:n], n_boot=2000)})
    if v.size not in [o["n"] for o in out]:
        out.append({"n": int(v.size), **bootstrap_cep(v, n_boot=4000)})
    return out


def stable_at(conv: list) -> dict:
    """The count at which the CEP stops moving, or an honest report that it
    has not."""
    if not conv:
        return {"converged": False, "reason": "no sample points"}
    final = conv[-1]
    _base = {"converged": False, "final_n": final["n"],
             "final_cep_m": final["cep_m"],
             "final_ci_m": [final.get("cep_lo_m"), final.get("cep_hi_m")],
             "half_width_m": 0.5 * (final.get("cep_hi_m", float("nan"))
                                    - final.get("cep_lo_m", float("nan"))),
             "half_width_pct": (50.0 * (final.get("cep_hi_m", float("nan"))
                                        - final.get("cep_lo_m", float("nan")))
                                / final["cep_m"]) if final["cep_m"] else
                               float("nan")}
    if len(conv) < 3:
        return {**_base, "reason": "too few sample points to test"}
    lo, hi = final["cep_lo_m"], final["cep_hi_m"]
    # At least three estimates must sit inside the final interval before the
    # CEP is called converged. Without that floor the last point is trivially
    # inside its own interval and EVERY campaign reports convergence at its
    # largest sample count, which is a tautology and not evidence.
    for i, c in enumerate(conv[:-2]):
        if all(lo <= later["cep_m"] <= hi for later in conv[i:]):
            return {"converged": True, "stable_at_n": c["n"],
                    "final_n": final["n"], "final_cep_m": final["cep_m"],
                    "final_ci_m": [lo, hi],
                    "half_width_m": 0.5 * (hi - lo),
                    "half_width_pct": 100.0 * 0.5 * (hi - lo) / final["cep_m"]}
    return {**_base,
            "reason": "no count beyond which every later estimate stays "
                      "inside the final interval"}


# ===========================================================================
# 5. Flying
# ===========================================================================
def _imap(pool, fn, cases, label):
    t0 = time.time()
    out = []
    n = len(cases)
    for i, r in enumerate(pool.imap_unordered(fn, cases, chunksize=1), 1):
        out.append(r)
        if i % 16 == 0 or i == n:
            el = time.time() - t0
            print(f"  {label}: {i}/{n}  {el:7.1f} s  "
                  f"(eta {el / i * (n - i):6.1f} s)", flush=True)
    return out


def fly(pool, cases, label):
    rows = _imap(pool, nc.run_guided_nav, cases, label)
    return [r for r in rows if r is not None]


def unguided(args) -> dict:
    """
    One round with NO KIT AT ALL: where this shell, this charge and this
    atmosphere would put it unguided.

    This is what Task U measures the dispersion of, and it is the quantity
    docs/ARCHITECTURE-DECISION.md section 2 assumed at CEP 200 m without ever
    generating it.
    """
    base = args["baseline"]
    proj = nc.round_projectile(args)
    world = nc.round_base_model(args, args.get("met"))
    launch = pr.LaunchConditions.from_mils(
        base["muzzle_velocity"] + args.get("dmv", 0.0),
        base["qe_mils"] + args.get("dqe_mils", 0.0),
        azimuth=args.get("daz", 0.0))
    y0 = dyn.initial_state(proj, launch)
    res = ig.integrate(y0, world, dt=args.get("dt", base["dt"]),
                       log_every=10 ** 9, t_max=300.0)
    return {
        "draw": args.get("draw"),
        "range_m": float(res.range_m), "drift_m": float(res.drift_m),
        "tof_s": float(res.impact_time),
        "d_range_m": float(res.range_m - base["uncorrected_range"]),
        "d_defl_m": float(res.drift_m - base["uncorrected_drift"]),
        **(args.get("_draw_scalars") or {}),
    }


# ===========================================================================
# TASK U -- what the physical draws actually disperse to
# ===========================================================================
def met_component(met, which: str):
    """
    A profile carrying only ONE of a met profile's three fields.

    Used to decompose the atmospheric contribution. `None` in, `None` out.
    """
    if met is None or met.is_standard:
        return met
    n = len(met.grid)
    zero_w = (0.0,) * n
    one = (1.0,) * n
    if which == "wind":
        return replace(met, density_ratio=one, temperature_ratio=one,
                       label="wind only")
    if which == "density":
        return replace(met, wind_north=zero_w, wind_east=zero_w,
                       temperature_ratio=one, label="density only")
    if which == "temperature":
        return replace(met, wind_north=zero_w, wind_east=zero_w,
                       density_ratio=one, label="temperature only")
    raise ValueError(which)


#: What each Task U variant keeps. `met` implies the fire-control solution
#: that goes with it -- a variant that flies a real atmosphere is laid on the
#: message for it, because that is what a gun does.
U_VARIANTS = (
    ("all", ("dmv", "laying", "shell", "met")),
    ("muzzle_velocity", ("dmv",)),
    ("laying", ("laying",)),
    ("shell", ("shell",)),
    ("met_message_error", ("met",)),
    ("met_wind", ("met_wind",)),
    ("met_density", ("met_density",)),
    ("met_temperature", ("met_temperature",)),
    # The control: the same atmospheres, flown with the gun laid at the
    # NOMINAL quadrant elevation. This is a round fired with no met message at
    # all, and the difference between it and `met_message_error` is what the
    # met correction is worth.
    ("no_met_correction", ("met_uncorrected",)),
)


def _u_case(ctx: dict, d: dict, keep) -> dict:
    base = ctx["base"]
    c = {"baseline": base, "draw": d["index"], "dt": base["dt"],
         "_draw_scalars": d["_draw_scalars"]}
    sc = d["_draw_scalars"]
    if "dmv" in keep:
        c["dmv"] = d["dmv"]
    if "laying" in keep:
        c["dqe_mils"] = sc["dqe_err_rad"] / pr.MIL_TO_RAD
        c["daz"] = sc["daz_err"]
    if "shell" in keep:
        c["projectile_overrides"] = d["projectile_overrides"]
    if "met" in keep:
        # laid on the message, flying the air
        c["met"] = d["met"]
        c["dqe_mils"] = c.get("dqe_mils", 0.0) + sc["lay_dqe_mils"]
        c["daz"] = c.get("daz", 0.0) + sc["lay_daz"]
    if "met_uncorrected" in keep:
        c["met"] = d["met"]
    for f in ("wind", "density", "temperature"):
        if f"met_{f}" in keep:
            part = met_component(d["met"], f)
            c["met"] = part
            sol = lay_gun(ctx["base"], met_component(d["met_message"], f))
            c["dqe_mils"] = c.get("dqe_mils", 0.0) + sol["dqe_mils"]
            c["daz"] = c.get("daz", 0.0) + sol["daz"]
    return c


def task_u(pool, mapdata: dict, label: str, n: int, met_scale: float,
           met_age, variants=None) -> dict:
    """
    Fly the drawn rounds UNGUIDED and measure the dispersion they produce.

    The project has assumed an uncorrected CEP of 200 m since step 2.5 and
    never generated one. This is the first measurement of it, and every
    sampling decision downstream depends on the answer.
    """
    ctx = context(mapdata, label)
    print(f"[U] uncorrected dispersion, {label}, n={n}, "
          f"met x{met_scale}, message {met_age} h old", flush=True)
    out = {"label": label, "n": n, "met_scale": met_scale,
           "met_age_hours": met_age, "components": {}}
    draws = draw_set(n, ctx["base"], met_scale, met_age)

    for vname, keep in U_VARIANTS:
        if variants is not None and vname not in variants:
            continue
        cases = [_u_case(ctx, d, keep) for d in draws]
        rows = _imap(pool, unguided, cases, f"U/{vname}")
        rows.sort(key=lambda x: x["draw"])
        r = np.array([x["d_range_m"] for x in rows])
        f = np.array([x["d_defl_m"] for x in rows])
        out["components"][vname] = {
            "n": len(rows),
            "sigma_range_m": float(r.std(ddof=1)),
            "sigma_defl_m": float(f.std(ddof=1)),
            "bias_range_m": float(r.mean()),
            "bias_defl_m": float(f.mean()),
            "cep_m": float(np.median(np.hypot(r - r.mean(), f - f.mean()))),
            "rows": [{"draw": x["draw"], "d_range_m": x["d_range_m"],
                      "d_defl_m": x["d_defl_m"]} for x in rows],
        }
        print(f"    {vname:20s} sigma_r {r.std(ddof=1):8.2f} m  "
              f"sigma_d {f.std(ddof=1):7.2f} m  bias_r {r.mean():8.2f} m",
              flush=True)

    tot = out["components"]["all"]
    scale = ctx["base"]["uncorrected_range"] / gc.DISPERSION_REFERENCE_RANGE
    want_r, want_d = gc.sigmas_for(TARGET_UNCORRECTED_CEP * scale,
                                   TARGET_UNCORRECTED_RATIO)
    need_r = math.sqrt(max(0.0, want_r ** 2 - tot["sigma_range_m"] ** 2))
    need_d = math.sqrt(max(0.0, want_d ** 2 - tot["sigma_defl_m"] ** 2))
    dr_dmv = _range_per_mv(ctx["base"])
    out["target"] = {
        "cep_m": TARGET_UNCORRECTED_CEP * scale,
        "sigma_range_m": want_r, "sigma_defl_m": want_d,
        "modelled_sigma_range_m": tot["sigma_range_m"],
        "modelled_sigma_defl_m": tot["sigma_defl_m"],
        "modelled_fraction_of_variance_range":
            float(tot["sigma_range_m"] ** 2 / want_r ** 2),
        "residual_sigma_range_m": need_r,
        "residual_sigma_defl_m": need_d,
        "d_range_per_mv": dr_dmv,
        "inflate": {"sigma_mv": need_r / dr_dmv,
                    "sigma_az": need_d / ctx["base"]["uncorrected_range"]},
    }
    print(f"  [U] modelled {tot['sigma_range_m']:.1f} m of {want_r:.1f} m "
          f"range sigma "
          f"({100 * tot['sigma_range_m'] ** 2 / want_r ** 2:.1f} % of variance)"
          f"; residual to add {need_r:.1f} m", flush=True)
    return out


def _range_per_mv(base: dict) -> float:
    """d(range)/d(muzzle velocity), m per m/s, from the reduced-order model."""
    r0 = gc._mpmm_range(base, 0.0)
    return (gc._mpmm_range(base, 2.0) - r0) / 2.0


# ===========================================================================
# THE FIRE CONTROL SOLUTION
# ===========================================================================
# A real gun is not laid at the standard atmosphere and then fired into a real
# one. It is laid using THE MET MESSAGE, and the residual error is the
# message's error and not the atmosphere's anomaly. Leaving this out was worth
# a factor of five: the first pass at Task U measured 196 m of range sigma
# from the atmosphere, which is the dispersion of a gun fired with no met
# correction at all -- a real quantity, and the reason met messages exist, but
# not the one an accuracy budget wants.
#
# So the sequence per round is:
#   1. the met message is what fire control has;
#   2. it lays the gun -- quadrant elevation and azimuth -- so that the
#      reduced-order model lands the round on the aim point IN THAT
#      ATMOSPHERE;
#   3. the laying error, the muzzle-velocity error and the shell are drawn on
#      top of that solution;
#   4. the round flies through the REAL atmosphere.
# What is left is the met message's error, expressed as a miss. It is
# `analysis.monte_carlo`'s single most consequential structural choice and
# docs/ATMOSPHERIC-ERROR.md section 3 is the measurement of what it is worth.
_LAY_CACHE: dict = {}

#: The last (profile, model) pair, held as a STRONG reference pair rather than
#: keyed on id(). Keying a cache on id() of a temporary is a latent
#: correctness bug, not a style point: a freed MetProfile's address is
#: reusable, so a later profile allocated at the same address would silently
#: be flown through the previous one's atmosphere. It was written that way
#: first and `test_a_perfect_message_lands_the_lay_on_the_aim_point` caught it.
_LAST_MET = object()
_LAST_MODEL = None


def _mpmm_for(met):
    """The reduced-order model in `met`'s atmosphere, cached for one profile."""
    global _LAST_MET, _LAST_MODEL
    if met is _LAST_MET and _LAST_MODEL is not None:
        return _LAST_MODEL
    from models import mpmm
    from sim import aerodata
    kw = {}
    if met is not None and not met.is_standard:
        kw = {"wind": met.wind, "atmosphere": met.scalars}
    model = mpmm.MpmmModel(projectile=pr.M107,
                           aero=aerodata.make_m107_table(),
                           environment=rs.base_model().environment,
                           iterate_yaw=True, **kw)
    _LAST_MET, _LAST_MODEL = met, model
    return model


def _mpmm_impact(base: dict, dqe_mils: float, daz: float, met,
                 dt: float = 0.2) -> tuple:
    """(range, deflection) of the reduced-order model under `met`."""
    from models import mpmm
    launch = pr.LaunchConditions.from_mils(base["muzzle_velocity"],
                                           base["qe_mils"] + dqe_mils,
                                           azimuth=daz)
    y0 = mpmm.initial_state(pr.M107, launch)
    r = mpmm.propagate_to_impact(y0, _mpmm_for(met), dt=dt)
    return float(r.range_m), float(r.drift_m)


def lay_gun(base: dict, met, tol_m: float = 0.5, max_iter: int = 8) -> dict:
    """
    The quadrant elevation and azimuth fire control would order, given this
    met message.

    Range by a secant solve on quadrant elevation, deflection by rotating the
    azimuth -- the same two knobs and the same method
    `guidance_cep.perturbation_for` uses, applied to the fire-control problem
    rather than to realising a drawn miss.

    `met = None` is the standard atmosphere and returns the nominal lay, which
    is what every step before 6 fired.
    """
    want_r = base["uncorrected_range"]
    want_d = base["uncorrected_drift"]
    if met is None or met.is_standard:
        return {"dqe_mils": 0.0, "daz": 0.0, "lay_iterations": 0,
                "lay_residual_m": 0.0}
    # TWO OUTER PASSES, and the second one is not decoration. Rotating the
    # gun by the azimuth that removes the deflection error also shortens the
    # DOWNRANGE component by r(1 - cos daz), which at this engagement is 3.5 m
    # -- larger than the range tolerance the secant is solving to. Solving the
    # two once each and stopping leaves that behind as a deterministic bias in
    # every met-corrected round.
    x1, daz, f1, it = 0.0, 0.0, 0.0, 0
    for _pass in range(2):
        x0 = x1
        r0, _d0 = _mpmm_impact(base, x0, daz, met)
        f0 = r0 - want_r
        x1 = x0 - f0 / _qe_sensitivity(base)
        for it in range(1, max_iter + 1):
            r1, d1 = _mpmm_impact(base, x1, daz, met)
            f1 = r1 - want_r
            if abs(f1) < tol_m or f1 == f0:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            x0, f0 = x1, f1
            x1 = x2
        else:
            r1, d1 = _mpmm_impact(base, x1, daz, met)
            f1 = r1 - want_r
        daz += (want_d - d1) / max(r1, 1.0)
    r1, d1 = _mpmm_impact(base, x1, daz, met)
    return {"dqe_mils": float(x1), "daz": float(daz), "lay_iterations": it,
            "lay_residual_m": float(math.hypot(r1 - want_r, d1 - want_d))}


def fuze_setting(base: dict, met, dqe_mils: float,
                 dt: float = 0.1) -> float:
    """
    The deployment time fire control would order, from ITS OWN solution.

    The kit deploys at a fixed fraction of apogee time -- 0.25 at the adopted
    engagement (docs/ARCHITECTURE-DECISION.md section 1) -- and apogee time is
    a property of the trajectory fire control has computed, which is the one
    in the MET MESSAGE'S atmosphere. A message that moves the ordered quadrant
    elevation by 15 mils moves apogee time by several tenths of a second, and
    that is LARGER than the setting increment, so it cannot be treated as
    noise about the nominal.

    The consequence is that the deployment Mach, altitude and roll phase all
    vary round to round through the met message as well as through the gun.
    """
    from models import mpmm
    from sim import aerodata
    if met is None or met.is_standard:
        return float(base["deploy_time"])
    key = ("apogee_nominal", base["label"])
    ta0 = _LAY_CACHE.get(key)
    if ta0 is None:
        ta0 = _apogee_time(base, None, 0.0, dt)
        _LAY_CACHE[key] = ta0
    frac = float(base["deploy_time"]) / ta0
    return frac * _apogee_time(base, met, dqe_mils, dt)


def _apogee_time(base: dict, met, dqe_mils: float, dt: float = 0.1) -> float:
    from models import mpmm
    from sim import aerodata
    kw = {}
    if met is not None and not met.is_standard:
        kw = {"wind": met.wind, "atmosphere": met.scalars}
    model = mpmm.MpmmModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                           environment=rs.base_model().environment,
                           iterate_yaw=True, **kw)
    launch = pr.LaunchConditions.from_mils(base["muzzle_velocity"],
                                           base["qe_mils"] + dqe_mils)
    y0 = mpmm.initial_state(pr.M107, launch)
    r = mpmm.propagate_to_impact(y0, model, dt=dt, log_every=1)
    z = np.asarray(r.position)[:, 2]
    return float(np.asarray(r.t)[int(np.argmin(z))])


def _qe_sensitivity(base: dict) -> float:
    """d(range)/d(QE), m per mil, at the nominal. Cached per engagement."""
    key = ("dr_dqe", base["label"])
    v = _LAY_CACHE.get(key)
    if v is None:
        r0, _ = _mpmm_impact(base, 0.0, 0.0, None)
        r1, _ = _mpmm_impact(base, 2.0, 0.0, None)
        v = (r1 - r0) / 2.0
        _LAY_CACHE[key] = v
    return v


# ===========================================================================
# TASK C -- the atmospheric knowledge term
# ===========================================================================
#: Met message ages swept, hours. "perfect" is a message equal to the truth --
#: a fuze that knows the air exactly -- and it is the REFERENCE every other
#: age is paired against, because the difference from it is the knowledge
#: error and nothing else. `None` is no message at all: the fuze flies the
#: standard atmosphere, which is what every step before 6 assumed.
#: 0, 1, 3 and 6 hours are the sweep the session brief asks for. 2 hours is
#: added because it is the age the headline campaign flies (`MET_AGE_HOURS`),
#: and a budget line for the adopted configuration should be measured rather
#: than interpolated between two that were.
MET_AGES = ("perfect", 0.0, 1.0, 2.0, 3.0, 6.0, None)


def task_c(pool, mapdata: dict, label: str, n: int, met_scale: float,
           use_nav: bool = False, inflate: dict = None, ages=None) -> dict:
    """
    What the met message's error costs at the impact point.

    Truth-fed by default -- the guidance law reads the true state -- so the
    term is isolated from the 30 m of navigation contribution that would
    otherwise swamp it. Every age flies THE SAME ATMOSPHERES, the same shells
    and the same muzzle velocities as the perfect-message reference, because
    the draws are keyed on the round index; only the message changes. The
    difference is therefore the message.

    Note what varies with the message and not only the predictor: fire control
    lays the gun on it and sets the fuze from it, so a stale message moves the
    quadrant elevation, the azimuth and the deployment time as well as the
    onboard impact-point prediction. All four are part of the term.
    """
    ctx = context(mapdata, label)
    print(f"[C] atmospheric knowledge, {label}, n={n}, met x{met_scale}, "
          f"{'nav in loop' if use_nav else 'truth-fed'}", flush=True)
    out = {"label": label, "n": n, "met_scale": met_scale,
           "use_nav": use_nav, "inflate": inflate,
           "ages_flown": [("perfect" if a == "perfect" else
                           ("none" if a is None else a))
                          for a in (ages if ages is not None else MET_AGES)],
           "ages": {}}
    flown = {}
    for age in (ages if ages is not None else MET_AGES):
        key = "none" if age is None else (
            "perfect" if age == "perfect" else f"{age:g}h")
        draws = draw_set(n, ctx["base"], met_scale, age, inflate)
        cases = [case_for(ctx, d, use_nav=use_nav) for d in draws]
        rows = fly(pool, cases, f"C/{key}")
        flown[key] = rows
        out["ages"][key] = {"age_hours": age, **summarise(rows)}
        s = out["ages"][key]
        print(f"    {key:9s} CEP {s['cep_m']:7.2f} m  "
              f"sd_range {s['sd_range_m']:7.2f}  sd_defl {s['sd_defl_m']:6.2f}  "
              f"bias_range {s['bias_range_m']:8.2f}", flush=True)

    # The term itself: the PAIRED difference against a perfect message.
    ref = flown["perfect"]
    by_ref = {r["draw"]: r for r in ref}
    for key, rows in flown.items():
        if key == "perfect":
            continue
        dr, dd = [], []
        for r in rows:
            b = by_ref.get(r["draw"])
            if b is None:
                continue
            dr.append(r["miss_range_m"] - b["miss_range_m"])
            dd.append(r["miss_defl_m"] - b["miss_defl_m"])
        a, b_ = np.array(dr), np.array(dd)
        out["ages"][key]["knowledge_term"] = {
            "n": int(a.size),
            "sigma_range_m": float(a.std(ddof=1)),
            "sigma_defl_m": float(b_.std(ddof=1)),
            "bias_range_m": float(a.mean()),
            "bias_defl_m": float(b_.mean()),
            "rms_range_m": float(np.sqrt((a ** 2).mean())),
            "rms_defl_m": float(np.sqrt((b_ ** 2).mean())),
            "cep_contribution_m": float(np.median(np.hypot(a, b_))),
        }
        k = out["ages"][key]["knowledge_term"]
        print(f"    -> {key:9s} knowledge term  1 sigma range "
              f"{k['sigma_range_m']:7.2f} m  deflection "
              f"{k['sigma_defl_m']:6.2f} m", flush=True)
    out["rows"] = {k: _thin(v) for k, v in flown.items()}
    return out


def _thin(rows: list) -> list:
    """Per-round fields worth keeping in the campaign JSON."""
    keep = ("draw", "seed", "miss_m", "miss_range_m", "miss_defl_m",
            "range_m", "drift_m", "tof_s", "deploy_time", "t_dep_actual",
            "stage2_delay_s", "g_authority_ok", "g_authority_fault_time",
            "g_ever_saturated", "g_terminal_saturated", "g_saturated_fraction",
            "g_armed_cycles", "max_total_aoa_deg", "acquisition_s")
    return [{k: r[k] for k in keep if k in r} for r in rows]


def task_cs(pool, mapdata: dict, label: str, n: int, met_age,
            scales=(0.5, 1.0, 2.0), inflate: dict = None) -> dict:
    """
    How the atmospheric knowledge term moves with the ASSUMED variability.

    Every sigma in `sim.atmosphere` is multiplied by one factor, so this
    answers the question a reader who disagrees with section 2.2 will ask
    without making them re-run anything. If the term is linear in the scale
    the whole of section 2.2's uncertainty reduces to multiplying one number,
    and how close to linear it is is measured here rather than assumed.

    Paired at each scale against a fuze given a perfect message AT THAT SCALE,
    which is the only reference that isolates the message from the air.
    """
    ctx = context(mapdata, label)
    print(f"[CS] knowledge term vs assumed variability, {label}, n={n}, "
          f"message {met_age} h old", flush=True)
    out = {"label": label, "n": n, "met_age_hours": met_age, "scales": {}}
    for sc in scales:
        pair = {}
        for tag, age in (("perfect", "perfect"), ("aged", met_age)):
            draws = draw_set(n, ctx["base"], sc, age, inflate)
            pair[tag] = fly(pool, [case_for(ctx, d, use_nav=False)
                                   for d in draws], f"CS/x{sc:g}/{tag}")
        by = {r["draw"]: r for r in pair["perfect"]}
        dr = np.array([r["miss_range_m"] - by[r["draw"]]["miss_range_m"]
                       for r in pair["aged"] if r["draw"] in by])
        dd = np.array([r["miss_defl_m"] - by[r["draw"]]["miss_defl_m"]
                       for r in pair["aged"] if r["draw"] in by])
        out["scales"][f"x{sc:g}"] = {
            "scale": sc, "n": int(dr.size),
            "sigma_range_m": float(dr.std(ddof=1)),
            "sigma_defl_m": float(dd.std(ddof=1)),
            "bias_range_m": float(dr.mean()),
            "cep_contribution_m": float(np.median(np.hypot(dr, dd))),
            "cep_aged_m": summarise(pair["aged"])["cep_m"],
            "cep_perfect_m": summarise(pair["perfect"])["cep_m"],
        }
        s = out["scales"][f"x{sc:g}"]
        print(f"    x{sc:<4g} term range 1s {s['sigma_range_m']:7.2f} m  "
              f"defl {s['sigma_defl_m']:6.2f} m   "
              f"CEP {s['cep_perfect_m']:6.2f} -> {s['cep_aged_m']:6.2f} m",
              flush=True)
    # Is it linear in the scale? Reported as the ratio of the term at each
    # scale to the term at 1.0 divided by the scale itself: 1.0 is linear.
    ref = out["scales"].get("x1")
    if ref and ref["sigma_range_m"] > 0:
        for k, v in out["scales"].items():
            v["linearity"] = float(v["sigma_range_m"]
                                   / (ref["sigma_range_m"] * v["scale"]))
        print("    linearity (1.0 = the term scales with the assumed sigma): "
              + ", ".join(f"{k} {v['linearity']:.3f}"
                          for k, v in out["scales"].items()), flush=True)
    return out


# ===========================================================================
# TASK D -- the authority monitor, re-derived with navigation in the loop
# ===========================================================================
#: The thresholds swept. `monitor_fraction` is how much of the map's promised
#: movement must be observed; `monitor_floor_m` is the promise below which the
#: test is not applied at all.
MONITOR_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
MONITOR_FLOORS = (10.0, 20.0, 32.3, 50.0, 80.0, 120.0)


def would_fault(mlog: list, fraction: float, floor: float) -> float:
    """
    The time this round's monitor WOULD have faulted at, or None.

    Replays the recorded per-window test statistic. Exact for a round that was
    allowed to fly to impact without faulting, which is why the sweep is flown
    with the monitor disabled and only shadow-logged.
    """
    for w in mlog or ():
        e = w["expected_m"]
        if e >= floor and w["promise_m"] > 0.0 and w["along_m"] < fraction * e:
            return float(w["t"])
    return None


def task_d(pool, mapdata: dict, label: str, n: int, met_scale: float,
           met_age, inflate: dict = None, confirm=None) -> dict:
    """
    The false-alarm and missed-detection rates, as a pair, at every threshold.

    METHOD. Both populations -- healthy rounds and rounds whose stage-2
    mechanism failed -- are flown ONCE with the monitor DISABLED and its
    arithmetic shadow-logged. Every window of every round therefore exists,
    and any (fraction, floor) pair can be evaluated on the recorded statistic
    without re-flying. That is exact, because a round that never faulted has
    the full window sequence; it would not be exact if the monitor had acted,
    since a fault ends the flight the windows describe.

    What the sweep CANNOT give is the CEP at a chosen threshold, because a
    fault changes the trajectory. So the operating points in `confirm` are
    re-flown with the monitor really on.

    REPORTED AS A PAIR, ALWAYS. Raising the threshold suppresses the false
    alarm and degrades detection of the mechanical failure the monitor exists
    for; docs/DEGRADATION-LADDER.md section 5 measured that detection at 69 %.
    A re-tune reporting only the CEP improvement would be hiding half of what
    it did.
    """
    ctx = context(mapdata, label)
    # THE SETTING IN FORCE IS THE ENGAGEMENT'S OWN, NOT THE CLASS DEFAULT.
    # `guidance_cep.monitor_options` sizes `monitor_floor_m` from the measured
    # prediction error at each engagement -- 32.3 m at `long`, 16.3 m at
    # `middle` -- and overrides `GuidanceConfig.monitor_floor_m` (20.0). A
    # re-tune measured against 20.0 would be comparing itself with a setting
    # nothing has ever flown.
    adopted = (float(ctx["monitor"]["monitor_fraction"]),
               float(ctx["monitor"]["monitor_floor_m"]))
    print(f"[D] monitor re-derivation, {label}, n={n}, navigation IN the loop; "
          f"the setting in force is fraction {adopted[0]:.2f}, floor "
          f"{adopted[1]:.2f} m, start {ctx['monitor']['monitor_start_s']:.2f} s",
          flush=True)
    draws = draw_set(n, ctx["base"], met_scale, met_age, inflate)
    out = {"label": label, "n": n, "sweep": [], "confirm": {},
           "adopted": {"monitor_fraction": adopted[0],
                       "monitor_floor_m": round(adopted[1], 3),
                       "monitor_start_s": ctx["monitor"]["monitor_start_s"],
                       "monitor_window_s": ctx["monitor"]["monitor_window_s"]}}

    shadow = {}
    for mode in ("full", "stage2_fail"):
        cases = [case_for(ctx, d, mode=mode, use_nav=True,
                          authority_monitor=False, monitor_shadow=True)
                 for d in draws]
        shadow[mode] = fly(pool, cases, f"D/shadow/{mode}")
        w = [len(r.get("monitor_log") or ()) for r in shadow[mode]]
        print(f"    {mode:12s} {len(shadow[mode])} rounds, "
              f"{np.mean(w):.1f} monitor windows each", flush=True)
        out[f"unmonitored_{mode}"] = summarise(shadow[mode])

    # The adopted floor is swept explicitly as well as the round numbers, so
    # the setting in force appears in the table rather than being interpolated.
    floors = sorted(set(MONITOR_FLOORS) | {round(adopted[1], 3)})
    fractions = sorted(set(MONITOR_FRACTIONS) | {adopted[0]})
    for fr in fractions:
        for fl in floors:
            fa = [would_fault(r.get("monitor_log"), fr, fl)
                  for r in shadow["full"]]
            md = [would_fault(r.get("monitor_log"), fr, fl)
                  for r in shadow["stage2_fail"]]
            det = [t for t in md if t is not None]
            dep = float(np.mean([r["t_dep_actual"]
                                 for r in shadow["stage2_fail"]]))
            out["sweep"].append({
                "monitor_fraction": fr, "monitor_floor_m": fl,
                "false_alarm_rate": float(np.mean([t is not None for t in fa])),
                "detection_rate": float(np.mean([t is not None for t in md])),
                "missed_detection_rate":
                    float(np.mean([t is None for t in md])),
                "median_detection_delay_s":
                    float(np.median([t for t in det]) - dep) if det else None,
                "n_false_alarms": int(sum(t is not None for t in fa)),
                "n_detections": int(len(det)),
            })
    # The adopted floor goes into the sweep ROUNDED (it is a dict key and a
    # label), so the row lookup must use the rounded value or it matches
    # nothing -- 32.307 against 32.3069987 is 1.3e-6 apart.
    _print_trade(out["sweep"], round(adopted[1], 3))

    # -- CHOOSE THE OPERATING POINTS FROM THE SWEEP, NOT FROM A GUESS ------
    # Three, and the criterion for each is stated rather than implied:
    #   the step-3 setting, because the retune has to be measured against
    #     what it replaces;
    #   the best detection available at a false-alarm rate of 2 % or less,
    #     which is the point a re-tune would actually adopt;
    #   the best detection available at ANY false-alarm rate, which is what
    #     the monitor could do if false alarms were free -- and they are not,
    #     so it is quoted to show the price rather than to be adopted.
    points = [(adopted[0], round(adopted[1], 3),
               "the setting in force since step 3")]
    ok = [r for r in out["sweep"] if r["false_alarm_rate"] <= 0.02]
    if ok:
        b = max(ok, key=lambda r: (r["detection_rate"],
                                   -(r["median_detection_delay_s"] or 1e9)))
        points.append((b["monitor_fraction"], b["monitor_floor_m"],
                       "best detection at FA <= 2 %"))
    # A THIRD point -- the best detection available at any false-alarm rate --
    # was in an earlier version and is not flown. The trade curve already
    # reports the false-alarm and detection rates at every threshold, so the
    # only thing re-flying that point adds is its CEP, and it is a setting
    # nobody would adopt. `out["sweep"]` carries it for anyone who wants it.
    best_det = max(r["detection_rate"] for r in out["sweep"])
    b2 = min([r for r in out["sweep"] if r["detection_rate"] >= best_det],
             key=lambda r: r["false_alarm_rate"])
    out["best_detection_available"] = dict(b2)
    seen = set()
    chosen = []
    for fr, fl, why in points:
        if (fr, fl) in seen:
            continue
        seen.add((fr, fl))
        chosen.append((fr, fl, why))
    out["chosen"] = [{"monitor_fraction": f, "monitor_floor_m": l,
                      "criterion": w} for f, l, w in chosen]
    print(f"    confirming {len(chosen)} operating points: "
          f"{[(f, l, w) for f, l, w in chosen]}", flush=True)

    # -- the operating points, re-flown with the monitor really acting -----
    for fr, fl, why in chosen:
        key = f"f{fr:g}_floor{fl:g}"
        rows = {}
        for mode in ("full", "stage2_fail"):
            cases = [case_for(ctx, d, mode=mode, use_nav=True,
                              authority_monitor=True,
                              monitor_fraction=fr, monitor_floor_m=fl)
                     for d in draws]
            rows[mode] = fly(pool, cases, f"D/{key}/{mode}")
        out["confirm"][key] = {
            "monitor_fraction": fr, "monitor_floor_m": fl,
            "criterion": why,
            "healthy": summarise(rows["full"]),
            "stage2_fail": summarise(rows["stage2_fail"]),
            "healthy_vs_unmonitored":
                paired_delta(rows["full"], shadow["full"]),
            "detected_fraction": float(np.mean(
                [not r["g_authority_ok"] for r in rows["stage2_fail"]])),
            "false_alarm_fraction": float(np.mean(
                [not r["g_authority_ok"] for r in rows["full"]])),
        }
        c = out["confirm"][key]
        print(f"    confirm {key}: healthy CEP {c['healthy']['cep_m']:.2f} m, "
              f"FA {100 * c['false_alarm_fraction']:.1f} %, "
              f"detect {100 * c['detected_fraction']:.1f} %", flush=True)
    out["rows"] = {f"shadow_{k}": _thin(v) for k, v in shadow.items()}
    return out


def _print_trade(sweep: list, floor: float) -> None:
    print(f"    trade curve at the adopted floor ({floor:.1f} m):", flush=True)
    print("      fraction   FA %   detect %   median delay s", flush=True)
    for r in sorted(sweep, key=lambda x: x["monitor_fraction"]):
        if abs(r["monitor_floor_m"] - floor) > 1e-3:
            continue
        d = r["median_detection_delay_s"]
        print(f"      {r['monitor_fraction']:8.2f} {100 * r['false_alarm_rate']:6.1f} "
              f"{100 * r['detection_rate']:10.1f}   "
              f"{'--' if d is None else f'{d:8.2f}'}", flush=True)


# ===========================================================================
# TASK E -- staged versus single stage, over the DISTRIBUTIONS
# ===========================================================================
def task_e(pool, mapdata: dict, label: str, n: int, met_scale: float,
           met_age, inflate: dict = None, baseline_rows=None,
           monitor: dict = None) -> dict:
    """
    Settle the staging decision by running both configurations over the full
    dispersion, paired on common random numbers.

    docs/STAGED-DEPLOYMENT.md section 8.2 recorded one number that could
    reverse the adoption and said explicitly that step 6 should settle it "by
    running the CEP projection over both DISTRIBUTIONS rather than over both
    means". This is that, with the loop closed and navigation in it rather
    than through the projection.

    Both configurations see the same atmospheres, shells, muzzle velocities,
    fuze offsets and sensor seeds. The tails are reported alongside the
    medians because step 5.5 found excess kurtosis 5.94 with one round in 48
    carrying 42 % of the variance -- on a distribution like that a comparison
    of means is the wrong test.
    """
    ctx = context(mapdata, label)
    print(f"[E] staged vs single stage, {label}, n={n}, paired", flush=True)
    draws = draw_set(n, ctx["base"], met_scale, met_age, inflate)
    mon = monitor or {}
    out = {"label": label, "n": n, "configs": {}}
    rows = {}
    for cname, kw in (("staged", {}),
                      ("single_stage", {"staging": {"enabled": False}})):
        if cname == "staged" and baseline_rows is not None:
            rows[cname] = [r for r in baseline_rows if r["draw"] < n]
            print(f"    staged: reusing {len(rows[cname])} rounds already "
                  f"flown for the headline campaign", flush=True)
        else:
            cases = [case_for(ctx, d, use_nav=True, **mon, **kw) for d in draws]
            rows[cname] = fly(pool, cases, f"E/{cname}")
        out["configs"][cname] = summarise(rows[cname])
        s = out["configs"][cname]
        print(f"    {cname:13s} CEP {s['cep_m']:7.2f} "
              f"[{s['cep_lo_m']:6.2f}, {s['cep_hi_m']:6.2f}]  "
              f"p90 {s['p90_m']:7.2f}  max {s['max_m']:8.2f}  "
              f"kurt {s['kurtosis_range']:5.2f}", flush=True)
    out["paired"] = paired_delta(rows["single_stage"], rows["staged"])
    p = out["paired"]
    print(f"    single minus staged: {p['delta_cep_m']:+.2f} m "
          f"[{p['delta_cep_lo_m']:+.2f}, {p['delta_cep_hi_m']:+.2f}] "
          f"(single worse on {100 * p['fraction_a_worse']:.0f} % of rounds)",
          flush=True)
    out["rows"] = {k: _thin(v) for k, v in rows.items()}
    return out


# ===========================================================================
# TASK A/F -- the headline campaign
# ===========================================================================
def task_a(pool, mapdata: dict, label: str, n: int, met_scale: float,
           met_age, inflate: dict = None, monitor: dict = None,
           extra: dict = None, tag: str = "headline") -> dict:
    """
    The headline CEP at one engagement, with the convergence evidence.

    Everything aleatoric is in and drawn per round; nothing epistemic is.
    Guided and unguided are both flown on the same draws, because the scatter
    plot needs both and pairing them is free.
    """
    ctx = context(mapdata, label)
    print(f"[A:{tag}] headline, {label}, n={n}", flush=True)
    draws = draw_set(n, ctx["base"], met_scale, met_age, inflate)
    kw = dict(monitor or {})
    kw.update(extra or {})
    cases = [case_for(ctx, d, use_nav=True, **kw) for d in draws]
    rows = fly(pool, cases, f"A/{tag}/{label}")
    rows.sort(key=lambda r: r["draw"])

    ucases = [_u_case(ctx, d, ("dmv", "laying", "shell", "met"))
              for d in draws]
    urows = _imap(pool, unguided, ucases, f"A/{tag}/{label}/unguided")
    urows.sort(key=lambda r: r["draw"])
    umiss = np.array([math.hypot(r["d_range_m"], r["d_defl_m"]) for r in urows])

    conv = convergence(rows)
    out = {
        "label": label, "n": n, "tag": tag,
        "met_scale": met_scale, "met_age_hours": met_age,
        "inflate": inflate, "monitor": monitor, "extra": extra,
        "guided": summarise(rows),
        "unguided": {
            "n": len(urows),
            "cep_m": float(np.median(umiss)),
            "sigma_range_m": float(np.std([r["d_range_m"] for r in urows],
                                          ddof=1)),
            "sigma_defl_m": float(np.std([r["d_defl_m"] for r in urows],
                                         ddof=1)),
        },
        "convergence": conv,
        "converged": stable_at(conv),
        "rows": _thin(rows),
        "unguided_rows": [{"draw": r["draw"], "d_range_m": r["d_range_m"],
                           "d_defl_m": r["d_defl_m"]} for r in urows],
    }
    g = out["guided"]
    print(f"  [A:{tag}] CEP {g['cep_m']:.2f} m "
          f"[{g['cep_lo_m']:.2f}, {g['cep_hi_m']:.2f}] at n={g['n']}; "
          f"unguided {out['unguided']['cep_m']:.1f} m; "
          f"{100 * g['within_30m']:.1f} % within 30 m", flush=True)
    print(f"  [A:{tag}] convergence: {out['converged']}", flush=True)
    return out


def task_n(pool, mapdata: dict, label: str, n: int, met_scale: float,
           met_age, inflate: dict = None, monitor: dict = None,
           baseline_rows=None) -> dict:
    """
    The NAVIGATION contribution under the full step-6 dispersion, paired.

    The same rounds are flown truth-fed -- the guidance law reads the true
    6-DOF state -- and differenced against the headline campaign's
    navigation-fed rounds. That difference IS the navigation contribution,
    and it is the same construction step 5 used.

    WHAT IS DIFFERENT FROM STEP 5 IS THE THING STEP 5.5 SAID WAS WRONG WITH
    IT. Step 5 drew the per-flight sensor bias vector once per SEED and reused
    it across 24 draws, so 72 rounds were three samples of the quantity that
    set the answer and the 30.4 m it reported is resolved only to about
    +-10 m. Here every round has its own sensor seed, so n rounds are n
    samples of it, and the interval below is the first one that means what it
    says.
    """
    ctx = context(mapdata, label)
    print(f"[N] navigation contribution, {label}, n={n}, truth-fed pair",
          flush=True)
    draws = draw_set(n, ctx["base"], met_scale, met_age, inflate)
    mon = monitor or {}
    if baseline_rows is None:
        nav_rows = fly(pool, [case_for(ctx, d, use_nav=True, **mon)
                              for d in draws], "N/navigation")
    else:
        nav_rows = [r for r in baseline_rows if r["draw"] < n]
        print(f"    navigation-fed: reusing {len(nav_rows)} rounds", flush=True)
    truth_rows = fly(pool, [case_for(ctx, d, use_nav=False, **mon)
                            for d in draws], "N/truth")
    out = {"label": label, "n": n,
           "navigation_fed": summarise(nav_rows),
           "truth_fed": summarise(truth_rows),
           "paired": paired_delta(nav_rows, truth_rows)}
    by = {r["draw"]: r for r in truth_rows}
    for axis, key in (("range", "miss_range_m"), ("defl", "miss_defl_m")):
        d = np.array([r[key] - by[r["draw"]][key] for r in nav_rows
                      if r["draw"] in by])
        out[f"contribution_{axis}"] = {
            "n": int(d.size), "sigma_m": float(d.std(ddof=1)),
            "bias_m": float(d.mean()), "rms_m": float(np.sqrt((d ** 2).mean())),
            **_boot_sigma(d),
        }
    p_ = out["paired"]
    print(f"    CEP {out['truth_fed']['cep_m']:.2f} m truth-fed -> "
          f"{out['navigation_fed']['cep_m']:.2f} m with navigation "
          f"(paired {p_['delta_cep_m']:+.2f} m "
          f"[{p_['delta_cep_lo_m']:+.2f}, {p_['delta_cep_hi_m']:+.2f}])",
          flush=True)
    for axis in ("range", "defl"):
        c = out[f"contribution_{axis}"]
        print(f"    {axis:5s} contribution 1 sigma {c['sigma_m']:6.2f} m "
              f"[{c['sigma_lo_m']:.2f}, {c['sigma_hi_m']:.2f}], "
              f"bias {c['bias_m']:+.2f} m", flush=True)
    out["rows"] = {"truth_fed": _thin(truth_rows)}
    return out


def _boot_sigma(d: np.ndarray, n_boot: int = 4000, seed: int = 8821) -> dict:
    """Bootstrap interval on a 1 sigma."""
    if d.size < 4:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    s = d[idx].std(axis=1, ddof=1)
    return {"sigma_se_m": float(s.std(ddof=1)),
            "sigma_lo_m": float(np.percentile(s, 2.5)),
            "sigma_hi_m": float(np.percentile(s, 97.5))}


# ===========================================================================
# TASK B -- the epistemic band
# ===========================================================================
def task_b(pool, mapdata: dict, label: str, n: int, met_scale: float,
           met_age, inflate: dict = None, monitor: dict = None,
           baseline_rows=None, which=None) -> dict:
    """
    The CEP at the edges of every epistemic range, paired against the nominal.

    NOT a term inside the CEP. Every round of a lot flies with the same
    canards, the same bearing and the same coil, so an uncertainty in them
    makes the CEP a DIFFERENT NUMBER for each possible lot rather than a wider
    number for all of them. What is reported is the interval the CEP lies in,
    and the corners are flown together as well as one at a time because a band
    built from single-term excursions understates its own width.
    """
    ctx = context(mapdata, label)
    print(f"[B] epistemic band, {label}, n={n}", flush=True)
    draws = draw_set(n, ctx["base"], met_scale, met_age, inflate)
    mon = monitor or {}
    cases_of = lambda kw: [case_for(ctx, d, use_nav=True, **mon, **kw)
                           for d in draws]
    variants = dict(EPISTEMIC)
    variants.update(EPISTEMIC_CORNERS)
    if which:
        variants = {k: v for k, v in variants.items()
                    if k in which or k == "nominal"}
    out = {"label": label, "n": n, "variants": {}}
    rows = {}
    for vname, kw in variants.items():
        if vname == "nominal" and baseline_rows is not None:
            rows[vname] = [r for r in baseline_rows if r["draw"] < n]
            print(f"    nominal: reusing {len(rows[vname])} rounds already "
                  f"flown for the headline campaign", flush=True)
        else:
            rows[vname] = fly(pool, cases_of(kw), f"B/{vname}")
        out["variants"][vname] = {"settings": _settings_repr(kw),
                                  **summarise(rows[vname])}
        s = out["variants"][vname]
        print(f"    {vname:20s} CEP {s['cep_m']:7.2f} "
              f"[{s['cep_lo_m']:6.2f}, {s['cep_hi_m']:6.2f}]  "
              f"within30 {100 * s['within_30m']:5.1f} %", flush=True)
    for vname in variants:
        if vname == "nominal":
            continue
        out["variants"][vname]["paired_vs_nominal"] = paired_delta(
            rows[vname], rows["nominal"])
    ceps = {k: v["cep_m"] for k, v in out["variants"].items()}
    single = {k: v for k, v in ceps.items() if not k.startswith("corner_")}
    out["band"] = {
        "nominal_cep_m": ceps["nominal"],
        "min_cep_m": float(min(ceps.values())),
        "max_cep_m": float(max(ceps.values())),
        "argmin": min(ceps, key=ceps.get), "argmax": max(ceps, key=ceps.get),
        "single_term_min_m": float(min(single.values())),
        "single_term_max_m": float(max(single.values())),
    }
    b = out["band"]
    print(f"  [B] CEP is {b['nominal_cep_m']:.1f} m and lies between "
          f"{b['min_cep_m']:.1f} m ({b['argmin']}) and "
          f"{b['max_cep_m']:.1f} m ({b['argmax']})", flush=True)
    out["rows"] = {k: _thin(v) for k, v in rows.items()}
    return out


def _settings_repr(kw: dict) -> dict:
    return {k: (v if not isinstance(v, dict)
                else {kk: float(vv) for kk, vv in v.items()})
            for k, v in kw.items()}


# ===========================================================================
# TASK G -- the two "cheap levers", re-measured across many sensor seeds
# ===========================================================================
#: docs/NAV-CEP.md sections 3 and 4 priced two mounting decisions and step 5.5
#: found that their RANGE figures do not reproduce: the antenna measured
#: -7.5 m in one campaign and +1.8 +- 1.3 m in the other, the magnetometer
#: floor -4.6 m and +3.2 +- 2.0 m. The cause was named -- the per-flight
#: sensor bias vector was drawn once per SEED and only three seeds were flown,
#: so 72 rounds were three samples of the quantity that set the answer.
#:
#: This re-measures them with a DIFFERENT SENSOR SEED ON EVERY ROUND, paired
#: on common random numbers, which is the condition step 5.5 said the figures
#: should be treated as unmeasured until they met.
LEVERS = {
    "antenna_on_axis": {"antenna_transverse": 0.0},
    "mag_floor_0.1pct": {"mag_residual_fraction": 0.001},
}
#: The two together were flown in step 5 and are not re-flown here. The
#: question this task exists to settle is whether EACH figure reproduces, and
#: a combination cannot answer that for either of them.


def task_g(pool, mapdata: dict, label: str, n: int, met_scale: float,
           met_age, inflate: dict = None, monitor: dict = None,
           baseline_rows=None) -> dict:
    """
    Re-measure the two cheap levers, or withdraw them.

    Reported as a PAIRED difference with its bootstrap interval, per axis,
    because that is the only form in which the previous figures failed and the
    only form in which a replacement can be trusted. A lever whose interval
    straddles zero is reported as not measured rather than as zero.
    """
    ctx = context(mapdata, label)
    print(f"[G] cheap levers re-measured, {label}, n={n}, "
          f"one sensor seed per round", flush=True)
    draws = draw_set(n, ctx["base"], met_scale, met_age, inflate)
    mon = monitor or {}
    out = {"label": label, "n": n, "levers": {}}
    rows = {}
    if baseline_rows is not None:
        rows["baseline"] = [r for r in baseline_rows if r["draw"] < n]
        print(f"    baseline: reusing {len(rows['baseline'])} rounds",
              flush=True)
    else:
        rows["baseline"] = fly(pool, [case_for(ctx, d, use_nav=True, **mon)
                                      for d in draws], "G/baseline")
    out["levers"]["baseline"] = summarise(rows["baseline"])

    for lname, kw in LEVERS.items():
        rows[lname] = fly(pool, [case_for(ctx, d, use_nav=True, **mon, **kw)
                                 for d in draws], f"G/{lname}")
        out["levers"][lname] = {"settings": _settings_repr(kw),
                                **summarise(rows[lname])}
        # Per axis, paired, with the interval. The range figure is the one
        # that failed to reproduce and the deflection figure is the one that
        # did, so they are reported separately and not as a radial miss.
        for axis, key in (("range", "miss_range_m"), ("defl", "miss_defl_m")):
            b = {r["draw"]: r for r in rows["baseline"]}
            d = np.array([r[key] - b[r["draw"]][key] for r in rows[lname]
                          if r["draw"] in b])
            base_sd = np.std([b[r["draw"]][key] for r in rows[lname]
                              if r["draw"] in b], ddof=1)
            new_sd = np.std([r[key] for r in rows[lname] if r["draw"] in b],
                            ddof=1)
            boot = _boot_sd_delta(
                np.array([b[r["draw"]][key] for r in rows[lname]
                          if r["draw"] in b]),
                np.array([r[key] for r in rows[lname] if r["draw"] in b]))
            out["levers"][lname][f"{axis}_delta"] = {
                "n": int(d.size),
                "baseline_sigma_m": float(base_sd),
                "lever_sigma_m": float(new_sd),
                "delta_sigma_m": float(new_sd - base_sd),
                **boot,
                "delta_bias_m": float(d.mean()),
            }
        r = out["levers"][lname]["range_delta"]
        f = out["levers"][lname]["defl_delta"]
        print(f"    {lname:18s} range 1s {r['baseline_sigma_m']:6.2f} -> "
              f"{r['lever_sigma_m']:6.2f}  ({r['delta_sigma_m']:+6.2f} "
              f"[{r['delta_sigma_lo_m']:+6.2f}, {r['delta_sigma_hi_m']:+6.2f}])"
              f"   defl {f['baseline_sigma_m']:6.2f} -> {f['lever_sigma_m']:6.2f}"
              f"  ({f['delta_sigma_m']:+6.2f} "
              f"[{f['delta_sigma_lo_m']:+6.2f}, {f['delta_sigma_hi_m']:+6.2f}])",
              flush=True)
        out["levers"][lname]["reproduces"] = {
            "range": bool(r["delta_sigma_hi_m"] < 0.0),
            "deflection": bool(f["delta_sigma_hi_m"] < 0.0),
        }
    out["rows"] = {k: _thin(v) for k, v in rows.items()}
    return out


def _boot_sd_delta(base: np.ndarray, lever: np.ndarray, n_boot: int = 4000,
                   seed: int = 3311) -> dict:
    """
    Bootstrap interval on the CHANGE in 1 sigma, resampling the two arms
    TOGETHER by round.

    Differencing two independently quoted sigmas throws away the pairing and
    is what made the original figures irreproducible: at 24 draws each sigma
    carries about 15 % of standard error, which is +-5 m on a 32 m number, and
    every lever step 5 found was smaller than that.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, base.size, size=(n_boot, base.size))
    d = lever[idx].std(axis=1, ddof=1) - base[idx].std(axis=1, ddof=1)
    return {
        "delta_sigma_se_m": float(d.std(ddof=1)),
        "delta_sigma_lo_m": float(np.percentile(d, 2.5)),
        "delta_sigma_hi_m": float(np.percentile(d, 97.5)),
    }


def _write(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=_jsonable)
    os.replace(tmp, path)
    print(f"wrote {path}", flush=True)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, atm.MetProfile):
        return o.summary()
    return str(o)


# ===========================================================================
def keep_awake() -> str:
    """
    Ask Windows not to sleep while the campaign runs.

    A campaign is hours long and unattended, and the development machine
    suspended for 4.8 hours in the middle of Task D -- which does not corrupt
    anything, because the pool simply resumes, but it makes a twelve-hour run
    unschedulable and it made one progress line read 547 s per round.

    `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` is a
    PER-PROCESS request. It is released automatically when this process exits,
    it does not change the machine's power scheme, and it leaves the display
    free to sleep. Nothing outside this process is affected and nothing has to
    be put back.
    """
    if os.name != "nt":
        return "not Windows; no request made"
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        prev = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        return "system sleep suppressed for this process" if prev else "refused"
    except Exception as exc:                       # pragma: no cover
        return f"unavailable ({type(exc).__name__})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="u",
                    help="u,c,d,e,a,b -- run in that order, they depend on "
                         "each other in it")
    ap.add_argument("--engagement", default="long")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--n-c", type=int, default=None, help="Task C override")
    ap.add_argument("--n-d", type=int, default=None)
    ap.add_argument("--n-e", type=int, default=None)
    ap.add_argument("--n-b", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--met-scale", type=float, default=MET_SCALE)
    ap.add_argument("--met-age", type=float, default=MET_AGE_HOURS)
    ap.add_argument("--ages", default=None,
                    help="Task C: comma-separated message ages to fly, e.g. "
                         "'perfect,2.0'. Default is the full sweep.")
    ap.add_argument("--u-quick", action="store_true",
                    help="Task U: fly only the `all` variant, which is all "
                         "that is needed to size the dispersion top-up. The "
                         "decomposition is run once, at the adopted "
                         "engagement.")
    ap.add_argument("--no-inflate", action="store_true",
                    help="do not top the physical dispersion up to the "
                         "project's quoted 200 m uncorrected CEP")
    ap.add_argument("--monitor-fraction", type=float, default=None)
    ap.add_argument("--monitor-floor", type=float, default=None)
    ap.add_argument("--single-stage", action="store_true")
    ap.add_argument("--tag", default="headline")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    mapdata = gc.load_maps()
    out = {}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            out = json.load(fh)
    out.setdefault("config", {})
    out["config"].update({
        "sigma_mv": SIGMA_MV, "sigma_qe_rad": SIGMA_QE, "sigma_az_rad": SIGMA_AZ,
        "sigma_mass_frac": SIGMA_MASS_FRAC,
        "sigma_i_axial_frac": SIGMA_IAXIAL_FRAC,
        "sigma_i_transverse_frac": SIGMA_ITRANS_FRAC,
        "fuze_increment_s": FUZE_INCREMENT_S,
        "met_scale": args.met_scale, "met_age_hours": args.met_age,
        "wind_sigma_anchors": list(atm.WIND_SIGMA_ANCHORS),
        "density_sigma_anchors": list(atm.DENSITY_SIGMA_ANCHORS),
        "temperature_sigma_anchors": list(atm.TEMPERATURE_SIGMA_ANCHORS),
        "decorrelation_hours": dict(atm.DECORRELATION_HOURS),
        "fresh_message_sigma": dict(atm.FRESH_MESSAGE_SIGMA),
    })

    # The calibrated top-up, from Task U if it has run.
    inflate = None
    if not args.no_inflate:
        u = (out.get("u") or {}).get(args.engagement)
        if u:
            inflate = u["target"]["inflate"]
            print(f"using Task U inflation {inflate}", flush=True)
        elif set(tasks) - {"u"}:
            print("WARNING: no Task U result for this engagement; the "
                  "uncorrected dispersion will be whatever the physical "
                  "draws produce and will NOT match the project's quoted "
                  "200 m. Run --tasks u first.", flush=True)

    monitor = {}
    if args.monitor_fraction is not None:
        monitor["monitor_fraction"] = args.monitor_fraction
    if args.monitor_floor is not None:
        monitor["monitor_floor_m"] = args.monitor_floor
    extra = {"staging": {"enabled": False}} if args.single_stage else None

    print(f"keep-awake: {keep_awake()}", flush=True)
    t_start = time.time()
    with Pool(args.workers) as pool:
        if "u" in tasks:
            out.setdefault("u", {})[args.engagement] = task_u(
                pool, mapdata, args.engagement, args.n,
                args.met_scale, args.met_age,
                variants=(("all",) if args.u_quick else None))
            _write(args.out, out)
            # Task U is what sizes the top-up, so a run that does U and then
            # anything else must pick it up HERE and not from the file it was
            # read from before U existed.
            if not args.no_inflate:
                inflate = out["u"][args.engagement]["target"]["inflate"]
                print(f"using Task U inflation {inflate}", flush=True)

        if "c" in tasks:
            out.setdefault("c", {})[args.engagement] = task_c(
                pool, mapdata, args.engagement, args.n_c or args.n,
                args.met_scale, inflate=inflate,
                ages=(tuple(("perfect" if a == "perfect" else
                             (None if a == "none" else float(a)))
                            for a in args.ages.split(","))
                      if args.ages else None))
            _write(args.out, out)

        if "cs" in tasks:
            out.setdefault("cs", {})[args.engagement] = task_cs(
                pool, mapdata, args.engagement, args.n_c or args.n,
                args.met_age, inflate=inflate)
            _write(args.out, out)

        if "d" in tasks:
            out.setdefault("d", {})[args.engagement] = task_d(
                pool, mapdata, args.engagement, args.n_d or args.n,
                args.met_scale, args.met_age, inflate)
            _write(args.out, out)

        if "a" in tasks:
            out.setdefault("a", {}).setdefault(args.tag, {})[args.engagement] =                 task_a(pool, mapdata, args.engagement, args.n, args.met_scale,
                       args.met_age, inflate, monitor, extra, args.tag)
            _write(args.out, out)

        base_rows = None
        head = ((out.get("a") or {}).get(args.tag) or {}).get(args.engagement)
        if head and (head.get("monitor") or None) == (monitor or None)                 and (head.get("extra") or None) == (extra or None):
            # Common random numbers means the headline campaign's rounds ARE
            # this comparison's baseline arm, round for round. Re-flying them
            # would produce identical numbers at full cost.
            base_rows = head["rows"]

        if "e" in tasks:
            out.setdefault("e", {})[args.engagement] = task_e(
                pool, mapdata, args.engagement, args.n_e or args.n,
                args.met_scale, args.met_age, inflate,
                baseline_rows=base_rows, monitor=monitor)
            _write(args.out, out)

        if "n" in tasks:
            out.setdefault("n", {})[args.engagement] = task_n(
                pool, mapdata, args.engagement, args.n_e or args.n,
                args.met_scale, args.met_age, inflate, monitor,
                baseline_rows=base_rows)
            _write(args.out, out)

        if "g" in tasks:
            out.setdefault("g", {})[args.engagement] = task_g(
                pool, mapdata, args.engagement, args.n_b or args.n,
                args.met_scale, args.met_age, inflate, monitor,
                baseline_rows=base_rows)
            _write(args.out, out)

        if "b" in tasks:
            out.setdefault("b", {})[args.engagement] = task_b(
                pool, mapdata, args.engagement, args.n_b or args.n,
                args.met_scale, args.met_age, inflate, monitor,
                baseline_rows=base_rows)
            _write(args.out, out)

    print(f"total {time.time() - t_start:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
