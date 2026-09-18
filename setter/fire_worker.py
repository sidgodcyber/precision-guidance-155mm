"""
FIRE's subprocess worker: one real guided round, navigation in the loop,
through the actual 6-DOF engine -- not the reduced-order model the ground
track elsewhere in the app uses.

Invoked as `python -m setter.fire_worker` (B3/B4: subprocess, never
exec/import-and-call from the main Streamlit process) with the current
Mission Control configuration and an output path; writes one JSON result
and exits. `setter/pages/mission_control.py`'s FIRE fragment launches this
via `subprocess.Popen` and polls for it.

THE REAL ENTRY POINT, TRACED (not assumed) from analysis/monte_carlo.py:
`task_a` (the campaign function that produces this project's headline CEP)
flies each round via `case_for()` -> `analysis.nav_common.run_guided_nav`.
That IS "what actually runs a single guided round" -- called here exactly
as the campaign calls it, unmodified.

WHAT THIS DOES AND DOES NOT REUSE FROM MISSION CONTROL
--------------------------------------------------------
The met profile, the fire-control solve (lay_gun/fuze_setting) and the
target are the SAME ones Mission Control already computed and is
displaying -- this worker does not redraw them, so "predicted impact" and
"actual impact" are directly comparable (same aim, same message). What
DOES get freshly drawn, every call, is the round's own physical
realisation: muzzle-velocity error, laying error, mass/inertia variation
and fuze-phase jitter -- using the exact sigma constants
`analysis.monte_carlo.sample_round` draws a campaign round with
(SIGMA_MV, SIGMA_QE, SIGMA_AZ, SIGMA_MASS_FRAC, SIGMA_IAXIAL_FRAC,
SIGMA_ITRANS_FRAC, FUZE_INCREMENT_S) -- because that IS what firing the
same fire-control solution twice means physically: same order, same gun,
different round.

TARGET OFFSET. Mission Control's "target offset" control is realised here
via `aim_off`/`aim_off_deflection` -- the engine's own, already-public
mechanism for shifting the guidance target away from the gun's
uncorrected impact point (see `run_guided_nav`: `target =
(uncorrected_range - aim_off, uncorrected_drift - aim_off_deflection)`).
Passing the negative of the operator's range/deflection offset makes the
guided round's target equal to uncorrected + offset, i.e. what the setter
message's `target_position` already states.

FUZE MODE IS NOT A SIMULATION INPUT HERE. `run_guided_nav` takes a `mode`
argument too, but it means something else entirely --
`gnc.guidance.GuidanceConfig.mode` ("full" / "reversionary" /
"stage2_fail", the guidance LAW's own operating mode), not the fuze event
engine's trigger type (time/motion/proximity/combined,
`fuze.config.EVENT_KINDS`). CLAUDE.md is explicit that `fuze/` is
simulated "against trajectories the engine already produces," i.e.
downstream of a flown round, not an input that changes how one flies. So
the fired round's mode stays "full" regardless of the Mission Control
fuze-mode selection; that selection is carried through to the result JSON
as metadata only, for display, not passed into the engine.

THE FULL 6-DOF STATE TRAJECTORY -- RESOLVED, WITH AN APPROVED FROZEN-CODE
CHANGE. `run_guided_nav` originally returned only a processed summary
dict; the `Trajectory` its own integration produces (position, velocity,
quaternion, Mach, ... at every logged step) was computed but discarded.
Traced and confirmed by reading the function, not assumed -- and reported
before any code was written, per the working rules for `analysis/`. The
resolution, approved explicitly rather than decided unilaterally: a fifth
`keep_state_trajectory` flag was added to `run_guided_nav`, in exactly the
shape of the four `keep_*_log` flags already there (serialises an
already-computed structure; changes nothing when absent). Verified
bit-for-bit identical output with the flag absent, both before and after
the change, on a fully deterministic case (fixed seeds throughout, no
`hash()`-derived seeding -- see the identity check this verification used).
This worker sets it, so `state_trajectory` below is the GUIDED-PHASE
trajectory (deployment to impact) of the actual fired round -- not a
reduced-order stand-in. It does not cover the pre-deployment leg (launch
to deployment), which `run_guided_nav` still logs too coarsely to extract;
Flight Deck's ground track therefore starts at deployment, not at the
muzzle, and should say so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from analysis import monte_carlo as mc
from analysis import nav_common as nc
from sim import projectile as pr
from setter import simulation_adapter as sa


def fire_one_round(engagement: str, met_age_bucket: str,
                    range_offset_m: float, defl_offset_m: float,
                    fuze_mode: str, fuze_event_time_s: float,
                    fuze_motion_threshold: float, fuze_proximity_trigger_m: float,
                    seed: int) -> dict:
    t0 = time.perf_counter()

    base = sa.baseline_for(engagement)
    ctx = sa.engagement_context(engagement)
    met_profile = sa.met_profile_for_age(met_age_bucket, seed=hash(engagement) & 0xFFFF)
    lay = sa.lay_gun(engagement, met_profile)
    deploy_time = sa.fuze_setting(engagement, met_profile, lay["dqe_mils"])

    rng = np.random.default_rng(seed)
    dmv = float(rng.normal(0.0, mc.SIGMA_MV))
    dqe_err = float(rng.normal(0.0, mc.SIGMA_QE))
    daz_err = float(rng.normal(0.0, mc.SIGMA_AZ))
    mass_f = 1.0 + float(rng.normal(0.0, mc.SIGMA_MASS_FRAC))
    ia_f = mass_f * (1.0 + float(rng.normal(0.0, mc.SIGMA_IAXIAL_FRAC)))
    it_f = mass_f * (1.0 + float(rng.normal(0.0, mc.SIGMA_ITRANS_FRAC)))
    d_fuze = float(rng.uniform(-0.5, 0.5) * mc.FUZE_INCREMENT_S)

    case = {
        "baseline": ctx["base"], "map": ctx["map"],
        "scheduler": "proportional", "scheduler_opts": ctx["scheduler_opts"],
        "met": met_profile, "use_nav": True,
        "dqe_mils": lay["dqe_mils"] + dqe_err / pr.MIL_TO_RAD,
        "daz": lay["daz"] + daz_err,
        "dmv": dmv,
        "deploy_time": deploy_time + d_fuze,
        "aim_off": -range_offset_m,
        "aim_off_deflection": -defl_offset_m,
        "projectile_overrides": {
            "mass": pr.M107.mass * mass_f,
            "I_axial": pr.M107.I_axial * ia_f,
            "I_transverse": pr.M107.I_transverse * it_f,
        },
        "seed": seed,
        "draw": 0,
        "keep_guidance_log": True,
        "keep_state_trajectory": True,
    }

    out = nc.run_guided_nav(case)
    elapsed_s = time.perf_counter() - t0

    return {
        "ok": True,
        "elapsed_s": elapsed_s,
        "engagement": engagement,
        "met_age_bucket": met_age_bucket,
        "range_offset_m": range_offset_m,
        "defl_offset_m": defl_offset_m,
        "fuze_mode_context_only": fuze_mode,
        "fuze_event_time_s": fuze_event_time_s,
        "fuze_motion_threshold": fuze_motion_threshold,
        "fuze_proximity_trigger_m": fuze_proximity_trigger_m,
        "seed": seed,
        "miss_range_m": out["miss_range_m"],
        "miss_defl_m": out["miss_defl_m"],
        "miss_m": out["miss_m"],
        "range_m": out["range_m"],
        "drift_m": out["drift_m"],
        "tof_s": out["tof_s"],
        "t_dep_actual": out["t_dep_actual"],
        "target_range": out["target_range"],
        "target_defl": out["target_defl"],
        "dqe_mils": out["dqe_mils"],
        "daz": out["daz"],
        "dmv": out["dmv"],
        "g_log": out.get("g_log"),
        "g_saturated_fraction": out.get("g_saturated_fraction"),
        "g_authority_ok": out.get("g_authority_ok"),
        "g_ever_saturated": out.get("g_ever_saturated"),
        "state_trajectory_available": True,
        "state_trajectory": out["state_trajectory"],
        "state_trajectory_note": (
            "Guided-phase only (deployment to impact) -- the pre-deployment "
            "leg (launch to deployment) is not covered; see this module's "
            "docstring."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engagement", required=True)
    ap.add_argument("--met-age", required=True)
    ap.add_argument("--range-offset", type=float, default=0.0)
    ap.add_argument("--defl-offset", type=float, default=0.0)
    ap.add_argument("--fuze-mode", default="time")
    ap.add_argument("--fuze-event-time-s", type=float, default=30.0)
    ap.add_argument("--fuze-motion-threshold", type=float, default=50.0)
    ap.add_argument("--fuze-proximity-trigger-m", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        result = fire_one_round(
            args.engagement, args.met_age, args.range_offset,
            args.defl_offset, args.fuze_mode, args.fuze_event_time_s,
            args.fuze_motion_threshold, args.fuze_proximity_trigger_m, args.seed)
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
