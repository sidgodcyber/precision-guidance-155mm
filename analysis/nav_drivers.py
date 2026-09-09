"""
Step 5: WHICH error drives the navigation contribution to CEP?

Task F measures the contribution. `analysis.nav_antenna` tested the first
hypothesis -- that it is GNSS velocity error amplified over the time to go --
and only half confirmed it: moving the antenna onto the spin axis cuts the
velocity error five-fold, from 0.53 m/s to 0.10, and cuts the contribution by
only a fifth. So velocity is A driver and not THE driver, and something else
is carrying the rest.

This module removes each error source in turn, one at a time, and reports what
the contribution becomes. It is a decomposition by ABLATION rather than by
assumption, which is the only kind available when the terms interact through a
closed loop.

  baseline        everything on
  perfect_roll    the magnetometer calibration residual set to 0.1 %, which
                  removes the roll bias and most of the roll noise
  on_axis         the GNSS antenna phase centre on the spin axis
  both            both of the above

WHY ROLL IS A CANDIDATE AT ALL. docs/CONTROL-CHARACTERISATION.md section 4.3
measured that a roll error costs `1 - cos(e)` of the commanded correction --
59 parts per million at the servo's own 0.55 deg -- and concluded it is not a
CEP term. That is right about the component ALONG the commanded direction and
silent about the one PERPENDICULAR to it. A correction of 246 m applied 2.6 deg
off points 11 m sideways, and a roll BIAS points the same 11 m sideways every
cycle, so the loop cannot null it: it converges to a miss it has no direction
left to remove.

Run:  python -m analysis.nav_drivers
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from gnc import sensors as sn
from analysis import guidance_cep as gc, nav_cep as ncep, nav_common as nc

#: The ablations. Each key becomes a `run_guided_nav` argument override.
VARIANTS = {
    "baseline": {},
    "perfect_roll": {"mag_residual_fraction": 0.001},
    "on_axis": {"antenna_transverse": 0.0},
    "both": {"mag_residual_fraction": 0.001, "antenna_transverse": 0.0},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engagement", default="long")
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default="docs/nav_drivers.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args(argv)

    sn.warn_low_confidence()
    md = gc.load_maps()
    ctx = gc.engagement_context(md, args.engagement)
    opts = gc.scheduler_options(md, args.engagement)["proportional"]
    draws = gc.make_draws(ctx, args.draws)
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    t0 = time.time()

    out = {"engagement": args.engagement, "n_draws": args.draws,
           "n_seeds": args.seeds, "variants": {}}
    with Pool(args.workers) as pool:
        truth = ncep._imap(pool, nc.run_guided_nav,
                           ncep._cases(ctx, draws, use_nav=False,
                                       scheduler_opts=opts), "truth")
        out["truth_cep"] = ncep._cep(truth)
        for name in want:
            kw = VARIANTS[name]
            cases = []
            for s in range(args.seeds):
                cases += ncep._cases(ctx, draws, use_nav=True, seed=s,
                                     scheduler_opts=opts, keep_nav_log=True,
                                     **kw)
            rows = ncep._imap(pool, nc.run_guided_nav, cases, name)
            d = ncep.paired_navigation_delta(truth, rows)
            vel = [r["n_vel_rms_ms"] for r in rows if "n_vel_rms_ms" in r]
            rb = [abs(r["n_roll_bias_deg"]) for r in rows
                  if r.get("n_roll_bias_deg") is not None]
            rs = [r["n_roll_sd_deg"] for r in rows
                  if r.get("n_roll_sd_deg") is not None]
            out["variants"][name] = {
                "overrides": kw, "delta": d, "cep": ncep._cep(rows),
                "velocity_rms_ms": ([float(v) for v in
                                     np.sqrt((np.array(vel) ** 2).mean(axis=0))]
                                    if vel else None),
                "roll_bias_deg": float(np.mean(rb)) if rb else None,
                "roll_sd_deg": float(np.mean(rs)) if rs else None,
                "servo_error_deg": float(np.mean(
                    [r["servo_error_deg"] for r in rows])),
            }
            v = out["variants"][name]
            print(f"  {name:14s} range 1 sigma {d['range']['sigma_m']:6.2f} m, "
                  f"deflection {d['deflection']['sigma_m']:6.2f} m | "
                  f"roll bias {v['roll_bias_deg']:5.2f} deg, "
                  f"sd {v['roll_sd_deg']:5.2f} | "
                  f"vel rms {np.round(v['velocity_rms_ms'], 3)}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
