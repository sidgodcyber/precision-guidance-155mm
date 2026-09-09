"""
Step 5, the sensitivity that follows from Task F: what does the SPINNING
ANTENNA cost the CEP?

Task F measures a navigation contribution that grows with range -- 4.3 m at
4 km to 30.4 m at 15.8 km, 1 sigma in range -- and the shape of that growth
points at one term. The guidance law's impact-point predictor propagates the
estimated state to impact, so a VELOCITY error of `dv` becomes a predicted
impact-point error of roughly `dv * t_go`, and the time to go is 12 s at the
short engagement and 43 s at the long one. Position error enters once;
velocity error enters multiplied by the time to go.

And the velocity error is not the receiver's specification. A patch antenna
10 mm off the spin axis at 1308 rad/s moves at 13 m/s; the receiver averages
most of that away over its ~20 ms integration, but what survives is about
**1.0 m/s against a 0.1 m/s open-sky figure** (docs/SENSOR-MODELS.md section
3.5). The spin costs an order of magnitude in velocity accuracy.

THIS MODULE TESTS THAT RATHER THAN ASSERTING IT. It re-flies the Task F
comparison with the antenna phase centre moved onto the spin axis -- which is
what mounting it in the DESPUN section, or centring it, would achieve -- and
reports the contribution that results. If the hypothesis is right the
contribution collapses; if it is wrong, something else is driving Task F and
this says so.

What the change would cost is stated with the result: an on-axis or despun
antenna does not sweep, so the C/N0 roll source of Task C disappears with it.

Run:  python -m analysis.nav_antenna
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

from gnc import sensors as sn
from analysis import guidance_cep as gc, nav_cep as ncep, nav_common as nc

#: Antenna transverse phase-centre offsets to compare, m. 10 mm is the adopted
#: fuze-well geometry; 0 mm is an antenna on the spin axis or in the despun
#: section; 40 mm is a conformal patch on the body, for the other bound.
OFFSETS = (0.0, 0.010, 0.040)


def _case(a) -> dict:
    """One run with the antenna offset overridden."""
    args, offset = a
    args = dict(args)
    args["antenna_transverse"] = offset
    return nc.run_guided_nav(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engagement", default="long")
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="docs/nav_antenna.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args(argv)

    sn.warn_low_confidence()
    md = gc.load_maps()
    ctx = gc.engagement_context(md, args.engagement)
    opts = gc.scheduler_options(md, args.engagement)["proportional"]
    draws = gc.make_draws(ctx, args.draws)
    t0 = time.time()

    out = {"engagement": args.engagement, "offsets_m": list(OFFSETS),
           "n_draws": args.draws, "n_seeds": args.seeds, "results": {}}
    with Pool(args.workers) as pool:
        truth = ncep._imap(pool, nc.run_guided_nav,
                           ncep._cases(ctx, draws, use_nav=False,
                                       scheduler_opts=opts),
                           "truth")
        for off in OFFSETS:
            cases = []
            for s in range(args.seeds):
                cases += [(c, off) for c in
                          ncep._cases(ctx, draws, use_nav=True, seed=s,
                                      scheduler_opts=opts,
                                      keep_nav_log=True)]
            rows = ncep._imap(pool, _case, cases, f"antenna {1000 * off:.0f} mm")
            d = ncep.paired_navigation_delta(truth, rows)
            vel = [r["n_vel_rms_ms"] for r in rows if "n_vel_rms_ms" in r]
            out["results"][f"{off:.3f}"] = {
                "delta": d, "cep": ncep._cep(rows),
                "velocity_rms_ms": ([float(v) for v in
                                     np.sqrt((np.array(vel) ** 2).mean(axis=0))]
                                    if vel else None),
            }
            v = out["results"][f"{off:.3f}"]["velocity_rms_ms"]
            print(f"  antenna {1000 * off:5.1f} mm:  range 1 sigma "
                  f"{d['range']['sigma_m']:6.2f} m, deflection "
                  f"{d['deflection']['sigma_m']:6.2f} m, velocity rms "
                  f"{'-' if v is None else np.round(v, 3)}", flush=True)
    out["truth_cep"] = ncep._cep(truth)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
