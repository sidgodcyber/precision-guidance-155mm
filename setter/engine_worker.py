"""
The Engine's live model-comparison worker: one engagement, flown from launch
through BOTH the actual 6-DOF engine and the reduced-order MPMM model, so the
two ground tracks and their divergence can be shown side by side.

Invoked as `python -m setter.engine_worker` (B3/B4: subprocess, never
exec/import-and-call from the main Streamlit process). `setter/pages/
engine.py`'s live-comparison fragment launches this via `subprocess.Popen`
and polls for it, exactly as `setter/fire_worker.py` does for FIRE.

THE REAL ENTRY POINT, TRACED (not assumed) from analysis/mpmm_compare.py:
`run_case` is what produced every stored number in `docs/mpmm_results.json`
-- this worker reproduces its "from launch" comparison (6-DOF integrated by
`sim.integrate.integrate`, MPMM by `models.mpmm.propagate_to_impact`, same
projectile, same coefficient table, same environment, coriolis off, matching
every stored case) for exactly one engagement, keeping the full time series
`run_case` computes but discards after reducing it to `max_divergence_m`.
Deterministic: no randomness anywhere in this path, so a live run for an
engagement already in `docs/mpmm_results.json` reproduces that row's
`max_divergence_m`/`t_of_max_divergence` bit-for-bit.

Engagement -> (charge, muzzle_velocity, qe_mils) resolves through
`setter.validation_data.ENGAGEMENT_TO_FIRING_TABLE_INDEX` against
`analysis.mpmm_compare.FIRING_TABLE` -- the single source of truth for that
mapping; this worker does not re-derive it.
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

from analysis import mpmm_compare as mc
from analysis import roll_servo as rs
from models import mpmm as M
from sim import dynamics as dyn, integrate as ig, projectile as pr

from setter import validation_data as vd
from setter.config import SUPPORTED_ENGAGEMENTS

SIXDOF_DT = mc.SIXDOF_DT
MPMM_DT = mc.MPMM_DT
LOG_EVERY = mc.LOG_EVERY


def run_comparison(engagement: str) -> dict:
    t0 = time.perf_counter()
    if engagement not in SUPPORTED_ENGAGEMENTS:
        raise ValueError(f"unsupported engagement {engagement!r}; supported: {SUPPORTED_ENGAGEMENTS}")
    idx = vd.ENGAGEMENT_TO_FIRING_TABLE_INDEX[engagement]
    charge, mv, qe_mils, ft_range, ft_tof, ft_drift = mc.FIRING_TABLE[idx]

    # Same environment for both models, per fuze/trajectory_adapter.py's own
    # precedent (`rs.base_model().environment`) -- coriolis off, matching
    # every stored docs/mpmm_results.json case (run with coriolis=False).
    six_model = rs.base_model()
    mpmm_model = M.MpmmModel(
        projectile=pr.M107, aero=six_model.aero, environment=six_model.environment)

    launch = pr.LaunchConditions.from_mils(mv, qe_mils)

    y6 = dyn.initial_state(pr.M107, launch)
    six = ig.integrate(y6, six_model, dt=SIXDOF_DT, log_every=LOG_EVERY, t_max=200.0)
    tr = six.trajectory

    y0 = M.initial_state(pr.M107, launch)
    red = M.propagate_to_impact(y0, mpmm_model, dt=MPMM_DT, log_every=1)

    div = np.full(tr.t.size, np.nan)
    t_common = np.array([])
    if red.t.size > 2:
        t_common = tr.t[tr.t <= red.t[-1]]
        px = np.interp(t_common, red.t, red.position[:, 0])
        py = np.interp(t_common, red.t, red.position[:, 1])
        pz = np.interp(t_common, red.t, red.position[:, 2])
        n = t_common.size
        div[:n] = np.sqrt((px - tr.position[:n, 0]) ** 2
                           + (py - tr.position[:n, 1]) ** 2
                           + (pz - tr.position[:n, 2]) ** 2)
    finite = np.isfinite(div)
    max_div = float(np.nanmax(div)) if finite.any() else float("nan")
    t_max_div = float(tr.t[int(np.nanargmax(div))]) if finite.any() else float("nan")

    elapsed_s = time.perf_counter() - t0
    return {
        "ok": True,
        "engagement": engagement,
        "charge": charge, "muzzle_velocity": mv, "qe_mils": qe_mils,
        "elapsed_s": elapsed_s,
        "six_dof": {
            "t": tr.t.tolist(),
            "downrange_m": tr.position[:, 0].tolist(),
            "crossrange_m": tr.position[:, 1].tolist(),
            "range_m": six.range_m, "drift_m": six.drift_m, "tof_s": six.impact_time,
        },
        "mpmm": {
            "t": red.t.tolist(),
            "downrange_m": red.position[:, 0].tolist(),
            "crossrange_m": red.position[:, 1].tolist(),
            "range_m": red.range_m, "drift_m": red.drift_m, "tof_s": red.impact_time,
        },
        "divergence": {
            "t": t_common.tolist() if red.t.size > 2 else [],
            "distance_m": div[:t_common.size].tolist() if red.t.size > 2 else [],
        },
        "max_divergence_m": max_div,
        "t_of_max_divergence": t_max_div,
        "stored_comparison": {
            "max_divergence_m": vd.mpmm_point(engagement).max_divergence_m,
            "t_of_max_divergence": vd.mpmm_point(engagement).t_of_max_divergence,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engagement", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        result = run_comparison(args.engagement)
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
