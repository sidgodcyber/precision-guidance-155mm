"""
MPMM versus 6-DOF: validation, model error, and the drift-hypothesis test.

Produces the numbers behind docs/MPMM-VALIDATION.md (Task B),
docs/MODEL-ERROR.md (Task C) and docs/DRIFT-RESOLUTION.md (Task D).

Both models are driven by the SAME coefficient table, the same atmosphere, the
same projectile and the same environment. No fitting factors are used.

Run:  python -m analysis.mpmm_compare
"""

from __future__ import annotations

import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np

from models import mpmm as M
from sim import aerodata, atmosphere as atm
from sim import dynamics as dyn, integrate as ig, projectile as pr

LATITUDE_DEG = 45.0
SIXDOF_DT = 2.0e-4
MPMM_DT = 0.01
LOG_EVERY = 500          # 6-DOF logging cadence: one sample per 0.1 s

# Firing-table envelope used in step 1 (FT 155-AM-2 via Lim NPS 2016).
# (charge, muzzle_velocity, QE_mils, FT range, FT tof, FT drift)
FIRING_TABLE = [
    (4, 337.0, 97.2, 2000.0, 6.4, 3.2),
    (4, 337.0, 152.0, 3000.0, 9.8, 7.8),
    (4, 337.0, 211.6, 4000.0, 13.5, 15.2),
    (5, 397.0, 118.1, 3000.0, 8.8, 7.5),
    (5, 397.0, 280.4, 6000.0, 19.6, 36.6),
    (5, 397.0, 420.6, 8000.0, 28.2, 79.2),
    (6, 474.0, 258.4, 7000.0, 20.8, 45.5),
    (6, 474.0, 378.6, 9000.0, 28.9, 88.2),
    (6, 474.0, 539.9, 11000.0, 39.1, 169.4),
    (7, 568.0, 177.6, 7000.0, 17.6, 37.1),
    (7, 568.0, 319.8, 10000.0, 28.6, 94.0),
    (7, 568.0, 520.7, 13000.0, 42.6, 211.9),
    (8, 684.0, 141.6, 8000.0, 17.0, 37.6),
    (8, 684.0, 248.4, 11000.0, 27.2, 92.4),
    (8, 684.0, 525.3, 16000.0, 48.9, 292.8),
]


def _models(coriolis: bool):
    env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=coriolis)
    six = dyn.FlightModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), environment=env
    )
    red = M.MpmmModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), environment=env
    )
    #: The same model with one fixed-point pass on the yaw of repose, so that
    #: the dv/dt driving alpha_e includes the lift and Magnus accelerations
    #: alpha_e itself produces. Same coefficients, same fitting factors (all
    #: unity) -- the only difference is how the standard's own formula is
    #: evaluated. See docs/MPMM-COMPUTE.md.
    red_it = M.MpmmModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), environment=env,
        iterate_yaw=True,
    )
    return six, red, red_it


def run_case(args):
    """One engagement: 6-DOF, MPMM from launch, MPMM from apogee."""
    charge, mv, qe_mils, ft_range, ft_tof, ft_drift, coriolis = args
    six_model, mpmm_model, mpmm_iter = _models(coriolis)
    launch = pr.LaunchConditions.from_mils(mv, qe_mils)

    # ---- 6-DOF, the reference -------------------------------------------
    y6 = dyn.initial_state(pr.M107, launch)
    six = ig.integrate(y6, six_model, dt=SIXDOF_DT, log_every=LOG_EVERY, t_max=200.0)
    tr = six.trajectory

    # ---- MPMM from the same muzzle conditions ---------------------------
    y0 = M.initial_state(pr.M107, launch)
    red = M.propagate_to_impact(y0, mpmm_model, dt=MPMM_DT, log_every=1)

    # ---- trajectory divergence: MPMM interpolated onto 6-DOF sample times
    div = np.full(tr.t.size, np.nan)
    if red.t.size > 2:
        t_common = tr.t[tr.t <= red.t[-1]]
        px = np.interp(t_common, red.t, red.position[:, 0])
        py = np.interp(t_common, red.t, red.position[:, 1])
        pz = np.interp(t_common, red.t, red.position[:, 2])
        n = t_common.size
        div[:n] = np.sqrt(
            (px - tr.position[:n, 0]) ** 2
            + (py - tr.position[:n, 1]) ** 2
            + (pz - tr.position[:n, 2]) ** 2
        )
    max_div = float(np.nanmax(div)) if np.any(np.isfinite(div)) else float("nan")
    t_max_div = float(tr.t[int(np.nanargmax(div))]) if np.any(np.isfinite(div)) else float("nan")

    # ---- Task C: MPMM initialised from the 6-DOF state at apogee --------
    alt = -tr.position[:, 2]
    i_ap = int(np.argmax(alt))
    y6_ap = np.empty(dyn.STATE_SIZE)
    y6_ap[0:3] = tr.position[i_ap]
    y6_ap[3:6] = tr.velocity[i_ap]
    y6_ap[6:10] = tr.quaternion[i_ap]
    y6_ap[10:13] = tr.omega[i_ap]
    y_ap = M.state_from_sixdof(y6_ap)
    red_ap = M.propagate_to_impact(
        y_ap, mpmm_model, dt=MPMM_DT, t0=float(tr.t[i_ap]), log_every=0
    )

    # ---- the same two runs with the yaw-of-repose iteration on -----------
    red_it = M.propagate_to_impact(y0, mpmm_iter, dt=MPMM_DT, log_every=0)
    red_it_ap = M.propagate_to_impact(
        y_ap, mpmm_iter, dt=MPMM_DT, t0=float(tr.t[i_ap]), log_every=0
    )

    # ---- Task D: MPMM algebraic alpha_e evaluated at 6-DOF states -------
    alpha_alg, aoa_6dof, mach_s = [], [], []
    for i in range(tr.t.size):
        ys = M.pack(tr.position[i], tr.velocity[i], tr.omega[i, 0])
        _x, _y, _z, a = M.yaw_of_repose(float(tr.t[i]), ys, mpmm_model)
        alpha_alg.append(a)
        aoa_6dof.append(float(tr.total_aoa[i]))
        mach_s.append(float(tr.mach[i]))
    alpha_alg = np.array(alpha_alg)
    aoa_6dof = np.array(aoa_6dof)
    mach_s = np.array(mach_s)
    sub = mach_s < 0.7
    frac_sub = float(np.mean(sub))
    min_mach = float(mach_s.min())
    # where does the 6-DOF yaw exceed the algebraic value, and by how much?
    early = tr.t < 0.25 * tr.t[-1]
    late = tr.t > 0.5 * tr.t[-1]

    return {
        "charge": charge,
        "muzzle_velocity": mv,
        "qe_mils": qe_mils,
        "qe_deg": qe_mils * 360.0 / 6400.0,
        "coriolis": coriolis,
        "ft_range": ft_range,
        "ft_tof": ft_tof,
        "ft_drift": ft_drift,
        # 6-DOF
        "six_range": six.range_m,
        "six_drift": six.drift_m,
        "six_tof": six.impact_time,
        "six_vimp": six.impact_velocity,
        "six_angle_deg": math.degrees(six.impact_angle_rad),
        "six_maxord": six.max_ordinate,
        "six_max_aoa_deg": math.degrees(six.max_total_aoa),
        # MPMM from launch
        "mpmm_range": red.range_m,
        "mpmm_drift": red.drift_m,
        "mpmm_tof": red.impact_time,
        "mpmm_vimp": red.impact_velocity,
        "mpmm_angle_deg": math.degrees(red.impact_angle_rad),
        "mpmm_maxord": red.max_ordinate,
        "mpmm_steps": red.steps,
        # differences, MPMM minus 6-DOF
        "d_range": red.range_m - six.range_m,
        "d_range_pct": 100.0 * (red.range_m - six.range_m) / six.range_m,
        "d_drift": red.drift_m - six.drift_m,
        "d_drift_pct": 100.0 * (red.drift_m - six.drift_m) / six.drift_m,
        "d_tof": red.impact_time - six.impact_time,
        "d_vimp": red.impact_velocity - six.impact_velocity,
        "d_angle_deg": math.degrees(red.impact_angle_rad - six.impact_angle_rad),
        "max_divergence_m": max_div,
        "t_of_max_divergence": t_max_div,
        # Task C, from apogee
        "apogee_time": float(tr.t[i_ap]),
        "apogee_alt": float(alt[i_ap]),
        "apogee_mach": float(tr.mach[i_ap]),
        "ap_mpmm_range": red_ap.range_m,
        "ap_mpmm_drift": red_ap.drift_m,
        "ap_mpmm_tof": red_ap.impact_time,
        "ap_d_range": red_ap.range_m - six.range_m,
        "ap_d_drift": red_ap.drift_m - six.drift_m,
        "ap_d_tof": red_ap.impact_time - six.impact_time,
        # iterated yaw of repose
        "it_range": red_it.range_m,
        "it_drift": red_it.drift_m,
        "it_d_range": red_it.range_m - six.range_m,
        "it_d_range_pct": 100.0 * (red_it.range_m - six.range_m) / six.range_m,
        "it_d_drift": red_it.drift_m - six.drift_m,
        "it_d_drift_pct": 100.0 * (red_it.drift_m - six.drift_m) / six.drift_m,
        "it_drift_vs_ft_pct": 100.0 * (red_it.drift_m - ft_drift) / ft_drift,
        "ap_it_range": red_it_ap.range_m,
        "ap_it_drift": red_it_ap.drift_m,
        "ap_it_d_range": red_it_ap.range_m - six.range_m,
        "ap_it_d_drift": red_it_ap.drift_m - six.drift_m,
        # Task D
        "mean_alpha_alg_deg": float(np.degrees(np.mean(alpha_alg))),
        "mean_aoa_6dof_deg": float(np.degrees(np.mean(aoa_6dof))),
        "aoa_ratio": float(np.mean(aoa_6dof) / np.mean(alpha_alg)) if np.mean(alpha_alg) else float("nan"),
        "mean_alpha_alg_sub_deg": float(np.degrees(np.mean(alpha_alg[sub]))) if sub.any() else float("nan"),
        "mean_aoa_6dof_sub_deg": float(np.degrees(np.mean(aoa_6dof[sub]))) if sub.any() else float("nan"),
        "frac_time_subsonic_M07": frac_sub,
        "min_mach": min_mach,
        "aoa_ratio_early": float(np.mean(aoa_6dof[early]) / np.mean(alpha_alg[early])),
        "aoa_ratio_late": float(np.mean(aoa_6dof[late]) / np.mean(alpha_alg[late])),
        # FT comparisons
        "six_drift_err_pct": 100.0 * (six.drift_m - ft_drift) / ft_drift,
        "mpmm_drift_err_pct": 100.0 * (red.drift_m - ft_drift) / ft_drift,
        "six_range_err_pct": 100.0 * (six.range_m - ft_range) / ft_range,
        "mpmm_range_err_pct": 100.0 * (red.range_m - ft_range) / ft_range,
    }


def main():
    t0 = time.time()
    cases = [(c, mv, qe, r, tf, dr, False) for c, mv, qe, r, tf, dr in FIRING_TABLE]
    jobs = max(1, min(7, os.cpu_count() or 2))
    print(f"MPMM vs 6-DOF over {len(cases)} engagements, {jobs} workers ...", flush=True)
    with Pool(jobs) as pool:
        rows = pool.map(run_case, cases)

    os.makedirs("docs", exist_ok=True)
    with open("docs/mpmm_results.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)

    print("\n=== TASK B: MPMM minus 6-DOF, from identical muzzle conditions ===")
    print("chg   QE    TOF6    dRange     dR%    dDrift   dDrift%    dTOF     dVimp   dAngle   maxDiv   @t")
    for r in rows:
        print(f"{r['charge']:3d} {r['qe_mils']:6.1f} {r['six_tof']:6.2f} "
              f"{r['d_range']:+9.2f} {r['d_range_pct']:+7.3f} {r['d_drift']:+9.2f} "
              f"{r['d_drift_pct']:+8.2f} {r['d_tof']:+8.4f} {r['d_vimp']:+8.3f} "
              f"{r['d_angle_deg']:+7.3f} {r['max_divergence_m']:8.2f} {r['t_of_max_divergence']:6.1f}")

    print("\n=== TASK C: impact-point error, MPMM initialised at 6-DOF apogee ===")
    print("chg   QE   apogee_t  apogee_alt  M_ap   dRange(m)  dDrift(m)  dTOF(s)")
    for r in rows:
        print(f"{r['charge']:3d} {r['qe_mils']:6.1f} {r['apogee_time']:8.2f} "
              f"{r['apogee_alt']:10.1f} {r['apogee_mach']:5.2f} "
              f"{r['ap_d_range']:+10.2f} {r['ap_d_drift']:+10.2f} {r['ap_d_tof']:+8.4f}")
    dr = np.array([r["ap_d_range"] for r in rows])
    dd = np.array([r["ap_d_drift"] for r in rows])
    print(f"\n  range      mean {dr.mean():+7.2f} m   1sigma {dr.std(ddof=1):6.2f} m   "
          f"max|.| {np.abs(dr).max():6.2f} m")
    print(f"  deflection mean {dd.mean():+7.2f} m   1sigma {dd.std(ddof=1):6.2f} m   "
          f"max|.| {np.abs(dd).max():6.2f} m")

    print("\n=== TASK D: drift against the FT column, both models, no fitting ===")
    print("chg   QE    FT_drift   6DOF     err%     MPMM     err%    MPMM-6DOF%")
    for r in rows:
        print(f"{r['charge']:3d} {r['qe_mils']:6.1f} {r['ft_drift']:9.1f} "
              f"{r['six_drift']:8.2f} {r['six_drift_err_pct']:+7.2f} "
              f"{r['mpmm_drift']:8.2f} {r['mpmm_drift_err_pct']:+7.2f} "
              f"{r['d_drift_pct']:+10.2f}")
    s6 = np.array([r["six_drift_err_pct"] for r in rows])
    sm = np.array([r["mpmm_drift_err_pct"] for r in rows])
    print(f"\n  6-DOF vs FT : mean {s6.mean():+6.2f} %   MPMM vs FT : mean {sm.mean():+6.2f} %")

    print("\n=== TASK D: yaw comparison at matched states ===")
    print("chg   QE   mean_alpha_alg  mean_aoa_6dof  ratio  ratio_early ratio_late  minMach  frac<M0.7")
    for r in rows:
        print(f"{r['charge']:3d} {r['qe_mils']:6.1f} {r['mean_alpha_alg_deg']:13.4f} "
              f"{r['mean_aoa_6dof_deg']:14.4f} {r['aoa_ratio']:7.3f} "
              f"{r['aoa_ratio_early']:11.3f} {r['aoa_ratio_late']:10.3f} "
              f"{r['min_mach']:8.3f} {r['frac_time_subsonic_M07']:10.3f}")

    print()
    print("=== YAW-OF-REPOSE ITERATION: one fixed-point pass, same formula ===")
    print("chg   QE    --------- from launch ---------   ---- from apogee (model error) ----")
    print("            dR_plain  dR_iter   dD_plain  dD_iter   dR_plain  dR_iter  dD_plain  dD_iter")
    for r in rows:
        print(f"{r['charge']:3d} {r['qe_mils']:6.1f} "
              f"{r['d_range']:+9.2f} {r['it_d_range']:+8.2f} {r['d_drift']:+10.2f} {r['it_d_drift']:+8.2f} "
              f"{r['ap_d_range']:+10.2f} {r['ap_it_d_range']:+8.2f} {r['ap_d_drift']:+9.2f} {r['ap_it_d_drift']:+8.2f}")
    for tag, ka, kb in (("from launch, range", "d_range", "it_d_range"),
                        ("from launch, deflection", "d_drift", "it_d_drift"),
                        ("from apogee, range", "ap_d_range", "ap_it_d_range"),
                        ("from apogee, deflection", "ap_d_drift", "ap_it_d_drift")):
        a = np.array([r[ka] for r in rows])
        b = np.array([r[kb] for r in rows])
        print(f"  {tag:<26} plain: mean {a.mean():+7.2f} 1sig {a.std(ddof=1):6.2f} "
              f"rms {np.sqrt((a**2).mean()):6.2f}  |  iterated: mean {b.mean():+7.2f} "
              f"1sig {b.std(ddof=1):6.2f} rms {np.sqrt((b**2).mean()):6.2f}")
    si = np.array([r["it_drift_vs_ft_pct"] for r in rows])
    print(f"  iterated MPMM vs the FT drift column: mean {si.mean():+6.2f} %")

    print(f"\nWrote docs/mpmm_results.json  ({time.time()-t0:.1f} s)")


if __name__ == "__main__":
    main()
