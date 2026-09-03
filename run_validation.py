"""
The six-rung validation ladder from SIXDOFSPEC.md section 10, plus a
timestep-convergence check.

Every rung reports NUMBERS, not a pass/fail assertion. Run:

    python run_validation.py            # full ladder, parallel
    python run_validation.py --quick    # coarser dt, fewer firing-table points

Results are written to docs/validation_results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from analysis.pointmass3dof import integrate_point_mass
from sim import aerodata, atmosphere as atm, diagnostics as dg
from sim import dynamics as dyn, integrate as ig, projectile as pr

# ---------------------------------------------------------------------------
# Firing table data.
#
# Source: Firing Table FT 155-AM-2 (155 mm Howitzer M185 on M109A1/A2/A3 and
# M199 on M198), for the 155 mm M107 HE projectile, as tabulated by
# W. Y. Lim, "Predicting the Accuracy of Unguided Artillery Projectiles",
# M.S. thesis, Naval Postgraduate School, September 2016 (DTIC AD1029824),
# Tables 15-19, columns headed "FT".
#
# Fields: (charge, muzzle_velocity_ms, QE_mils, range_m, tof_s, drift_m,
#          impact_velocity_ms, max_ordinate_m)
#
# QE is in NATO mils (6400 to the circle). Drift is positive to the RIGHT and
# is the ballistic (yaw-of-repose) drift; firing tables handle the Coriolis
# deflection as a separate correction, so the comparison below is made
# against a simulation run WITHOUT Coriolis, and the Coriolis contribution is
# reported separately.
# ---------------------------------------------------------------------------
FIRING_TABLE = [
    (4, 337.0, 97.2, 2000.0, 6.4, 3.2, 300.0, 50.0),
    (4, 337.0, 152.0, 3000.0, 9.8, 7.8, 290.0, 119.0),
    (4, 337.0, 211.6, 4000.0, 13.5, 15.2, 281.0, 224.0),
    (5, 397.0, 118.1, 3000.0, 8.8, 7.5, 310.0, 96.0),
    (5, 397.0, 280.4, 6000.0, 19.6, 36.6, 287.0, 477.0),
    (5, 397.0, 420.6, 8000.0, 28.2, 79.2, 280.0, 988.0),
    (6, 474.0, 258.4, 7000.0, 20.8, 45.5, 298.0, 548.0),
    (6, 474.0, 378.6, 9000.0, 28.9, 88.2, 293.0, 1059.0),
    (6, 474.0, 539.9, 11000.0, 39.1, 169.4, 294.0, 1924.0),
    (7, 568.0, 177.6, 7000.0, 17.6, 37.1, 313.0, 383.0),
    (7, 568.0, 319.8, 10000.0, 28.6, 94.0, 302.0, 1055.0),
    (7, 568.0, 520.7, 13000.0, 42.6, 211.9, 307.0, 2335.0),
    (8, 684.0, 141.6, 8000.0, 17.0, 37.6, 338.0, 352.0),
    (8, 684.0, 248.4, 11000.0, 27.2, 92.4, 309.0, 930.0),
    (8, 684.0, 525.3, 16000.0, 48.9, 292.8, 318.0, 3116.0),
]

DEFAULT_DT = 2.0e-4
LATITUDE_DEG = 45.0


def _model(coriolis=True, **kw):
    env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=coriolis)
    return dyn.FlightModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), environment=env, **kw
    )


# ===========================================================================
# Rung 1 -- vacuum against the analytic parabola
# ===========================================================================
def rung1_vacuum(dt=1e-3):
    """
    Zero all aero, constant gravity, no Coriolis. Compare with

        R = V^2 sin(2 theta) / g,   T = 2 V sin(theta) / g

    Must match to better than 0.1 %. Catches integrator and frame errors.
    """
    out = []
    for qe_deg in (15.0, 30.0, 45.0, 60.0, 75.0):
        env = pr.Environment.from_degrees(
            LATITUDE_DEG, include_coriolis=False, include_inverse_square_gravity=False
        )
        model = dyn.FlightModel(
            projectile=pr.M107,
            aero=aerodata.make_m107_table(),
            environment=env,
            aero_enabled=False,
        )
        launch = pr.LaunchConditions.from_degrees(684.0, qe_deg)
        y0 = dyn.initial_state(pr.M107, launch)
        res = ig.integrate(y0, model, dt=dt, log_every=100000, t_max=300.0)

        th = math.radians(qe_deg)
        g = 9.80665
        v = 684.0
        r_exact = v * v * math.sin(2.0 * th) / g
        t_exact = 2.0 * v * math.sin(th) / g
        h_exact = (v * math.sin(th)) ** 2 / (2.0 * g)
        out.append(
            {
                "qe_deg": qe_deg,
                "range_sim": res.range_m,
                "range_exact": r_exact,
                "range_err_pct": 100.0 * (res.range_m - r_exact) / r_exact,
                "tof_sim": res.impact_time,
                "tof_exact": t_exact,
                "tof_err_pct": 100.0 * (res.impact_time - t_exact) / t_exact,
                "apogee_sim": res.max_ordinate,
                "apogee_exact": h_exact,
                "apogee_err_pct": 100.0 * (res.max_ordinate - h_exact) / h_exact,
                "drift_m": res.drift_m,
            }
        )
    return out


# ===========================================================================
# Rung 2 -- drag only against an independent 3-DOF point mass
# ===========================================================================
def rung2_drag_only(dt=2e-4):
    """
    6-DOF with the angle of attack forced to zero and all moments off,
    against analysis/pointmass3dof.py, which shares no code with sim/.
    """
    table = aerodata.make_m107_table()

    def cd(m):
        return table.lookup(m)[0]

    out = []
    for qe_deg in (15.0, 30.0, 45.0, 60.0):
        env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=False)
        model = dyn.FlightModel(
            projectile=pr.M107,
            aero=aerodata.make_m107_table(),
            environment=env,
            alpha_zero_drag_only=True,
        )
        launch = pr.LaunchConditions.from_degrees(684.0, qe_deg)
        y0 = dyn.initial_state(pr.M107, launch)
        six = ig.integrate(y0, model, dt=dt, log_every=100000, t_max=300.0)

        three = integrate_point_mass(
            muzzle_velocity=684.0,
            qe_rad=math.radians(qe_deg),
            mass=pr.M107.mass,
            diameter=pr.M107.diameter,
            cd_of_mach=cd,
            dt=dt,
            latitude_rad=None,
            inverse_square_gravity=True,
        )
        out.append(
            {
                "qe_deg": qe_deg,
                "range_6dof": six.range_m,
                "range_3dof": three.range_m,
                "range_diff_m": six.range_m - three.range_m,
                "range_diff_pct": 100.0 * (six.range_m - three.range_m) / three.range_m,
                "tof_6dof": six.impact_time,
                "tof_3dof": three.time_of_flight,
                "tof_diff_s": six.impact_time - three.time_of_flight,
                "vimp_6dof": six.impact_velocity,
                "vimp_3dof": three.impact_speed,
            }
        )
    return out


# ===========================================================================
# Rungs 3-6 come out of full-physics trajectories
# ===========================================================================
def _run_case(args):
    """Worker: one full-physics trajectory. Must be top level for Windows."""
    if len(args) == 8:
        (charge, mv, qe_mils, dt, coriolis, rate_scale, drag_scale, want_history) = args
        splice = True
    else:
        (charge, mv, qe_mils, dt, coriolis, rate_scale, drag_scale, want_history,
         splice) = args
    env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=coriolis)
    model = dyn.FlightModel(
        projectile=pr.M107,
        aero=aerodata.make_m107_table(splice_cnalpha=splice),
        environment=env,
        rate_coefficient_scale=rate_scale,
        drag_scale=drag_scale,
    )
    launch = pr.LaunchConditions.from_mils(mv, qe_mils)
    y0 = dyn.initial_state(pr.M107, launch)
    ms = dg.muzzle_stability(y0, model)
    res = ig.integrate(y0, model, dt=dt, log_every=200, t_max=200.0)
    traj = res.trajectory

    stab = dg.stability_history(traj, model)
    sg = [s.Sg for s in stab]
    sg_req = [s.Sg_required for s in stab]
    repose = [s.yaw_of_repose_rad for s in stab]

    rec = {
        "charge": charge,
        "muzzle_velocity": mv,
        "qe_mils": qe_mils,
        "qe_deg": qe_mils * 360.0 / 6400.0,
        "dt": dt,
        "coriolis": coriolis,
        "rate_scale": rate_scale,
        "drag_scale": drag_scale,
        "splice": splice,
        "range_m": res.range_m,
        "drift_m": res.drift_m,
        "tof_s": res.impact_time,
        "impact_velocity": res.impact_velocity,
        "impact_angle_deg": math.degrees(res.impact_angle_rad),
        "max_ordinate": res.max_ordinate,
        "max_total_aoa_deg": math.degrees(res.max_total_aoa),
        "max_quat_norm_err_per_step": res.max_quat_norm_error,
        "logged_quat_norm_dev": float(
            np.max(np.abs(np.linalg.norm(traj.quaternion, axis=1) - 1.0))
        ),
        "any_nan": bool(not np.all(np.isfinite(traj.position))),
        "steps": res.steps,
        "terminated": res.terminated,
        "muzzle_mach": ms.mach,
        "muzzle_spin": ms.spin,
        "muzzle_Sg": ms.Sg,
        "muzzle_Sd": ms.Sd,
        "muzzle_Sg_required": ms.Sg_required,
        "Sg_min": min(sg),
        "Sg_max": max(sg),
        "Sg_final": sg[-1],
        "Sg_min_margin": min(a - b for a, b in zip(sg, sg_req)),
        "all_gyro_stable": all(s > 1.0 for s in sg),
        "all_dyn_stable": all(a > b for a, b in zip(sg, sg_req)),
        "spin_final": float(traj.omega[-1, 0]),
        "spin_ratio": float(traj.omega[-1, 0] / traj.omega[0, 0]),
        "max_yaw_repose_deg": math.degrees(max(repose)),
        "mean_aoa_deg": float(math.degrees(np.mean(traj.total_aoa))),
    }
    if want_history:
        n = traj.t.size
        keep = max(1, n // 400)
        rec["history"] = {
            "t": traj.t[::keep].tolist(),
            "aoa_deg": np.degrees(traj.total_aoa[::keep]).tolist(),
            "Sg": sg[::keep],
            "mach": traj.mach[::keep].tolist(),
            "altitude": (-traj.position[::keep, 2]).tolist(),
            "spin": traj.omega[::keep, 0].tolist(),
        }
    return rec


def firing_table_cases(dt, quick=False):
    rows = FIRING_TABLE
    if quick:
        rows = [r for r in rows if r[0] in (7, 8)]
    # Coriolis OFF: firing-table drift is the ballistic drift only.
    return [(r[0], r[1], r[2], dt, False, 1.0, 1.0, False) for r in rows]


def max_range_cases(dt):
    return [(8, 684.0, qe, dt, False, 1.0, 1.0, False) for qe in (700, 750, 800, 850, 900, 950)]


# ===========================================================================
# Rung 5b -- the fully specified ASAT-13 section 4.3 trajectory
# ===========================================================================
#
# Every input is given by the source, so nothing is left to inference. Time of
# flight and summit time are unambiguous text-stated scalars and are much
# harder to satisfy by accident than a range figure, which a drag form factor
# can always be tuned to hit.
#
# Inputs (ASAT-13 section 4.3):
#   theta0 = 44 deg, V0 = 684.3 m/s, p0 = 175.48 rps
#   m = 43 kg, d = 0.155 m, L = 698 mm, CG = 0.459 m from nose
#   Ix = 0.144, Iy = Iz = 1.216 kg m^2
#
# Published outputs, with the source's own reliability:
#   TOTAL FLIGHT TIME     66.67 s     text-stated  -> tight
#   SUMMIT TIME           ~31 s       text-stated  -> tight
#   INITIAL AXIAL DECEL   4.45 g      text-stated  -> tight
#   summit altitude       ~5700 m     read off Fig 4  -> approximate
#   range                 ~16500 m    read off Fig 3  -> approximate
#   max total AoA         ~1.3 deg at t ~ 32 s, read off Fig 10 -> approximate
#   drift direction       right       text-stated
#
# NOTE ON p0: 684.3 / 175.48 / 0.155 = 25.16 calibres per turn. That is an
# M114-era twist paired with an M185-era muzzle velocity, which is not
# self-consistent for any single US tube (see sim/projectile.py TUBES). It is
# used here regardless, because the point of this rung is to reproduce the
# published case exactly as specified.
ASAT_5B = {
    "qe_deg": 44.0,
    "muzzle_velocity": 684.3,
    "p0_rps": 175.48,
    "tof_s": 66.67,
    "summit_time_s": 31.0,
    "summit_alt_m": 5700.0,
    "range_m": 16500.0,
    "initial_axial_decel_g": 4.45,
    "max_aoa_deg": 1.3,
    "max_aoa_time_s": 32.0,
}


def rung5b_asat(dt=2e-4):
    """Reproduce the ASAT-13 section 4.3 trajectory from its own inputs."""
    shell = pr.M107_ASAT
    spec = ASAT_5B
    # Coriolis off: ASAT does not state a firing latitude or azimuth.
    env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=False)
    model = dyn.FlightModel(
        projectile=shell, aero=aerodata.make_m107_table(), environment=env
    )
    launch = pr.LaunchConditions.from_degrees(spec["muzzle_velocity"], spec["qe_deg"])
    y0 = dyn.initial_state(shell, launch)

    # Check the muzzle spin reproduces ASAT's stated p0.
    p0_rps = float(y0[10]) / (2.0 * math.pi)

    # Initial axial deceleration: the aerodynamic axial force plus the
    # component of gravity along the body axis. ASAT quotes 4.45 g; the drag
    # term alone is only 3.77 g, so the quoted figure must include gravity.
    st = dyn.aero_state(0.0, y0, model)
    axial_drag_g = -st.force_body[0] / shell.mass / atm.G0
    gravity_axial_g = math.sin(math.radians(spec["qe_deg"]))
    total_axial_g = axial_drag_g + gravity_axial_g

    res = ig.integrate(y0, model, dt=dt, log_every=200, t_max=200.0)
    traj = res.trajectory
    alt = -traj.position[:, 2]
    i_sum = int(np.argmax(alt))
    i_aoa = int(np.argmax(traj.total_aoa))

    return {
        "p0_rps_model": p0_rps,
        "p0_rps_spec": spec["p0_rps"],
        "twist_calibers": shell.twist_calibers,
        "axial_drag_g": axial_drag_g,
        "gravity_axial_g": gravity_axial_g,
        "total_axial_g": total_axial_g,
        "spec_axial_g": spec["initial_axial_decel_g"],
        "tof_s": res.impact_time,
        "spec_tof_s": spec["tof_s"],
        "summit_time_s": float(traj.t[i_sum]),
        "spec_summit_time_s": spec["summit_time_s"],
        "summit_alt_m": float(alt[i_sum]),
        "spec_summit_alt_m": spec["summit_alt_m"],
        "range_m": res.range_m,
        "spec_range_m": spec["range_m"],
        "drift_m": res.drift_m,
        "max_aoa_deg": math.degrees(res.max_total_aoa),
        "spec_max_aoa_deg": spec["max_aoa_deg"],
        "max_aoa_time_s": float(traj.t[i_aoa]),
        "spec_max_aoa_time_s": spec["max_aoa_time_s"],
        "impact_velocity": res.impact_velocity,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    args = ap.parse_args()
    dt = args.dt

    aerodata.warn_unvalidated(stream=sys.stdout)
    t_start = time.time()
    results = {"dt": dt, "latitude_deg": LATITUDE_DEG}

    print("\n[rung 1] vacuum vs analytic parabola ...", flush=True)
    results["rung1"] = rung1_vacuum()
    for r in results["rung1"]:
        print(
            f"  QE {r['qe_deg']:5.1f} deg  R_sim {r['range_sim']:10.2f}  "
            f"R_exact {r['range_exact']:10.2f}  err {r['range_err_pct']:+.5f} %  "
            f"TOF err {r['tof_err_pct']:+.5f} %"
        )

    print("\n[rung 2] drag-only 6-DOF vs independent 3-DOF point mass ...", flush=True)
    results["rung2"] = rung2_drag_only(dt=max(dt, 1e-3))
    for r in results["rung2"]:
        print(
            f"  QE {r['qe_deg']:5.1f} deg  6DOF {r['range_6dof']:9.2f}  "
            f"3DOF {r['range_3dof']:9.2f}  diff {r['range_diff_m']:+8.3f} m "
            f"({r['range_diff_pct']:+.5f} %)  TOF diff {r['tof_diff_s']:+.5f} s"
        )

    print(f"\n[rungs 3-6] firing-table trajectories, dt={dt}, {args.jobs} workers ...", flush=True)
    cases = firing_table_cases(dt, args.quick)
    with Pool(args.jobs) as pool:
        ft_runs = pool.map(_run_case, cases)
    results["firing_table"] = []
    for run, ref in zip(ft_runs, [r for r in FIRING_TABLE if not args.quick or r[0] in (7, 8)]):
        _, _, _, ft_range, ft_tof, ft_drift, ft_vimp, ft_maxord = ref
        rec = dict(run)
        rec.update(
            {
                "ft_range": ft_range,
                "ft_tof": ft_tof,
                "ft_drift": ft_drift,
                "ft_impact_velocity": ft_vimp,
                "ft_max_ordinate": ft_maxord,
                "range_err_pct": 100.0 * (run["range_m"] - ft_range) / ft_range,
                "tof_err_pct": 100.0 * (run["tof_s"] - ft_tof) / ft_tof,
                "drift_err_pct": 100.0 * (run["drift_m"] - ft_drift) / ft_drift,
                "vimp_err_pct": 100.0 * (run["impact_velocity"] - ft_vimp) / ft_vimp,
                "maxord_err_pct": 100.0 * (run["max_ordinate"] - ft_maxord) / ft_maxord,
            }
        )
        results["firing_table"].append(rec)
        print(
            f"  chg {rec['charge']} QE {rec['qe_mils']:6.1f} mil  "
            f"R {rec['range_m']:9.1f} vs {ft_range:8.1f} ({rec['range_err_pct']:+6.2f} %)  "
            f"TOF {rec['tof_s']:6.2f} vs {ft_tof:5.1f} ({rec['tof_err_pct']:+6.2f} %)  "
            f"drift {rec['drift_m']:7.2f} vs {ft_drift:6.1f} ({rec['drift_err_pct']:+7.2f} %)"
        )

    print("\n[rung 5b] ASAT-13 section 4.3 fully specified trajectory ...", flush=True)
    r5b = rung5b_asat(dt=dt)
    results["rung5b"] = r5b
    print(f"  twist implied by the ASAT p0: 1 turn in {r5b['twist_calibers']:.3f} calibres")
    print(f"  muzzle spin   model {r5b['p0_rps_model']:8.2f} rps   spec {r5b['p0_rps_spec']:8.2f} rps")
    print(f"  initial axial decel: drag {r5b['axial_drag_g']:.3f} g + gravity along axis "
          f"{r5b['gravity_axial_g']:.3f} g = {r5b['total_axial_g']:.3f} g   "
          f"spec {r5b['spec_axial_g']:.2f} g   "
          f"({100*(r5b['total_axial_g']-r5b['spec_axial_g'])/r5b['spec_axial_g']:+.2f} %)")
    for key, spec_key, unit, tight in [
        ("tof_s", "spec_tof_s", "s", True),
        ("summit_time_s", "spec_summit_time_s", "s", True),
        ("summit_alt_m", "spec_summit_alt_m", "m", False),
        ("range_m", "spec_range_m", "m", False),
        ("max_aoa_deg", "spec_max_aoa_deg", "deg", False),
        ("max_aoa_time_s", "spec_max_aoa_time_s", "s", False),
    ]:
        got, want = r5b[key], r5b[spec_key]
        tag = "text-stated" if tight else "figure-read"
        print(f"  {key:16s} {got:10.3f} {unit:4s} vs {want:9.2f}  "
              f"({100*(got-want)/want:+7.2f} %)  [{tag}]")
    print(f"  drift {r5b['drift_m']:+.2f} m  (spec: to the right)")

    print("\n[rung 5c] maximum-range sweep, charge 8 ...", flush=True)
    with Pool(args.jobs) as pool:
        mr = pool.map(_run_case, max_range_cases(dt))
    results["max_range_sweep"] = mr
    best = max(mr, key=lambda r: r["range_m"])
    results["max_range"] = {"qe_mils": best["qe_mils"], "qe_deg": best["qe_deg"], "range_m": best["range_m"]}
    for r in mr:
        print(f"  QE {r['qe_mils']:6.1f} mil ({r['qe_deg']:5.2f} deg)  range {r['range_m']:9.1f} m")
    print(f"  -> maximum {best['range_m']:.0f} m at QE {best['qe_mils']:.0f} mils "
          f"(published M107 max range ~18100 m)")

    print("\n[coriolis] contribution to drift, charge 8 ...", flush=True)
    cor_cases = [
        (8, 684.0, 141.6, dt, True, 1.0, 1.0, False),
        (8, 684.0, 248.4, dt, True, 1.0, 1.0, False),
        (8, 684.0, 525.3, dt, True, 1.0, 1.0, False),
    ]
    with Pool(min(3, args.jobs)) as pool:
        cor = pool.map(_run_case, cor_cases)
    results["coriolis_on"] = cor
    for c, base in zip(cor, [r for r in results["firing_table"] if r["charge"] == 8]):
        print(
            f"  QE {c['qe_mils']:6.1f}  drift with Coriolis {c['drift_m']:7.2f} m, "
            f"without {base['drift_m']:7.2f} m, Coriolis contributes {c['drift_m']-base['drift_m']:+6.2f} m; "
            f"range shift {c['range_m']-base['range_m']:+6.2f} m"
        )

    print("\n[convergence] timestep halving, charge 8 QE 525.3 mils ...", flush=True)
    conv_cases = [(8, 684.0, 525.3, d, False, 1.0, 1.0, False) for d in (8e-4, 4e-4, 2e-4, 1e-4)]
    with Pool(min(4, args.jobs)) as pool:
        conv = pool.map(_run_case, conv_cases)
    results["convergence"] = conv
    ref = conv[-1]
    for c in conv:
        spr = 1.0 / (c["muzzle_spin"] / (2 * math.pi) * c["dt"])
        print(
            f"  dt {c['dt']:.1e} ({spr:5.1f} samples/rev at muzzle)  "
            f"R {c['range_m']:9.2f} ({c['range_m']-ref['range_m']:+7.3f} m vs dt=1e-4)  "
            f"drift {c['drift_m']:7.3f} ({c['drift_m']-ref['drift_m']:+6.3f})  "
            f"TOF {c['tof_s']:.4f}"
        )

    print("\n[sensitivity] C_Nalpha splice, twist, rate convention, drag form factor ...",
          flush=True)
    sens_cases = [
        # (charge, mv, qe, dt, coriolis, rate_scale, drag_scale, history, splice)
        (8, 684.0, 525.3, dt, False, 1.0, 1.0, False, False),   # splice OFF
        (8, 684.0, 141.6, dt, False, 1.0, 1.0, False, False),
        (8, 684.0, 525.3, dt, False, 2.0, 1.0, False, True),    # rate coeffs x2
        (8, 684.0, 141.6, dt, False, 2.0, 1.0, False, True),
        (8, 684.0, 525.3, dt, False, 1.0, 0.9076, False, True), # source form factor
        (8, 684.0, 141.6, dt, False, 1.0, 0.9076, False, True),
    ]
    with Pool(min(6, args.jobs)) as pool:
        sens = pool.map(_run_case, sens_cases)
    results["sensitivity"] = sens
    for s in sens:
        if not s["splice"]:
            tag = "splice OFF (raw ASAT)"
        elif s["rate_scale"] == 2.0:
            tag = "rate coeffs x2      "
        else:
            tag = "drag x0.9076        "
        print(
            f"  QE {s['qe_mils']:6.1f}  {tag}  R {s['range_m']:9.1f}  "
            f"drift {s['drift_m']:7.2f}  maxAoA {s['max_total_aoa_deg']:.3f} deg"
        )

    print("\n[sensitivity] twist: M185 1/20 (nominal) vs M1 1/25 ...", flush=True)
    twist_rows = []
    for qe, ft_drift in ((141.6, 37.6), (525.3, 292.8)):
        for tube, n in (("M185", 20.0), ("M1", 25.0)):
            shell = pr.M107.perturbed(twist_calibers=n)
            env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=False)
            mdl = dyn.FlightModel(
                projectile=shell, aero=aerodata.make_m107_table(), environment=env
            )
            y0 = dyn.initial_state(shell, pr.LaunchConditions.from_mils(684.0, qe))
            ms = dg.muzzle_stability(y0, mdl)
            # log_every must stay small: max_total_aoa is a maximum over
            # logged samples, so a coarse cadence under-reports the peak.
            rr = ig.integrate(y0, mdl, dt=dt, log_every=200, t_max=200.0)
            row = {
                "qe_mils": qe, "tube": tube, "twist": n,
                "p0_rps": ms.spin / (2 * math.pi), "Sg_muzzle": ms.Sg,
                "range_m": rr.range_m, "drift_m": rr.drift_m,
                "ft_drift": ft_drift,
                "drift_err_pct": 100.0 * (rr.drift_m - ft_drift) / ft_drift,
                "max_aoa_deg": math.degrees(rr.max_total_aoa),
            }
            twist_rows.append(row)
            print(
                f"  QE {qe:6.1f}  {tube:5s} 1/{n:.0f}  p0 {row['p0_rps']:7.2f} rps  "
                f"Sg {row['Sg_muzzle']:6.3f}  drift {row['drift_m']:7.2f} vs FT "
                f"{ft_drift:6.1f} ({row['drift_err_pct']:+7.2f} %)  "
                f"maxAoA {row['max_aoa_deg']:.3f} deg"
            )
    results["twist_sensitivity"] = twist_rows

    print("\n[rung 6] one long trajectory with full history ...", flush=True)
    detail = _run_case((8, 684.0, 525.3, dt, True, 1.0, 1.0, True))
    results["detail_run"] = detail
    print(
        f"  max |q|-1 over logged samples : {detail['logged_quat_norm_dev']:.3e}\n"
        f"  max per-step norm drift removed by renormalisation : "
        f"{detail['max_quat_norm_err_per_step']:.3e}\n"
        f"  NaNs present : {detail['any_nan']}\n"
        f"  max total angle of attack : {detail['max_total_aoa_deg']:.4f} deg\n"
        f"  mean total angle of attack : {detail['mean_aoa_deg']:.4f} deg\n"
        f"  Sg range over flight : {detail['Sg_min']:.3f} to {detail['Sg_max']:.3f}\n"
        f"  spin {detail['muzzle_spin']:.1f} -> {detail['spin_final']:.1f} rad/s "
        f"({100*detail['spin_ratio']:.1f} % retained)"
    )

    results["wall_clock_s"] = time.time() - t_start
    os.makedirs("docs", exist_ok=True)
    with open("docs/validation_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nWrote docs/validation_results.json  ({results['wall_clock_s']:.1f} s)")


if __name__ == "__main__":
    main()
