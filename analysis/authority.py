"""
Open-loop correction-authority envelope for the fuze-well canard kit.

Produces every number in docs/AUTHORITY-ENVELOPE.md and the measured ratios
quoted in docs/CANARD-MODEL.md.

METHOD
------
For each firing-table engagement:

  1. Fly UNCORRECTED -- the step-1 ballistic model, no nose degree of freedom,
     no canards. Record the impact point and the apogee time.
  2. Fly again with the canards deployed at a fixed time and the nose HELD at
     a fixed earth-referenced roll angle phi. Record the impact point.
  3. The difference is the impact shift, in range and deflection.
  4. Sweep phi over the circle; sweep the deployment time.

There is no controller anywhere in this file. The nose is pinned by an ideal
kinematic constraint, which is the open-loop idealisation of a servo that step
4 has to build; the brake torque is identically zero throughout.

WHY THE SHIFT AND NOT THE ABSOLUTE IMPACT
-----------------------------------------
Step 1 closed with a known +14 % bias in absolute lateral deflection against
the firing table. Every number in this file is a DIFFERENCE between two runs
of the same model from the same muzzle conditions, so that bias cancels
exactly, which is the regime step 1 and step 2 both certified the model for.

Run:  python -m analysis.authority              full sweep, ~20 min on 8 cores
      python -m analysis.authority --quick      one engagement, coarse
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

from sim import aerodata, canards as cn
from sim import dynamics as dyn, integrate as ig, projectile as pr

LATITUDE_DEG = 45.0
#: The sweeps run at 5e-4. The headline table is confirmed at 2e-4 by
#: `--converge`, which reports the change in the SHIFT rather than in the
#: absolute impact point -- see docs/AUTHORITY-ENVELOPE.md section 8.
SWEEP_DT = 5.0e-4
FINE_DT = 2.0e-4
LOG_EVERY = 400
#: Baseline logging cadence: 0.05 s at the sweep step size. Corrected runs
#: restart from one of these samples, so this also sets how finely a
#: deployment time can be requested.
RESTART_LOG_EVERY = 100

# (charge, muzzle_velocity, QE_mils, FT range) -- the step-1 envelope.
FIRING_TABLE = [
    (4, 337.0, 97.2, 2000.0),
    (4, 337.0, 152.0, 3000.0),
    (4, 337.0, 211.6, 4000.0),
    (5, 397.0, 118.1, 3000.0),
    (5, 397.0, 280.4, 6000.0),
    (5, 397.0, 420.6, 8000.0),
    (6, 474.0, 258.4, 7000.0),
    (6, 474.0, 378.6, 9000.0),
    (6, 474.0, 539.9, 11000.0),
    (7, 568.0, 177.6, 7000.0),
    (7, 568.0, 319.8, 10000.0),
    (7, 568.0, 520.7, 13000.0),
    (8, 684.0, 141.6, 8000.0),
    (8, 684.0, 248.4, 11000.0),
    (8, 684.0, 525.3, 16000.0),
]

#: Three engagements carrying the deployment-time and geometry sweeps: a
#: short low-angle shot, a mid-range one and the maximum-range case.
REPRESENTATIVE = (0, 6, 14)

PHI_SWEEP_DEG = tuple(range(0, 360, 30))
PHI_CARDINAL_DEG = (0.0, 90.0, 180.0, 270.0)


def _base_model():
    """The step-1 ballistic model: no nose, no canards, no Coriolis."""
    env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=False)
    return dyn.FlightModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), environment=env
    )


def _fly(launch, model, dt, log_every=LOG_EVERY):
    y0 = dyn.initial_state(pr.M107, launch)
    return ig.integrate(y0, model, dt=dt, log_every=log_every, t_max=200.0)


def _apogee_time(res) -> float:
    alt = -res.trajectory.position[:, 2]
    return float(res.trajectory.t[int(np.argmax(alt))])


# ===========================================================================
# Baselines, computed once per engagement and shared
# ===========================================================================
def baseline(args) -> dict:
    """
    The uncorrected trajectory, plus every state a corrected run might need to
    restart from.

    Flying the unguided leg once and restarting the corrected runs from a
    logged state is worth about a factor of three over re-flying the whole
    trajectory for every commanded roll angle, and it is exact: a logged
    sample is an integrator state, not an interpolation.
    """
    charge = args["charge"]; mv = args["mv"]; qe = args["qe_mils"]
    dt = args.get("dt", SWEEP_DT)
    launch = pr.LaunchConditions.from_mils(mv, qe)
    res = _fly(launch, _base_model(), dt, log_every=args.get("log_every", RESTART_LOG_EVERY))
    tr = res.trajectory
    t_ap = _apogee_time(res)
    return {
        "charge": charge, "mv": mv, "qe_mils": qe, "dt": dt,
        "range": res.range_m, "drift": res.drift_m, "tof": res.impact_time,
        "apogee_time": t_ap, "max_ordinate": res.max_ordinate,
        "impact_velocity": res.impact_velocity,
        "log_t": tr.t.tolist(),
        "log_y": _states_from(tr).tolist(),
        "log_alpha": tr.alpha.tolist(),
        "log_beta": tr.beta.tolist(),
        "log_total_aoa": tr.total_aoa.tolist(),
        "log_mach": tr.mach.tolist(),
    }


def _states_from(tr) -> np.ndarray:
    """(n, STATE_SIZE) array of full states from a logged trajectory."""
    n = tr.t.size
    y = np.zeros((n, dyn.STATE_SIZE))
    y[:, 0:3] = tr.position
    y[:, 3:6] = tr.velocity
    y[:, 6:10] = tr.quaternion
    y[:, 10:13] = tr.omega
    y[:, 13] = tr.nose_angle
    y[:, 14] = tr.nose_rate
    return y


def _deploy_index(base: dict, deploy_fraction: float, deploy_time=None):
    """
    Index of the logged sample to restart from, and its exact time.

    fraction 0 = apogee, -1 = muzzle, +1 = impact. The requested time is
    snapped to the nearest logged sample so that the restart is exact rather
    than interpolated; the snapped time is what every result reports.
    """
    t_ap = base["apogee_time"]
    t_imp = base["tof"]
    if deploy_time is None:
        f = deploy_fraction
        deploy_time = t_ap + f * ((t_imp - t_ap) if f >= 0 else t_ap)
    deploy_time = max(0.0, min(deploy_time, t_imp - 1.0))
    t = np.asarray(base["log_t"])
    i = int(np.argmin(np.abs(t - deploy_time)))
    return i, float(t[i])


# ===========================================================================
# One case
# ===========================================================================
def run_case(args) -> dict:
    """
    One (engagement, deployment time, phi, geometry) point, restarted from the
    shared uncorrected leg in `args["baseline"]`.
    """
    base = args["baseline"]
    dt = base["dt"]
    geom = args.get("geometry", cn.NOMINAL_GEOMETRY)
    phi_deg = args["phi_deg"]

    i, t_dep = _deploy_index(base, args.get("deploy_fraction", 0.0),
                             args.get("deploy_time"))
    y_dep = np.asarray(base["log_y"][i])

    nose = cn.NoseAssembly(deploy_time=t_dep, hold_angle=math.radians(phi_deg))
    guided = cn.guided_model(_base_model(), geom, nose)
    cor = ig.integrate(y_dep, guided, dt=dt, log_every=LOG_EVERY,
                       t_max=200.0, t_start=t_dep)

    return {
        "charge": base["charge"],
        "mv": base["mv"],
        "qe_mils": base["qe_mils"],
        "phi_deg": float(phi_deg),
        "deploy_time": t_dep,
        "deploy_fraction": float(args.get("deploy_fraction", 0.0)),
        "apogee_time": base["apogee_time"],
        "uncorrected_range": base["range"],
        "uncorrected_drift": base["drift"],
        "uncorrected_tof": base["tof"],
        "corrected_range": cor.range_m,
        "corrected_drift": cor.drift_m,
        "corrected_tof": cor.impact_time,
        "d_range": cor.range_m - base["range"],
        "d_deflection": cor.drift_m - base["drift"],
        "d_tof": cor.impact_time - base["tof"],
        "max_total_aoa_deg": math.degrees(cor.max_total_aoa),
        "steering_deflection_deg": math.degrees(geom.steering_deflection),
        "station_from_nose_m": geom.station_from_nose,
        "carryover": geom.carryover,
        "panel_area_m2": geom.panel_area,
        "dt": dt,
    }


def run_free_nose(args) -> dict:
    """
    Task D.2: a FREE nose. The two-state degree of freedom integrated with
    zero brake torque and no hold, to see what the nose actually does when
    nothing commands it.
    """
    base = args["baseline"]
    dt = base["dt"]
    geom = args.get("geometry", cn.NOMINAL_GEOMETRY)
    i, t_dep = _deploy_index(base, args.get("deploy_fraction", 0.0))
    y_dep = np.asarray(base["log_y"][i])

    nose = cn.NoseAssembly(deploy_time=t_dep, hold_angle=None)
    guided = cn.guided_model(_base_model(), geom, nose)
    cor = ig.integrate(y_dep, guided, dt=dt, log_every=LOG_EVERY,
                       t_max=200.0, t_start=t_dep)
    tr = cor.trajectory
    p_nose = tr.nose_spin

    settled = p_nose[tr.t >= t_dep + 2.0]
    return {
        "charge": base["charge"],
        "qe_mils": base["qe_mils"],
        "deploy_time": t_dep,
        "body_spin_at_deploy": float(tr.spin[0]),
        "uncorrected_range": base["range"],
        "corrected_range": cor.range_m,
        "d_range": cor.range_m - base["range"],
        "d_deflection": cor.drift_m - base["drift"],
        "nose_spin_mean_after_settle": float(np.mean(settled)) if settled.size else float("nan"),
        "nose_spin_min": float(np.min(settled)) if settled.size else float("nan"),
        "nose_spin_max": float(np.max(settled)) if settled.size else float("nan"),
        "body_spin_at_impact": float(cor.impact_state[10]),
        "p_rel_at_impact": float(cor.impact_state[14]),
        "despin_time_to_10pct_s": _despin_time(tr, t_dep),
        "bearing_slip_power_W": _slip_power(tr, nose),
        "t": tr.t.tolist(),
        "nose_spin": p_nose.tolist(),
        "body_spin": tr.spin.tolist(),
    }


def _despin_time(tr, deploy_time: float) -> float:
    """Time from deployment for |p_nose| to fall below 10 % of the body spin."""
    m = tr.t >= deploy_time
    if not np.any(m):
        return float("nan")
    t = tr.t[m]
    pn = np.abs(tr.nose_spin[m])
    pb = np.abs(tr.spin[m])
    below = np.nonzero(pn < 0.1 * pb)[0]
    if below.size == 0:
        return float("nan")
    return float(t[below[0]] - deploy_time)


def _slip_power(tr, nose) -> float:
    """
    Mean power dissipated in the bearing while the nose is despun, W.

    Reported because it is large enough to matter and nothing else in this
    project would surface it: the bearing slips at nearly the full body spin
    for the whole guided phase.
    """
    if tr.nose_rate.size == 0:
        return float("nan")
    p_rel = tr.nose_rate
    tq = np.array([nose.friction_torque(float(x)) for x in p_rel])
    return float(np.mean(np.abs(tq * p_rel)))


# ===========================================================================
# Task B -- the direct force and the induced body force, measured
# ===========================================================================
def induced_vs_direct(args) -> dict:
    """
    Measure the ratio of the induced body force to the direct canard force
    along a trajectory, and compare it with the closed-form prediction in
    canards.trim_force_factor().

    The induced force is isolated by DIFFERENCING. Guided and unguided are
    flown from identical conditions, the angle-of-attack vector is sampled
    from both at matched times, and the difference is the trim angle of
    attack the canards produced. The body normal force that follows from it,
    through the MEASURED C_Nalpha, is the induced force.

    Nothing in sim/canards.py computes this. It is a property of the measured
    body deck acting through the 6-DOF, and that is the point: the term that
    turns out to dominate is not the estimated one.
    """
    base = args["baseline"]
    dt = base["dt"]
    geom = args.get("geometry", cn.NOMINAL_GEOMETRY)
    phi_deg = args.get("phi_deg", 0.0)

    i, t_dep = _deploy_index(base, args.get("deploy_fraction", 0.0))
    y_dep = np.asarray(base["log_y"][i])
    nose = cn.NoseAssembly(deploy_time=t_dep, hold_angle=math.radians(phi_deg))
    guided = cn.guided_model(_base_model(), geom, nose)
    cor = ig.integrate(y_dep, guided, dt=dt, log_every=LOG_EVERY,
                       t_max=200.0, t_start=t_dep)

    kit = guided.control
    tg = cor.trajectory
    tu_t = np.asarray(base["log_t"])
    tu_a = np.asarray(base["log_alpha"])
    tu_b = np.asarray(base["log_beta"])
    tu_d = np.asarray(base["log_total_aoa"])
    tab = aerodata.make_m107_table()
    S = pr.M107.reference_area
    yg = _states_from(tg)

    rows = []
    for k, t in enumerate(tg.t):
        if t < t_dep + 2.0 or t > tu_t[-1]:
            continue
        y = yg[k]
        st = dyn.aero_state(float(t), y, guided)
        F, M, T = kit(float(t), y, st)
        direct = math.hypot(float(F[1]), float(F[2]))
        if direct <= 0.0:
            continue

        a_u = float(np.interp(t, tu_t, tu_a))
        b_u = float(np.interp(t, tu_t, tu_b))
        d_aoa = math.hypot(float(tg.alpha[k]) - a_u, float(tg.beta[k]) - b_u)
        c = tab.coefficients_at(st.mach)
        induced = st.dynamic_pressure * S * c.C_Nalpha * d_aoa

        cla = cn.canard_lift_curve_slope(st.mach, geom.aspect_ratio_effective)
        pred = cn.trim_force_factor(geom, pr.M107, c.C_Nalpha, c.C_Malpha, cla)

        rows.append({
            "t": float(t),
            "mach": st.mach,
            "qbar": st.dynamic_pressure,
            "direct_N": direct,
            "induced_N": induced,
            "ratio_induced_over_direct": induced / direct,
            "trim_aoa_deg": math.degrees(d_aoa),
            "total_aoa_deg": math.degrees(float(tg.total_aoa[k])),
            "ballistic_aoa_deg": math.degrees(float(np.interp(t, tu_t, tu_d))),
            "predicted_ratio": -pred["induced_over_direct"],
            "predicted_net_factor": pred["trim_force_factor"],
            "x_cp_m": pred["x_cp_body_m"],
            "canard_roll_torque_Nm": float(T),
        })

    return {"charge": base["charge"], "qe_mils": base["qe_mils"],
            "phi_deg": phi_deg, "deploy_time": t_dep, "rows": rows}


# ===========================================================================
# Envelope statistics
# ===========================================================================
def envelope_stats(rows: list) -> dict:
    """
    Reduce one engagement's phi sweep to the numbers step 3 needs.

    The shift decomposes into

        bias      the mean over phi -- what deployment costs regardless of
                  how the kit is pointed, and therefore not steerable
        envelope  the variation about that mean -- the steerable authority

    and the envelope is fitted as an ellipse. A constant transverse force of
    fixed magnitude swept through 360 degrees maps to an ELLIPSE in the
    range/deflection plane, not a circle, because a vertical force and a
    horizontal one of the same size do not move the impact point equally. The
    semi-axes and the tilt are exactly what a guidance law has to know.

    `harmonic_residual_m` is the RMS departure of the measured sweep from a
    pure first harmonic in phi. It is the honest measure of how far the plant
    is from the linear "one force, one direction" picture: a large residual
    would mean the response has structure the ellipse does not capture.
    """
    rows = sorted(rows, key=lambda r: r["phi_deg"])
    phi = np.array([math.radians(r["phi_deg"]) for r in rows])
    dR = np.array([r["d_range"] for r in rows])
    dD = np.array([r["d_deflection"] for r in rows])
    n = phi.size

    biasR = float(np.mean(dR))
    biasD = float(np.mean(dD))
    xR = dR - biasR
    xD = dD - biasD

    c = np.cos(phi)
    s = np.sin(phi)
    aR = 2.0 * float(np.dot(xR, c)) / n
    bR = 2.0 * float(np.dot(xR, s)) / n
    aD = 2.0 * float(np.dot(xD, c)) / n
    bD = 2.0 * float(np.dot(xD, s)) / n

    resid = float(np.sqrt(np.mean((xR - (aR * c + bR * s)) ** 2
                                  + (xD - (aD * c + bD * s)) ** 2)))
    amp = float(np.sqrt(np.mean(xR ** 2 + xD ** 2)))

    # phi -> (aR cos + bR sin, aD cos + bD sin) is the unit circle mapped
    # through [[aR, bR], [aD, bD]]; the ellipse semi-axes are its singular
    # values and the major-axis direction is the first left singular vector.
    Jm = np.array([[aR, bR], [aD, bD]])
    u, sv, _ = np.linalg.svd(Jm)
    major, minor = float(sv[0]), float(sv[1])
    tilt = math.degrees(math.atan2(u[1, 0], u[0, 0]))

    total = np.hypot(dR, dD)
    i_max = int(np.argmax(total))
    rng = rows[0]["uncorrected_range"]
    aoas = [r.get("max_total_aoa_deg") for r in rows if r.get("max_total_aoa_deg") is not None]
    return {
        "n_phi": n,
        "bias_range_m": biasR,
        "bias_deflection_m": biasD,
        "bias_magnitude_m": float(math.hypot(biasR, biasD)),
        "semi_axis_major_m": major,
        "semi_axis_minor_m": minor,
        "axis_ratio": (major / minor) if minor > 0 else float("inf"),
        "major_axis_tilt_deg": tilt,
        "rms_amplitude_m": amp,
        "harmonic_residual_m": resid,
        "max_total_shift_m": float(total[i_max]),
        "phi_at_max_total_deg": float(rows[i_max]["phi_deg"]),
        "peak_range_swing_m": float(np.max(np.abs(xR))),
        "peak_deflection_swing_m": float(np.max(np.abs(xD))),
        "uncorrected_range_m": rng,
        "steerable_pct_of_range": 100.0 * major / rng,
        "bias_pct_of_range": 100.0 * math.hypot(biasR, biasD) / rng,
        "max_total_pct_of_range": 100.0 * float(total[i_max]) / rng,
        "deploy_time": rows[0]["deploy_time"],
        "apogee_time": rows[0]["apogee_time"],
        "max_total_aoa_deg": max(aoas) if aoas else float("nan"),
    }


# ===========================================================================
# Drivers
# ===========================================================================
def _pool(n=None):
    return Pool(processes=n or max(1, (os.cpu_count() or 2) - 1))


def _baselines(indices, dt=SWEEP_DT):
    jobs = [
        {"charge": FIRING_TABLE[k][0], "mv": FIRING_TABLE[k][1],
         "qe_mils": FIRING_TABLE[k][2], "dt": dt}
        for k in indices
    ]
    with _pool() as p:
        return dict(zip(indices, p.map(baseline, jobs)))


def _group(jobs, out, key):
    g = {}
    for j, r in zip(jobs, out):
        g.setdefault(key(j, r), []).append(r)
    return g


def _tag(r) -> str:
    return "c%d_qe%g" % (r["charge"], r["qe_mils"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join("docs", "authority_results.json"))
    args = ap.parse_args()

    t0 = time.time()
    results = {}
    all_idx = list(range(len(FIRING_TABLE)))
    rep = list(REPRESENTATIVE)

    if args.quick:
        bl = _baselines([rep[-1]])
        b = bl[rep[-1]]
        jobs = [{"baseline": b, "phi_deg": float(p), "deploy_fraction": 0.0}
                for p in PHI_CARDINAL_DEG]
        with _pool() as p:
            rows = p.map(run_case, jobs)
        for r in rows:
            print(" phi %3.0f  dR %+8.2f  dD %+8.2f  aoa %.2f deg"
                  % (r["phi_deg"], r["d_range"], r["d_deflection"],
                     r["max_total_aoa_deg"]))
        print(json.dumps(envelope_stats(rows), indent=1, default=float))
        print("wall %.0f s" % (time.time() - t0))
        return

    print("[1/8] uncorrected baselines, 15 engagements ...", flush=True)
    BL = _baselines(all_idx)

    print("[2/8] main phi sweep at apogee ...", flush=True)
    jobs = [{"baseline": BL[k], "phi_deg": float(ph), "deploy_fraction": 0.0}
            for k in all_idx for ph in PHI_SWEEP_DEG]
    with _pool() as p:
        out = p.map(run_case, jobs)
    g = _group(jobs, out, lambda j, r: _tag(r))
    results["main"] = {k: {"rows": v, "stats": envelope_stats(v)} for k, v in g.items()}

    print("[3/8] deployment-time sweep ...", flush=True)
    fracs = (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5)
    jobs = [{"baseline": BL[k], "phi_deg": float(ph), "deploy_fraction": f}
            for k in rep for f in fracs for ph in PHI_SWEEP_DEG]
    with _pool() as p:
        out = p.map(run_case, jobs)
    g = _group(jobs, out, lambda j, r: "%s_f%+.2f" % (_tag(r), r["deploy_fraction"]))
    results["deploy"] = {k: {"rows": v, "stats": envelope_stats(v)} for k, v in g.items()}

    print("[4/8] deflection linearity ...", flush=True)
    deltas = (0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0)
    jobs = [{"baseline": BL[rep[-1]], "phi_deg": float(ph), "deploy_fraction": 0.0,
             "geometry": replace(cn.NOMINAL_GEOMETRY,
                                 steering_deflection=math.radians(d))}
            for d in deltas for ph in PHI_CARDINAL_DEG]
    with _pool() as p:
        out = p.map(run_case, jobs)
    g = _group(jobs, out, lambda j, r: "%g" % r["steering_deflection_deg"])
    results["deflection"] = {k: {"rows": v, "stats": envelope_stats(v)} for k, v in g.items()}

    print("[5/8] canard station ...", flush=True)
    stations = (0.050, 0.070, 0.090, 0.110, 0.130, 0.160, 0.200)
    jobs = [{"baseline": BL[rep[-1]], "phi_deg": float(ph), "deploy_fraction": 0.0,
             "geometry": replace(cn.NOMINAL_GEOMETRY, station_from_nose=x)}
            for x in stations for ph in PHI_CARDINAL_DEG]
    with _pool() as p:
        out = p.map(run_case, jobs)
    g = _group(jobs, out, lambda j, r: "%.3f" % r["station_from_nose_m"])
    results["station"] = {k: {"rows": v, "stats": envelope_stats(v)} for k, v in g.items()}

    print("[6/8] panel size and carryover ...", flush=True)
    variants = {
        "nominal": cn.NOMINAL_GEOMETRY,
        "half_area": replace(cn.NOMINAL_GEOMETRY, span_exposed=0.021, chord=0.032),
        "double_area": replace(cn.NOMINAL_GEOMETRY, span_exposed=0.042, chord=0.064),
        "quadruple_area": replace(cn.NOMINAL_GEOMETRY, span_exposed=0.060, chord=0.090),
        "no_carryover": replace(cn.NOMINAL_GEOMETRY, include_carryover=False),
    }
    jobs = [{"baseline": BL[rep[-1]], "phi_deg": float(ph), "deploy_fraction": 0.0,
             "geometry": gm, "variant": name}
            for name, gm in variants.items() for ph in PHI_CARDINAL_DEG]
    with _pool() as p:
        out = p.map(run_case, jobs)
    g = _group(jobs, out, lambda j, r: j["variant"])
    results["size"] = {k: {"rows": v, "stats": envelope_stats(v)} for k, v in g.items()}

    print("[7/8] free-nose despin, and induced vs direct ...", flush=True)
    jobs = [{"baseline": BL[k], "deploy_fraction": 0.0} for k in rep]
    with _pool() as p:
        results["free_nose"] = p.map(run_free_nose, jobs)
    jobs = [{"baseline": BL[k], "phi_deg": 0.0, "deploy_fraction": 0.0} for k in rep]
    with _pool() as p:
        results["induced"] = p.map(induced_vs_direct, jobs)

    print("[8/8] step-size convergence of the SHIFT ...", flush=True)
    fine = _baselines([rep[-1]], dt=FINE_DT)[rep[-1]]
    jobs = [{"baseline": fine, "phi_deg": float(ph), "deploy_fraction": 0.0}
            for ph in PHI_CARDINAL_DEG]
    with _pool() as p:
        out_f = p.map(run_case, jobs)
    key = _tag({"charge": FIRING_TABLE[rep[-1]][0], "qe_mils": FIRING_TABLE[rep[-1]][2]})
    coarse = sorted(
        [r for r in results["main"][key]["rows"] if r["phi_deg"] in PHI_CARDINAL_DEG],
        key=lambda r: r["phi_deg"],
    )
    out_f = sorted(out_f, key=lambda r: r["phi_deg"])
    results["convergence"] = [
        {"phi_deg": a["phi_deg"],
         "dt_coarse": a["dt"], "dt_fine": b["dt"],
         "d_range_coarse": a["d_range"], "d_range_fine": b["d_range"],
         "d_range_change": b["d_range"] - a["d_range"],
         "d_defl_coarse": a["d_deflection"], "d_defl_fine": b["d_deflection"],
         "d_defl_change": b["d_deflection"] - a["d_deflection"],
         "abs_range_change": b["corrected_range"] - a["corrected_range"]}
        for a, b in zip(coarse, out_f)
    ]

    results["meta"] = {
        "sweep_dt": SWEEP_DT,
        "fine_dt": FINE_DT,
        "restart_log_every": RESTART_LOG_EVERY,
        "geometry": cn.NOMINAL_GEOMETRY.describe(pr.M107),
        "nose": {
            "inertia": cn.NOMINAL_NOSE.inertia,
            "viscous": cn.NOMINAL_NOSE.viscous,
            "coulomb": cn.NOMINAL_NOSE.coulomb,
            "brake_max": cn.NOMINAL_NOSE.brake_max,
        },
        "wall_seconds": time.time() - t0,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print("wrote %s in %.0f s" % (args.out, time.time() - t0))


if __name__ == "__main__":
    main()
