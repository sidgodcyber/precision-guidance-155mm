"""
Task E: compute cost of one MPMM propagation to impact.

The onboard model runs at about 10 Hz, propagating from the current state to
ground impact every cycle, on a Cortex-M7-class microcontroller. This measures
what one such propagation costs and what accuracy is lost by coarsening the
step or dropping terms.

Everything here is measured on the development machine in CPython. That is not
the target, and the absolute numbers do not transfer -- what transfers is the
DERIVATIVE COUNT and the relative cost of each term, which is what step 7 needs
to size the C implementation.

Run:  python -m analysis.mpmm_compute
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from models import mpmm as M
from sim import aerodata, atmosphere as atm, dynamics as dyn, integrate as ig, projectile as pr

LATITUDE_DEG = 45.0
#: The reference engagement: charge 8, QE 525.3 mils, the longest shot in the
#: firing-table envelope and therefore the worst case for propagation cost.
REF_MV, REF_QE = 684.0, 525.3


def _timed(fn, budget=1.5, max_reps=50):
    """Call fn() repeatedly for about `budget` seconds and return the mean
    wall time in seconds, plus the last result. One warm-up call first."""
    r = fn()
    t0 = time.perf_counter()
    fn()
    one = time.perf_counter() - t0
    reps = max(1, min(max_reps, int(budget / max(one, 1e-6))))
    t0 = time.perf_counter()
    for _ in range(reps):
        r = fn()
    return (time.perf_counter() - t0) / reps, r


def _model(**kw):
    env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=False)
    return M.MpmmModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), environment=env, **kw
    )


def apogee_state():
    """True 6-DOF state at apogee for the reference engagement."""
    env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=False)
    six_model = dyn.FlightModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), environment=env
    )
    launch = pr.LaunchConditions.from_mils(REF_MV, REF_QE)
    y6 = dyn.initial_state(pr.M107, launch)
    six = ig.integrate(y6, six_model, dt=2e-4, log_every=500, t_max=200.0)
    tr = six.trajectory
    i = int(np.argmax(-tr.position[:, 2]))
    y6ap = np.empty(dyn.STATE_SIZE)
    y6ap[0:3] = tr.position[i]
    y6ap[3:6] = tr.velocity[i]
    y6ap[6:10] = tr.quaternion[i]
    y6ap[10:13] = tr.omega[i]
    return M.state_from_sixdof(y6ap), float(tr.t[i]), six


def main():
    out = {}
    print("Building the reference apogee state from a 6-DOF run ...", flush=True)
    y_ap, t_ap, six = apogee_state()
    print(f"  apogee at t = {t_ap:.2f} s, altitude {-y_ap[2]:.1f} m, "
          f"remaining flight {six.impact_time - t_ap:.2f} s")
    out["apogee_time"] = t_ap
    out["apogee_alt"] = float(-y_ap[2])
    out["sixdof_impact_range"] = six.range_m
    out["sixdof_impact_drift"] = six.drift_m

    model = _model()

    # ---- reference propagation at the finest step ------------------------
    ref = M.propagate_to_impact(y_ap, model, dt=0.001, t0=t_ap)
    print(f"\nReference (dt = 0.001 s): range {ref.range_m:.3f} m  "
          f"drift {ref.drift_m:.3f} m  {ref.steps} steps")

    # ---- Task E.1: cost and accuracy versus step size --------------------
    print("\n=== COST AND ACCURACY VERSUS STEP SIZE (propagation from apogee) ===")
    print("   dt(s)   steps   derivs   wall(ms)   us/deriv   dRange(m)  dDrift(m)")
    rows = []
    for dt in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        el, r = _timed(lambda dt=dt: M.propagate_to_impact(y_ap, model, dt=dt, t0=t_ap))
        derivs = 4 * r.steps
        rows.append(dict(dt=dt, steps=r.steps, derivs=derivs, wall_ms=el * 1e3,
                         us_per_deriv=el * 1e6 / max(1, derivs),
                         d_range=r.range_m - ref.range_m,
                         d_drift=r.drift_m - ref.drift_m,
                         range_m=r.range_m, drift_m=r.drift_m))
        print(f"  {dt:6.3f} {r.steps:7d} {derivs:8d} {el*1e3:10.3f} "
              f"{el*1e6/max(1,derivs):10.2f} {r.range_m-ref.range_m:+10.3f} "
              f"{r.drift_m-ref.drift_m:+10.3f}")
    out["step_size"] = rows

    # ---- Task E.2: term ablation ----------------------------------------
    print("\n=== TERM ABLATION at dt = 0.01 s (cost saved vs accuracy lost) ===")
    base_s, base = _timed(lambda: M.propagate_to_impact(y_ap, model, dt=0.01, t0=t_ap))
    base_ms = base_s * 1e3

    print(f"  {'term dropped':<28} {'wall(ms)':>9} {'saved':>7} "
          f"{'dRange(m)':>11} {'dDrift(m)':>11}")
    print(f"  {'(none, baseline)':<28} {base_ms:9.3f} {'--':>7} "
          f"{0.0:11.3f} {0.0:11.3f}")
    ablations = []
    for label, kw in (
        ("Magnus force", dict(include_magnus=False)),
        ("yaw drag (alpha^2 on C_D)", dict(include_yaw_drag=False)),
        ("lift (yaw of repose)", dict(include_lift=False)),
        ("Coriolis (ADDED, not dropped)", None),
    ):
        if kw is None:
            env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=False)
            m2 = M.MpmmModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                             environment=env)
            # Coriolis is already off in this comparison; measure it by turning
            # it ON in the baseline instead, and report the difference.
            env_on = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=True)
            m_on = M.MpmmModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                               environment=env_on)
            ms_s, r_on = _timed(
                lambda: M.propagate_to_impact(y_ap, m_on, dt=0.01, t0=t_ap))
            ms = ms_s * 1e3
            dr, dd = base.range_m - r_on.range_m, base.drift_m - r_on.drift_m
            print(f"  {label:<28} {ms:9.3f} {base_ms-ms:+7.3f} "
                  f"{dr:11.3f} {dd:11.3f}")
            ablations.append(dict(term=label, wall_ms=ms, d_range=dr, d_drift=dd))
            continue
        m2 = _model(**kw)
        ms_s, r2 = _timed(lambda m2=m2: M.propagate_to_impact(y_ap, m2, dt=0.01, t0=t_ap))
        ms = ms_s * 1e3
        dr = r2.range_m - base.range_m
        dd = r2.drift_m - base.drift_m
        print(f"  {label:<28} {ms:9.3f} {base_ms-ms:+7.3f} {dr:11.3f} {dd:11.3f}")
        ablations.append(dict(term=label, wall_ms=ms, saved_ms=base_ms - ms,
                              d_range=dr, d_drift=dd))
    out["ablation"] = ablations
    out["baseline_ms"] = base_ms

    # ---- Task E.2b: where the time actually goes, per derivative call ----
    # The end-to-end ablation above cannot resolve a single force term: the
    # include_* flags skip only a handful of multiply-adds, because alpha_e
    # has to be formed for all three of them anyway. Timing the derivative
    # directly, over many calls, resolves what the propagation timing cannot.
    print()
    print("=== PER-DERIVATIVE COST BREAKDOWN (mid-descent state) ===")
    y_mid = M.pack(y_ap[0:3], y_ap[3:6], y_ap[6])

    def bench(fn, n=20000):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1e6      # microseconds

    # NOTE: the models are built ONCE, outside the timed lambda. Building an
    # MpmmModel constructs a coefficient table; doing that inside the loop
    # measures table construction, not the derivative.
    variant_models = [
        ("full model", model),
        ("no Magnus", _model(include_magnus=False)),
        ("no yaw drag", _model(include_yaw_drag=False)),
        ("no lift", _model(include_lift=False)),
        ("no yaw terms at all (3-DOF)", _model(include_magnus=False,
                                               include_yaw_drag=False,
                                               include_lift=False)),
    ]
    variants = [(label, bench(lambda m=m: M.derivative(0.0, y_mid, m)))
                for label, m in variant_models]
    full_us = variants[0][1]
    # the two shared sub-costs every variant pays
    alt = float(-y_mid[2])
    V = float(np.linalg.norm(y_mid[3:6]))
    _T, _pp, _rho, a_snd = atm.isa_scalars(alt)
    mach = V / a_snd
    atm_us = bench(lambda: atm.isa_scalars(alt), 200000)
    aero_us = bench(lambda: model.aero.lookup(mach), 200000)

    print(f"  {'variant':<32} {'us/deriv':>9} {'vs full':>9}")
    micro = []
    for label, us in variants:
        print(f"  {label:<32} {us:9.3f} {us-full_us:+9.3f}")
        micro.append(dict(variant=label, us=us, delta_us=us - full_us))
    print(f"  {'-- of which: ISA atmosphere':<32} {atm_us:9.3f} "
          f"{100*atm_us/full_us:8.1f} %")
    print(f"  {'-- of which: aero table lookup':<32} {aero_us:9.3f} "
          f"{100*aero_us/full_us:8.1f} %")
    out["micro"] = micro
    out["full_us_per_deriv"] = full_us
    out["atm_us"] = atm_us
    out["aero_us"] = aero_us

    # ---- Task E.3: is the yaw-of-repose fixed-point iteration worth it? --
    print()
    print("=== YAW-OF-REPOSE FIXED-POINT ITERATION ===")
    ys = M.pack(y_ap[0:3], y_ap[3:6], y_ap[6])
    a0 = M.yaw_of_repose(t_ap, ys, model, iterate=False)[3]
    a1 = M.yaw_of_repose(t_ap, ys, model, iterate=True)[3]
    print(f"  alpha_e at apogee, no iteration {math.degrees(a0):.6f} deg")
    print(f"  alpha_e at apogee, one pass     {math.degrees(a1):.6f} deg")
    print(f"  relative change {100*(a1-a0)/a0:+.4f} %")
    out["alpha_no_iter_deg"] = math.degrees(a0)
    out["alpha_one_iter_deg"] = math.degrees(a1)

    # and what that is worth at the impact point, end to end
    m_it = _model(iterate_yaw=True)
    r_it = M.propagate_to_impact(y_ap, m_it, dt=0.01, t0=t_ap)
    it_s, _ = _timed(lambda: M.propagate_to_impact(y_ap, m_it, dt=0.01, t0=t_ap))
    it_us = bench(lambda: M.derivative(0.0, y_mid, m_it))
    print(f"  impact with iteration: range {r_it.range_m:.3f} m  "
          f"drift {r_it.drift_m:.3f} m")
    print(f"  vs non-iterated:       dRange {r_it.range_m-base.range_m:+.3f} m  "
          f"dDrift {r_it.drift_m-base.drift_m:+.3f} m")
    print(f"  cost: {it_us:.3f} us/deriv vs {full_us:.3f} "
          f"({100*(it_us-full_us)/full_us:+.1f} %), "
          f"{it_s*1e3:.1f} ms vs {base_ms:.1f} ms per propagation")
    out["iterate"] = dict(range_m=r_it.range_m, drift_m=r_it.drift_m,
                          d_range=r_it.range_m - base.range_m,
                          d_drift=r_it.drift_m - base.drift_m,
                          us_per_deriv=it_us, wall_ms=it_s * 1e3)

    # ---- Task E.4: 10 Hz duty cycle -------------------------------------
    print("\n=== 10 Hz DUTY CYCLE ===")
    for row in out["step_size"]:
        duty = row["wall_ms"] / 100.0 * 100.0   # ms per cycle / 100 ms budget
        print(f"  dt {row['dt']:6.3f}: {row['wall_ms']:8.3f} ms per propagation "
              f"= {duty:7.2f} % of a 100 ms budget (CPython, dev machine); "
              f"{row['derivs']} derivative evaluations")

    os.makedirs("docs", exist_ok=True)
    with open("docs/mpmm_compute.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nWrote docs/mpmm_compute.json")


if __name__ == "__main__":
    main()
