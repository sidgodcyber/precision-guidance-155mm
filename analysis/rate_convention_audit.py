"""
Convention audit for the rate-dependent coefficients.

Two separate questions, kept apart on purpose:

  PART 1  Why does `rate coeffs x2` INCREASE the angle of attack, when
          doubling damping derivatives should reduce it? Answered by
          decomposing the switch into its four coefficients and scaling one
          at a time.

  PART 2  What is each observable's sensitivity to REDUCED_RATE_FACTOR, the
          pd/V vs pd/(2V) choice?

  PART 3  THE DISCRIMINATOR. Reproduce the ASAT-13 section 4.3 trajectory --
          the coefficient source's OWN published trajectory, computed by the
          source with the deck under test -- under both conventions. This is
          the only test available that is independent of the firing table.

  PART 4  What each convention does to drift across all 15 firing-table
          points, reported because the answer must be stated whichever way
          it falls.

Nothing here changes the model. It is measurement.

Findings are written up in docs/RATE-CONVENTION.md.

Run:  python -m analysis.rate_convention_audit
"""

from __future__ import annotations

import json
import math
import os
from multiprocessing import Pool

import numpy as np

from sim import aerodata, diagnostics as dg, dynamics as dyn, integrate as ig
from sim import projectile as pr

LATITUDE_DEG = 45.0
DT = 2.0e-4
LOG_EVERY = 200          # 0.04 s: max_total_aoa is a max over logged samples

#: Column index of each coefficient inside AeroTable.values.
COL = {name: i for i, name in enumerate(aerodata.COEFFICIENT_NAMES)}

#: The four rate-dependent coefficients -- exactly what rate_coefficient_scale
#: multiplies in dynamics.py.
RATE_COEFFS = ("C_Ypalpha", "C_lp", "C_mq", "C_Mpalpha")


def scaled_table(scales: dict) -> aerodata.AeroTable:
    """A fresh M107 table with individual coefficient columns scaled."""
    rows = aerodata.apply_measured_cnalpha(aerodata._M107_ROWS).copy()
    for name, k in scales.items():
        rows[:, 1 + COL[name]] *= k
    return aerodata.AeroTable(rows, name="M107 scaled", source="audit")


def run(qe_mils, scales=None, rate_factor=None, mv=684.0):
    """One trajectory. `scales` scales named coefficients; `rate_factor`
    overrides REDUCED_RATE_FACTOR for the whole run."""
    saved = aerodata.REDUCED_RATE_FACTOR
    if rate_factor is not None:
        aerodata.REDUCED_RATE_FACTOR = rate_factor
        dyn.REDUCED_RATE_FACTOR = rate_factor
    try:
        env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=False)
        model = dyn.FlightModel(
            projectile=pr.M107,
            aero=scaled_table(scales or {}),
            environment=env,
        )
        launch = pr.LaunchConditions.from_mils(mv, qe_mils)
        y0 = dyn.initial_state(pr.M107, launch)
        ms = dg.muzzle_stability(y0, model)
        res = ig.integrate(y0, model, dt=DT, log_every=LOG_EVERY, t_max=200.0)
        tr = res.trajectory
        stab = dg.stability_history(tr, model)
        sg = np.array([s.Sg for s in stab])
        sd = np.array([s.Sd for s in stab])
        need = np.array([s.Sg_required for s in stab])
        spin = np.array([s.spin for s in stab])
        margin = sg - need
        return dict(
            qe_mils=qe_mils,
            range_m=res.range_m,
            drift_m=res.drift_m,
            tof_s=res.impact_time,
            max_aoa_deg=math.degrees(res.max_total_aoa),
            p0_rps=float(spin[0]) / (2 * math.pi),
            p_imp_rps=float(spin[-1]) / (2 * math.pi),
            spin_retained=float(spin[-1] / spin[0]),
            Sg_muzzle=float(sg[0]),
            Sg_min=float(sg.min()),
            Sg_impact=float(sg[-1]),
            Sd_muzzle=float(sd[0]),
            Sd_max=float(np.nanmax(sd)),
            Sg_required_max=float(np.nanmax(need)),
            margin_min=float(np.nanmin(margin)),
            frac_unstable=float(np.mean(margin < 0.0)),
        )
    finally:
        aerodata.REDUCED_RATE_FACTOR = saved
        dyn.REDUCED_RATE_FACTOR = saved


def _job(a):
    qe, scales, rf, label = a
    r = run(qe, scales, rf)
    r["label"] = label
    return r


def _set_rrf(v):
    """Override the convention everywhere it is bound at import time."""
    import sim.aerodata, sim.dynamics, sim.diagnostics
    for mod in (sim.aerodata, sim.dynamics, sim.diagnostics):
        mod.REDUCED_RATE_FACTOR = v


def part3_asat():
    """
    THE DISCRIMINATOR.

    ASAT-13 section 4.3 publishes a trajectory it computed with the very deck
    this package uses, and states the peak total angle of attack AND the time
    at which it occurs. The peak TIME is the diagnostic: it is a shape feature
    of the yaw history, not a magnitude that a coefficient scale can slide.
    """
    import run_validation as rv
    out = []
    for v in (0.5, 1.0):
        _set_rrf(v)
        r = rv.rung5b_asat(dt=DT)
        r["reduced_rate_factor"] = v
        out.append(r)
    _set_rrf(0.5)

    spec = rv.ASAT_5B
    print()
    print("=" * 118)
    print("PART 3 -- ASAT-13 SECTION 4.3: the source's own published trajectory")
    print("=" * 118)
    print(f"{'quantity':<32}{'pd/(2V)':>12}{'pd/V':>12}{'ASAT published':>18}")
    for label, key, sk in (
        ("initial axial decel (g)", "total_axial_g", "initial_axial_decel_g"),
        ("total flight time (s)", "tof_s", "tof_s"),
        ("summit time (s)", "summit_time_s", "summit_time_s"),
        ("MAX TOTAL AoA (deg)", "max_aoa_deg", "max_aoa_deg"),
        ("TIME OF MAX AoA (s)", "max_aoa_time_s", "max_aoa_time_s"),
    ):
        print(f"{label:<32}{out[0][key]:12.3f}{out[1][key]:12.3f}{spec[sk]:18.3f}")
    return out


def part4_firing_table():
    """What each convention does to drift over the whole FT envelope."""
    from run_validation import LATITUDE_DEG  # noqa: F401  (documents the site)
    FT = [(4, 337.0, 97.2, 2000.0, 3.2), (4, 337.0, 152.0, 3000.0, 7.8),
          (4, 337.0, 211.6, 4000.0, 15.2), (5, 397.0, 118.1, 3000.0, 7.5),
          (5, 397.0, 280.4, 6000.0, 36.6), (5, 397.0, 420.6, 8000.0, 79.2),
          (6, 474.0, 258.4, 7000.0, 45.5), (6, 474.0, 378.6, 9000.0, 88.2),
          (6, 474.0, 539.9, 11000.0, 169.4), (7, 568.0, 177.6, 7000.0, 37.1),
          (7, 568.0, 319.8, 10000.0, 94.0), (7, 568.0, 520.7, 13000.0, 211.9),
          (8, 684.0, 141.6, 8000.0, 37.6), (8, 684.0, 248.4, 11000.0, 92.4),
          (8, 684.0, 525.3, 16000.0, 292.8)]
    jobs = [(qe, None, rrf, "ft", mv, ftr, ftd, chg)
            for rrf in (0.5, 1.0) for (chg, mv, qe, ftr, ftd) in FT]
    with Pool(min(8, os.cpu_count() or 2)) as pool:
        res = pool.map(_ft_job, jobs)
    print()
    print("=" * 118)
    print("PART 4 -- DRIFT vs FT 155-AM-2 UNDER EACH CONVENTION")
    print("=" * 118)
    a = [r for r in res if r["rrf"] == 0.5]
    b = [r for r in res if r["rrf"] == 1.0]
    print(f"{'chg':>3}{'QE':>8}{'FT drift':>10} | {'pd/(2V)':>9}{'err%':>9}"
          f"{'maxAoA':>9} | {'pd/V':>9}{'err%':>9}{'maxAoA':>9}")
    for x, y in zip(a, b):
        print(f"{x['chg']:3d}{x['qe']:8.1f}{x['ft_drift']:10.1f} | "
              f"{x['drift']:9.2f}{x['err']:+9.2f}{x['aoa']:9.3f} | "
              f"{y['drift']:9.2f}{y['err']:+9.2f}{y['aoa']:9.3f}")
    for lbl, sel in (("pd/(2V) current", a), ("pd/V    (x2)", b)):
        e = np.array([r["err"] for r in sel])
        g = np.array([100 * (r["range_m"] - r["ft_range"]) / r["ft_range"] for r in sel])
        print(f"  {lbl:<16} drift vs FT: mean {e.mean():+6.2f} %  rms {np.sqrt((e**2).mean()):5.2f} %"
              f"  span {e.min():+6.2f} to {e.max():+6.2f}"
              f"   |  range vs FT: mean {g.mean():+5.2f} %  rms {np.sqrt((g**2).mean()):4.2f} %")
    return res


def _ft_job(a):
    qe, _sc, rrf, _tag, mv, ft_range, ft_drift, chg = a
    _set_rrf(rrf)
    r = run(qe, {}, rrf, mv=mv)
    r.update(rrf=rrf, mv=mv, ft_range=ft_range, ft_drift=ft_drift, chg=chg, qe=qe,
             drift=r["drift_m"], aoa=r["max_aoa_deg"],
             err=100.0 * (r["drift_m"] - ft_drift) / ft_drift)
    return r


def main():
    jobs = []

    # ---- PART 1: decompose the x2 switch, one coefficient at a time ------
    for qe in (141.6, 525.3):
        jobs.append((qe, {}, None, "baseline"))
        for c in RATE_COEFFS:
            jobs.append((qe, {c: 2.0}, None, "x2 " + c))
        jobs.append((qe, {c: 2.0 for c in RATE_COEFFS}, None, "x2 ALL FOUR"))
        # the two damping terms together, without the Magnus pair
        jobs.append((qe, {"C_lp": 2.0, "C_mq": 2.0}, None, "x2 damping only"))
        # the two Magnus terms together, without the damping pair
        jobs.append((qe, {"C_Ypalpha": 2.0, "C_Mpalpha": 2.0}, None, "x2 Magnus only"))

    # ---- PART 2: the convention switch itself ----------------------------
    for qe in (141.6, 525.3):
        jobs.append((qe, {}, 1.0, "REDUCED_RATE_FACTOR = 1.0 (pd/V)"))

    with Pool(min(8, os.cpu_count() or 2)) as pool:
        rows = pool.map(_job, jobs)

    os.makedirs("docs", exist_ok=True)
    for qe in (141.6, 525.3):
        sel = [r for r in rows if r["qe_mils"] == qe]
        base = sel[0]
        print()
        print("=" * 118)
        print(f"CHARGE 8, QE {qe} mils    (FT drift "
              f"{37.6 if qe == 141.6 else 292.8} m)")
        print("=" * 118)
        print(f"{'case':<34}{'range':>9}{'drift':>9}{'dDrift%':>9}"
              f"{'maxAoA':>9}{'dAoA%':>8}{'p_imp':>8}{'spin%':>7}"
              f"{'Sg_min':>8}{'Sd_max':>8}{'margin':>9}")
        for r in sel:
            print(f"{r['label']:<34}{r['range_m']:9.1f}{r['drift_m']:9.2f}"
                  f"{100*(r['drift_m']-base['drift_m'])/base['drift_m']:+9.2f}"
                  f"{r['max_aoa_deg']:9.3f}"
                  f"{100*(r['max_aoa_deg']-base['max_aoa_deg'])/base['max_aoa_deg']:+8.1f}"
                  f"{r['p_imp_rps']:8.1f}{100*r['spin_retained']:7.1f}"
                  f"{r['Sg_min']:8.2f}{r['Sd_max']:8.3f}{r['margin_min']:+9.2f}")
    asat_rows = part3_asat()
    ft_rows = part4_firing_table()
    with open("docs/rate_convention_audit.json", "w", encoding="utf-8") as fh:
        json.dump({"decomposition": rows, "asat": asat_rows,
                   "firing_table": ft_rows}, fh, indent=1)

    print("\nWrote docs/rate_convention_audit.json")


if __name__ == "__main__":
    main()
