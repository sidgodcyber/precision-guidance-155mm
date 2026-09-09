"""
Gyroscopic and dynamic stability WITH THE CANARDS DEPLOYED.

[CANARD-MODEL.md](../docs/CANARD-MODEL.md) limitation 7 records that step 2.5
never computed this:

  "Gyroscopic stability with canards deployed is reduced, and this session did
   not check it properly. ... `sim/diagnostics.py` still computes `Sg` from
   `Ix*p` and the body-alone `C_Malpha`, so the deployed-configuration `Sg` is
   NOT reported anywhere in this project. A design that deployed near muzzle
   exit would need it."

Phase 3 of the authority session proposes exactly that -- deployment on the
ASCENDING branch, at a quarter of the apogee time -- so the check can no
longer be deferred. Two effects reduce Sg together:

  * the canards add their own `C_Malpha` at the canard station, which is far
    ahead of the CG, so the combined overturning moment slope rises;
  * despinning the nose removes `I_nose * p` from the axial angular momentum,
    and it is the TOTAL axial angular momentum `H_x`, not `Ix * p`, that sets
    gyroscopic stiffness.

        Sg_deployed = H_x^2 / (2 rho S d It V^2 C_Malpha_total)
        H_x         = (Ix - I_nose) p + I_nose p_nose,   p_nose ~ 0 despun

This module evaluates both along the UNCORRECTED trajectory, which is the
right reference: it asks "if the kit were deployed at this instant, would the
shell still be stable?" for every instant of the flight.

Run:  python -m analysis.deployed_stability
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from analysis.authority import FIRING_TABLE, LATITUDE_DEG, _apogee_time
from sim import aerodata
from sim import canards as cn
from sim import diagnostics as dg
from sim import dynamics as dyn
from sim import integrate as ig
from sim import projectile as pr

DT = 5.0e-4


def _model():
    env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=False)
    return dyn.FlightModel(projectile=pr.M107, aero=aerodata.make_m107_table(),
                           environment=env)


def canard_moment_slope(mach: float, geometry: cn.CanardGeometry,
                        projectile: pr.Projectile) -> tuple:
    """(delta C_Nalpha, delta C_Malpha) contributed by the four-panel array."""
    S = projectile.reference_area
    d = projectile.diameter
    a_c = geometry.moment_arm(projectile)
    C_La = cn.canard_lift_curve_slope(mach, geometry.aspect_ratio_effective)
    dC_N = 2.0 * geometry.panel_area * C_La * geometry.carryover / S
    return dC_N, dC_N * a_c / d


def stability_along(charge_index: int, geometry=None, nose=None,
                    dt: float = DT) -> dict:
    geometry = geometry or cn.NOMINAL_GEOMETRY
    nose = nose or cn.NOMINAL_NOSE
    chg, mv, qe, _ = FIRING_TABLE[charge_index]
    launch = pr.LaunchConditions.from_mils(mv, qe)
    model = _model()
    res = ig.integrate(dyn.initial_state(pr.M107, launch), model, dt=dt,
                       log_every=200, t_max=200.0)
    tr = res.trajectory
    t_ap = _apogee_time(res)
    proj = pr.M107
    I_nose = nose.inertia
    Ix_body = proj.I_axial - I_nose
    S, d, It = proj.reference_area, proj.diameter, proj.I_transverse

    rows = []
    for i in range(tr.t.size):
        mach = float(tr.mach[i])
        V = float(tr.airspeed[i])
        rho = float(tr.density[i])
        p = float(tr.omega[i, 0])
        c = model.aero.coefficients_at(mach)
        dC_N, dC_M = canard_moment_slope(mach, geometry, proj)

        Sg_stowed = dg.gyroscopic_stability_factor(proj, rho, V, p, c.C_Malpha)
        Hx = Ix_body * p          # nose despun: p_nose ~ 0
        den = 2.0 * rho * S * d * It * V * V * (c.C_Malpha + dC_M)
        Sg_deployed = (Hx * Hx / den) if den > 0 else math.inf

        Sd_stowed = dg.dynamic_stability_factor(proj, c.C_Nalpha, c.C_X0,
                                                c.C_Mpalpha, c.C_mq)
        Sd_deployed = dg.dynamic_stability_factor(proj, c.C_Nalpha + dC_N,
                                                  c.C_X0, c.C_Mpalpha, c.C_mq)
        rows.append({
            "t": float(tr.t[i]), "t_over_t_apogee": float(tr.t[i]) / t_ap,
            "mach": mach, "airspeed": V, "density": rho, "spin": p,
            "C_Malpha_body": c.C_Malpha, "delta_C_Malpha": dC_M,
            "C_Malpha_total": c.C_Malpha + dC_M,
            "Sg_stowed": Sg_stowed, "Sg_deployed": Sg_deployed,
            "Sg_ratio": Sg_deployed / Sg_stowed if Sg_stowed else float("nan"),
            "Sd_stowed": Sd_stowed, "Sd_deployed": Sd_deployed,
            "Sg_required_stowed": dg.dynamic_stability_limit(Sd_stowed),
            "Sg_required_deployed": dg.dynamic_stability_limit(Sd_deployed),
        })
    return {"charge": chg, "qe_mils": qe, "range_m": res.range_m,
            "tof": res.impact_time, "apogee_time": t_ap,
            "station_from_nose_m": geometry.station_from_nose,
            "panel_area_m2": geometry.panel_area, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/deployed_stability.json")
    ap.add_argument("--engagements", type=int, nargs="*", default=[0, 6, 14])
    ap.add_argument("--stations", type=float, nargs="*", default=[0.090, 0.025])
    args = ap.parse_args(argv)

    from dataclasses import replace
    out = {"cases": []}
    for station in args.stations:
        geom = replace(cn.NOMINAL_GEOMETRY, station_from_nose=station)
        for k in args.engagements:
            r = stability_along(k, geom)
            r["engagement_index"] = k
            out["cases"].append(r)
            rows = r["rows"]
            print(f"\n=== charge {r['charge']} QE {r['qe_mils']} "
                  f"({r['range_m'] / 1000:.1f} km), canards at "
                  f"{station * 1000:.0f} mm ===")
            print(f"{'t/t_ap':>7} {'t s':>7} {'M':>6} {'Sg stowed':>10} "
                  f"{'Sg deployed':>12} {'ratio':>7} {'Sg req':>8} {'margin':>8}")
            for f in (0.0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50):
                j = min(range(len(rows)),
                        key=lambda i: abs(rows[i]["t_over_t_apogee"] - f))
                w = rows[j]
                need = w["Sg_required_deployed"]
                print(f"{w['t_over_t_apogee']:7.2f} {w['t']:7.1f} {w['mach']:6.3f} "
                      f"{w['Sg_stowed']:10.3f} {w['Sg_deployed']:12.3f} "
                      f"{w['Sg_ratio']:7.3f} {need:8.3f} "
                      f"{w['Sg_deployed'] / need if need > 0 else float('inf'):8.3f}")
            worst = min(rows, key=lambda w: w["Sg_deployed"])
            print(f"  minimum Sg_deployed = {worst['Sg_deployed']:.3f} at "
                  f"t/t_ap = {worst['t_over_t_apogee']:.2f} (M {worst['mach']:.3f}); "
                  f"stowed there = {worst['Sg_stowed']:.3f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {args.out}")
    return out


if __name__ == "__main__":
    main()
