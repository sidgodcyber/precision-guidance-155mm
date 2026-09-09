"""
PHASE 3 -- the design sweep, and the aerodynamic sensitivity that Phase 0
showed has to go with it.

Step 2.5 swept the canard station, the panel area and the deflection and read
the results against a closed form that assumed the net correction force is
ANTI-PARALLEL to the canard force. Phase 0 of this session measured the
direction in the 6-DOF and found it is not: it is rotated by 70-80 degrees,
because the direct canard force and the induced body normal force cancel to a
few per cent while the MAGNUS force -- which is perpendicular to the angle of
attack, and therefore perpendicular to both of them -- is about 15 per cent of
either. The residue and the Magnus term are comparable, and the Magnus term is
INDEPENDENT of the canard station.

That changes what a design sweep means, so this module sweeps the same
parameters step 2.5 did plus the one it did not:

  1. canard station, forward only, bounded by the fuze-cavity envelope
  2. deployment point, with the Mach and time-remaining effects separated
  3. panel area
  4. steering deflection, reported against the trim angle of attack
  5. the Magnus force coefficient C_Ypalpha, which the deck itself flags as
     its lowest-confidence entry and which Phase 0 showed is now a leading
     term in the answer

Method is step 2.5's: fly uncorrected, fly again with the nose held at a fixed
earth-referenced roll angle, difference the impact points, sweep the roll
angle, fit the ellipse. `analysis.authority.envelope_stats` does the fit, so
the reduction is identical and the numbers are comparable row for row.

The aero variants change the UNCORRECTED trajectory too, so a baseline is
flown per (engagement, aero variant) rather than shared across variants.

Run:  python -m analysis.design_sweep            full, ~8 cores
      python -m analysis.design_sweep --quick    one engagement, station only
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

from analysis.authority import (FIRING_TABLE, LATITUDE_DEG, LOG_EVERY,
                                RESTART_LOG_EVERY, SWEEP_DT, _apogee_time,
                                _deploy_index, _states_from, envelope_stats)
from sim import aerodata
from sim import canards as cn
from sim import dynamics as dyn
from sim import integrate as ig
from sim import projectile as pr

#: Four cardinal roll angles. The step-2.5 twelve-point sweep established that
#: the response is a pure first harmonic to 0.6 % of amplitude (median, 2.8 %
#: worst), so four points determine the ellipse exactly and the residual the
#: twelve-point fit reports is already known. This is a 3x saving per point
#: and it is the only reason a sweep this wide fits in the session.
PHI_CARDINAL = (0.0, 90.0, 180.0, 270.0)

#: The three engagements step 2.5 carried its sweeps on: a short low-angle
#: shot, a mid-range one and the maximum-range case.
REPRESENTATIVE = (0, 14)

# ---------------------------------------------------------------------------
# The fuze-cavity bound on the canard station
# ---------------------------------------------------------------------------
#: A fuze-well guidance kit can only carry canards on the part of itself that
#: protrudes AHEAD of the projectile's nose thread; everything aft of that is
#: inside the shell's own ogive. The M1156 PGK, the fielded kit of exactly
#: this class, has an overall length of 8.67 in (220.2 mm) of which
#: 3.75 in = 95.25 mm is visible ahead of the shell and 4.91 in = 124.7 mm
#: intrudes into the fuze cavity. 95 mm from the tip is therefore the AFT
#: bound on any canard station for a kit of this class, and it is a hard
#: geometric bound rather than an aerodynamic preference.
KIT_VISIBLE_LENGTH_M = 0.09525
#: Forward bound: the panel needs its own chord plus a nose cap ahead of it,
#: and the kit tapers, so the local radius runs out. 25 mm from the tip is
#: about where a 45 mm chord stops fitting on a 60 mm-diameter section.
STATION_MIN_M = 0.025

#: Radius of the kit's visible section where it meets the shell's nose thread.
KIT_BASE_RADIUS_M = 0.030
#: Radius of its blunt tip. A fuze-well kit of this class carries a GPS patch
#: antenna in the nose, so the tip is truncated rather than pointed.
KIT_TIP_RADIUS_M = 0.012
#: Deployed panel tip radius. Held at the step-2.5 value, which is 17.5 mm
#: inside the shell's own 77.5 mm radius so the panels stow within the
#: projectile envelope.
DEPLOYED_TIP_RADIUS_M = 0.060

STATION_SWEEP_M = (0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095)


def kit_local_radius(station: float) -> float:
    """
    Local radius of the guidance kit at `station` metres from the tip.

    Linear taper of the kit's VISIBLE section from a 12 mm-radius blunt tip to
    the 30 mm radius it has where it meets the shell's nose thread at
    95.25 mm. This is the constraint that makes moving the canards forward
    cost something: there is less kit to mount them on.
    """
    if station >= KIT_VISIBLE_LENGTH_M:
        return KIT_BASE_RADIUS_M
    f = max(0.0, station) / KIT_VISIBLE_LENGTH_M
    return KIT_TIP_RADIUS_M + f * (KIT_BASE_RADIUS_M - KIT_TIP_RADIUS_M)
AREA_SCALE_SWEEP = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DEFLECTION_SWEEP_DEG = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
DEPLOY_FRACTION_SWEEP = (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5)
#: C_Ypalpha scale factors. The deck's own confidence note records that BRL
#: measured C_Npalpha = -0.15 to -0.55 for this shell family in the pd/V
#: normalisation, which is -0.30 to -1.10 in this package's pd/(2V) -- so the
#: measured spread brackets the deck's -0.77 to -1.08 by roughly a factor of
#: 2 in each direction. 0.0 is not a physical case; it is the diagnostic that
#: isolates how much of the answer the Magnus force is carrying.
CYPA_SCALE_SWEEP = (0.0, 0.5, 1.0, 2.0)
#: The same scaling applied to the Magnus MOMENT as well, to separate the
#: Magnus FORCE (which acts on the trajectory directly) from the Magnus
#: MOMENT (which acts only through the trim angle).
CYPA_CMPA_SCALE_SWEEP = (0.0, 2.0)


def scaled_aero(c_ypa_scale: float = 1.0, c_mpa_scale: float = 1.0):
    """The measured M107 deck with the two Magnus coefficients scaled."""
    tbl = aerodata.make_m107_table()
    if c_ypa_scale == 1.0 and c_mpa_scale == 1.0:
        return tbl
    rows = np.column_stack([tbl.mach, tbl.values])
    j_ypa = 1 + list(aerodata.COEFFICIENT_NAMES).index("C_Ypalpha")
    j_mpa = 1 + list(aerodata.COEFFICIENT_NAMES).index("C_Mpalpha")
    rows[:, j_ypa] *= c_ypa_scale
    rows[:, j_mpa] *= c_mpa_scale
    return aerodata.AeroTable(
        rows, name=f"{tbl.name} [C_Ypa x{c_ypa_scale}, C_Mpa x{c_mpa_scale}]",
        source=tbl.source)


def _base_model(c_ypa_scale=1.0, c_mpa_scale=1.0):
    env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=False)
    return dyn.FlightModel(projectile=pr.M107,
                           aero=scaled_aero(c_ypa_scale, c_mpa_scale),
                           environment=env)


def _geometry(station=None, area_scale=1.0, deflection_deg=None, taper=False):
    """
    A canard geometry.

    `taper=False` moves the station alone, holding the nominal 30 mm local
    radius and 30 mm exposed semi-span, so the result is directly comparable
    with the step-2.5 station sweep.

    `taper=True` is the geometry a real kit would have at that station: the
    local radius follows `kit_local_radius`, and the exposed semi-span grows
    to keep the deployed tip radius at 60 mm. Forward stations therefore get
    a LARGER panel on a THINNER body, which raises the exposed area and lowers
    the slender-body carryover at the same time.
    """
    g = cn.NOMINAL_GEOMETRY
    kw = {}
    if station is not None:
        kw["station_from_nose"] = float(station)
        if taper:
            r = kit_local_radius(float(station))
            kw["body_radius_local"] = r
            kw["span_exposed"] = DEPLOYED_TIP_RADIUS_M - r
    if deflection_deg is not None:
        kw["steering_deflection"] = math.radians(deflection_deg)
    if area_scale != 1.0:
        # Hold the aspect ratio: scale span and chord by sqrt(area_scale), so
        # the panel lift-curve slope is unchanged and the sweep measures area
        # alone. The slender-body carryover (1 + r/s)^2 still falls as the
        # span grows, which is physical, and the tip radius eventually leaves
        # the shell profile -- `fits_within_envelope` flags that.
        r = math.sqrt(area_scale)
        kw["span_exposed"] = g.span_exposed * r
        kw["chord"] = g.chord * r
    return replace(g, **kw) if kw else g


# ===========================================================================
# One (engagement, aero variant) baseline
# ===========================================================================
def baseline(args) -> dict:
    launch = pr.LaunchConditions.from_mils(args["mv"], args["qe_mils"])
    model = _base_model(args.get("c_ypa_scale", 1.0), args.get("c_mpa_scale", 1.0))
    y0 = dyn.initial_state(pr.M107, launch)
    res = ig.integrate(y0, model, dt=SWEEP_DT, log_every=RESTART_LOG_EVERY,
                       t_max=200.0)
    tr = res.trajectory
    return {"charge": args["charge"], "mv": args["mv"], "qe_mils": args["qe_mils"],
            "dt": SWEEP_DT, "range": res.range_m, "drift": res.drift_m,
            "tof": res.impact_time, "apogee_time": _apogee_time(res),
            "c_ypa_scale": args.get("c_ypa_scale", 1.0),
            "c_mpa_scale": args.get("c_mpa_scale", 1.0),
            "log_t": tr.t.tolist(), "log_y": _states_from(tr).tolist(),
            "log_mach": tr.mach.tolist()}


def run_case(args) -> dict:
    base = args["baseline"]
    geom = args["geometry"]
    phi_deg = args["phi_deg"]
    i, t_dep = _deploy_index(base, args.get("deploy_fraction", 0.0),
                             args.get("deploy_time"))
    y_dep = np.asarray(base["log_y"][i])
    mach_dep = float(base["log_mach"][i])
    nose = cn.NoseAssembly(deploy_time=t_dep, hold_angle=math.radians(phi_deg))
    model = _base_model(base["c_ypa_scale"], base["c_mpa_scale"])
    guided = cn.guided_model(model, geom, nose)
    cor = ig.integrate(y_dep, guided, dt=SWEEP_DT, log_every=LOG_EVERY,
                       t_max=200.0, t_start=t_dep)
    return {"charge": base["charge"], "qe_mils": base["qe_mils"],
            "phi_deg": float(phi_deg), "deploy_time": t_dep,
            "deploy_fraction": float(args.get("deploy_fraction", 0.0)),
            "apogee_time": base["apogee_time"],
            "mach_at_deploy": mach_dep,
            "time_remaining_s": base["tof"] - t_dep,
            "uncorrected_range": base["range"],
            "uncorrected_drift": base["drift"],
            "uncorrected_tof": base["tof"],
            "d_range": cor.range_m - base["range"],
            "d_deflection": cor.drift_m - base["drift"],
            "max_total_aoa_deg": math.degrees(cor.max_total_aoa),
            "station_from_nose_m": geom.station_from_nose,
            "panel_area_m2": geom.panel_area,
            "steering_deflection_deg": math.degrees(geom.steering_deflection),
            "span_exposed_m": geom.span_exposed,
            "body_radius_local_m": geom.body_radius_local,
            "carryover": geom.carryover,
            "fits_envelope": bool(geom.fits_within_envelope(pr.M107)),
            "c_ypa_scale": base["c_ypa_scale"],
            "c_mpa_scale": base["c_mpa_scale"],
            "label": args["label"]}


# ===========================================================================
# Driver
# ===========================================================================
def _pool(n=None):
    return Pool(processes=n or max(1, (os.cpu_count() or 2) - 1))


def _points(quick: bool) -> list:
    """(label, kwargs) for every configuration point in the sweep."""
    pts = []
    for s in STATION_SWEEP_M:
        pts.append((f"station_{s:.3f}", dict(station=s)))
    for s in STATION_SWEEP_M:
        pts.append((f"stationtaper_{s:.3f}", dict(station=s, taper=True)))
    if quick:
        return pts
    for a in AREA_SCALE_SWEEP:
        pts.append((f"area_x{a:g}", dict(area_scale=a)))
    for dd in DEFLECTION_SWEEP_DEG:
        pts.append((f"deflection_{dd:g}", dict(deflection_deg=dd)))
    for f in DEPLOY_FRACTION_SWEEP:
        pts.append((f"deploy_f{f:+.2f}", dict(deploy_fraction=f)))
    for c in CYPA_SCALE_SWEEP:
        pts.append((f"cypa_x{c:g}", dict(c_ypa_scale=c)))
    for c in CYPA_CMPA_SCALE_SWEEP:
        pts.append((f"cypa_and_cmpa_x{c:g}", dict(c_ypa_scale=c, c_mpa_scale=c)))
    # the two combinations worth having: best station with early deployment,
    # and best station with early deployment and a larger panel
    pts.append(("combo_fwd_early", dict(station=STATION_MIN_M,
                                        deploy_fraction=-0.5)))
    pts.append(("combo_fwd_early_area2", dict(station=STATION_MIN_M,
                                              deploy_fraction=-0.5, area_scale=2.0)))
    pts.append(("combo_fwd_early_area4", dict(station=STATION_MIN_M,
                                              deploy_fraction=-0.5, area_scale=4.0)))
    pts.append(("combo_fwd_area4", dict(station=STATION_MIN_M, area_scale=4.0)))
    return pts


def _combination_points() -> list:
    """
    The candidate best configurations, and the authority-versus-trim-angle
    trade at each. Run with --combos.

    The first sweep showed the two levers that actually move the answer are
    the canard STATION (forward) and the DEPLOYMENT POINT (early). This
    combines them, and then walks the steering deflection down from 5 deg to
    find where the induced trim angle of attack re-enters the range in which
    the measured aerodynamic deck was validated.
    """
    pts = []
    for f in (-0.75, -0.5):
        for dd in (1.0, 2.0, 3.0, 4.0, 5.0):
            pts.append((f"fwd{f:+.2f}_defl{dd:g}",
                        dict(station=STATION_MIN_M, deploy_fraction=f,
                             deflection_deg=dd)))
        pts.append((f"fwdtaper{f:+.2f}_defl5",
                    dict(station=STATION_MIN_M, deploy_fraction=f, taper=True)))
        pts.append((f"fwdtaper{f:+.2f}_defl3",
                    dict(station=STATION_MIN_M, deploy_fraction=f, taper=True,
                         deflection_deg=3.0)))
        pts.append((f"nom{f:+.2f}_defl3",
                    dict(deploy_fraction=f, deflection_deg=3.0)))
    # deployment even earlier than a quarter of the apogee time, at the
    # forward station: how far does this go before the trim angle or the
    # envelope stops it?
    for f in (-0.90, -0.85):
        pts.append((f"fwd{f:+.2f}_defl5",
                    dict(station=STATION_MIN_M, deploy_fraction=f)))
        pts.append((f"fwd{f:+.2f}_defl3",
                    dict(station=STATION_MIN_M, deploy_fraction=f,
                         deflection_deg=3.0)))
    return pts


def _best_config_points() -> list:
    """
    The C_Ypalpha sensitivity re-measured at the BEST configuration rather
    than at the step-2.5 nominal. It matters because the Magnus term dominates
    the residue only subsonically: the best configuration deploys at M 1.5,
    where the in-line term is far larger, so the sensitivity should be
    smaller. Run with --best.
    """
    pts = []
    for c in (0.0, 0.35, 0.5, 1.0, 1.28, 2.0):
        pts.append((f"best_cypa_x{c:g}",
                    dict(station=STATION_MIN_M, deploy_fraction=-0.75,
                         deflection_deg=3.0, c_ypa_scale=c)))
    for c in (0.35, 1.28):
        pts.append((f"apogee_cypa_x{c:g}", dict(c_ypa_scale=c)))
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--engagements", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="docs/design_sweep.json")
    ap.add_argument("--best", action="store_true",
                    help="C_Ypalpha sensitivity at the best configuration")
    ap.add_argument("--combos", action="store_true",
                    help="run the combination points instead of the main sweep")
    args = ap.parse_args(argv)
    t0 = time.time()

    idx = args.engagements if args.engagements else (
        [14] if args.quick else list(REPRESENTATIVE))
    if args.best:
        points = _best_config_points()
    elif args.combos:
        points = _combination_points()
    else:
        points = _points(args.quick)

    # baselines: one per (engagement, aero variant)
    variants = sorted({(p[1].get("c_ypa_scale", 1.0), p[1].get("c_mpa_scale", 1.0))
                       for p in points})
    bjobs = [{"charge": FIRING_TABLE[k][0], "mv": FIRING_TABLE[k][1],
              "qe_mils": FIRING_TABLE[k][2],
              "c_ypa_scale": cy, "c_mpa_scale": cm}
             for k in idx for (cy, cm) in variants]
    print(f"{len(bjobs)} baselines ({len(idx)} engagements x {len(variants)} "
          f"aero variants), {len(points)} configuration points")
    with _pool() as p:
        blist = p.map(baseline, bjobs)
    bases = {(j["charge"], j["qe_mils"], j["c_ypa_scale"], j["c_mpa_scale"]): b
             for j, b in zip(bjobs, blist)}
    print(f"  baselines done at {time.time() - t0:.0f} s")

    jobs = []
    for k in idx:
        chg, mv, qe, _ = FIRING_TABLE[k]
        for label, kw in points:
            cy = kw.get("c_ypa_scale", 1.0)
            cm = kw.get("c_mpa_scale", 1.0)
            base = bases[(chg, qe, cy, cm)]
            geom = _geometry(kw.get("station"), kw.get("area_scale", 1.0),
                             kw.get("deflection_deg"), kw.get("taper", False))
            for phi in PHI_CARDINAL:
                jobs.append({"baseline": base, "geometry": geom,
                             "phi_deg": phi,
                             "deploy_fraction": kw.get("deploy_fraction", 0.0),
                             "label": f"{k}|{label}"})
    print(f"  {len(jobs)} trajectories")
    with _pool() as p:
        rows = p.map(run_case, jobs, chunksize=1)
    print(f"  trajectories done at {time.time() - t0:.0f} s")

    grouped = {}
    for r in rows:
        grouped.setdefault(r["label"], []).append(r)

    out = {"kit_visible_length_m": KIT_VISIBLE_LENGTH_M,
           "station_min_m": STATION_MIN_M, "engagements": idx,
           "phi_deg": list(PHI_CARDINAL), "dt": SWEEP_DT, "points": []}
    for label, rs in sorted(grouped.items()):
        st = envelope_stats(rs)
        st["label"] = label
        st["engagement_index"] = int(label.split("|")[0])
        st["variant"] = label.split("|")[1]
        st["station_from_nose_m"] = rs[0]["station_from_nose_m"]
        st["panel_area_m2"] = rs[0]["panel_area_m2"]
        st["steering_deflection_deg"] = rs[0]["steering_deflection_deg"]
        st["c_ypa_scale"] = rs[0]["c_ypa_scale"]
        st["c_mpa_scale"] = rs[0]["c_mpa_scale"]
        st["span_exposed_m"] = rs[0]["span_exposed_m"]
        st["body_radius_local_m"] = rs[0]["body_radius_local_m"]
        st["carryover"] = rs[0]["carryover"]
        st["fits_envelope"] = rs[0]["fits_envelope"]
        st["mach_at_deploy"] = rs[0]["mach_at_deploy"]
        st["time_remaining_s"] = rs[0]["time_remaining_s"]
        st["deploy_fraction"] = rs[0]["deploy_fraction"]
        out["points"].append(st)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n{'variant':<24} {'eng':>4} {'M dep':>6} {'T rem':>6} "
          f"{'a m':>9} {'b m':>8} {'% rng':>7} {'bias m':>8} {'AoA deg':>8}")
    for st in out["points"]:
        print(f"{st['variant']:<24} {st['engagement_index']:>4} "
              f"{st['mach_at_deploy']:6.3f} {st['time_remaining_s']:6.1f} "
              f"{st['semi_axis_major_m']:9.2f} {st['semi_axis_minor_m']:8.2f} "
              f"{st['steerable_pct_of_range']:7.3f} "
              f"{st['bias_range_m']:8.1f} {st['max_total_aoa_deg']:8.2f}")
    print(f"\nwrote {args.out} in {time.time() - t0:.0f} s")
    return out


if __name__ == "__main__":
    main()
