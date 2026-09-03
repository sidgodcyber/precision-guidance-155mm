"""
Driver for the 6-DOF ballistic simulator.

Runs one uncorrected ballistic trajectory for a 155 mm M107 shell, prints a
summary against firing-table data where a matching point exists, and writes
plots: trajectory profile, ground track, angle-of-attack history, and the
stability/spin diagnostics.

    python run_ballistic.py                       # charge 8, QE 525.3 mils
    python run_ballistic.py --qe-mils 248.4
    python run_ballistic.py --charge 7 --qe-deg 30 --dt 1e-4
    python run_ballistic.py --wind-north -10      # 10 m/s head wind
    python run_ballistic.py --no-plots
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from sim import aerodata, atmosphere as atm, diagnostics as dg
from sim import dynamics as dyn, integrate as ig, projectile as pr

# Firing-table reference points, FT 155-AM-2 for the M107, as tabulated in
# Lim (NPS 2016, DTIC AD1029824) Tables 15-19. Keyed by (charge, QE in mils).
# Values: (range m, TOF s, drift m, impact velocity m/s, max ordinate m).
FT_POINTS = {
    (4, 97.2): (2000.0, 6.4, 3.2, 300.0, 50.0),
    (4, 152.0): (3000.0, 9.8, 7.8, 290.0, 119.0),
    (4, 211.6): (4000.0, 13.5, 15.2, 281.0, 224.0),
    (5, 118.1): (3000.0, 8.8, 7.5, 310.0, 96.0),
    (5, 280.4): (6000.0, 19.6, 36.6, 287.0, 477.0),
    (5, 420.6): (8000.0, 28.2, 79.2, 280.0, 988.0),
    (6, 258.4): (7000.0, 20.8, 45.5, 298.0, 548.0),
    (6, 378.6): (9000.0, 28.9, 88.2, 293.0, 1059.0),
    (6, 539.9): (11000.0, 39.1, 169.4, 294.0, 1924.0),
    (7, 177.6): (7000.0, 17.6, 37.1, 313.0, 383.0),
    (7, 319.8): (10000.0, 28.6, 94.0, 302.0, 1055.0),
    (7, 520.7): (13000.0, 42.6, 211.9, 307.0, 2335.0),
    (8, 141.6): (8000.0, 17.0, 37.6, 338.0, 352.0),
    (8, 248.4): (11000.0, 27.2, 92.4, 309.0, 930.0),
    (8, 525.3): (16000.0, 48.9, 292.8, 318.0, 3116.0),
}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--charge", type=int, default=8, choices=sorted(pr.CHARGE_TABLE))
    g = p.add_mutually_exclusive_group()
    g.add_argument("--qe-mils", type=float, default=None, help="quadrant elevation, NATO mils")
    g.add_argument("--qe-deg", type=float, default=None, help="quadrant elevation, degrees")
    p.add_argument("--dt", type=float, default=2e-4, help="fixed RK4 step, s")
    p.add_argument("--latitude", type=float, default=45.0, help="firing site latitude, deg")
    p.add_argument("--azimuth", type=float, default=0.0, help="azimuth of fire, deg from north")
    p.add_argument("--site-altitude", type=float, default=0.0, help="muzzle height above MSL, m")
    p.add_argument("--wind-north", type=float, default=0.0, help="air velocity north component, m/s")
    p.add_argument("--wind-east", type=float, default=0.0, help="air velocity east component, m/s")
    p.add_argument("--no-coriolis", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--outdir", default="docs/figures")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # The coefficient-confidence banner prints on EVERY run, by design.
    aerodata.warn_unvalidated(stream=sys.stdout)

    mv = pr.CHARGE_TABLE[args.charge]
    if args.qe_deg is not None:
        qe_mils = args.qe_deg * 6400.0 / 360.0
    elif args.qe_mils is not None:
        qe_mils = args.qe_mils
    else:
        qe_mils = 525.3

    shell = pr.M107
    table = aerodata.make_m107_table()
    env = pr.Environment.from_degrees(
        args.latitude,
        include_coriolis=not args.no_coriolis,
        site_altitude=args.site_altitude,
    )
    wind = (
        atm.no_wind
        if (args.wind_north == 0.0 and args.wind_east == 0.0)
        else atm.constant_wind(args.wind_north, args.wind_east)
    )
    model = dyn.FlightModel(
        projectile=shell, aero=table, environment=env, wind=wind, control=None
    )
    launch = pr.LaunchConditions.from_mils(
        mv, qe_mils, azimuth=math.radians(args.azimuth)
    )
    y0 = dyn.initial_state(shell, launch)

    print("\n" + "=" * 78)
    print("155 mm M107 -- uncorrected ballistic flight, full 6-DOF")
    print("=" * 78)
    print(f"  projectile      {shell.name}")
    print(f"  mass            {shell.mass:.3f} kg")
    print(f"  diameter        {shell.diameter:.3f} m")
    print(f"  Ix / It         {shell.I_axial:.5f} / {shell.I_transverse:.5f} kg m^2")
    print(f"  CG from nose    {shell.x_cg:.4f} m ({shell.x_cg_calibers:.2f} cal)")
    print(f"  rifling twist   1 turn in {shell.twist_calibers:g} calibres (right hand)")
    print(f"  charge          {args.charge}, muzzle velocity {mv:.1f} m/s")
    print(f"  quadrant elev   {qe_mils:.1f} mils = {launch.qe_degrees:.3f} deg")
    print(f"  azimuth         {args.azimuth:.1f} deg, latitude {args.latitude:.1f} deg")
    print(f"  Coriolis        {'off' if args.no_coriolis else 'on'}")
    print(f"  wind (air vel)  N {args.wind_north:+.1f}, E {args.wind_east:+.1f} m/s")
    print(f"  timestep        {args.dt:g} s")

    ms = dg.muzzle_stability(y0, model)
    spin_rev = ms.spin / (2.0 * math.pi)
    print(f"\n  muzzle spin     {ms.spin:.1f} rad/s ({spin_rev:.1f} rev/s)")
    print(f"  samples/rev     {1.0 / (spin_rev * args.dt):.1f} at the muzzle")
    print(f"  muzzle Mach     {ms.mach:.3f}")
    print(f"  muzzle Sg       {ms.Sg:.3f}   (needs > 1)")
    print(f"  muzzle Sd       {ms.Sd:.3f}   requires Sg > {ms.Sg_required:.3f}")

    print("\nintegrating ...", flush=True)
    res = ig.integrate(y0, model, dt=args.dt, log_every=max(1, int(0.02 / args.dt)), t_max=200.0)
    traj = res.trajectory
    stab = dg.stability_history(traj, model)
    sg = np.array([s.Sg for s in stab])
    sg_req = np.array([s.Sg_required for s in stab])

    print(f"  {res.steps} steps, terminated on {res.terminated}\n")
    print("-" * 78)
    print("IMPACT")
    print("-" * 78)
    print(f"  range              {res.range_m:12.2f} m")
    print(f"  drift (right +)    {res.drift_m:12.2f} m")
    print(f"  time of flight     {res.impact_time:12.3f} s")
    print(f"  impact velocity    {res.impact_velocity:12.2f} m/s")
    print(f"  impact angle       {math.degrees(res.impact_angle_rad):12.2f} deg below horizontal")
    print(f"  maximum ordinate   {res.max_ordinate:12.2f} m")

    key = (args.charge, round(qe_mils, 1))
    if key in FT_POINTS:
        ftR, ftT, ftD, ftV, ftO = FT_POINTS[key]
        print("\n  versus FT 155-AM-2 (firing table):")
        print(f"    range            {ftR:10.1f} m   error {100*(res.range_m-ftR)/ftR:+7.2f} %")
        print(f"    time of flight   {ftT:10.1f} s   error {100*(res.impact_time-ftT)/ftT:+7.2f} %")
        print(f"    drift            {ftD:10.1f} m   error {100*(res.drift_m-ftD)/ftD:+7.2f} %"
              + ("   (simulated drift includes Coriolis; the table does not)"
                 if not args.no_coriolis else ""))
        print(f"    impact velocity  {ftV:10.1f} m/s error {100*(res.impact_velocity-ftV)/ftV:+7.2f} %")
        print(f"    max ordinate     {ftO:10.1f} m   error {100*(res.max_ordinate-ftO)/ftO:+7.2f} %")

    print("\n" + "-" * 78)
    print("STABILITY AND NUMERICAL HEALTH")
    print("-" * 78)
    print(f"  Sg               {sg.min():.3f} to {sg.max():.3f} (final {sg[-1]:.3f})")
    print(f"  Sg > 1 always    {bool(np.all(sg > 1.0))}")
    print(f"  Sg > 1/(Sd(2-Sd)) always  {bool(np.all(sg > sg_req))}")
    print(f"  total AoA        max {math.degrees(res.max_total_aoa):.4f} deg, "
          f"mean {math.degrees(float(np.mean(traj.total_aoa))):.4f} deg")
    norms = np.linalg.norm(traj.quaternion, axis=1)
    print(f"  |q| - 1          max {np.max(np.abs(norms - 1.0)):.3e} over logged samples")
    print(f"  per-step norm drift removed by renormalisation: "
          f"{res.max_quat_norm_error:.3e}")
    print(f"  NaNs             {not np.all(np.isfinite(traj.position))}")
    print(f"  spin             {traj.omega[0,0]:.1f} -> {traj.omega[-1,0]:.1f} rad/s "
          f"({100*traj.omega[-1,0]/traj.omega[0,0]:.1f} % retained)")
    if table.extrapolated_above or table.extrapolated_below:
        print(f"  NOTE: Mach left the tabulated range [{table.mach_min}, {table.mach_max}]; "
              f"end values held flat.")

    if not args.no_plots:
        make_plots(res, stab, args, qe_mils, mv)

    return res


def make_plots(res, stab, args, qe_mils, mv):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available; skipping plots.")
        return

    os.makedirs(args.outdir, exist_ok=True)
    traj = res.trajectory
    tag = f"c{args.charge}_qe{qe_mils:.0f}"
    title = f"155 mm M107, charge {args.charge} ({mv:.0f} m/s), QE {qe_mils:.1f} mils"

    # --- 1. trajectory profile -------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(traj.downrange / 1000.0, traj.altitude, lw=1.6)
    ax.set_xlabel("downrange (km)")
    ax.set_ylabel("altitude above muzzle (m)")
    ax.set_title(f"Trajectory profile -- {title}")
    ax.grid(alpha=0.3)
    ax.annotate(
        f"impact {res.range_m:.0f} m\nTOF {res.impact_time:.1f} s\n"
        f"apogee {res.max_ordinate:.0f} m",
        xy=(res.range_m / 1000.0, 0),
        xytext=(0.62, 0.62), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/trajectory_{tag}.png", dpi=140)
    plt.close(fig)

    # --- 2. ground track --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(traj.downrange / 1000.0, traj.crossrange, lw=1.6)
    ax.axhline(0.0, color="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("downrange (km)")
    ax.set_ylabel("crossrange (m), positive to the RIGHT")
    ax.set_title(f"Ground track -- {title}")
    ax.grid(alpha=0.3)
    ax.annotate(
        f"drift {res.drift_m:+.1f} m to the right\n"
        f"(right-hand rifling must drift right)",
        xy=(0.05, 0.80), xycoords="axes fraction", fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/ground_track_{tag}.png", dpi=140)
    plt.close(fig)

    # --- 3. angle of attack ----------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(traj.t, np.degrees(traj.total_aoa), lw=0.8)
    axes[0].set_ylabel("total angle of attack (deg)")
    axes[0].set_title(f"Angle of attack -- {title}")
    axes[0].grid(alpha=0.3)
    axes[1].plot(traj.t, np.degrees(traj.alpha), lw=0.7, label="alpha (pitch)")
    axes[1].plot(traj.t, np.degrees(traj.beta), lw=0.7, label="beta (sideslip)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("angle (deg)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/angle_of_attack_{tag}.png", dpi=140)
    plt.close(fig)

    # --- 4. diagnostics ---------------------------------------------------
    sg = [s.Sg for s in stab]
    sg_req = [s.Sg_required for s in stab]
    ts = [s.t for s in stab]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(ts, sg, lw=1.3, label="Sg")
    axes[0, 0].plot(ts, sg_req, lw=1.0, ls="--", label="1/(Sd(2-Sd))")
    axes[0, 0].axhline(1.0, color="r", lw=0.8, ls=":", label="Sg = 1")
    axes[0, 0].set_ylabel("gyroscopic stability factor")
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(traj.t, traj.mach, lw=1.2)
    axes[0, 1].axhline(1.0, color="r", lw=0.8, ls=":")
    axes[0, 1].set_ylabel("Mach")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(traj.t, traj.omega[:, 0], lw=1.2)
    axes[1, 0].set_ylabel("axial spin p (rad/s)")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(traj.t, np.degrees([s.yaw_of_repose_rad for s in stab]), lw=1.2,
                    label="analytic yaw of repose")
    axes[1, 1].plot(traj.t, np.degrees(traj.total_aoa), lw=0.5, alpha=0.5,
                    label="6-DOF total AoA")
    axes[1, 1].set_ylabel("angle (deg)")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Diagnostics -- {title}")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/diagnostics_{tag}.png", dpi=140)
    plt.close(fig)

    print(f"\nplots written to {args.outdir}/  (trajectory, ground_track, "
          f"angle_of_attack, diagnostics -- suffix {tag})")


if __name__ == "__main__":
    main()
