"""
Turn docs/authority_results.json into the tables in docs/AUTHORITY-ENVELOPE.md.

Reporting only: it computes nothing that analysis/authority.py did not already
measure, so a table in the document and the JSON behind it cannot drift apart.

Run:  python -m analysis.authority_report
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from sim import aerodata, canards as cn, projectile as pr

DEFAULT_JSON = os.path.join("docs", "authority_results.json")


def _load(path=DEFAULT_JSON):
    with open(path) as fh:
        return json.load(fh)


def _order_main(res):
    """Engagements in firing-table order."""
    from analysis.authority import FIRING_TABLE

    keys = []
    for charge, mv, qe, ft in FIRING_TABLE:
        k = "c%d_qe%g" % (charge, qe)
        if k in res["main"]:
            keys.append((k, charge, qe, ft))
    return keys


def _deploy_mach(path=os.path.join("docs", "deploy_mach.json")):
    """Mach at the deployment point, from analysis/_deploy_mach.py. Optional."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except OSError:
        return {"apogee": {}, "deploy": {}}


def table_main(res) -> str:
    dm = _deploy_mach()["apogee"]
    lines = [
        "| Chg | QE mils | Range m | Deploy s | **M at deploy** | Steerable a m | b m | a/b | Tilt deg | **Steerable %** | Bias m | Max shift m | Max % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, charge, qe, ft in _order_main(res):
        s = res["main"][k]["stats"]
        mach = dm.get(k, {}).get("mach")
        lines.append(
            "| %d | %g | %.0f | %.1f | %s | **%.1f** | %.1f | %.2f | %+.1f | **%.3f** | %+.1f | %.1f | %.3f |"
            % (charge, qe, s["uncorrected_range_m"], s["deploy_time"],
               "--" if mach is None else "%.3f" % mach,
               s["semi_axis_major_m"], s["semi_axis_minor_m"], s["axis_ratio"],
               s["major_axis_tilt_deg"], s["steerable_pct_of_range"],
               s["bias_range_m"], s["max_total_shift_m"], s["max_total_pct_of_range"])
        )
    return "\n".join(lines)


def table_authority_vs_mach(res) -> str:
    """The main table re-sorted by the variable that actually explains it."""
    dm = _deploy_mach()["apogee"]
    rows = []
    for k, charge, qe, ft in _order_main(res):
        s = res["main"][k]["stats"]
        m = dm.get(k, {}).get("mach")
        if m is not None:
            rows.append((m, k, s))
    rows.sort(reverse=True)
    lines = ["| M at deploy | Engagement | Range km | Steerable % of range | Steerable m |",
             "|---|---|---|---|---|"]
    for m, k, s in rows:
        lines.append("| %.3f | %s | %.1f | **%.3f** | %.1f |" % (
            m, k, 1e-3 * s["uncorrected_range_m"], s["steerable_pct_of_range"],
            s["semi_axis_major_m"]))
    return "\n".join(lines)


def summary_main(res) -> str:
    st = [res["main"][k]["stats"] for k, _, _, _ in _order_main(res)]
    pct = np.array([s["steerable_pct_of_range"] for s in st])
    ar = np.array([s["axis_ratio"] for s in st])
    resid = np.array([s["harmonic_residual_m"] for s in st])
    amp = np.array([s["rms_amplitude_m"] for s in st])
    bias = np.array([s["bias_range_m"] for s in st])
    aoa = np.array([s["max_total_aoa_deg"] for s in st])
    return "\n".join([
        "steerable authority, %% of range : min %.3f  median %.3f  max %.3f"
        % (pct.min(), float(np.median(pct)), pct.max()),
        "envelope axis ratio             : min %.2f  median %.2f  max %.2f"
        % (ar.min(), float(np.median(ar)), ar.max()),
        "first-harmonic residual / amp   : median %.4f  max %.4f"
        % (float(np.median(resid / amp)), float(np.max(resid / amp))),
        "range bias (drag), m            : min %+.1f  median %+.1f  max %+.1f"
        % (bias.min(), float(np.median(bias)), bias.max()),
        "peak total angle of attack, deg : min %.2f  median %.2f  max %.2f"
        % (aoa.min(), float(np.median(aoa)), aoa.max()),
    ])


def table_phi(res, key) -> str:
    rows = sorted(res["main"][key]["rows"], key=lambda r: r["phi_deg"])
    s = res["main"][key]["stats"]
    lines = ["| phi deg | dRange m | dDeflection m | about the bias: dR m | dD m |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %.0f | %+.2f | %+.2f | %+.2f | %+.2f |" % (
            r["phi_deg"], r["d_range"], r["d_deflection"],
            r["d_range"] - s["bias_range_m"], r["d_deflection"] - s["bias_deflection_m"]))
    return "\n".join(lines)


def table_deploy(res) -> str:
    dm = _deploy_mach()["deploy"]
    lines = ["| Chg | QE mils | Deploy s | t / t_apogee | M at deploy | Steerable a m | b m | **% of range** | Bias m | Peak AoA deg |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(res["deploy"], key=lambda k: (k.split("_f")[0], float(k.split("_f")[1]))):
        s = res["deploy"][k]["stats"]
        r0 = res["deploy"][k]["rows"][0]
        mach = dm.get(k, {}).get("mach")
        lines.append("| %d | %g | %.1f | %.2f | %s | **%.1f** | %.1f | **%.3f** | %+.1f | %.2f |" % (
            r0["charge"], r0["qe_mils"], s["deploy_time"],
            s["deploy_time"] / s["apogee_time"],
            "--" if mach is None else "%.3f" % mach,
            s["semi_axis_major_m"], s["semi_axis_minor_m"],
            s["steerable_pct_of_range"], s["bias_range_m"], s["max_total_aoa_deg"]))
    return "\n".join(lines)


def table_deflection(res) -> str:
    keys = sorted(res["deflection"], key=float)
    lines = ["| delta deg | Steerable a m | a per degree | linearity vs 1 deg | peak total AoA deg | Bias m |",
             "|---|---|---|---|---|---|"]
    ref = None
    for k in keys:
        s = res["deflection"][k]["stats"]
        d = float(k)
        per = s["semi_axis_major_m"] / d if d > 0 else float("nan")
        if d == 1.0:
            ref = per
        lin = per / ref if (ref and d > 0) else float("nan")
        lines.append("| %g | %.2f | %.3f | %s | %.2f | %+.1f |" % (
            d, s["semi_axis_major_m"], per,
            "--" if math.isnan(lin) else "%.3f" % lin,
            s["max_total_aoa_deg"], s["bias_range_m"]))
    return "\n".join(lines)


def table_station(res) -> str:
    tab = aerodata.make_m107_table()
    lines = ["| x_c m | x_c cal | arm to CG m | Steerable a m | % of range | predicted trim factor at M 0.85 |",
             "|---|---|---|---|---|---|"]
    c = tab.coefficients_at(0.85)
    for k in sorted(res["station"], key=float):
        s = res["station"][k]["stats"]
        x = float(k)
        from dataclasses import replace
        g = replace(cn.NOMINAL_GEOMETRY, station_from_nose=x)
        cla = cn.canard_lift_curve_slope(0.85, g.aspect_ratio_effective)
        pred = cn.trim_force_factor(g, pr.M107, c.C_Nalpha, c.C_Malpha, cla)
        lines.append("| %.3f | %.2f | %.3f | %.1f | %.3f | %+.3f |" % (
            x, x / pr.M107.diameter, pr.M107.x_cg - x,
            s["semi_axis_major_m"], s["steerable_pct_of_range"],
            pred["trim_force_factor"]))
    return "\n".join(lines)


def table_size(res) -> str:
    lines = ["| Variant | panel area mm2 | area vs nominal | Steerable a m | % of range | a per unit area |",
             "|---|---|---|---|---|---|"]
    nom = res["size"]["nominal"]["rows"][0]["panel_area_m2"]
    nom_a = res["size"]["nominal"]["stats"]["semi_axis_major_m"]
    for k in ("half_area", "nominal", "double_area", "quadruple_area", "no_carryover"):
        if k not in res["size"]:
            continue
        s = res["size"][k]["stats"]
        a = res["size"][k]["rows"][0]["panel_area_m2"]
        lines.append("| %s | %.0f | %.2f | %.1f | %.3f | %.2f |" % (
            k.replace("_", " "), 1e6 * a, a / nom, s["semi_axis_major_m"],
            s["steerable_pct_of_range"], (s["semi_axis_major_m"] / nom_a) / (a / nom)))
    return "\n".join(lines)


def table_induced(res) -> str:
    lines = ["| Chg | QE mils | t s | Mach | qbar kPa | direct N | induced N | measured ratio | closed-form ratio | trim AoA deg |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for blk in res["induced"]:
        rows = blk["rows"]
        if not rows:
            continue
        pick = [rows[0], rows[len(rows) // 2], rows[-1]]
        for r in pick:
            lines.append("| %d | %g | %.1f | %.3f | %.1f | %.2f | %.2f | **%.2f** | %.2f | %.2f |" % (
                blk["charge"], blk["qe_mils"], r["t"], r["mach"], 1e-3 * r["qbar"],
                r["direct_N"], r["induced_N"], r["ratio_induced_over_direct"],
                r["predicted_ratio"], r["trim_aoa_deg"]))
    return "\n".join(lines)


def summary_induced(res) -> str:
    out = []
    for blk in res["induced"]:
        rows = blk["rows"]
        if not rows:
            continue
        meas = np.array([r["ratio_induced_over_direct"] for r in rows])
        pred = np.array([r["predicted_ratio"] for r in rows])
        aoa = np.array([r["trim_aoa_deg"] for r in rows])
        out.append(
            "c%d QE %g : measured induced/direct %.2f-%.2f (median %.2f), "
            "closed form %.2f-%.2f, trim AoA %.2f-%.2f deg, "
            "median measured/predicted %.2f"
            % (blk["charge"], blk["qe_mils"], meas.min(), meas.max(),
               float(np.median(meas)), pred.min(), pred.max(),
               aoa.min(), aoa.max(), float(np.median(meas / pred)))
        )
    return "\n".join(out)


def table_free_nose(res) -> str:
    lines = ["| Chg | QE mils | Deploy s | Body spin at deploy rad/s | Nose inertial rate after settling rad/s | min | max | Despin time s | Bearing slip power W | dRange m |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["free_nose"]:
        lines.append("| %d | %g | %.1f | %.0f | **%+.1f** | %+.1f | %+.1f | %.2f | %.0f | %+.1f |" % (
            r["charge"], r["qe_mils"], r["deploy_time"], r["body_spin_at_deploy"],
            r["nose_spin_mean_after_settle"], r["nose_spin_min"], r["nose_spin_max"],
            r["despin_time_to_10pct_s"], r["bearing_slip_power_W"], r["d_range"]))
    return "\n".join(lines)


def table_convergence(res) -> str:
    lines = ["| phi deg | dRange at 5e-4 m | at 2e-4 m | change m | dDefl at 5e-4 m | at 2e-4 m | change m | absolute impact change m |",
             "|---|---|---|---|---|---|---|---|"]
    for r in res["convergence"]:
        lines.append("| %.0f | %+.2f | %+.2f | **%+.3f** | %+.2f | %+.2f | **%+.3f** | %+.2f |" % (
            r["phi_deg"], r["d_range_coarse"], r["d_range_fine"], r["d_range_change"],
            r["d_defl_coarse"], r["d_defl_fine"], r["d_defl_change"],
            r["abs_range_change"]))
    return "\n".join(lines)


def table_cp(res=None) -> str:
    """The mechanism table: where the CP is, and what it does to the factor."""
    tab = aerodata.make_m107_table()
    g = cn.NOMINAL_GEOMETRY
    lines = ["| Mach | C_Nalpha | C_Malpha | x_cp m | x_cp cal | x_c - x_cp mm | trim factor | induced/direct |",
             "|---|---|---|---|---|---|---|---|"]
    for m in (0.7, 0.8, 0.9, 1.0, 1.2, 1.6, 2.0):
        c = tab.coefficients_at(m)
        cla = cn.canard_lift_curve_slope(m, g.aspect_ratio_effective)
        d = cn.trim_force_factor(g, pr.M107, c.C_Nalpha, c.C_Malpha, cla)
        lines.append("| %.1f | %.3f | %.3f | %.4f | %.2f | %+.1f | %+.3f | %+.2f |" % (
            m, c.C_Nalpha, c.C_Malpha, d["x_cp_body_m"], d["x_cp_body_cal"],
            1e3 * (g.station_from_nose - d["x_cp_body_m"]),
            d["trim_force_factor"], d["induced_over_direct"]))
    return "\n".join(lines)


def figure(res, path=os.path.join("docs", "figures", "authority_envelope.png")):
    """
    The envelope in the range/deflection plane, for step 3 to look at.

    Left: four representative engagements' phi sweeps, plotted as reached
    impact points relative to the uncorrected one. The uncorrected point is
    the origin; the cross marks the roll-independent drag bias, which is the
    centre the reachable set is arranged about. That the set is NOT centred
    on the origin is the most important thing on the figure.

    Right: steerable semi-major axis against deployment time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    keys = [k for k, _, _, _ in _order_main(res)]
    show = [keys[i] for i in (0, 6, 12, 14) if i < len(keys)]
    colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(show)))
    for k, col in zip(show, colours):
        blk = res["main"][k]
        rows = sorted(blk["rows"], key=lambda r: r["phi_deg"])
        s = blk["stats"]
        dR = [r["d_range"] for r in rows] + [rows[0]["d_range"]]
        dD = [r["d_deflection"] for r in rows] + [rows[0]["d_deflection"]]
        lbl = "%s  (%.1f km)" % (k, 1e-3 * s["uncorrected_range_m"])
        ax.plot(dD, dR, "o-", color=col, ms=3.5, lw=1.3, label=lbl)
        ax.plot([s["bias_deflection_m"]], [s["bias_range_m"]], "x",
                color=col, ms=9, mew=2)
    ax.plot([0], [0], "k+", ms=13, mew=2)
    ax.annotate("uncorrected", (0, 0), textcoords="offset points",
                xytext=(6, 6), fontsize=8)
    ax.axhline(0, color="0.8", lw=0.7, zorder=0)
    ax.axvline(0, color="0.8", lw=0.7, zorder=0)
    ax.set_xlabel("deflection shift, m   (positive right)")
    ax.set_ylabel("range shift, m   (positive long)")
    ax.set_title("Reachable impact points, deployment at apogee\n"
                 "x marks the roll-independent drag bias", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)

    by_eng = {}
    for k, blk in res.get("deploy", {}).items():
        eng = k.rsplit("_f", 1)[0]
        s = blk["stats"]
        by_eng.setdefault(eng, []).append(
            (s["deploy_time"] / s["apogee_time"], s["steerable_pct_of_range"])
        )
    cols2 = plt.cm.viridis(np.linspace(0.05, 0.85, max(1, len(by_eng))))
    for (eng, pts), col in zip(sorted(by_eng.items()), cols2):
        pts.sort()
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                 color=col, ms=4, lw=1.4, label=eng)
    ax2.axvline(1.0, color="0.5", ls="--", lw=1.0)
    ax2.set_xlabel("deployment time / apogee time      (1.0 = apogee)")
    ax2.set_ylabel("steerable semi-major axis, % of range")
    ax2.set_title("Authority against deployment time", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    res = _load(args.json)

    blocks = [
        ("MECHANISM: centre of pressure and the trim-force factor", table_cp(res)),
        ("MAIN TABLE: 15 engagements, deployment at apogee", table_main(res)),
        ("MAIN SUMMARY", summary_main(res)),
        ("AUTHORITY SORTED BY MACH AT DEPLOYMENT", table_authority_vs_mach(res)),
        ("PHI SWEEP, maximum-range engagement", table_phi(res, "c8_qe525.3")),
        ("DEPLOYMENT TIME", table_deploy(res)),
        ("DEFLECTION LINEARITY", table_deflection(res)),
        ("CANARD STATION", table_station(res)),
        ("PANEL SIZE AND CARRYOVER", table_size(res)),
        ("INDUCED VS DIRECT", table_induced(res)),
        ("INDUCED VS DIRECT, SUMMARY", summary_induced(res)),
        ("FREE NOSE DESPIN", table_free_nose(res)),
        ("STEP-SIZE CONVERGENCE OF THE SHIFT", table_convergence(res)),
    ]
    for title, body in blocks:
        print("\n### " + title + "\n")
        print(body)
    print("\n### META\n")
    print(json.dumps(res["meta"], indent=1))
    if not args.no_figure:
        print("\nfigure: " + figure(res))


if __name__ == "__main__":
    main()
