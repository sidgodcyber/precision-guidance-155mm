"""
Markdown tables for docs/CONTROL-CHARACTERISATION.md and
docs/CONTROL-ROBUSTNESS.md, generated from docs/roll_servo.json and
docs/roll_robustness.json.

Nothing here computes anything: it formats. Every number it prints came out of
`analysis.roll_servo` or `analysis.roll_robustness`, so a table in the
documents and the JSON step 3 reads cannot disagree.

Run:  python -m analysis.roll_servo_report            -> docs/roll_servo_tables.md
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


def _f(x, spec=".2f", dash="--"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return dash
    return format(x, spec)


def _table(headers, rows) -> list:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    out.append("")
    return out


# ===========================================================================
# Task A
# ===========================================================================
def task_a_tables(d: dict) -> list:
    out = ["## Task A - the one-sided actuator", ""]
    ta = d["task_a"]
    t_dep = ta["deploy_time"]

    for key in sorted(ta["variants"], key=float):
        v = ta["variants"][key]
        out.append(f"### Brake capacity {v['brake_max_Nm']:.2f} N m")
        out.append("")
        out.append(
            f"Hold possible over **{100*v['hold_possible_fraction']:.1f} %** of "
            f"the guided phase; dead window after deployment "
            f"**{_f(v['dead_window_s'], '.2f')} s**; worst saturation margin "
            f"**{v['min_saturation_margin']:.2f}**.")
        out.append("")
        rows = []
        for r in v["rows"]:
            since = r["time_s"] - t_dep
            if not any(abs(since - x) < 0.03 for x in
                       (0, 0.5, 1, 2, 3, 5, 8, 12, 17, 22, 27, 32, 37, 41, 42.5)):
                continue
            rows.append([
                f"{since:.1f}", f"{r['qbar_kPa']:.1f}", f"{r['mach']:.3f}",
                f"{r['body_spin_rads']:.0f}", f"{r['cant_torque_Nm']:.3f}",
                f"{r['hold_command_Nm']:.3f}", f"{r['saturation_margin']:.2f}",
                f"{r['rate_full_degs']:.0f}", f"{r['rate_free_degs']:.0f}",
                f"{r['accel_forward_rads2']:.0f}",
                f"{r['accel_return_rads2']:.0f}",
                f"{r['time_constant_s']:.3f}",
                "yes" if r["can_hold"] else "**no**",
            ])
        out += _table(
            ["t - t_dep, s", "q&#772;, kPa", "M", "spin, rad/s", "T_cant, N m",
             "u_hold, N m", "margin", "forward, deg/s", "return, deg/s",
             "fwd accel, rad/s2", "ret accel, rad/s2", "tau, s", "can hold"],
            rows)

    if "task_a_verification" in d:
        out.append("### Closed form against the 6-DOF")
        out.append("")
        rows = []
        for v in d["task_a_verification"]:
            for r in v["rows"]:
                rows.append([f"{r['brake_Nm']:.2f}", f"{r['time_s']:.2f}",
                             f"{r['sixdof_nose_spin_rads']:.2f}",
                             f"{r['reduced_equilibrium_rads']:.2f}",
                             f"{100*r['relative_error']:.2f}"])
        out += _table(["brake, N m", "t, s", "6-DOF p_nose, rad/s",
                       "closed form, rad/s", "difference, %"], rows)
    return out


# ===========================================================================
# Task C
# ===========================================================================
def task_c_tables(d: dict) -> list:
    out = ["## Task C - slew, settling, tracking error, bandwidth", ""]
    for name in ("sized_brake", "nominal_brake"):
        tc = d["task_c"][name]
        out.append(f"### {name.replace('_', ' ')} "
                   f"({tc['brake_max_Nm']:.2f} N m)")
        out.append("")
        by_off = {}
        for s in tc["steps"]:
            by_off.setdefault(s["since_deploy_s"], {})[s["step_deg"]] = s
        rows = []
        for off in sorted(by_off):
            g = by_off[off]
            cells = [f"{off:.1f}", f"{list(g.values())[0]['qbar_kPa']:.1f}"]
            for step in (90.0, -90.0, 179.0, -179.0):
                s = g.get(step)
                cells.append(_f(s["rise_s"], ".3f") if s else "--")
            for step in (90.0, -90.0, 179.0, -179.0):
                s = g.get(step)
                cells.append(_f(s["settle_s"], ".3f") if s else "--")
            worst = max((s["overshoot_pct"] or 0.0) for s in g.values())
            cells.append(f"{worst:.2f}")
            rows.append(cells)
        out += _table(
            ["t - t_dep, s", "q&#772;, kPa",
             "rise +90", "rise -90", "rise +179", "rise -179",
             "settle +90", "settle -90", "settle +179", "settle -179",
             "worst overshoot, %"], rows)

        out.append("Steady-state hold, reduced model:")
        out.append("")
        rows = [[f"{h['since_deploy_s']:.1f}", f"{h['qbar_kPa']:.1f}",
                 f"{h['mean_abs_error_deg']:.4f}",
                 f"{h['max_abs_error_deg']:.4f}",
                 f"{h['mean_brake_Nm']:.3f}",
                 f"{100*h['brake_duty_of_capacity']:.1f}"]
                for h in tc["holds"]]
        out += _table(["t - t_dep, s", "q&#772;, kPa", "mean |e|, deg",
                       "max |e|, deg", "brake, N m", "% of capacity"], rows)

        bands = [b for b in tc["bandwidth"] if b["amplitude_deg"] == 10.0]
        if bands:
            out.append("Closed-loop bandwidth:")
            out.append("")
            rows = []
            for amp in (10.0, 45.0):
                for b in tc["bandwidth"]:
                    if b["amplitude_deg"] != amp:
                        continue
                    rows.append([f"{b['since_deploy_s']:.1f}",
                                 f"{b['amplitude_deg']:.0f}",
                                 f"{b['qbar_kPa']:.1f}",
                                 _f(b["bandwidth_hz"], ".2f"),
                                 _f(b["phase_90_hz"], ".2f")])
            out += _table(["t - t_dep, s", "amplitude, deg", "q&#772;, kPa",
                           "-3 dB, Hz", "-90 deg, Hz"], rows)

    if "acquisition" in d:
        out.append("### Acquisition from deployment")
        out.append("")
        rows = []
        for name, a in d["acquisition"].items():
            rows.append([name.replace("_", " "), f"{a['brake_max_Nm']:.2f}",
                         f"{a['acquired']}/{a['of']}",
                         _f(a["min_s"], ".2f"), _f(a["median_s"], ".2f"),
                         _f(a["max_s"], ".2f"), _f(a["median_1deg_s"], ".2f")])
        out += _table(["case", "brake, N m", "acquired", "min, s", "median, s",
                       "max, s", "median to 1 deg, s"], rows)

    if "actuator_lag" in d:
        al = d["actuator_lag"]
        out.append(f"### What a faster brake coil would buy "
                   f"(at q&#772; = {al['qbar_kPa']:.1f} kPa)")
        out.append("")
        rows = [[f"{1e3*r['actuator_tau_s']:.0f}",
                 f"{r['rate_bandwidth_rads']:.0f}",
                 f"{r['angle_bandwidth_rads']:.1f}",
                 _f(r["bandwidth_hz"], ".2f")] for r in al["rows"]]
        out += _table(["coil tau, ms", "rate loop, rad/s", "angle loop, rad/s",
                       "closed-loop -3 dB, Hz"], rows)

    if "fidelity" in d:
        out.append("### The reduced model against the 6-DOF")
        out.append("")
        rows = [[f["label"],
                 f"{f['rms_angle_difference_settled_deg']:.4f}",
                 f"{f['max_abs_angle_difference_deg']:.3f}",
                 f"{f['max_abs_rate_difference_rads']:.3f}",
                 _f(f["sixdof_rms_error_deg"], ".4f"),
                 _f(f["reduced_rms_error_deg"], ".4f")]
                for f in d["fidelity"]]
        out += _table(["case", "rms angle difference, deg",
                       "max angle difference, deg", "max rate difference, rad/s",
                       "6-DOF rms tracking error, deg",
                       "reduced rms tracking error, deg"], rows)

    if "sixdof_checkpoints" in d:
        out.append("### Tracking error measured in the 6-DOF")
        out.append("")
        rows = [[s["label"], f"{s['brake_max_Nm']:.2f}",
                 _f(s["acquisition_s"], ".2f"),
                 _f(s["mean_abs_error_deg_settled"], ".3f"),
                 _f(s["rms_error_deg_settled"], ".3f"),
                 _f(s["p95_abs_error_deg_settled"], ".3f"),
                 _f(s["correction_retained_settled"], ".6f"),
                 f"{s['bearing_slip_energy_kJ']:.2f}"]
                for s in d["sixdof_checkpoints"]]
        out += _table(["case", "brake, N m", "acquisition, s", "mean |e|, deg",
                       "rms e, deg", "p95 |e|, deg", "cos(e) retained",
                       "bearing energy, kJ"], rows)
    return out


# ===========================================================================
# Servo cost and Task D
# ===========================================================================
def servo_cost_tables(d: dict) -> list:
    out = ["## What the servo costs against an ideal hold", ""]
    rows = []
    for tag, label in (("servo_cost", "engagement gate on"),
                       ("servo_cost_ungated", "gate off")):
        if tag not in d:
            continue
        sc = d[tag]
        rows.append([label,
                     f"{sc['closed_loop']['semi_major_m']:.1f}",
                     f"{sc['closed_loop']['semi_minor_m']:.1f}",
                     f"{sc['closed_loop']['axis_ratio']:.2f}",
                     f"{sc['closed_loop']['rms_amplitude_m']:.1f}",
                     f"{100*sc['retained_semi_major']:.1f}",
                     f"{100*sc['retained_rms']:.1f}",
                     f"{sc['closed_max_aoa_deg']:.2f}"])
    sc = d["servo_cost"]
    rows.insert(0, ["ideal kinematic hold",
                    f"{sc['ideal']['semi_major_m']:.1f}",
                    f"{sc['ideal']['semi_minor_m']:.1f}",
                    f"{sc['ideal']['axis_ratio']:.2f}",
                    f"{sc['ideal']['rms_amplitude_m']:.1f}",
                    "100.0", "100.0", f"{sc['ideal_max_aoa_deg']:.2f}"])
    out += _table(["case", "semi-major, m", "semi-minor, m", "axis ratio",
                   "rms amplitude, m", "% semi-major", "% rms",
                   "peak AoA, deg"], rows)

    out.append("Per commanded angle, engagement gate on:")
    out.append("")
    rows = [[f"{p['angle_deg']:.0f}",
             f"{p['closed_range_m']:.1f}", f"{p['closed_deflection_m']:.1f}",
             f"{p['ideal_range_m']:.1f}", f"{p['ideal_deflection_m']:.1f}",
             _f(p["acquisition_s"], ".2f"), _f(p["tracking_rms_deg"], ".3f")]
            for p in sc["per_angle"]]
    out += _table(["phi, deg", "closed range, m", "closed deflection, m",
                   "ideal range, m", "ideal deflection, m", "acquisition, s",
                   "tracking rms, deg"], rows)
    return out


def task_d_tables(d: dict) -> list:
    td = d["task_d"]
    out = ["## Task D - the duty-cycle mode", "",
           f"Reference: a full hold at phi = {td['reference_angle_deg']:.0f} "
           f"degrees moves the impact point by "
           f"**{td['reference_magnitude_m']:.1f} m** relative to a free nose.",
           ""]

    out.append("### Duty fraction against delivered correction")
    out.append("")
    rows = []
    for r in sorted(td["duty_rows"], key=lambda x: (x["period_s"], x["duty"])):
        rows.append([f"{r['period_s']:.1f}", f"{r['duty']:.3f}",
                     f"{r['steerable_range_m']:.1f}",
                     f"{r['steerable_deflection_m']:.1f}",
                     f"{r['projected_m']:.1f}",
                     f"{r['fraction_of_full']:.3f}",
                     f"{100*(r['duty'] - r['fraction_of_full']):.1f}"])
    out += _table(["period, s", "duty", "range, m", "deflection, m",
                   "projected, m", "fraction of full", "loss vs ideal, pp"],
                  rows)

    out.append("### Linearity")
    out.append("")
    rows = [[f"{v['period_s']:.1f}", f"{v['slope']:.3f}", f"{v['intercept']:.3f}",
             f"{v['max_abs_residual']:.3f}", f"{v['rms_residual']:.4f}"]
            for v in td["linearity"].values()]
    out += _table(["period, s", "slope", "intercept", "max residual",
                   "rms residual"], rows)

    out.append("### The correction quantum")
    out.append("")
    rows = [[f"{q['start_s']:.1f}", f"{q['length_s']:.2f}",
             f"{q['steerable_range_m']:.2f}", f"{q['steerable_deflection_m']:.2f}",
             f"{q['magnitude_m']:.2f}", f"{q['per_second_m']:.2f}",
             _f(q["acquisition_s"], ".2f")]
            for q in sorted(td["quanta"], key=lambda x: (x["start_s"], x["length_s"]))]
    out += _table(["hold start, s after deploy", "hold length, s", "range, m",
                   "deflection, m", "magnitude, m", "per second, m",
                   "acquisition, s"], rows)

    if "switching" in td:
        out.append("### The cost of switching")
        out.append("")
        rows = [[f"{s['period_s']:.1f}", f"{s['duty']:.3f}", f"{s['n_arcs']}",
                 f"{s['n_acquired']}",
                 _f(s["median_reacquire_s"], ".3f"),
                 _f(s["max_reacquire_s"], ".3f"),
                 f"{100*s['mean_inside_fraction']:.1f}",
                 f"{s['mean_cos_error']:.3f}",
                 f"{s['switching_overhead_pct']:.1f}"]
                for s in sorted(td["switching"],
                                key=lambda x: (x["period_s"], x["duty"]))]
        out += _table(["period, s", "duty", "hold arcs", "arcs captured",
                       "median reacquire, s", "max reacquire, s",
                       "% of arc on target", "mean cos(e)",
                       "switching overhead, %"], rows)
    return out


# ===========================================================================
# Robustness
# ===========================================================================
def robustness_tables(r: dict) -> list:
    out = ["## Task E - robustness", "",
           "Pass line, taken from the nominal characterisation:", ""]
    out += _table(["criterion", "value"],
                  [[k.replace("_", " "), _f(v, ".3f")]
                   for k, v in r["targets"].items()])

    for name in ("uninformed", "informed"):
        out.append(f"### Controller {name}")
        out.append("")
        rows = []
        for c in r[name]:
            rows_ = c["probe"]["rows"]
            worst_rise_f = max((x["rise_forward_s"] or math.inf) for x in rows_)
            worst_rise_r = max((x["rise_return_s"] or math.inf) for x in rows_)
            worst_settle = max(
                (x["settle_forward_s"] or math.inf,
                 x["settle_return_s"] or math.inf) for x in rows_)
            rows.append([
                c["family"].replace("_", " "), f"{c['scale']:.2f}",
                "yes" if all(x["can_hold"] for x in rows_) else "**no**",
                _f(worst_rise_f, ".3f"), _f(worst_rise_r, ".3f"),
                _f(max(worst_settle), ".3f"),
                f"{max(x['hold_mean_error_deg'] for x in rows_):.4f}",
                _f(c["acquisition"]["median_s"], ".2f"),
                _f(c["bandwidth_hz"], ".2f"),
                "PASS" if c["verdict"]["passes"]
                else "FAIL: " + ", ".join(c["verdict"]["failed"]),
            ])
        out += _table(["family", "scale", "holds everywhere", "worst rise fwd, s",
                       "worst rise ret, s", "worst settle, s",
                       "worst hold error, deg", "acquisition, s",
                       "bandwidth, Hz", "verdict"], rows)

    out.append("### Stacked corners, controller uninformed")
    out.append("")
    rows = []
    for name, c in r["corners"].items():
        rows_ = c["probe"]["rows"]
        rows.append([name.replace("_", " "),
                     ", ".join(f"{k.replace('_scale','')} x{v}"
                               for k, v in c["perturbation"].items()),
                     "yes" if all(x["can_hold"] for x in rows_) else "**no**",
                     _f(c["acquisition"]["median_s"], ".2f"),
                     _f(c["bandwidth_hz"], ".2f"),
                     "PASS" if c["verdict"]["passes"]
                     else "FAIL: " + ", ".join(c["verdict"]["failed"])])
    out += _table(["corner", "perturbation", "holds everywhere",
                   "acquisition, s", "bandwidth, Hz", "verdict"], rows)

    out.append("### Where it breaks: the scale factors over which the angle "
               "can be held at all")
    out.append("")
    rows = []
    for k, b in r.get("holding_bands", {}).items():
        fam, brake = k.split("@")
        if not b["nominal_holds"]:
            rows.append([fam.replace("_", " "), brake, "**the nominal plant "
                         "cannot be held**", "--", "--"])
            continue
        rows.append([
            fam.replace("_", " "), brake,
            _f(b["lower"], ".3f", "none"), _f(b["upper"], ".3f", "none"),
            b["upper_reason"] if b["upper"] else b.get("lower_reason", ""),
        ])
    out += _table(["family", "brake, N m", "lower scale", "upper scale",
                   "what the upper bound is"], rows)

    si = r.get("sign_inversion")
    if si:
        out.append(
            f"The bearing drag beats the cant torque at "
            f"**x{si['bearing_scale_at_inversion']:.2f}**, at "
            f"t + {si['since_deploy_s']:.1f} s and "
            f"{si['at_qbar_kPa']:.1f} kPa. Above it the nose creeps forward "
            f"with the brake released and no brake capacity helps.")
        out.append("")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--servo", default=os.path.join("docs", "roll_servo.json"))
    ap.add_argument("--robustness",
                    default=os.path.join("docs", "roll_robustness.json"))
    ap.add_argument("--out", default=os.path.join("docs", "roll_servo_tables.md"))
    args = ap.parse_args(argv)

    with open(args.servo, encoding="utf-8") as f:
        d = json.load(f)

    lines = ["# Roll servo - generated tables", "",
             "Generated by `python -m analysis.roll_servo_report`. Do not edit.",
             ""]
    cfg = d["configuration"]
    lines += ["Configuration: canards at "
              f"{1e3*cfg['station_from_nose_m']:.0f} mm, "
              f"delta_s = {cfg['steering_deflection_deg']:.0f} deg, "
              f"cant {cfg['cant_angle_deg']:.1f} deg, deployment at "
              f"t/t_apogee = {cfg['deploy_fraction_of_apogee']:.2f}; "
              f"charge {cfg['charge']}, QE {cfg['qe_mils']} mils.", ""]
    lines += task_a_tables(d)
    lines += task_c_tables(d)
    lines += servo_cost_tables(d)
    if "task_d" in d:
        lines += task_d_tables(d)
    if os.path.exists(args.robustness):
        with open(args.robustness, encoding="utf-8") as f:
            lines += robustness_tables(json.load(f))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
