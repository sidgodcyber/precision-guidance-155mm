"""
Generated tables for step 5.

Reads the JSON the campaigns wrote and emits `docs/nav_tables.md`, the same
way `analysis.roll_servo_report` and `analysis.guidance_report` do for steps 4
and 3. Every table in the step-5 documents that is longer than a few rows is
generated here rather than typed, so it cannot drift from the data.

Run:  python -m analysis.nav_report
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

OUT = "docs/nav_tables.md"


def _load(p):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def _tbl(head, rows) -> list:
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    out.append("")
    return out


def sensors_tables(d) -> list:
    L = ["## Sensor exposure (Task A)", "",
         f"Engagement `{d['engagement']}`. Source: `docs/nav_sensors.json`.", ""]
    s = d["saturation"]
    L += ["### Gyro saturation", ""]
    L += _tbl(["full scale, deg/s", "body: % of flight pinned",
               "despun: clear after, s", "despun: % pinned after t_dep+2.2 s"],
              [[f"{float(k):.0f}",
                f"{100 * v['body']['fraction_saturated']:.0f}",
                f"{v['despun']['clear_after_s']:.2f}",
                f"{100 * v['despun']['fraction_saturated_after_despin']:.1f}"]
               for k, v in s["classes"].items()])
    L += ["### Lever-arm accelerations at the IMU station", "",
          f"Flight signal for comparison: "
          f"{d['lever']['flight_specific_force_mean']:.2f} m/s^2 mean. "
          f"Nose inertia {1000 * d['lever']['nose_inertia_kg_m2']:.1f} g m^2, "
          f"peak angular acceleration "
          f"{d['lever']['nose_angular_accel_rad_s2']['max']:.0f} rad/s^2.", ""]
    L += _tbl(["transverse offset, mm", "body centripetal max, g",
               "despun centripetal rms, m/s^2", "despun tangential rms, m/s^2",
               "despun tangential max, m/s^2"],
              [[f"{1000 * float(k):.0f}", f"{v['centripetal_body_max_g']:.0f}",
                f"{v['centripetal_despun_rms']:.3f}",
                f"{v['tangential_rms']:.2f}", f"{v['tangential_max']:.2f}"]
               for k, v in d["lever"]["offsets"].items()])
    L += [f"Offset at which a body-mounted accelerometer would stay inside "
          f"+-40 g: **{d['lever']['body_mounted_offset_for_40g_mm']:.3f} mm**.",
          ""]
    L += ["### Roll observability by firing azimuth", ""]
    L += _tbl(["azimuth, deg", "min field-to-axis, deg", "median, deg",
               "median sin", "% of flight below 15 deg", "roll 1 sigma, deg"],
              [[r["azimuth_deg"], f"{r['angle_deg']['min']:.1f}",
                f"{r['angle_deg']['median']:.1f}", f"{r['sin_median']:.3f}",
                f"{100 * r['fraction_below_15deg']:.1f}",
                f"{r['roll_sigma_median_deg']:.2f}"]
               for r in d["observability"]["rows"]])
    e = d["euler_identity"]
    L += [f"Euler identity: max roll error **{e['max_roll_error_rad']:.2e} rad**, "
          f"pitch {e['max_pitch_error_rad']:.1e}, yaw {e['max_yaw_error_rad']:.1e}, "
          f"over {e['samples']} samples.", ""]
    a = d["alignment"]
    L += [f"Velocity attitude reference: total angle of attack "
          f"{a['total_aoa_deg']['mean']:.3f} deg mean, "
          f"**{a['total_aoa_deg']['rms']:.3f} deg rms**, "
          f"{a['total_aoa_deg']['max']:.2f} worst; mean components "
          f"{a['yaw_component_deg']['mean']:+.4f} deg yaw, "
          f"{a['pitch_component_deg']['mean']:+.4f} deg pitch.", ""]
    return L


def consistency_tables(d) -> list:
    c = d.get("campaign")
    if not c:
        return []
    L = ["## Filter consistency (Task E)", "",
         f"{c['n_runs']} runs: {len(c['engagements'])} engagements x "
         f"{c['seeds']} sensor seeds. Source: `docs/nav_consistency.json`.", ""]
    L += _tbl(["test", "dof", "mean", "median", "p95", "95 % bounds", "verdict"],
              [[name, r["dof"], f"{r['mean']:.3f}", f"{r['median']:.3f}",
                f"{r['p95']:.2f}",
                f"[{r['bound_lo']:.2f}, {r['bound_hi']:.2f}]",
                "PASS" if r["consistent"] else "FAIL"]
               for name, r in (("NEES, position + velocity", c["nees_pv"]),
                               ("NEES, 9 navigation states", c["nees_nav"]),
                               ("NEES, all 15 states", c["nees"]))])
    L += ["### Per engagement", ""]
    L += _tbl(["engagement", "NEES 15", "NEES 9", "NEES 6"],
              [[k, f"{v['nees']['mean']:.2f}", f"{v['nees_nav']['mean']:.2f}",
                f"{v['nees_pv']['mean']:.2f}"]
               for k, v in c["per_engagement"].items()])
    L += ["### NIS, per measurement stream", ""]
    L += _tbl(["stream", "dof", "mean", "ratio to dof", "95 % bounds", "verdict"],
              [[k, v["dof"], f"{v['mean']:.3f}", f"{v['ratio_to_dof']:.2f}",
                f"[{v['bound_lo']:.2f}, {v['bound_hi']:.2f}]",
                "PASS" if v["consistent"] else "FAIL"]
               for k, v in c["nis"].items()])
    L += ["### Innovation whiteness", ""]
    L += _tbl(["stream", "lag-1 autocorrelation", "95 % band", "verdict"],
              [[k, f"{v['lag1']:+.4f}", f"{v['band']:.4f}",
                "WHITE" if v["white"] else "CORRELATED"]
               for k, v in c["whiteness"].items()])
    a = c["accuracy"]
    L += ["### Accuracy, pooled", ""]
    L += _tbl(["quantity", "X (range)", "Y (deflection)", "Z (vertical)"],
              [["position rms, m"] + [f"{v:.2f}" for v in a["position_rms_m"]],
               ["position bias, m"] + [f"{v:+.2f}" for v in a["position_bias_m"]],
               ["velocity rms, m/s"] + [f"{v:.3f}" for v in a["velocity_rms_ms"]],
               ["attitude rms, deg"] + [f"{v:.2f}" for v in a["attitude_rms_deg"]],
               ["attitude bias, deg"] + [f"{v:+.2f}" for v in a["attitude_bias_deg"]]])
    return L


def cep_tables(d) -> list:
    L = []
    if "task_d" in d:
        t = d["task_d"]
        L += ["## Warm start (Task D)", "",
              f"{t['n']} seeds per mode. Deployment at {t['deploy_time_s']:.2f} s; "
              f"the law may not command a direction before "
              f"**{t['deadline_s']:.2f} s**.", ""]
        L += _tbl(["mode", "median time to a usable solution, s",
                   "fraction usable by the deadline",
                   "position error at the deadline, median / p90 / max, m"],
                  [[name,
                    "0.00" if m["first_usable_s"]["median"] is None
                    else f"{m['first_usable_s']['median']:.2f}",
                    f"{100 * m['fraction_usable_by_deadline']:.0f} %",
                    " / ".join("-" if m["position_error_at_deadline_m"][k] is None
                               else f"{m['position_error_at_deadline_m'][k]:.1f}"
                               for k in ("median", "p90", "max"))]
                   for name, m in t["modes"].items()])
    if "task_f" in d:
        f = d["task_f"]["engagements"]
        L += ["## The navigation contribution to CEP (Task F)", "",
              f"{d['task_f']['n_draws']} paired draws x "
              f"{d['task_f']['n_seeds']} sensor seeds per engagement.", ""]
        L += _tbl(["engagement", "truth-fed CEP, m", "nav-fed CEP, m",
                   "range bias, m", "range 1 sigma, m",
                   "deflection bias, m", "deflection 1 sigma, m",
                   "radial median, m"],
                  [[k, f"{v['truth_cep']['cep_m']:.2f}",
                    f"{v['nav_cep']['cep_m']:.2f}",
                    f"{v['delta']['range']['bias_m']:+.2f}",
                    f"{v['delta']['range']['sigma_m']:.2f}",
                    f"{v['delta']['deflection']['bias_m']:+.2f}",
                    f"{v['delta']['deflection']['sigma_m']:.2f}",
                    f"{v['delta']['radial']['median_m']:.2f}"]
                   for k, v in f.items()])
    if "task_g" in d:
        g = d["task_g"]["cases"]
        L += ["## Degradation (Task G)", "",
              f"{d['task_g']['n_draws']} draws x {d['task_g']['seeds']} seeds, "
              f"engagement `{d['task_g']['engagement']}`, guided phase "
              f"{d['task_g']['guided_phase_s']:.1f} s.", ""]
        rows = []
        for k, v in g.items():
            pd = v.get("paired_vs_nominal") or {}
            rows.append([k, f"{v['cep']['cep_m']:.2f}",
                         f"{100 * v['cep']['within_30m']:.0f} %",
                         f"{v['cep']['p90_m']:.1f}",
                         ("-" if not pd else
                          f"{pd['median_delta_m']:+.2f} +- {pd['se_median_m']:.2f}")])
        L += _tbl(["case", "CEP, m", "P(<=30 m)", "p90, m",
                   "paired delta vs nominal, m"], rows)
    return L


def main(argv=None) -> int:
    parts = ["# Step 5 — generated tables", "",
             "Generated by `python -m analysis.nav_report`. Do not edit: every",
             "number here is read from the campaign JSON, so it cannot drift",
             "from the data that produced it.", ""]
    s = _load("docs/nav_sensors.json")
    if s:
        parts += sensors_tables(s)
    c = _load("docs/nav_consistency.json")
    if c:
        parts += consistency_tables(c)
    p = _load("docs/nav_cep.json")
    if p:
        parts += cep_tables(p)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts) + "\n")
    print(f"wrote {OUT} ({len(parts)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
