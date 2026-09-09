"""
Markdown tables for docs/STAGED-DEPLOYMENT.md, generated from
docs/staged_deployment.json.

Nothing here computes anything except the formatting of a distribution: every
number came out of `analysis.staged_deployment`, so a table in the document
and the JSON step 3 reads cannot disagree.

Run:  python -m analysis.staged_deployment_report
          -> docs/staged_deployment_tables.md
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
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    out.append("")
    return out


def _dist(s: dict, spec=".3f") -> str:
    """A distribution as mean +/- sd [min, max], n."""
    if not s.get("n"):
        return "--"
    return (f"{_f(s['mean'], spec)} ± {_f(s['sd'], spec)} "
            f"[{_f(s['min'], spec)}, {_f(s['max'], spec)}], n={s['n']}")


# ===========================================================================
# Task B
# ===========================================================================
def task_b_tables(d: dict) -> list:
    b = d["task_b"]
    out = ["## Task B - two panels against four", "",
           f"Brake {b['brake_max_Nm']:.2f} N m. `T_cant` sums over the cant "
           "pair and is unchanged; `c_aero` sums over the deployed panels and "
           "is halved.", ""]
    rows = []
    for a, c in zip(b["four_panel"], b["two_panel"]):
        rows.append([
            _f(a["t_after_deploy_s"], ".1f"), _f(a["qbar_kPa"], ".1f"),
            _f(a["T_cant_Nm"], ".4f"), _f(a["u_hold_Nm"], ".4f"),
            _f(a["saturation_margin"], ".2f"),
            f"{a['c_aero_Nms'] * 1e3:.3f}", f"{c['c_aero_Nms'] * 1e3:.3f}",
            _f(a["time_constant_s"], ".3f"), _f(c["time_constant_s"], ".3f"),
            _f(a["p_free_rads"], ".1f"), _f(c["p_free_rads"], ".1f"),
            _f(a["p_full_rads"], ".1f"), _f(c["p_full_rads"], ".1f"),
        ])
    out += _table(["t-t_dep, s", "q̄, kPa", "T_cant, N m", "u_hold, N m",
                   "margin", "c_aero×10³ (4)", "c_aero×10³ (2)",
                   "τ, s (4)", "τ, s (2)",
                   "p_free (4)", "p_free (2)", "p_full (4)", "p_full (2)"],
                  rows)

    s = b["despin_settling"]
    rows = [[k.replace("_", " "), _f(v["tau_s"], ".3f"),
             _f(v["p_inf_rads"], ".1f"), _f(v["t_to_within_5_rads"], ".2f")]
            for k, v in s.items()]
    out += ["Despin from the body rate, first order:", ""]
    out += _table(["case", "τ, s", "p_∞, rad/s", "t to within 5 rad/s"], rows)

    out += [
        f"- stage 1 still despins: **{b['stage1_still_despins']}**",
        f"- stage 1 still holdable at every probe: **{b['stage1_still_holdable']}**",
        f"- largest `u_hold` difference between two panels and four: "
        f"**{b['max_u_hold_difference_Nm']:.3e} N m**",
        f"- largest `T_cant` difference: **{b['max_T_cant_difference_Nm']:.3e} N m**",
        f"- damping ratio two-panel / four-panel: "
        f"**{min(b['damping_ratio_two_over_four']):.6f} - "
        f"{max(b['damping_ratio_two_over_four']):.6f}**",
        f"- brake sizing rule of CONTROL-ROBUSTNESS.md §4.3 unchanged: "
        f"**{b['sizing_rule_unchanged']}**",
        "",
    ]
    return out


# ===========================================================================
# Task D -- the ensembles
# ===========================================================================
def ensemble_tables(d: dict) -> list:
    out = ["## Task D - the phase ensembles", ""]
    cfg = d["configuration"]
    ens = d["ensemble"]
    out += [f"{ens['n']} deployment phases, {360.0 / ens['n']:.1f}° of body "
            f"roll apart, spanning one full roll period "
            f"({ens['roll_period_s'] * 1e3:.3f} ms) over which dynamic "
            f"pressure moves by {ens['qbar_spread_relative'] * 100:.3f} %. "
            f"{len(cfg['angles_deg'])} commanded angles each.", ""]

    ideal = d["ideal"]["pooled_ellipse"]
    out += [f"Denominator, the ideal kinematic hold pooled over "
            f"{len(d['ideal']['per_phase'])} phases: semi-major "
            f"**{ideal['semi_major_m']:.1f} m**, semi-minor "
            f"{ideal['semi_minor_m']:.1f} m, rms amplitude "
            f"**{ideal['rms_amplitude_m']:.1f} m**, axis ratio "
            f"{ideal['axis_ratio']:.2f}. Per-phase semi-major "
            f"{_dist(d['ideal']['semi_major_m'], '.1f')}.", ""]

    for tag, title in (("single", "Single stage"), ("staged", "Staged")):
        e = d.get(f"ensemble_{tag}")
        if not e:
            continue
        out += [f"### {title}", ""]
        rows = [[_f(p["body_roll_deg"], ".0f"), _f(p["semi_major_m"], ".1f"),
                 _f(p["rms_amplitude_m"], ".1f"),
                 _f(100 * p["retained_semi_major"], ".1f"),
                 _f(100 * p["retained_rms"], ".1f"),
                 _f(p["axis_ratio"], ".2f"), _f(p["max_aoa_deg"], ".2f"),
                 _f(p["max_acquisition_s"], ".2f"),
                 _f(p["stage2_delay_s"], ".2f"), p["stage2_reason"]]
                for p in e["per_phase"]]
        out += _table(["body roll, °", "semi-major, m", "rms, m",
                       "% ideal semi-major", "% ideal rms", "axis ratio",
                       "peak AoA, °", "acquisition, s", "stage 2, s", "reason"],
                      rows)
        out += _summary_block(e)

    if "comparison" in d:
        out += ["### Staged minus single stage", "",
                "Welch's t on the two independent samples. |t| below about 2 "
                "is not resolved by these samples.", ""]
        rows = []
        for k, v in d["comparison"].items():
            if not v.get("se_difference"):
                rows.append([k, "--", "--", "--", "--", "not resolved"])
                continue
            rows.append([k.replace("_", " "), _f(v["difference"], ".4f"),
                         _f(v["se_difference"], ".4f"), _f(v["t"], ".2f"),
                         f"[{_f(v['ci95_low'], '.4f')}, {_f(v['ci95_high'], '.4f')}]",
                         "**resolved**" if v["resolved"] else "not resolved"])
        out += _table(["quantity", "difference", "s.e.", "t", "95 % CI",
                       "verdict"], rows)
    return out


def _summary_block(e: dict) -> list:
    return [
        f"- retained rms amplitude: **{_dist(e['retained_rms'], '.4f')}**",
        f"- retained semi-major: **{_dist(e['retained_semi_major'], '.4f')}**",
        f"- axis ratio: {_dist(e['axis_ratio'], '.2f')}",
        f"- first-harmonic fit residual, as a fraction of the amplitude: "
        f"{_dist(e['harmonic_residual'], '.4f')}",
        f"- peak total angle of attack, per run: "
        f"**{_dist(e['aoa_per_run_deg'], '.2f')} °**",
        f"- acquisition, per run: {_dist(e['acquisition_per_run_s'], '.2f')} s",
        f"- steering available and pointed, per run: "
        f"**{_dist(e['steering_ready_per_run_s'], '.2f')} s**",
        f"- engagement, per run: {_dist(e['engage_per_run_s'], '.2f')} s",
        f"- stage-2 delay, per run: {_dist(e['stage2_delay_per_run_s'], '.2f')} s",
        f"- settled tracking rms: {_dist(e['tracking_rms_deg'], '.3f')} °",
        "",
    ]


# ===========================================================================
# Task C -- the delay sweep
# ===========================================================================
def delay_tables(d: dict) -> list:
    s = d.get("delay_sweep")
    if not s:
        return []
    out = ["## Task C - the stage-2 delay trade", "",
           f"Guided phase {s['guided_phase_s']:.1f} s. Four cardinal angles "
           f"at {s['by_delay'][0]['n_phases']} phases per delay; delay 0 is "
           "single stage to within one controller sample.", ""]
    rows = []
    for r in s["by_delay"]:
        rows.append([
            _f(r["delay_s"], ".2f"), _f(100 * r["fraction_of_guided_phase"], ".1f"),
            _f(100 * r["retained_rms"]["mean"], ".1f"),
            _f(100 * r["retained_rms"]["sd"], ".1f"),
            _f(100 * r["retained_rms"]["min"], ".1f"),
            _f(100 * r["retained_rms"]["max"], ".1f"),
            _f(100 * r["retained_semi_major"]["mean"], ".1f"),
            _f(r["aoa_per_run_deg"]["mean"], ".2f"),
            _f(r["aoa_per_run_deg"]["max"], ".2f"),
            _f(r["steering_ready_per_run_s"]["mean"], ".2f"),
            _f(r["axis_ratio"]["mean"], ".2f"),
        ])
    out += _table(["stage-2 delay, s", "% of guided phase",
                   "retained rms, % mean", "sd", "min", "max",
                   "retained semi-major, % mean",
                   "peak AoA, ° mean", "max", "steering ready, s", "axis ratio"],
                  rows)
    return out


# ===========================================================================
# The deadlock demonstration
# ===========================================================================
def deadlock_table(d: dict) -> list:
    rows = d.get("deadlock")
    if not rows:
        return []
    out = ["## The stage-2 backstop, in the 6-DOF", "",
           "Controller told the nominal canard aerodynamics, truth 15 % weaker, "
           "and the engagement gate's own model-free criterion disabled so that "
           "the gate locks out exactly as CONTROL-ROBUSTNESS.md §6 records. "
           "That round has no servo; the question is whether it ends up with a "
           "steering surface.", ""]
    body = []
    for r in sorted(rows, key=lambda r: r["label"]):
        body.append([r["label"], _f(r["stage2_delay_s"], ".2f"),
                     r["stage2_reason"] or "never released",
                     "yes" if r["engage_s"] is not None else "**no**",
                     _f(r["max_total_aoa_deg"], ".2f"),
                     _f(r["d_range_m"], ".1f"), _f(r["d_deflection_m"], ".1f")])
    out += _table(["case", "stage 2, s", "reason", "loop engaged",
                   "peak AoA, °", "Δrange, m", "Δdeflection, m"], body)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="docs/staged_deployment.json")
    ap.add_argument("--out", default="docs/staged_deployment_tables.md")
    args = ap.parse_args(argv)

    with open(args.results, encoding="utf-8") as f:
        d = json.load(f)

    cfg = d["configuration"]
    lines = ["# Staged deployment - generated tables", "",
             "Generated by `python -m analysis.staged_deployment_report` from "
             "`docs/staged_deployment.json`. Do not edit.", "",
             f"Brake {cfg['brake_max_Nm']:.2f} N m (step 4 measured at "
             f"{cfg['brake_max_step4_Nm']:.2f} N m); {cfg['n_phases']} "
             f"deployment phases; stage-2 backstop {cfg['max_delay_s']:.1f} s; "
             f"guided phase {cfg['guided_phase_s']:.1f} s.", ""]
    if "task_b" in d:
        lines += task_b_tables(d)
    lines += ensemble_tables(d)
    lines += delay_tables(d)
    lines += deadlock_table(d)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
