"""
Tables for the step-3 documents, from docs/guidance_map.json and
docs/guidance_cep.json.

Writes docs/guidance_tables.md. Every table in docs/INVERSE-MAP.md,
docs/GUIDANCE-DESIGN.md, docs/AIM-OFF.md, docs/DEGRADATION-LADDER.md and
docs/CEP-CLOSED-LOOP.md is generated here rather than typed, so a re-run of
the campaigns updates the documents rather than silently contradicting them.

Run:  python -m analysis.guidance_report
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from gnc.inverse_map import (AuthorityMap, ellipse_of,
                             ellipse_contains_fraction, containment_margin)


def _f(x, n=1, dash="—"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return dash
    return f"{x:.{n}f}"


def _table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


# ===========================================================================
def _dead_time(nodes: list):
    """
    Time from deployment to a directed steering force, s.

    Read from the LONGEST hold, not from the first node that has a number.
    The early nodes release the brake before the servo has captured, so their
    stage-2 gate falls through to the model-free backstop at 4.0 s and their
    acquisition never happens: those are properties of a round that never
    steers, and quoting them as the dead time would report the backstop rather
    than the mechanism.
    """
    for n in reversed(nodes):
        acq, st2 = n.get("acquisition_s"), n.get("stage2_delay_s")
        vals = [x for x in (acq, st2) if x is not None]
        if vals:
            return max(vals)
    return None


def inverse_map_tables(m: dict, label: str = "long") -> list:
    e = m["engagements"][label]
    amap = AuthorityMap.from_dict(e["map"])
    L = [f"## Inverse map — {label} engagement", ""]

    L.append("### The table the flight computer carries")
    L.append("")
    rows = []
    for n in e["nodes"]:
        rows.append([_f(n["t_end_offset"], 2), _f(n["t_go"], 2),
                     _f(n["mach"], 3), _f(n["semi_major_m"], 1),
                     _f(n["semi_minor_m"], 1), _f(n["axis_ratio"], 2),
                     _f(n["major_axis_tilt_deg"], 1),
                     _f(n["centre_range_m"], 1),
                     _f(n["centre_deflection_m"], 1),
                     _f(n["residual"], 3)])
    L += _table(rows, ["hold to, s after dep", "t_go, s", "M", "semi-major, m",
                       "semi-minor, m", "ratio", "tilt, deg",
                       "centre ΔR, m", "centre ΔD, m", "residual"])
    L.append("")
    L.append(f"{len(e['nodes'])} nodes, six floats each: "
             f"**{amap.size_floats} floats = {amap.size_bytes(4)} bytes** at "
             f"single precision, {amap.size_bytes(8)} bytes at double.")
    L.append("")

    L.append("### Pointing: desired direction to commanded roll angle")
    L.append("")
    rows = [[_f(p["desired_deg"], 0), _f(p["phi_deg"], 1),
             _f(p["rotation_deg"], 1), _f(p["reach_m"], 1)]
            for p in e["pointing"]]
    L += _table(rows, ["desired direction, deg", "commanded φ, deg",
                       "rotation, deg", "reach, m"])
    L.append("")
    return L


def pointing_tables(m: dict) -> list:
    """Task A validation: fly the map's own answers, out of sample."""
    L = ["## Task A validation — residual pointing error", "",
         "Twelve desired directions 30° apart, inverted through the map and "
         "flown. The commanded angles they invert to are not the eight the map "
         "was fitted through, so this is an out-of-sample test of the fit.", ""]
    rows = []
    for lbl, e in m["engagements"].items():
        pv = e.get("pointing_validation")
        if not pv:
            continue
        for frac, s in pv["summary"].items():
            rows.append([lbl, frac, s["n"], _f(s["pointing_rms_deg"], 2),
                         _f(s["pointing_max_deg"], 2),
                         _f(s["pointing_mean_deg"], 2),
                         _f(100 * s["magnitude_rms_frac"], 1),
                         _f(100 * s["magnitude_max_frac"], 1),
                         _f(s["cross_rms_m"], 2), _f(s["cross_max_m"], 2)])
    L += _table(rows, ["engagement", "hold fraction", "n",
                       "pointing rms, °", "worst, °", "mean, °",
                       "magnitude rms, %", "worst, %",
                       "cross-track rms, m", "worst, m"])
    L.append("")
    lbl = "long" if "long" in m["engagements"] else list(m["engagements"])[0]
    pv = m["engagements"][lbl].get("pointing_validation")
    if pv:
        L.append(f"### Per direction, {lbl}, full hold")
        L.append("")
        rows = [[_f(r["desired_deg"], 0), _f(r["phi_deg"], 1),
                 _f(r["predicted_m"], 1), _f(r["delivered_m"], 1),
                 _f(r["along_m"], 1), _f(r["cross_m"], 2),
                 _f(r["pointing_error_deg"], 2),
                 _f(100 * r["magnitude_error_frac"], 1)]
                for r in pv["rows"] if abs(r["hold_fraction"] - 1.0) < 1e-9]
        L += _table(rows, ["desired, °", "commanded φ, °", "predicted, m",
                           "delivered, m", "along, m", "cross, m",
                           "pointing error, °", "magnitude error, %"])
        L.append("")
    return L


def prediction_tables(m: dict) -> list:
    L = ["## Impact-point prediction error", ""]
    for lbl, e in m["engagements"].items():
        rows = [[_f(r["t_end_offset"], 2), _f(r["t_go"], 2),
                 _f(r["bias_range_m"], 2), _f(r["sd_range_m"], 2),
                 _f(r["bias_deflection_m"], 2), _f(r["sd_deflection_m"], 2),
                 _f(r["rms_m"], 2), _f(r["max_m"], 2)]
                for r in e["prediction_error"]["rows"]]
        if not rows:
            continue
        L.append(f"### {lbl}")
        L.append("")
        L += _table(rows, ["s after dep", "t_go, s", "range bias, m",
                           "sd, m", "deflection bias, m", "sd, m",
                           "rms, m", "worst, m"])
        L.append("")
    return L


def envelope_tables(m: dict) -> list:
    L = ["## The reachable set across the firing table", ""]
    rows = []
    for lbl, e in m["engagements"].items():
        b = e["baseline"]
        f = e["full_hold_ellipse"]
        i = e["ideal"]["ellipse"]
        nodes = e["nodes"]
        dead = _dead_time(nodes)
        rows.append([lbl, b["charge"], _f(b["qe_mils"], 1),
                     _f(b["uncorrected_range"], 0), _f(b["uncorrected_tof"], 2),
                     _f(b["deploy_time"], 2), _f(b["guided_phase_s"], 2),
                     _f(b["mach_at_deploy"], 3),
                     _f(b["qbar_at_deploy"] / 1e3, 1),
                     _f(dead, 2),
                     _f(100 * dead / b["guided_phase_s"], 1) if dead else "—",
                     _f(f["semi_major_m"], 1), _f(f["semi_minor_m"], 1),
                     _f(f["axis_ratio"], 2), _f(i["semi_major_m"], 1),
                     _f(100 * e["retained_rms"], 1),
                     _f(nodes[-1]["residual"], 3)])
    L += _table(rows, ["engagement", "charge", "QE mils", "range m", "tof s",
                       "deploy s", "guided s", "M dep", "q̄ kPa",
                       "dead time s", "dead %", "semi-major m",
                       "semi-minor m", "ratio", "ideal semi-major m",
                       "retained %", "residual"])
    L.append("")
    return L


def _one_scheduler_table(t: dict, heading: str) -> list:
    L = [heading, "",
         f"Engagement `{t['engagement']}`, {t['n_draws']} draws, "
         f"σ_range {t['dispersion']['sigma_range']:.1f} m, "
         f"σ_deflection {t['dispersion']['sigma_deflection']:.1f} m "
         f"(uncorrected CEP {t['dispersion']['cep_m']:.0f} m, "
         f"{t['dispersion']['ratio']:.0f}:1 range-dominated). "
         f"Aim-off {t.get('aim_off_m', float('nan')):.1f} m. Under the best "
         f"law (`{t['best']}`) **{100 * t.get('never_released', float('nan')):.0f} % "
         f"of rounds never released** — the hold ran to impact. The "
         f"`held to impact` column gives the same count for every law.", ""]
    rows = []
    for name, r in t["schedulers"].items():
        cp = r["cep"]
        hold = [x.get("hold_length_s") for x in r["rows"]
                if x.get("hold_length_s")]
        never = sum(1 for x in r["rows"] if x.get("g_release_time") is None)
        faults = sum(1 for x in r["rows"]
                     if x.get("g_authority_fault_time") is not None)
        rows.append([name, _f(cp["cep_m"]), _f(cp.get("cep_se_m"), 1),
                     _f(cp["mean_miss_m"]), _f(cp["p90_m"]),
                     _f(100 * cp["within_30m"], 1),
                     _f(100 * cp["within_50m"], 1),
                     _f(cp["bias_range_m"]), _f(cp["bias_defl_m"]),
                     _f(float(np.median(hold)) if hold else None, 1),
                     f"{never}/{len(r['rows'])}", faults])
    L += _table(rows, ["law", "CEP m", "SE m", "mean m", "p90 m", "≤30 m %",
                       "≤50 m %", "range bias m", "defl bias m",
                       "median hold s", "held to impact", "monitor fired"])
    L.append("")
    L.append("Paired against the best law, on the same draws:")
    L.append("")
    rows = [[name, d["n"], _f(d["median_delta_m"], 2), _f(d["se_median_m"], 2),
             _f(d["mean_delta_m"], 2), _f(100 * d["fraction_worse"], 0)]
            for name, d in t.get("paired_vs_best", {}).items()]
    L += _table(rows, ["law", "n", f"median Δmiss vs {t['best']}, m",
                       "SE, m", "mean Δ, m", "worse on, %"])
    L.append("")
    return L


def scheduler_tables(c: dict) -> list:
    L = []
    if c.get("task_c"):
        L += _one_scheduler_table(
            c["task_c"],
            "## Task C — the schedulers, against a dispersion the kit cannot "
            "cover")
    if c.get("task_c2"):
        L += _one_scheduler_table(
            c["task_c2"],
            "## Task C.2 — the schedulers, against a dispersion the kit CAN "
            "cover")
    if c.get("task_r"):
        t = c["task_r"]
        L += ["## Task C.3 — the guidance rate", "",
              f"Engagement `{t['engagement']}`, law `{t['scheduler']}`, "
              f"{t['n_draws']} draws COMMON to every rate. The paired column "
              f"is the evidence; the absolute CEPs carry a standard error of "
              f"the same order as the differences between them.", ""]
        rows = []
        for k in sorted(t["rates"], key=float):
            v = t["rates"][k]
            cp = v["cep"]
            d = v.get("paired_vs_1hz", {})
            rows.append([k, cp["n"], _f(cp["cep_m"]), _f(cp.get("cep_se_m"), 1),
                         _f(100 * cp["within_30m"], 1),
                         _f(d.get("median_delta_m"), 2),
                         _f(d.get("se_median_m"), 2),
                         _f(100 * d["fraction_worse"], 0) if d else "—",
                         _f(v["predictor_calls"], 0),
                         _f(v["derivative_calls"], 0)])
        L += _table(rows, ["rate, Hz", "n", "CEP m", "SE m", "≤30 m %",
                           "paired Δ vs 1 Hz, m", "SE, m", "worse on, %",
                           "propagations", "derivative evaluations"])
        L.append("")
    return L


def _cep_from(rows: list) -> dict:
    """Local copy of `analysis.guidance_cep.cep_of`, to avoid importing the
    campaign module (and its multiprocessing pool) into the report."""
    v = np.array([r["miss_m"] for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "cep_m": float("nan"), "within_30m": float("nan")}
    rng = np.random.default_rng(20260905)
    idx = rng.integers(0, v.size, size=(2000, v.size))
    return {"n": int(v.size), "cep_m": float(np.median(v)),
            "cep_se_m": float(np.median(v[idx], axis=1).std(ddof=1)),
            "within_30m": float((v <= 30.0).mean()),
            "within_50m": float((v <= 50.0).mean())}


def aimoff_tables(c: dict) -> list:
    d = c.get("task_d")
    if not d:
        return []
    L = ["## Task D — the aim-off, with the loop closed", "",
         "The paired column is against the published 224 m on the same draws. "
         "It is the evidence: the CEP column's own standard error is 18–28 m, "
         "so the interior of the sweep is not resolved by the medians alone.",
         ""]
    ref = d["offsets"].get("224.0")
    rows = []
    for k in sorted(d["offsets"], key=float):
        cp = d["offsets"][k]["cep"]
        pd = (_paired(d["offsets"][k]["rows"], ref["rows"]) if ref else {})
        rows.append([_f(float(k), 0), _f(cp["cep_m"]), _f(cp.get("cep_se_m"), 1),
                     _f(pd.get("median_delta_m"), 2), _f(pd.get("se_median_m"), 2),
                     _f(100 * pd["fraction_worse"], 0) if pd else "—",
                     _f(100 * cp["within_30m"], 1),
                     _f(100 * cp["within_50m"], 1),
                     _f(cp["bias_range_m"]), _f(cp["bias_defl_m"])])
    L += _table(rows, ["aim-off m", "CEP m", "SE m", "paired Δ vs 224 m",
                       "SE m", "worse on %", "≤30 m %", "≤50 m %",
                       "range bias m", "defl bias m"])
    L.append("")
    if "two_d" in d:
        t = d["two_d"]
        pd = _paired(t["rows"], d["offsets"][str(t["aim_off_range_m"])]["rows"])
        L.append(f"Two-dimensional aim-off, {t['aim_off_range_m']:.0f} m long "
                 f"and {t['aim_off_deflection_m']:+.1f} m across: "
                 f"**CEP {t['cep']['cep_m']:.1f} ± {t['cep']['cep_se_m']:.1f} m**, "
                 f"{100 * t['cep']['within_30m']:.1f} % within 30 m, residual "
                 f"deflection bias {t['cep']['bias_defl_m']:+.1f} m against "
                 f"{t['measured_defl_bias_m']:+.1f} m before. Paired against "
                 f"the range-only offset: {pd['median_delta_m']:+.2f} ± "
                 f"{pd['se_median_m']:.2f} m, better on "
                 f"{100 * (1 - pd['fraction_worse']):.0f} % of rounds.")
        L.append("")
    return L


def _paired(rows_a: list, rows_b: list) -> dict:
    """Paired difference in residual miss on common draws. Positive: a worse."""
    by_b = {r["draw"]: r for r in rows_b}
    v = np.array([r["miss_m"] - by_b[r["draw"]]["miss_m"]
                  for r in rows_a if r["draw"] in by_b], dtype=float)
    if v.size == 0:
        return {}
    rng = np.random.default_rng(20260905)
    idx = rng.integers(0, v.size, size=(2000, v.size))
    return {"n": int(v.size), "median_delta_m": float(np.median(v)),
            "se_median_m": float(np.median(v[idx], axis=1).std(ddof=1)),
            "mean_delta_m": float(v.mean()),
            "fraction_worse": float((v > 0).mean())}


def ladder_tables(c: dict) -> list:
    e = c.get("task_e")
    if not e:
        return []
    L = ["## Task E — the degradation ladder", ""]
    rows = []
    for k, r in e["rungs"].items():
        cp = r["cep"]
        rows.append([k, cp["n"], _f(cp["cep_m"]), _f(cp["mean_miss_m"]),
                     _f(cp["p90_m"]), _f(cp["max_m"]),
                     _f(100 * cp["within_30m"], 1),
                     _f(100 * cp["within_100m"], 1),
                     _f(cp["bias_range_m"]), _f(cp["bias_defl_m"])])
    L += _table(rows, ["state", "n", "CEP m", "mean m", "p90 m", "worst m",
                       "≤30 m %", "≤100 m %", "range bias m", "defl bias m"])
    L.append("")
    s = e.get("stage2_detection", {})
    if s:
        L.append(f"Stage-2 failure detected in **{s['detected']}/{s['of']}** "
                 f"rounds, median {_f(s['median_detect_s_after_deploy'], 2)} s "
                 f"after deployment.")
        L.append("")
    return L


def envelope_cep_tables(c: dict, m: dict) -> list:
    f = c.get("task_f")
    if not f:
        return []
    L = ["## Task F — CEP across the firing table", ""]
    rows = []
    order = sorted(f["engagements"],
                   key=lambda l: f["engagements"][l]["baseline"]["uncorrected_range"])
    for lbl in order:
        r = f["engagements"][lbl]
        b = r["baseline"]
        dead = _dead_time(m["engagements"][lbl]["nodes"])
        rows.append([lbl, _f(b["uncorrected_range"], 0), _f(b["uncorrected_tof"], 1),
                     _f(b["guided_phase_s"], 1),
                     _f(100 * dead / b["guided_phase_s"], 1) if dead else "—",
                     _f(r["ellipse"]["semi_major_m"], 1),
                     _f(r["dispersion"]["cep_m"], 0),
                     _f(r["unguided_cep"]["cep_m"]),
                     _f(r["cep"]["cep_m"]),
                     _f(r["unguided_cep"]["cep_m"] / max(r["cep"]["cep_m"], 1e-9), 2),
                     _f(100 * r["cep"]["within_30m"], 1),
                     _f(100 * r["cep"]["within_50m"], 1)])
    L += _table(rows, ["engagement", "range m", "tof s", "guided s", "dead %",
                       "semi-major m", "assumed unguided CEP m",
                       "measured unguided CEP m", "guided CEP m",
                       "improvement ×", "≤30 m %", "≤50 m %"])
    L.append("")
    return L


def monitor_tables(c: dict) -> list:
    """Task E.2: the monitor's start time, sized and measured."""
    e2 = c.get("task_e2")
    if not e2:
        return []
    e = c.get("task_e", {})
    L = ["## Task E.2 — the authority monitor with its start time sized", "",
         f"Monitor options: {e2['monitor']}. Compared against the first "
         f"version, which opened its first window as soon as the law armed.",
         ""]
    rows = []
    for tag in ("full", "stage2_fail"):
        a = e.get("rungs", {}).get(tag)
        b = e2["rungs"][tag]
        rows.append([tag,
                     _f(a["cep"]["cep_m"]) if a else "—",
                     _f(b["cep"]["cep_m"]),
                     f"{a['authority_faults']}/{a['cep']['n']}" if a else "—",
                     f"{b['authority_faults']}/{b['cep']['n']}",
                     _f(b.get("median_detect_s_after_deploy"), 2)])
    L += _table(rows, ["rung", "CEP m, first version", "CEP m, start sized",
                       "monitor fired, first", "monitor fired, sized",
                       "median detection, s after deployment"])
    L.append("")
    return L


def envelope_cep2_tables(c: dict) -> list:
    """Task F.2: the firing table at the adopted configuration's own aim-off."""
    f2 = c.get("task_f2")
    f1 = c.get("task_f")
    if not f2 or not f1:
        return []
    L = ["## Task F.2 — the firing table at the staged aim-off", "",
         "Same draws as Task F. `paired Δ` is the staged aim-off against the "
         "published one, per round.", ""]
    order = sorted(f2["engagements"],
                   key=lambda l: f2["engagements"][l]["baseline"]["uncorrected_range"])
    rows = []
    for lbl in order:
        a, b = f1["engagements"][lbl], f2["engagements"][lbl]
        pd = _paired(b["rows"], a["rows"])
        rows.append([lbl,
                     _f(a["rows"][0]["aim_off"]), _f(b["rows"][0]["aim_off"]),
                     _f(a["cep"]["cep_m"]), _f(b["cep"]["cep_m"]),
                     _f(pd.get("median_delta_m"), 2), _f(pd.get("se_median_m"), 2),
                     _f(a["unguided_cep"]["cep_m"]), _f(b["unguided_cep"]["cep_m"]),
                     _f(a["cep"]["cep_m"] / a["unguided_cep"]["cep_m"], 2),
                     _f(b["cep"]["cep_m"] / b["unguided_cep"]["cep_m"], 2)])
    L += _table(rows, ["engagement", "aim-off published m", "aim-off staged m",
                       "guided CEP published m", "guided CEP staged m",
                       "paired Δ m", "SE m", "unguided published m",
                       "unguided staged m", "ratio published", "ratio staged"])
    L.append("")
    return L


def containment_tables(c: dict) -> list:
    g = c.get("task_g")
    if not g:
        return []
    L = ["## Task G — containment against the published scalar criterion", ""]
    rows = []
    for r in g["containment"]:
        rows.append([r["dispersion"], r["set"], _f(r["sigma_range"], 0),
                     _f(r["sigma_deflection"], 0), _f(r["semi_major_m"], 1),
                     _f(r["semi_minor_m"], 1), _f(r["axis_ratio"], 2),
                     _f(100 * r["p_contains"], 1),
                     _f(r["scale_for_50pc"], 2),
                     _f(r["scalar_required_a_m"], 1),
                     "yes" if r["scalar_passes"] else "**no**"])
    L += _table(rows, ["miss shape", "reachable set", "σ_R m", "σ_D m",
                       "a m", "b m", "ratio", "P(contains target) %",
                       "scale for 50 %", "scalar requirement m",
                       "scalar passes"])
    L.append("")
    if g.get("circular"):
        cc = g["circular"]["cep"]
        L.append(f"Flown against the CIRCULAR dispersion the published "
                 f"criterion assumed: **CEP {cc['cep_m']:.1f} m**, "
                 f"{100 * cc['within_30m']:.1f} % within 30 m.")
        L.append("")
    return L


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="docs/guidance_map.json")
    ap.add_argument("--cep", default="docs/guidance_cep.json")
    ap.add_argument("--out", default="docs/guidance_tables.md")
    args = ap.parse_args(argv)

    with open(args.map) as fh:
        m = json.load(fh)
    c = {}
    if os.path.exists(args.cep):
        with open(args.cep) as fh:
            c = json.load(fh)

    L = ["# Step 3 — generated tables", "",
         "Generated by `analysis/guidance_report.py` from "
         "`docs/guidance_map.json` and `docs/guidance_cep.json`. Do not edit.",
         ""]
    L += envelope_tables(m)
    lbl = "long" if "long" in m["engagements"] else list(m["engagements"])[0]
    L += inverse_map_tables(m, lbl)
    L += pointing_tables(m)
    L += prediction_tables(m)
    L += scheduler_tables(c)
    L += aimoff_tables(c)
    L += ladder_tables(c)
    L += monitor_tables(c)
    L += envelope_cep_tables(c, m)
    L += envelope_cep2_tables(c)
    L += containment_tables(c)

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {args.out}  ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
