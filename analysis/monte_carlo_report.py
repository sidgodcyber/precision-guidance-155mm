"""
Generated tables for step 6.

Reads `docs/monte_carlo.json` and emits `docs/mc_tables.md`, the same way
`analysis.nav_report` does for step 5. Every table in the step-6 documents
that is longer than a few rows is generated here rather than typed, so it
cannot drift from the data that produced it.

Run:  python -m analysis.monte_carlo_report
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

SRC = "docs/monte_carlo.json"
OUT = "docs/mc_tables.md"


def _load(p=None):
    p = p or SRC
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _tbl(head, rows) -> list:
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    out.append("")
    return out


def _f(x, n=2, dash="—"):
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return dash if not math.isfinite(v) else f"{v:.{n}f}"


def _pct(x, n=1):
    return "—" if x is None else f"{100.0 * float(x):.{n}f} %"


def _ci(s):
    return f"[{_f(s.get('cep_lo_m'))}, {_f(s.get('cep_hi_m'))}]"


# ===========================================================================
def uncorrected(d) -> list:
    if "u" not in d:
        return []
    L = ["## The uncorrected dispersion, measured (Task U)", ""]
    for label, u in sorted(d["u"].items()):
        t = u["target"]
        L += [f"Engagement `{label}`, {u['n']} rounds, met message "
              f"{u['met_age_hours']} h old, atmospheric sigmas × "
              f"{u['met_scale']:g}.", "",
              "Each row keeps ONE cause and zeroes the rest. `all` is every "
              "cause together; `no_met_correction` is the same atmospheres "
              "with the gun laid at the nominal quadrant elevation, so the "
              "difference between it and `met_message_error` is what the met "
              "correction is worth.", ""]
        rows = []
        for k, c in u["components"].items():
            rows.append([f"`{k}`", c["n"], _f(c["sigma_range_m"]),
                         _f(c["sigma_defl_m"]), _f(c["bias_range_m"]),
                         _f(c["bias_defl_m"]), _f(c["cep_m"])])
        L += _tbl(["cause kept", "n", "range 1σ, m", "deflection 1σ, m",
                   "range bias, m", "deflection bias, m", "CEP, m"], rows)
        L += ["### Against the figure the project has assumed since step 2.5",
              ""]
        L += _tbl(["quantity", "value"], [
            ["assumed uncorrected CEP", f"{_f(t['cep_m'], 1)} m"],
            ["assumed range 1σ", f"{_f(t['sigma_range_m'], 1)} m"],
            ["**modelled** range 1σ", f"**{_f(t['modelled_sigma_range_m'], 1)} m**"],
            ["modelled share of the range variance",
             _pct(t["modelled_fraction_of_variance_range"])],
            ["assumed deflection 1σ", f"{_f(t['sigma_defl_m'], 1)} m"],
            ["**modelled** deflection 1σ", f"**{_f(t['modelled_sigma_defl_m'], 1)} m**"],
            ["residual added, range 1σ", f"{_f(t['residual_sigma_range_m'], 1)} m"],
            ["residual added, as muzzle velocity",
             f"{_f(t['inflate']['sigma_mv'])} m/s"],
            ["d(range)/d(muzzle velocity)", f"{_f(t['d_range_per_mv'])} m per m/s"],
        ])
    return L


# ===========================================================================
def atmospheric(d) -> list:
    if "c" not in d:
        return []
    L = ["## The atmospheric knowledge term (Task C)", ""]
    for label, c in sorted(d["c"].items()):
        L += [f"Engagement `{label}`, {c['n']} rounds per age, "
              f"{'navigation in the loop' if c['use_nav'] else 'truth-fed'}, "
              f"paired on common random numbers against a fuze given a "
              f"PERFECT met message.", "",
              "`perfect` is the reference: the fuze knows the air exactly. "
              "`none` is no met message at all — the fuze flies the standard "
              "atmosphere, which is what every step before 6 assumed.", ""]
        rows = []
        for k, a in c["ages"].items():
            kt = a.get("knowledge_term") or {}
            rows.append([f"`{k}`", a["n"], _f(a["cep_m"]), _ci(a),
                         _f(a["sd_range_m"]), _f(a["sd_defl_m"]),
                         _f(a["bias_range_m"]),
                         _f(kt.get("sigma_range_m")),
                         _f(kt.get("sigma_defl_m")),
                         _f(kt.get("bias_range_m"))])
        L += _tbl(["met message", "n", "CEP, m", "95 % CI", "range 1σ, m",
                   "defl 1σ, m", "range bias, m",
                   "**term** range 1σ", "**term** defl 1σ",
                   "**term** range bias"], rows)
    return L


# ===========================================================================
def monitor(d) -> list:
    if "d" not in d:
        return []
    L = ["## The authority monitor, re-derived (Task D)", ""]
    for label, m in sorted(d["d"].items()):
        L += [f"Engagement `{label}`, {m['n']} rounds per population, "
              f"navigation IN the loop.", "",
              "Both populations were flown ONCE with the monitor disabled and "
              "its test statistic shadow-logged, so every threshold is "
              "evaluated on the same flights. The operating points below the "
              "sweep were re-flown with the monitor really acting, because a "
              "fault changes the trajectory and the sweep cannot know that.",
              "",
              "**The two rates are a pair.** Raising the threshold to "
              "suppress the false alarm IS the act of degrading detection.",
              ""]
        floors = sorted({r["monitor_floor_m"] for r in m["sweep"]})
        for fl in floors:
            rows = []
            for r in sorted([x for x in m["sweep"]
                             if x["monitor_floor_m"] == fl],
                            key=lambda x: x["monitor_fraction"]):
                rows.append([_f(r["monitor_fraction"]),
                             _pct(r["false_alarm_rate"]),
                             _pct(r["detection_rate"]),
                             _pct(r["missed_detection_rate"]),
                             _f(r["median_detection_delay_s"])])
            L += [f"### `monitor_floor_m` = {fl:g} m", ""]
            L += _tbl(["`monitor_fraction`", "false alarms, healthy",
                       "detected, stage-2 fail", "missed",
                       "median detection delay, s"], rows)
        if m.get("confirm"):
            L += ["### The operating points, re-flown", ""]
            rows = []
            for k, c in m["confirm"].items():
                rows.append([f"`{k}`", c.get("criterion", ""),
                             _f(c["healthy"]["cep_m"]),
                             _ci(c["healthy"]),
                             _pct(c["false_alarm_fraction"]),
                             _pct(c["detected_fraction"]),
                             _f(c["healthy_vs_unmonitored"].get("delta_cep_m")),
                             _f(c["stage2_fail"]["cep_m"], 1)])
            rows.append(["monitor OFF", "the control",
                         _f(m["unmonitored_full"]["cep_m"]),
                         _ci(m["unmonitored_full"]), "0.0 %", "0.0 %", "—",
                         _f(m["unmonitored_stage2_fail"]["cep_m"], 1)])
            L += _tbl(["setting", "criterion", "healthy CEP, m", "95 % CI",
                       "false alarms", "detected",
                       "ΔCEP vs monitor off, m", "stage-2 fail CEP, m"], rows)
    return L


# ===========================================================================
def staging(d) -> list:
    if "e" not in d:
        return []
    L = ["## Staged versus single stage, over the distributions (Task E)", ""]
    for label, e in sorted(d["e"].items()):
        L += [f"Engagement `{label}`, {e['n']} rounds each, **paired on "
              f"common random numbers**: both configurations saw the same "
              f"atmospheres, shells, muzzle velocities, fuze offsets and "
              f"sensor seeds.", ""]
        rows = []
        for k, c in e["configs"].items():
            rows.append([f"`{k}`", c["n"], _f(c["cep_m"]), _ci(c),
                         _f(c["mean_miss_m"]), _f(c["p90_m"]), _f(c["p95_m"]),
                         _f(c["max_m"]), _pct(c["within_30m"]),
                         _f(c["kurtosis_range"]),
                         _pct(c["max_variance_share"])])
        L += _tbl(["configuration", "n", "CEP, m", "95 % CI", "mean, m",
                   "p90, m", "p95, m", "worst, m", "≤ 30 m",
                   "excess kurtosis", "worst round's variance share"], rows)
        p = e["paired"]
        L += ["### The paired difference, single stage minus staged", ""]
        L += _tbl(["quantity", "value"], [
            ["rounds paired", p["n"]],
            ["ΔCEP", f"**{_f(p['delta_cep_m'])} m**"],
            ["95 % CI on ΔCEP",
             f"[{_f(p['delta_cep_lo_m'])}, {_f(p['delta_cep_hi_m'])}]"],
            ["median per-round difference", f"{_f(p['median_paired_delta_m'])} m"],
            ["rounds on which single stage was worse",
             _pct(p["fraction_a_worse"])],
        ])
    return L


# ===========================================================================
def band(d) -> list:
    if "b" not in d:
        return []
    L = ["## The epistemic band (Task B)", ""]
    for label, b in sorted(d["b"].items()):
        L += [f"Engagement `{label}`, {b['n']} rounds per variant, paired "
              f"against the nominal.", "",
              "**These are not terms inside the CEP.** Every round of a lot "
              "flies the same canards, the same bearing and the same coil, so "
              "each row is the CEP of a DIFFERENT POSSIBLE LOT, not a wider "
              "CEP for all of them.", ""]
        rows = []
        for k, v in b["variants"].items():
            pv = v.get("paired_vs_nominal") or {}
            rows.append([f"`{k}`", _f(v["cep_m"]), _ci(v),
                         _f(pv.get("delta_cep_m")),
                         (f"[{_f(pv.get('delta_cep_lo_m'))}, "
                          f"{_f(pv.get('delta_cep_hi_m'))}]" if pv else "—"),
                         _pct(v["within_30m"]), _f(v["p90_m"])])
        L += _tbl(["variant", "CEP, m", "95 % CI", "Δ vs nominal, m",
                   "95 % CI on Δ", "≤ 30 m", "p90, m"], rows)
        bb = b["band"]
        L += ["", f"**CEP is {_f(bb['nominal_cep_m'], 1)} m, and lies between "
                  f"{_f(bb['min_cep_m'], 1)} m (`{bb['argmin']}`) and "
                  f"{_f(bb['max_cep_m'], 1)} m (`{bb['argmax']}`).**", ""]
    return L


# ===========================================================================
def levers(d) -> list:
    if "g" not in d:
        return []
    L = ["## The two cheap levers, re-measured (Task G3)", ""]
    for label, g in sorted(d["g"].items()):
        L += [f"Engagement `{label}`, {g['n']} rounds, **a different sensor "
              f"seed on every round**, paired.", "",
              "[NAV-CEP.md §4](NAV-CEP.md) priced the antenna at −7.5 m of "
              "range 1σ and §3 the magnetometer floor at −4.6 m; "
              "[STEP5.5-CLOSEOUT.md](STEP5.5-CLOSEOUT.md) found neither "
              "reproduced, because the sensor bias vector was drawn once per "
              "seed and only three seeds were flown. A lever whose interval "
              "straddles zero is **not measured**, and is reported as such "
              "rather than as zero.", ""]
        rows = []
        for k, v in g["levers"].items():
            if k == "baseline":
                continue
            for axis in ("range", "defl"):
                a = v[f"{axis}_delta"]
                rows.append([
                    f"`{k}`", axis, _f(a["baseline_sigma_m"]),
                    _f(a["lever_sigma_m"]), _f(a["delta_sigma_m"]),
                    f"[{_f(a['delta_sigma_lo_m'])}, {_f(a['delta_sigma_hi_m'])}]",
                    "**yes**" if v["reproduces"][
                        "range" if axis == "range" else "deflection"]
                    else "no"])
        L += _tbl(["lever", "axis", "baseline 1σ, m", "with lever 1σ, m",
                   "Δ1σ, m", "95 % CI on Δ1σ", "a real improvement?"], rows)
    return L


# ===========================================================================
def headline(d) -> list:
    if "a" not in d:
        return []
    L = ["## The headline campaign (Tasks A and F)", ""]
    for tag, per in sorted(d["a"].items()):
        L += [f"### `{tag}`", ""]
        rows = []
        for label, a in sorted(per.items(),
                               key=lambda kv: kv[1]["unguided"]["cep_m"]):
            g = a["guided"]
            rows.append([f"`{label}`", g["n"], _f(a["unguided"]["cep_m"], 1),
                         f"**{_f(g['cep_m'])}**", _ci(g), _f(g["mean_miss_m"]),
                         _f(g["p90_m"]), _f(g["max_m"], 1),
                         _pct(g["within_30m"]), _pct(g["within_50m"]),
                         _f(g["bias_range_m"]), _f(g["sd_range_m"]),
                         _f(g["sd_defl_m"])])
        L += _tbl(["engagement", "n", "unguided CEP, m", "guided CEP, m",
                   "95 % CI", "mean, m", "p90, m", "worst, m", "≤ 30 m",
                   "≤ 50 m", "range bias, m", "range 1σ, m", "defl 1σ, m"],
                  rows)
        for label, a in sorted(per.items()):
            L += [f"#### Convergence, `{label}`", ""]
            L += _tbl(["rounds", "CEP, m", "95 % CI", "half-width, m"],
                      [[c["n"], _f(c["cep_m"]), f"[{_f(c['cep_lo_m'])}, "
                        f"{_f(c['cep_hi_m'])}]",
                        _f(0.5 * (c["cep_hi_m"] - c["cep_lo_m"]))]
                       for c in a["convergence"]])
            cv = dict(a["converged"])
            cv.setdefault("final_n", a["convergence"][-1]["n"]
                          if a.get("convergence") else "?")
            cv.setdefault("final_cep_m", a["guided"]["cep_m"])
            cv.setdefault("half_width_m", float("nan"))
            cv.setdefault("half_width_pct", float("nan"))
            if cv.get("converged"):
                L += [f"**Converged at {cv['stable_at_n']} rounds.** At "
                      f"{cv['final_n']} the CEP is {_f(cv['final_cep_m'])} m "
                      f"± {_f(cv['half_width_m'])} m "
                      f"({_f(cv['half_width_pct'], 1)} %).", ""]
            else:
                L += [f"**Not converged at {cv['final_n']} rounds** — "
                      f"{cv.get('reason', '')}. The CEP is "
                      f"{_f(cv['final_cep_m'])} m with a residual 95 % "
                      f"half-width of {_f(cv['half_width_m'])} m "
                      f"({_f(cv['half_width_pct'], 1)} %).", ""]
    return L


# ===========================================================================
def atmospheric_scale(d) -> list:
    if "cs" not in d:
        return []
    L = ["## The atmospheric term against the ASSUMED variability (Task CS)",
         ""]
    for label, c in sorted(d["cs"].items()):
        L += [f"Engagement `{label}`, {c['n']} rounds per scale, message "
              f"{c['met_age_hours']} h old, paired at each scale against a "
              f"perfect message AT THAT SCALE.", "",
              "`linearity` is the term at that scale divided by (the term at "
              "1.0 × the scale): **1.0 means the answer scales with the "
              "assumed sigma**, so a reader who disagrees with "
              "[§2.2](ATMOSPHERIC-ERROR.md) can multiply rather than re-run.",
              ""]
        rows = []
        for k, v in sorted(c["scales"].items(), key=lambda kv: kv[1]["scale"]):
            rows.append([f"× {v['scale']:g}", v["n"],
                         _f(v["sigma_range_m"]), _f(v["sigma_defl_m"]),
                         _f(v["bias_range_m"]), _f(v["cep_perfect_m"]),
                         _f(v["cep_aged_m"]), _f(v.get("linearity"), 3)])
        L += _tbl(["atmospheric σ scale", "n", "term range 1σ, m",
                   "term defl 1σ, m", "term range bias, m",
                   "CEP, perfect message", "CEP, aged message", "linearity"],
                  rows)
    return L


# ===========================================================================
def budget(d, label="long", tag="headline") -> list:
    """
    The CEP budget: every line, measured or explicitly estimated, with the
    step that produced it.

    ADDED IN QUADRATURE, AND THE ASSUMPTION IS STATED RATHER THAN HIDDEN.
    Quadrature is exact only for independent terms, and these are not fully
    independent -- a round pushed outside the reachable set by the met message
    is a round on which the navigation error matters less, because the law
    holds to impact whatever it believes. So the named terms are measured
    PAIRED against the headline configuration, which gives each one's MARGINAL
    contribution in the presence of the others, and the residual line is what
    the quadrature does not account for. A residual close to the step-3
    truth-fed figure is the check that the decomposition is not fooling
    itself; a large NEGATIVE residual would mean the terms overlap and the
    quadrature is wrong.
    """
    a = ((d.get("a") or {}).get(tag) or {}).get(label)
    if not a:
        return []
    g = a["guided"]
    L = ["## The CEP budget", "",
         f"Engagement `{label}`, {g['n']} rounds, every aleatoric term in and "
         f"drawn per round.", "",
         "Read the residual line as **the authority limit**: it is what is "
         "left of a dispersion far larger than the reachable set after the "
         "loop has corrected what it can reach. It is not a list of small "
         "modelling errors.", ""]

    rows = []
    rows.append(["**uncorrected dispersion, as flown**",
                 _f(a["unguided"]["sigma_range_m"], 1),
                 _f(a["unguided"]["sigma_defl_m"], 1),
                 "the input, not a contributor", "step 6 (Task U)"])
    rows.append(["**guided, everything in**", _f(g["sd_range_m"], 1),
                 _f(g["sd_defl_m"], 1),
                 f"CEP {_f(g['cep_m'])} m {_ci(g)}", "step 6 (Task A)"])
    rows.append(["", "", "", "*decomposed:*", ""])

    named = []
    n = (d.get("n") or {}).get(label)
    if n:
        r = n["contribution_range"]
        f = n["contribution_defl"]
        named.append(("navigation, flying on the estimate",
                      r["sigma_m"], f["sigma_m"],
                      f"paired truth-fed, bias {_f(r['bias_m'])} m; "
                      f"step 5 said 30.4 m on three sensor seeds",
                      "step 6 (Task N)"))

    c = (d.get("c") or {}).get(label)
    if c:
        age = None
        for k in ("2h", "3h", "1h"):
            if k in c["ages"] and c["ages"][k].get("knowledge_term"):
                age = k
                break
        if age:
            k = c["ages"][age]["knowledge_term"]
            named.append((f"atmospheric knowledge, {age} met message",
                          k["sigma_range_m"], k["sigma_defl_m"],
                          f"paired against a perfect message, bias "
                          f"{_f(k['bias_range_m'])} m",
                          "step 6 (Task C)"))

    m = (d.get("d") or {}).get(label)
    if m and m.get("confirm"):
        ad = m.get("adopted") or {}
        key = f"f{ad.get('monitor_fraction', 0.3):g}_floor{round(ad.get('monitor_floor_m', 20.0), 3):g}"
        cf = m["confirm"].get(key)
        if cf:
            pd_ = cf["healthy_vs_unmonitored"]
            named.append(("authority-monitor false alarms",
                          None, None,
                          f"CEP {_f(pd_.get('delta_cep_m'))} m against the "
                          f"monitor disabled, paired; false-alarm rate "
                          f"{_pct(cf['false_alarm_fraction'])}",
                          "step 6 (Task D)"))

    for name, sr, sd, note, step in named:
        rows.append([name, _f(sr, 1) if sr is not None else "—",
                     _f(sd, 1) if sd is not None else "—", note, step])

    # The residual, in quadrature.
    tot_r = g["sd_range_m"] ** 2
    tot_d = g["sd_defl_m"] ** 2
    for _n, sr, sd, _x, _y in named:
        if sr is not None:
            tot_r -= sr ** 2
        if sd is not None:
            tot_d -= sd ** 2
    rows.append(["**everything else** — the authority limit above all, plus "
                 "the servo, the deployment phase, the impact-point "
                 "prediction, the law's own decisions **and the monitor's "
                 "false alarms**, which are quoted above as a CEP effect and "
                 "so are not subtracted per axis",
                 _f(math.copysign(math.sqrt(abs(tot_r)), tot_r), 1),
                 _f(math.copysign(math.sqrt(abs(tot_d)), tot_d), 1),
                 "by difference in quadrature", "steps 3, 4, 4.5"])
    L += _tbl(["term", "range 1σ, m", "deflection 1σ, m", "how", "source"],
              rows)
    return L


# ===========================================================================
def gap(d, label="long", tag="headline", target=30.0) -> list:
    """
    What would have to be removed to reach the requirement, in the order that
    removing it helps most.

    ARITHMETIC, NOT A PLAN. Each row takes the measured 1σ per axis of the
    headline campaign, removes ONE term's measured contribution in quadrature,
    and converts what is left back to a CEP by the same quadrature
    `analysis.cep_projection` uses. That is exact only if the terms are
    independent and the residual distribution stays roughly normal, and
    neither is exactly true — a round pushed outside the reachable set by one
    term is a round on which the others matter less. So these are *upper
    bounds on what the fix buys*, and the campaign that would settle any of
    them is named alongside.
    """
    from analysis import cep_projection as cp

    a = ((d.get("a") or {}).get(tag) or {}).get(label)
    if not a:
        return []
    g = a["guided"]
    sr, sd = g["sd_range_m"], g["sd_defl_m"]
    base_cep = float(cp.uncorrected_cep(sr, sd, n=401))

    actual = g["cep_m"]
    L = ["## What would have to be removed to reach 30 m", "",
         f"Engagement `{label}`. Each row removes ONE term's measured 1σ from "
         f"the campaign's spread in quadrature and converts what is left back "
         f"to a CEP.", "",
         f"**The normal model is used for the RATIO only, not for the level.** "
         f"The campaign's spread is {_f(sr, 1)} m in range and {_f(sd, 1)} m "
         f"in deflection, which under a bivariate normal would be a CEP of "
         f"{_f(base_cep, 1)} m — against the {_f(actual, 1)} m the rounds "
         f"actually give. The {_f(abs(base_cep - actual), 1)} m gap is the "
         f"**bimodality**: half the rounds are corrected to a few metres and "
         f"half are not reachable at all, and a normal has no way to be both. "
         f"So each row below is quoted as the measured CEP scaled by the "
         f"ratio the normal model predicts, which is the most this "
         f"approximation can honestly support.", ""]

    terms = []
    n = (d.get("n") or {}).get(label)
    if n:
        terms.append(("fly on truth instead of the estimate",
                      n["contribution_range"]["sigma_m"],
                      n["contribution_defl"]["sigma_m"],
                      "a perfect navigation solution — not achievable, "
                      "quoted as the bound", True))
    c = (d.get("c") or {}).get(label)
    if c:
        k0 = (c["ages"].get("0h") or {}).get("knowledge_term")
        k2 = (c["ages"].get("2h") or {}).get("knowledge_term")
        if k2:
            terms.append(("upload a PERFECT met message",
                          k2["sigma_range_m"], k2["sigma_defl_m"],
                          "the bound on any met improvement", True))
        if k0 and k2:
            dr = math.sqrt(max(0.0, k2["sigma_range_m"] ** 2
                               - k0["sigma_range_m"] ** 2))
            dd = math.sqrt(max(0.0, k2["sigma_defl_m"] ** 2
                               - k0["sigma_defl_m"] ** 2))
            # NOT combinable: it is a SUBSET of the perfect-message row
            # above, so adding both to the "together" line would remove the
            # same variance twice.
            terms.append(("upload a FRESH met message at fuze setting "
                          "(2 h → 0 h)", dr, dd,
                          "achievable: it is a procedure, not hardware",
                          False))

    scale = actual / base_cep if base_cep else 1.0
    rows = [["*nothing removed*", "—", "—", f"**{_f(actual, 1)}**",
             "—", f"the campaign, {g['n']} rounds"]]
    for name, tr, td, note, _comb in terms:
        r2 = max(0.0, sr ** 2 - (tr or 0.0) ** 2)
        d2 = max(0.0, sd ** 2 - (td or 0.0) ** 2)
        cep = scale * float(cp.uncorrected_cep(math.sqrt(r2) or 1e-6,
                                               math.sqrt(d2) or 1e-6, n=401))
        rows.append([name, _f(tr, 1), _f(td, 1), _f(cep, 1),
                     _f(actual - cep, 1), note])
    # Everything named, together.
    r2 = sr ** 2
    d2 = sd ** 2
    for _n, tr, td, _x, comb in terms:
        if not comb:
            continue
        r2 -= (tr or 0.0) ** 2
        d2 -= (td or 0.0) ** 2
    if r2 > 0 and d2 > 0:
        cep = scale * float(cp.uncorrected_cep(math.sqrt(r2), math.sqrt(d2),
                                               n=401))
        rows.append(["**perfect navigation AND a perfect met message**",
                     "—", "—",
                     f"**{_f(cep, 1)}**", _f(actual - cep, 1),
                     "and it still leaves the loop itself"])
    L += _tbl(["remove", "range 1σ removed, m", "defl 1σ removed, m",
               "CEP left, m", "CEP bought, m", "what it would take"], rows)
    L += ["", f"**The requirement is {target:.0f} m.** Any row above that "
              f"still does not reach it is a row where the remaining terms — "
              f"the authority limit, the deployment phase, the servo and the "
              f"impact-point prediction — are what stands between the design "
              f"and the requirement, and none of them is removed by better "
              f"data.", ""]
    return L


# ===========================================================================
def main(argv=None):
    d = _load()
    L = ["# Step 6 — generated tables", "",
         "Every table here is produced by "
         "[`analysis/monte_carlo_report.py`](../analysis/monte_carlo_report.py) "
         "from `docs/monte_carlo.json`. Nothing in it is typed, so it cannot "
         "drift from the campaign that produced it.", "",
         "Sampling: every per-round variable is drawn independently for every "
         "round and the sensor seed is the round index. Absolute figures are "
         "unpaired with a bootstrap interval; comparisons are paired on "
         "common random numbers with a bootstrap interval on the difference. "
         "See [MONTE-CARLO-DESIGN.md](MONTE-CARLO-DESIGN.md).", "",
         "---", ""]
    for fn in (headline, budget, gap, uncorrected, atmospheric,
               atmospheric_scale, monitor, staging, levers, band):
        part = fn(d)
        if part:
            L += part + ["---", ""]
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"wrote {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
