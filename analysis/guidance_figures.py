"""
Figures for step 3. Reads docs/guidance_map.json and docs/guidance_cep.json,
writes docs/figures/guidance_*.png.

Five, each carrying one finding a table states less well:

  guidance_inverse_map.png  the map itself: the rotation between commanded
                            angle and delivered direction, how it moves over
                            the flight, and how the reach varies with the
                            direction asked for. The rotation is the reason a
                            guidance law cannot just command the miss
                            direction, and the reach variation is the
                            anisotropy the scheduler must respect.
  guidance_prediction.png   impact-point-prediction error against time to go,
                            which is what sets when the direction can be
                            committed.
  guidance_schedulers.png   the four laws and the isotropic control, as miss
                            distributions rather than as four CEP numbers.
  guidance_aimoff.png       CEP against aim-off, with the published 224 m
                            marked.
  guidance_envelope.png     CEP across the firing table against authority and
                            against the dead time as a fraction of the guided
                            phase -- the short-range finding.

Run:  python -m analysis.guidance_figures
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


GUIDED = "#2E6DB4"
UNGUIDED = "#B4432E"
IDEAL = "#555555"
ACCENT = "#C88A00"
LAWS = {"proportional": "#2E6DB4", "deadband": "#4CA36B",
        "budget": "#B4432E", "single_shot": "#7A5AA8",
        "isotropic": "#999999"}


def _ecdf(ax, values, color, label, lw=1.8, ls="-"):
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0:
        return
    y = np.arange(1, v.size + 1) / v.size
    ax.step(v, y, where="post", color=color, lw=lw, ls=ls, label=label)


# ===========================================================================
def inverse_map_figure(m: dict, path: str, label: str = "long") -> str:
    plt = _mpl()
    from gnc.inverse_map import AuthorityMap

    e = m["engagements"][label]
    amap = AuthorityMap.from_dict(e["map"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # -- (a) rotation and reach against time to go -----------------------
    ax = axes[0]
    tgo, rot, reach_max, reach_min = [], [], [], []
    for n in amap.nodes:
        if n.t_go <= 0.5:
            continue
        A, _ = amap.remaining(n.t_go)
        s = np.linalg.svd(A, compute_uv=False)
        if s[1] < 1e-6:
            continue
        u = np.array([1.0, 0.0])
        phi = AuthorityMap.invert(A, u)
        if phi is None:
            continue
        tgo.append(n.t_go)
        rot.append(math.degrees(-phi))       # desired 0 deg minus commanded
        reach_max.append(float(s[0]))
        reach_min.append(float(s[1]))
    ax.plot(tgo, rot, color=GUIDED, lw=1.8, marker="o", ms=3)
    ax.set_xlabel("time to go, s")
    ax.set_ylabel("rotation for a pure range correction, deg")
    ax.set_title("(a) commanded angle is not the correction direction")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)

    # -- (b) the remaining set, at three times ---------------------------
    ax = axes[1]
    full = amap.nodes[0].t_go
    for frac, ls in ((1.0, "-"), (0.6, "--"), (0.3, ":")):
        t = full * frac
        A, _ = amap.remaining(t)
        th = np.linspace(0, 2 * math.pi, 361)
        pts = np.array([A @ [math.cos(x), math.sin(x)] for x in th])
        ax.plot(pts[:, 0], pts[:, 1], color=GUIDED, ls=ls, lw=1.6,
                label=f"t_go = {t:.0f} s")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("range, m")
    ax.set_ylabel("deflection, m")
    ax.set_title("(b) the set still reachable, as it expires")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # -- (c) reach against the direction asked for -----------------------
    ax = axes[2]
    A, _ = amap.remaining(full)
    th = np.linspace(0, 2 * math.pi, 361)
    r = [AuthorityMap.reach_along(A, [math.cos(x), math.sin(x)]) for x in th]
    ax.plot(np.degrees(th), r, color=GUIDED, lw=1.8)
    e_full = e["full_hold_ellipse"]
    ax.axhline(e_full["semi_major_m"], color=IDEAL, lw=0.9, ls="--",
               label=f"semi-major {e_full['semi_major_m']:.0f} m")
    ax.axhline(e_full["semi_minor_m"], color=IDEAL, lw=0.9, ls=":",
               label=f"semi-minor {e_full['semi_minor_m']:.0f} m")
    ax.set_xlabel("desired correction direction, deg from downrange")
    ax.set_ylabel("reach, m")
    ax.set_title("(c) the anisotropy a scheduler must respect")
    ax.set_xlim(0, 360)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"The analytic inverse map, {label} engagement", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def prediction_figure(m: dict, path: str) -> str:
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for lbl, col in zip(m["engagements"],
                        (UNGUIDED, ACCENT, "#4CA36B", GUIDED, "#7A5AA8")):
        e = m["engagements"][lbl]
        rows = e["prediction_error"]["rows"]
        if not rows:
            continue
        x = [r["t_go"] for r in rows]
        y = [r["rms_m"] for r in rows]
        ax.plot(x, y, marker="o", ms=3, lw=1.5, color=col, label=lbl)
    ax.set_xlabel("time to go, s")
    ax.set_ylabel("prediction error, rms over commanded angles, m")
    ax.set_title("(a) the predictor converges as the flight runs out")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    e = m["engagements"]["long"]
    rows = e["prediction_error"]["rows"]
    x = [r["t_end_offset"] for r in rows]
    ax.plot(x, [r["bias_range_m"] for r in rows], color=UNGUIDED, lw=1.6,
            marker="o", ms=3, label="range bias")
    ax.plot(x, [r["bias_deflection_m"] for r in rows], color=GUIDED, lw=1.6,
            marker="s", ms=3, label="deflection bias")
    ax.fill_between(x,
                    [r["bias_range_m"] - r["sd_range_m"] for r in rows],
                    [r["bias_range_m"] + r["sd_range_m"] for r in rows],
                    color=UNGUIDED, alpha=0.15)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("s after deployment")
    ax.set_ylabel("predicted minus actual, m")
    ax.set_title("(b) it is a bias, not scatter — long engagement")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def scheduler_figure(c: dict, path: str) -> str:
    plt = _mpl()
    t = c["task_c"]
    t2 = c.get("task_c2")
    fig, axes = plt.subplots(1, 3 if t2 else 2, figsize=(15.5 if t2 else 11.0, 4.4))

    for k, (tt, title) in enumerate([
            (t, "(a) a dispersion the kit CANNOT cover — uncorrected CEP 200 m"),
            (t2, "(b) one it CAN — uncorrected CEP 100 m")] if t2 else
            [(t, "(a) the laws as distributions, not as four numbers")]):
        ax = axes[k]
        for name, res in tt["schedulers"].items():
            _ecdf(ax, [r["miss_m"] for r in res["rows"]], LAWS.get(name, "#333"),
                  f"{name}  CEP {res['cep']['cep_m']:.1f} m",
                  ls=":" if name == "isotropic" else "-")
        ax.axvline(30.0, color="k", lw=0.8, ls="--")
        ax.text(31, 0.05, "30 m", fontsize=8)
        ax.set_xlabel("residual miss, m")
        ax.set_ylabel("cumulative fraction of rounds")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 260)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    ax = axes[-1]
    best = t["best"]
    rows = t["schedulers"][best]["rows"]
    ax.scatter([r["miss_range_m"] for r in rows], [r["miss_defl_m"] for r in rows],
               s=14, color=GUIDED, alpha=0.75, label=f"{best}")
    iso = t["schedulers"]["isotropic"]["rows"]
    ax.scatter([r["miss_range_m"] for r in iso], [r["miss_defl_m"] for r in iso],
               s=14, color="#999999", alpha=0.6, marker="x", label="isotropic")
    th = np.linspace(0, 2 * math.pi, 200)
    ax.plot(30 * np.cos(th), 30 * np.sin(th), color="k", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xlim(-160, 160)
    ax.set_ylim(-160, 160)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("residual range miss, m  (clipped to ±160 m)")
    ax.set_ylabel("residual deflection miss, m")
    ax.set_title("(c) where the residual sits, and in which axis", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def aimoff_figure(c: dict, path: str) -> str:
    plt = _mpl()
    d = c["task_d"]
    xs = sorted(float(k) for k in d["offsets"])
    cep = [d["offsets"][_k(k, d)]["cep"]["cep_m"] for k in xs]
    se = [d["offsets"][_k(k, d)]["cep"].get("cep_se_m", 0.0) for k in xs]
    bias = [d["offsets"][_k(k, d)]["cep"]["bias_range_m"] for k in xs]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # The CEP, WITH its standard error, because the interior of the sweep is
    # not resolved and a bare line would say otherwise.
    ax = axes[0]
    ax.errorbar(xs, cep, yerr=se, color=GUIDED, lw=1.9, marker="o", ms=4,
                capsize=3, elinewidth=1.0, label="CEP ± bootstrap SE")
    ax.axvline(224.0, color=UNGUIDED, lw=1.2, ls="--", label="published 224 m")
    ax.axvline(186.7, color=ACCENT, lw=1.4, ls="-",
               label="predicted by the map, 186.7 m")
    ax.set_xlabel("aim-off, m long")
    ax.set_ylabel("closed-loop CEP, m")
    ax.set_title("(a) the CEP — the interior is not resolved", fontsize=10)
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(alpha=0.3)

    # The residual range bias, which IS resolved and is the evidence.
    ax = axes[1]
    ax.plot(xs, bias, color=GUIDED, lw=1.9, marker="o", ms=4)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(224.0, color=UNGUIDED, lw=1.2, ls="--", label="published 224 m")
    ax.axvline(186.7, color=ACCENT, lw=1.4, ls="-",
               label="predicted by the map, 186.7 m")
    ax.set_xlabel("aim-off, m long")
    ax.set_ylabel("residual mean range bias, m")
    ax.set_title("(b) the residual range bias — which is", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.suptitle("Aim-off with the loop closed", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _k(x, d):
    for k in d["offsets"]:
        if abs(float(k) - x) < 1e-9:
            return k
    raise KeyError(x)


def envelope_figure(c: dict, m: dict, path: str) -> str:
    plt = _mpl()
    f = c["task_f"]
    labels = list(f["engagements"])
    rng = [f["engagements"][l]["baseline"]["uncorrected_range"] for l in labels]
    cep = [f["engagements"][l]["cep"]["cep_m"] for l in labels]
    unc = [f["engagements"][l]["unguided_cep"]["cep_m"] for l in labels]
    a = [f["engagements"][l]["ellipse"]["semi_major_m"] for l in labels]
    order = np.argsort(rng)
    rng = np.array(rng)[order]; cep = np.array(cep)[order]
    unc = np.array(unc)[order]; a = np.array(a)[order]
    labels = [labels[i] for i in order]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    ax.plot(rng / 1000, unc, color=UNGUIDED, lw=1.8, marker="s", ms=4,
            label="unguided (kit neutral)")
    ax.plot(rng / 1000, cep, color=GUIDED, lw=1.9, marker="o", ms=4,
            label="guided")
    ax.axhline(30.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("range, km")
    ax.set_ylabel("CEP, m")
    ax.set_yscale("log")
    ax.set_title("(a) CEP across the firing table")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(rng / 1000, a, color=GUIDED, lw=1.9, marker="o", ms=4,
            label="closed-loop semi-major")
    ideal = [m["engagements"][l]["ideal"]["ellipse"]["semi_major_m"]
             for l in labels]
    ax.plot(rng / 1000, ideal, color=IDEAL, lw=1.3, ls="--", marker="^", ms=4,
            label="ideal kinematic hold")
    ax.set_xlabel("range, km")
    ax.set_ylabel("reachable-set semi-major axis, m")
    ax.set_yscale("log")
    ax.set_title("(b) the authority that produces it")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    from analysis.guidance_report import _dead_time
    dead, frac = [], []
    for l in labels:
        b = f["engagements"][l]["baseline"]
        d = _dead_time(m["engagements"][l]["nodes"]) or 3.0
        dead.append(d)
        frac.append(100 * d / b["guided_phase_s"])
    # The RATIO, not the CEP: the assumed dispersion scales with range, so an
    # absolute CEP is smallest at the engagement where guidance does least.
    ratio = cep / unc
    order2 = np.argsort(frac)
    fr = np.array(frac)[order2]; rt = ratio[order2]
    lb = [labels[i] for i in order2]
    ax.plot(fr, rt, color=GUIDED, lw=1.9, marker="o", ms=5)
    for x, y, l in zip(fr, rt, lb):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(5, -10),
                    fontsize=8)
    ax.axhline(1.0, color=UNGUIDED, lw=1.0, ls="--")
    ax.text(fr.max() * 0.55, 1.02, "no improvement", color=UNGUIDED, fontsize=8)
    ax.set_xlabel("dead time as a per cent of the guided phase")
    ax.set_ylabel("guided CEP / unguided CEP")
    ax.set_ylim(0, 1.15)
    ax.set_title("(c) what the dead time costs")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def ladder_figure(c: dict, path: str) -> str:
    plt = _mpl()
    e = c["task_e"]
    # Derived from the data, not from a hard-coded list: the navigation-loss
    # fractions are a module constant and a stale list here silently drops
    # rows from the figure.
    keys = list(e["rungs"])
    deg = sorted(k for k in keys if k.startswith("degraded"))
    order = (["full"] + deg
             + [k for k in ("inhibit", "reversionary", "stage2_fail",
                            "stage2_fail_monitor_off") if k in keys]
             + [k for k in keys
                if k not in ("full", "inhibit", "reversionary", "stage2_fail",
                             "stage2_fail_monitor_off") and k not in deg])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ceps = [e["rungs"][k]["cep"]["cep_m"] for k in order]
    cols = [GUIDED if k == "full" else
            ACCENT if k.startswith("degraded") else
            IDEAL if k == "inhibit" else UNGUIDED for k in order]
    ax.barh(range(len(order)), ceps, color=cols)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(30.0, color="k", lw=0.8, ls="--")
    for i, v in enumerate(ceps):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=8)
    ax.set_xlabel("CEP, m")
    ax.set_title("(a) the degradation ladder")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    for k in order:
        _ecdf(ax, [r["miss_m"] for r in e["rungs"][k].get("rows", [])],
              GUIDED if k == "full" else
              ACCENT if k.startswith("degraded") else
              IDEAL if k == "inhibit" else UNGUIDED, k, lw=1.4,
              ls="-" if k in ("full", "reversionary") else "--")
    ax.set_xlabel("residual miss, m")
    ax.set_ylabel("cumulative fraction")
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(b) as distributions")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="docs/guidance_map.json")
    ap.add_argument("--cep", default="docs/guidance_cep.json")
    ap.add_argument("--outdir", default="docs/figures")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    with open(args.map) as fh:
        m = json.load(fh)
    c = None
    if os.path.exists(args.cep):
        with open(args.cep) as fh:
            c = json.load(fh)

    made = [inverse_map_figure(m, os.path.join(args.outdir,
                                               "guidance_inverse_map.png")),
            prediction_figure(m, os.path.join(args.outdir,
                                              "guidance_prediction.png"))]
    if c:
        if "task_c" in c:
            made.append(scheduler_figure(c, os.path.join(
                args.outdir, "guidance_schedulers.png")))
        if "task_d" in c:
            made.append(aimoff_figure(c, os.path.join(
                args.outdir, "guidance_aimoff.png")))
        if "task_e" in c:
            made.append(ladder_figure(c, os.path.join(
                args.outdir, "guidance_ladder.png")))
        if "task_f" in c:
            made.append(envelope_figure(c, m, os.path.join(
                args.outdir, "guidance_envelope.png")))
    for p in made:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
