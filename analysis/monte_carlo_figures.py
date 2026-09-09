"""
Figures for step 6.

Reads docs/monte_carlo.json and writes docs/figures/mc_*.png. Produces nothing
that is not already a number in that file -- a figure here exists to show a
SHAPE a table cannot, and where a table is enough there is no figure.

THE SCATTER PLOT IS THE SUBMISSION'S ARTEFACT AND IS BUILT TO BE LOOKED AT.
Unguided and guided on the same axes, at the same scale, with the CEP circle
and the 30 m requirement circle drawn on both. Same scale is the whole point:
a guided panel zoomed in to fill its own axes would hide the thing the figure
exists to show.

Run:  python -m analysis.monte_carlo_figures
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = "docs/figures"
SRC = "docs/monte_carlo.json"

plt.rcParams.update({"figure.dpi": 150, "font.size": 8,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 9, "legend.fontsize": 7,
                     "axes.axisbelow": True})

C_UNGUIDED = "#b03a2e"
C_GUIDED = "#1a5276"
C_REQ = "#117a65"


def _load(path=None):
    path = path or SRC
    with open(path) as fh:
        return json.load(fh)


def _cep_of(r, d):
    return float(np.median(np.hypot(r, d)))


# ===========================================================================
def _pick(d, label=None):
    """The engagement to draw. `long` when it is there, else whichever is."""
    if label and label in d:
        return label
    for k in ("long", "mid2", "middle", "short2", "short"):
        if k in d:
            return k
    return next(iter(d))


def figure_scatter(data, path, tag="headline", label=None):
    """
    Impact points, unguided and guided, with the CEP and the requirement.

    The two panels share axes and scale. The inset repeats the guided panel at
    a scale where the 30 m circle is legible, because the honest full-scale
    view makes the guided cloud a dot -- and both facts are worth showing.
    """
    label = _pick(data["a"][tag], label)
    a = data["a"][tag][label]
    g = a["rows"]
    u = a["unguided_rows"]
    gr = np.array([r["miss_range_m"] for r in g])
    gd = np.array([r["miss_defl_m"] for r in g])
    ur = np.array([r["d_range_m"] for r in u])
    ud = np.array([r["d_defl_m"] for r in u])
    cep_g = _cep_of(gr, gd)
    cep_u = _cep_of(ur, ud)

    lim = 1.05 * max(np.abs(np.concatenate([ur, ud, gr, gd])).max(), 1.0)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.9))

    for k, (x, y, cep, col, title, n) in enumerate((
            (ud, ur, cep_u, C_UNGUIDED,
             "Unguided — the round as it is fired today", len(u)),
            (gd, gr, cep_g, C_GUIDED,
             "Guided — the kit, everything aleatoric included", len(g)))):
        A = ax[k]
        A.scatter(x, y, s=9, alpha=0.55, color=col, linewidths=0,
                  label=f"{n} rounds")
        A.add_patch(Circle((0, 0), cep, fill=False, color=col, lw=1.6,
                           label=f"CEP {cep:.1f} m"))
        A.add_patch(Circle((0, 0), 30.0, fill=False, color=C_REQ, lw=1.6,
                           ls="--", label="requirement, 30 m"))
        A.plot(0, 0, "+", color="k", ms=9, mew=1.2)
        A.set_xlim(-lim, lim)
        A.set_ylim(-lim, lim)
        A.set_aspect("equal")
        A.set_xlabel("deflection error, m  (right positive)")
        if k == 0:
            A.set_ylabel("range error, m  (long positive)")
        A.set_title(title)
        A.legend(loc="upper right", framealpha=0.9)
        A.text(0.03, 0.03, f"{100 * np.mean(np.hypot(x, y) <= 30.0):.1f} % "
                           f"within 30 m",
               transform=A.transAxes, fontsize=7.5, color=col,
               bbox=dict(fc="white", ec=col, alpha=0.85, lw=0.6, pad=2.5))

    # The inset: the guided cloud at a scale where 30 m is a circle and not a
    # pixel. Drawn from the same array, not a re-run -- and drawn ONLY when it
    # is really a magnification. At the short engagements the guided cloud is
    # already the size of the requirement circle, and an "inset" that zoomed
    # OUT would be worse than none.
    # Sized so the CEP circle nearly fills the inset, which is also the scale
    # at which the 30 m requirement circle is legible.
    zl = 1.35 * max(cep_g, 30.0)
    if zl < 0.45 * lim:
        ins = ax[1].inset_axes([0.09, 0.63, 0.33, 0.33])
        ins.scatter(gd, gr, s=5, alpha=0.6, color=C_GUIDED, linewidths=0)
        ins.add_patch(Circle((0, 0), cep_g, fill=False, color=C_GUIDED,
                             lw=1.2))
        ins.add_patch(Circle((0, 0), 30.0, fill=False, color=C_REQ, lw=1.2,
                             ls="--"))
        ins.plot(0, 0, "+", color="k", ms=6, mew=1.0)
        ins.set_xlim(-zl, zl)
        ins.set_ylim(-zl, zl)
        ins.set_aspect("equal")
        ins.tick_params(labelsize=5.5, pad=1)
        ins.set_xticks([-round(zl / 2, -1), 0, round(zl / 2, -1)])
        ins.set_yticks([-round(zl / 2, -1), 0, round(zl / 2, -1)])
        ins.grid(alpha=0.25)
        ins.set_facecolor("white")
        ins.set_title(f"same rounds, ×{lim / zl:.0f}", fontsize=6.5, pad=2)
        ax[1].legend(loc="lower right", framealpha=0.9)

    eng = {"short": "2.0 km", "short2": "4.0 km", "middle": "9.0 km",
           "mid2": "13.0 km", "long": "15.8 km"}.get(label, label)
    fig.suptitle(f"155 mm M107 with the guidance kit — impact dispersion at "
                 f"{eng}\n"
                 f"unguided CEP {cep_u:.0f} m → guided CEP {cep_g:.1f} m; "
                 f"aleatoric terms only, epistemic band reported separately",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ===========================================================================
def figure_convergence(data, path, tag="headline"):
    """CEP against sample count, with the bootstrap interval."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label, d in sorted(data["a"][tag].items()):
        c = d["convergence"]
        n = [x["n"] for x in c]
        cep = [x["cep_m"] for x in c]
        lo = [x["cep_lo_m"] for x in c]
        hi = [x["cep_hi_m"] for x in c]
        ln, = ax.plot(n, cep, "o-", ms=3.5, lw=1.3, label=label)
        ax.fill_between(n, lo, hi, alpha=0.16, color=ln.get_color())
    ax.axhline(30.0, color=C_REQ, ls="--", lw=1.3, label="requirement, 30 m")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rounds")
    ax.set_ylabel("CEP, m")
    ax.set_title("The CEP against sample count, with its 95 % bootstrap "
                 "interval\n(independent draws, so the interval is valid)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ===========================================================================
def figure_monitor(data, path, label=None):
    """
    The false-alarm / missed-detection trade, as a curve rather than a point.

    Both axes, always, because raising the threshold to suppress the false
    alarm is the same act as degrading the detection.
    """
    d = data["d"][_pick(data["d"], label)]
    sw_all = d["sweep"]
    sw = sw_all
    # The adopted floor is swept both as a round number and as its exact
    # value (32.3 and 32.307). They are the same curve; plot one.
    seen = {}
    for r in sw:
        seen.setdefault(round(r["monitor_floor_m"], 1), []).append(r)
    sw = [r for rows in seen.values() for r in rows
          if r["monitor_floor_m"] == rows[0]["monitor_floor_m"]]
    floors = sorted({r["monitor_floor_m"] for r in sw})
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.9))

    cmap = plt.get_cmap("viridis")
    for i, fl in enumerate(floors):
        rows = sorted([r for r in sw if r["monitor_floor_m"] == fl],
                      key=lambda r: r["monitor_fraction"])
        fa = [100 * r["false_alarm_rate"] for r in rows]
        det = [100 * r["detection_rate"] for r in rows]
        col = cmap(i / max(len(floors) - 1, 1))
        ax[0].plot(fa, det, "o-", ms=3.5, lw=1.2, color=col,
                   label=f"floor {fl:g} m")
        ax[1].plot([r["monitor_fraction"] for r in rows], fa, "o-", ms=3,
                   lw=1.2, color=col)
        ax[1].plot([r["monitor_fraction"] for r in rows], det, "s--", ms=3,
                   lw=1.0, color=col, alpha=0.6)

    # The setting in force, marked. Read from the campaign rather than
    # written down, because it is the engagement's own sized floor.
    ad = d.get("adopted") or {}
    # Looked up in the UNDEDUPLICATED sweep: the deduplication above may have
    # kept the rounded floor and the adopted value is the exact one.
    op = next((r for r in sw_all
               if abs(r["monitor_fraction"] - ad.get("monitor_fraction", -1))
               < 1e-9
               and abs(r["monitor_floor_m"] - ad.get("monitor_floor_m", -1))
               < 1e-2), None)
    if op:
        ax[0].plot(100 * op["false_alarm_rate"], 100 * op["detection_rate"],
                   "*", ms=16, color="#c0392b", zorder=5,
                   label=f"in force since step 3 "
                         f"({ad['monitor_fraction']:.2f}, "
                         f"{ad['monitor_floor_m']:.0f} m)")
    # The chance diagonal. A test with no discriminating power sits on it, and
    # how far a curve lies ABOVE it is the whole of the monitor's worth.
    ax[0].plot([0, 100], [0, 100], color="#7f8c8d", lw=1.0, ls=":",
               zorder=0, label="no discriminating power")
    ax[0].set_xlabel("false alarms on HEALTHY rounds, %")
    ax[0].set_ylabel("stage-2 failures DETECTED, %")
    ax[0].set_title("The trade, with navigation in the loop")
    ax[0].legend(fontsize=6.5, loc="lower right")
    ax[1].set_xlabel("monitor_fraction")
    ax[1].set_ylabel("%")
    ax[1].set_title("solid: false alarm    dashed: detection")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ===========================================================================
def figure_atmosphere(data, path, label=None):
    """The atmospheric knowledge term against met message age."""
    d = data["c"][_pick(data["c"], label)]["ages"]
    order = [k for k in ("0h", "1h", "2h", "3h", "6h", "none") if k in d]
    x = list(range(len(order)))
    sr = [d[k]["knowledge_term"]["sigma_range_m"] for k in order]
    sd = [d[k]["knowledge_term"]["sigma_defl_m"] for k in order]
    br = [d[k]["knowledge_term"]["bias_range_m"] for k in order]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    w = 0.36
    ax[0].bar([v - w / 2 for v in x], sr, w, color=C_UNGUIDED, label="range 1σ")
    ax[0].bar([v + w / 2 for v in x], sd, w, color=C_GUIDED,
              label="deflection 1σ")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["fresh" if k == "0h" else
                           ("no met" if k == "none" else k) for k in order])
    ax[0].set_ylabel("1σ of the miss, m")
    ax[0].set_title("What the met message's error costs at the target")
    ax[0].legend()
    ax[1].plot(x, br, "o-", color=C_UNGUIDED)
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["fresh" if k == "0h" else
                           ("no met" if k == "none" else k) for k in order])
    ax[1].set_ylabel("mean range error, m")
    ax[1].set_title("and the bias it leaves")
    fig.suptitle("Atmospheric knowledge: paired against a fuze that knows the "
                 "air exactly", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ===========================================================================
def figure_band(data, path, label=None):
    """The epistemic band: the CEP at each edge, paired against the nominal."""
    v = data["b"][_pick(data["b"], label)]["variants"]
    names = [k for k in v if k != "nominal"]
    names.sort(key=lambda k: v[k]["cep_m"])
    nom = v["nominal"]["cep_m"]
    fig, ax = plt.subplots(figsize=(6.8, 0.34 * len(names) + 2.0))
    y = np.arange(len(names))
    cep = [v[k]["cep_m"] for k in names]
    lo = [v[k]["cep_m"] - v[k]["cep_lo_m"] for k in names]
    hi = [v[k]["cep_hi_m"] - v[k]["cep_m"] for k in names]
    cols = [("#7d3c98" if k.startswith("corner_") else C_GUIDED)
            for k in names]
    ax.barh(y, cep, color=cols, alpha=0.85)
    ax.errorbar(cep, y, xerr=[lo, hi], fmt="none", ecolor="k", lw=1, capsize=2)
    ax.axvline(nom, color="#b03a2e", lw=1.6,
               label=f"nominal, {nom:.1f} m")
    ax.axvline(30.0, color=C_REQ, ls="--", lw=1.4, label="requirement, 30 m")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("CEP, m")
    ax.set_title("The epistemic band — a band AROUND the CEP, not a term in "
                 "it\n(every round of a lot flies the same canards)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ===========================================================================
def figure_budget(data, path, label=None):
    """The uncorrected dispersion, decomposed into the causes that make it."""
    u = data["u"][_pick(data["u"], label)]
    comp = u["components"]
    order = [k for k in ("muzzle_velocity", "laying", "shell",
                         "met_message_error", "met_wind", "met_density",
                         "met_temperature", "all", "no_met_correction")
             if k in comp]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    y = np.arange(len(order))
    sr = [comp[k]["sigma_range_m"] for k in order]
    sd = [comp[k]["sigma_defl_m"] for k in order]
    h = 0.38
    ax.barh(y - h / 2, sr, h, color=C_UNGUIDED, label="range 1σ")
    ax.barh(y + h / 2, sd, h, color=C_GUIDED, label="deflection 1σ")
    ax.axvline(u["target"]["sigma_range_m"], color="k", ls=":", lw=1.4,
               label=f"assumed since step 2.5: {u['target']['sigma_range_m']:.0f} m")
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("_", " ") for k in order])
    ax.set_xlabel("1σ of the uncorrected impact point, m")
    ax.set_title("Where the uncorrected dispersion comes from\n"
                 "(measured for the first time; the project has assumed it "
                 "since step 2.5)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ===========================================================================
def figure_staging(data, path, label=None):
    """Staged against single stage over the distributions, not the means."""
    e = data["e"][_pick(data["e"], label)]
    rows = e["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for name, col in (("staged", C_GUIDED), ("single_stage", C_UNGUIDED)):
        v = np.array([r["miss_m"] for r in rows[name]])
        v = np.sort(v[np.isfinite(v)])
        ax[0].plot(v, np.arange(1, v.size + 1) / v.size, lw=1.6, color=col,
                   label=f"{name}, CEP {e['configs'][name]['cep_m']:.1f} m")
    ax[0].axvline(30.0, color=C_REQ, ls="--", lw=1.3, label="30 m")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("miss, m")
    ax[0].set_ylabel("fraction of rounds")
    ax[0].set_title("The whole distribution, not its median")
    ax[0].legend(fontsize=7)

    by = {r["draw"]: r for r in rows["staged"]}
    pr = [(by[r["draw"]]["miss_m"], r["miss_m"]) for r in rows["single_stage"]
          if r["draw"] in by]
    xs = np.array([p[0] for p in pr])
    ys = np.array([p[1] for p in pr])
    ax[1].scatter(xs, ys, s=10, alpha=0.55, color="#5d6d7e", linewidths=0)
    m = max(xs.max(), ys.max())
    ax[1].plot([0, m], [0, m], "k-", lw=0.9)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("staged miss, m")
    ax[1].set_ylabel("single-stage miss, m")
    ax[1].set_title("Paired on common random numbers\n"
                    "(above the line: single stage was worse on that round)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ===========================================================================
def main(argv=None):
    os.makedirs(OUT, exist_ok=True)
    d = _load()
    made = []
    # A scatter per headline tag: `headline` is the campaign at the dispersion
    # the project has assumed since step 2.5, `physical` the same thing at the
    # dispersion the modelled causes actually produce.
    extra = []
    for tag in (d.get("a") or {}):
        if tag != "headline":
            extra.append((f"mc_scatter_{tag}.png",
                          lambda dd, pp, t=tag: figure_scatter(dd, pp, tag=t),
                          ("a",)))
    for name, fn, need in extra + [
            ("mc_scatter.png", figure_scatter, ("a",)),
            ("mc_convergence.png", figure_convergence, ("a",)),
            ("mc_budget.png", figure_budget, ("u",)),
            ("mc_atmosphere.png", figure_atmosphere, ("c",)),
            ("mc_monitor.png", figure_monitor, ("d",)),
            ("mc_staging.png", figure_staging, ("e",)),
            ("mc_band.png", figure_band, ("b",)),
    ]:
        if not all(k in d for k in need):
            print(f"skip {name}: {need} not in {SRC} yet")
            continue
        try:
            fn(d, os.path.join(OUT, name))
            made.append(name)
        except (KeyError, IndexError, ValueError) as exc:
            print(f"skip {name}: {type(exc).__name__} {exc}")
    print(f"{len(made)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
