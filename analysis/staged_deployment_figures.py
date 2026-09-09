"""
Figures for the staged deployment. Reads docs/staged_deployment.json, writes
docs/figures/staged_*.png.

Three, each carrying one finding that a table states less well:

  staged_distributions.png  the two configurations as DISTRIBUTIONS over 24
                            deployment phases, not as two means. A
                            configuration that raises the mean while widening
                            the spread may be worse for CEP than one that does
                            not, and only this shows it.
  staged_delay.png          the trade: cleaner capture bought with a shorter
                            correction window, and where the optimum is.
  staged_despin.png         why staging costs time at all -- the two-panel
                            despin, at twice the time constant to twice the
                            equilibrium rate.

Run:  python -m analysis.staged_deployment_figures
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


SINGLE = "#B4432E"
STAGED = "#2E6DB4"
IDEAL = "#555555"


def _ecdf(ax, values, color, label):
    v = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, v.size + 1) / v.size
    ax.step(v, y, where="post", color=color, lw=1.8, label=label)
    ax.plot([v.mean(), v.mean()], [0, 1], color=color, lw=0.9, ls="--", alpha=0.7)


def distributions_figure(d: dict, path: str) -> str:
    plt = _mpl()
    a = d["ensemble_single"]
    b = d["ensemble_staged"]
    n = d["configuration"]["n_phases"]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))

    # -- retained rms amplitude, per phase -------------------------------
    ax = axes[0][0]
    _ecdf(ax, 100 * np.array(a["retained_rms"]["values"]), SINGLE, "single stage")
    _ecdf(ax, 100 * np.array(b["retained_rms"]["values"]), STAGED, "staged")
    ax.set_xlabel("retained rms amplitude, % of the ideal hold")
    ax.set_ylabel("fraction of deployment phases")
    ax.set_title(f"Reachable set, {n} phases each\n(dashed = mean)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # -- peak total angle of attack, per run ------------------------------
    ax = axes[0][1]
    _ecdf(ax, a["aoa_per_run_deg"]["values"], SINGLE, "single stage")
    _ecdf(ax, b["aoa_per_run_deg"]["values"], STAGED, "staged")
    if "ideal" in d:
        iv = d["ideal"]["aoa_per_run_deg"]["mean"]
        ax.axvline(iv, color=IDEAL, lw=1.2, ls=":", label=f"ideal hold {iv:.2f}°")
    ax.axvline(0.76, color="#8A6D3B", lw=1.2,
               label="step 1 validated 0.76°")
    ax.set_xlabel("peak total angle of attack, deg")
    ax.set_ylabel("fraction of runs")
    ax.set_title("Angle of attack, every run", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # -- retained rms against deployment phase ----------------------------
    ax = axes[1][0]
    for e, c, lab in ((a, SINGLE, "single stage"), (b, STAGED, "staged")):
        ph = [p["body_roll_deg"] for p in e["per_phase"]]
        rr = [100 * p["retained_rms"] for p in e["per_phase"]]
        order = np.argsort(ph)
        ax.plot(np.array(ph)[order], np.array(rr)[order], "o-", color=c,
                ms=3.5, lw=1.2, label=lab)
    ax.set_xlabel("body roll angle at deployment, deg")
    ax.set_ylabel("retained rms, %")
    ax.set_title("The quantity the fuze cannot choose", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # -- dead time ---------------------------------------------------------
    ax = axes[1][1]
    _ecdf(ax, a["steering_ready_per_run_s"]["values"], SINGLE, "single stage")
    _ecdf(ax, b["steering_ready_per_run_s"]["values"], STAGED, "staged")
    ax.set_xlabel("time from deployment to a correctly directed force, s")
    ax.set_ylabel("fraction of runs")
    ax.set_title("What the cleaner capture is bought with", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle("Staged against single-stage deployment, 0.85 N m brake",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def delay_figure(d: dict, path: str) -> str:
    plt = _mpl()
    s = d["delay_sweep"]
    rows = s["by_delay"]
    delay = np.array([r["delay_s"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))

    ax = axes[0]
    mean = np.array([100 * r["retained_rms"]["mean"] for r in rows])
    lo = np.array([100 * r["retained_rms"]["min"] for r in rows])
    hi = np.array([100 * r["retained_rms"]["max"] for r in rows])
    ax.fill_between(delay, lo, hi, color=STAGED, alpha=0.18, label="min-max over phases")
    ax.plot(delay, mean, "o-", color=STAGED, lw=1.6, label="mean")
    ax.set_xlabel("stage-2 delay, s")
    ax.set_ylabel("retained rms amplitude, %")
    ax.set_title("What the delay costs in reachable set", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    mean = np.array([r["aoa_per_run_deg"]["mean"] for r in rows])
    hi = np.array([r["aoa_per_run_deg"]["max"] for r in rows])
    ax.plot(delay, mean, "o-", color=STAGED, lw=1.6, label="mean")
    ax.plot(delay, hi, "s--", color=STAGED, lw=1.0, ms=4, alpha=0.7, label="worst")
    ax.axhline(0.76, color="#8A6D3B", lw=1.2, label="step 1 validated 0.76°")
    ax.set_xlabel("stage-2 delay, s")
    ax.set_ylabel("peak total angle of attack, deg")
    ax.set_title("What it buys", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    frac = np.array([100 * r["fraction_of_guided_phase"] for r in rows])
    ready = np.array([r["steering_ready_per_run_s"]["mean"] for r in rows])
    ax.plot(delay, frac, "o-", color="#666666", lw=1.4,
            label="delay, % of guided phase")
    ax.plot(delay, ready, "^-", color=SINGLE, lw=1.4,
            label="mean time to a directed force, s")
    ax.set_xlabel("stage-2 delay, s")
    ax.set_title("What it consumes", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"The stage-2 delay trade, guided phase {s['guided_phase_s']:.1f} s",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def despin_figure(d: dict, path: str) -> str:
    plt = _mpl()
    b = d["task_b"]
    t4 = [r["t_after_deploy_s"] for r in b["four_panel"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

    ax = axes[0]
    ax.plot(t4, [r["time_constant_s"] for r in b["four_panel"]], "o-",
            color=SINGLE, lw=1.6, label="four panels (stage 2 / single stage)")
    ax.plot(t4, [r["time_constant_s"] for r in b["two_panel"]], "s-",
            color=STAGED, lw=1.6, label="two panels (stage 1)")
    ax.set_xlabel("time after deployment, s")
    ax.set_ylabel("nose rate time constant  I_n / c_tot, s")
    ax.set_title("Halving the damping doubles the time constant", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for rows, c, lab, m in ((b["four_panel"], SINGLE, "four panels", "o"),
                            (b["two_panel"], STAGED, "two panels", "s")):
        ax.fill_between(t4, [r["p_free_rads"] for r in rows],
                        [r["p_full_rads"] for r in rows], color=c, alpha=0.15)
        ax.plot(t4, [r["p_free_rads"] for r in rows], m + "-", color=c, lw=1.3,
                ms=3.5, label=f"{lab}: free / full brake")
        ax.plot(t4, [r["p_full_rads"] for r in rows], m + "-", color=c, lw=1.3,
                ms=3.5)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xlabel("time after deployment, s")
    ax.set_ylabel("achievable nose inertial rate, rad/s")
    ax.set_title("Both ends roughly double, and the interval\n"
                 "still brackets zero, which is why staging is possible",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Stage 1: the cant pair alone, brake {b['brake_max_Nm']:.2f} N m",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="docs/staged_deployment.json")
    ap.add_argument("--outdir", default="docs/figures")
    args = ap.parse_args(argv)

    with open(args.results, encoding="utf-8") as f:
        d = json.load(f)
    os.makedirs(args.outdir, exist_ok=True)

    made = []
    if "task_b" in d:
        made.append(despin_figure(d, os.path.join(args.outdir, "staged_despin.png")))
    if "ensemble_single" in d and "ensemble_staged" in d:
        made.append(distributions_figure(
            d, os.path.join(args.outdir, "staged_distributions.png")))
    if "delay_sweep" in d:
        made.append(delay_figure(d, os.path.join(args.outdir, "staged_delay.png")))
    for p in made:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
