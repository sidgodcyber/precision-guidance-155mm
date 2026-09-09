"""
Figures for the roll servo. Reads docs/roll_servo.json, writes
docs/figures/roll_servo_*.png.

Three, each carrying one finding that a table states less well:

  roll_servo_envelope.png   the actuator is one-sided, and the asymmetry
                            REVERSES during the flight
  roll_servo_response.png   what the closed loop achieves, against what the
                            actuator could do if the loop asked for it
  roll_servo_duty.png       why the duty cycle does not work at short period

Run:  python -m analysis.roll_servo_figures
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


def envelope_figure(d: dict, path: str) -> str:
    """
    Left: the achievable nose-rate interval against flight time, for both
    brake capacities. The shaded band is what the actuator can reach; where it
    lies entirely below zero the angle cannot be held at all.

    Right: the peak angular ACCELERATION each way, which is what actually sets
    how long a step takes -- the loop never asks for anything near the steady
    rates. The two curves cross, and that crossing is why a 90 degree move is
    quicker backwards early in the flight and quicker forwards later.
    """
    plt = _mpl()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ta = d["task_a"]
    t_dep = ta["deploy_time"]

    colours = {"0.50": "tab:red", "0.75": "tab:blue"}
    for key in sorted(ta["variants"], key=float):
        v = ta["variants"][key]
        rows = v["rows"]
        t = np.array([r["time_s"] - t_dep for r in rows])
        lo = np.array([r["rate_free_degs"] for r in rows])
        hi = np.array([r["rate_full_degs"] for r in rows])
        col = colours.get(key, "tab:green")
        ax.fill_between(t, lo, hi, color=col, alpha=0.18)
        ax.plot(t, hi, color=col, lw=1.6,
                label=f"full brake, {v['brake_max_Nm']:.2f} N m")
        ax.plot(t, lo, color=col, lw=1.6, ls="--",
                label=f"released, {v['brake_max_Nm']:.2f} N m")
        if v["dead_window_s"]:
            ax.axvspan(0.0, v["dead_window_s"], color=col, alpha=0.10)
            ax.annotate(f"cannot hold\n(first {v['dead_window_s']:.1f} s)",
                        (v["dead_window_s"], 0.0), fontsize=8, color=col,
                        textcoords="offset points", xytext=(6, 40))

    ax.axhline(0, color="0.4", lw=0.9)
    ax.set_xlabel("time since deployment, s")
    ax.set_ylabel("achievable nose rate, deg/s")
    ax.set_title("The actuator is one-sided.\nBelow the dashed line is "
                 "unreachable; above the solid line is unreachable.",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    for key in sorted(ta["variants"], key=float):
        v = ta["variants"][key]
        rows = v["rows"]
        t = np.array([r["time_s"] - t_dep for r in rows])
        af = np.array([r["accel_forward_rads2"] for r in rows])
        ar = np.array([r["accel_return_rads2"] for r in rows])
        col = colours.get(key, "tab:green")
        ax2.plot(t, af, color=col, lw=1.6,
                 label=f"forward, {v['brake_max_Nm']:.2f} N m")
        ax2.plot(t, -ar, color=col, lw=1.6, ls="--",
                 label=f"return (magnitude), {v['brake_max_Nm']:.2f} N m")
        cross = np.nonzero(np.diff(np.sign(af + ar)))[0]
        if cross.size:
            ax2.axvline(t[cross[0]], color=col, lw=0.9, ls=":")
            ax2.annotate(f"equal both ways at {t[cross[0]]:.1f} s",
                         (t[cross[0]], af.max()), fontsize=8, color=col,
                         ha="left", textcoords="offset points",
                         xytext=(5, -14 - 14 * int(key == "0.50")))
    ax2.axhline(0, color="0.4", lw=0.9)
    ax2.set_xlabel("time since deployment, s")
    ax2.set_ylabel("peak angular acceleration, rad/s$^2$")
    ax2.set_title("Which direction is quicker REVERSES in flight",
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def response_figure(d: dict, path: str) -> str:
    """
    Left: time to move 90 degrees each way, against flight time, with the
    actuator-limited floor beneath it. The gap is the margin the loop
    deliberately leaves, and it is set by the brake coil, not by the gains.

    Right: the closed-loop bandwidth against the brake coil time constant.
    """
    plt = _mpl()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    tc = d["task_c"]["sized_brake"]
    for direction, colour, marker in (("forward", "tab:blue", "o"),
                                      ("return", "tab:red", "s")):
        pts = [s for s in tc["steps"]
               if s["direction"] == direction and abs(s["step_deg"]) == 90.0
               and s["rise_s"] is not None]
        pts.sort(key=lambda s: s["since_deploy_s"])
        ax.plot([s["since_deploy_s"] for s in pts],
                [s["rise_s"] for s in pts], marker + "-", color=colour,
                ms=4, lw=1.5, label=f"{direction}, closed loop")
        # The floor: 90 degrees at the steady rate limit.
        floor = [(s["since_deploy_s"],
                  90.0 / abs(s["rate_limit_degs"]))
                 for s in pts if abs(s["rate_limit_degs"]) > 1e-6]
        ax.plot([f[0] for f in floor], [f[1] for f in floor], ":",
                color=colour, lw=1.4, label=f"{direction}, actuator floor")
    ax.set_yscale("log")
    ax.set_ylim(4e-3, 1.2)
    ax.set_yticks([0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    ax.set_yticklabels(["0.005", "0.01", "0.05", "0.1", "0.5", "1.0"])
    ax.set_xlabel("time since deployment, s")
    ax.set_ylabel("time to move 90 degrees, s")
    ax.set_title("The loop is bandwidth-limited, not rate-limited.\n"
                 "The actuator could go an order of magnitude faster.",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    al = d.get("actuator_lag")
    if al:
        taus = [1e3 * r["actuator_tau_s"] for r in al["rows"]]
        bws = [r["bandwidth_hz"] for r in al["rows"]]
        ax2.plot(taus, bws, "o-", color="tab:purple", ms=5, lw=1.6)
        if taus and taus[0] == 0.0:
            ax2.annotate("no lag:\nsample-rate limited", (taus[0], bws[0]),
                         fontsize=8, ha="left",
                         textcoords="offset points", xytext=(6, -22))
        nominal = 10.0
        for x, y in zip(taus, bws):
            if abs(x - nominal) < 1e-6 and y:
                ax2.plot([x], [y], "o", ms=11, mfc="none", mec="k", mew=1.6)
                ax2.annotate(f"as designed\n{y:.2f} Hz", (x, y), fontsize=8,
                             textcoords="offset points", xytext=(10, 6))
        ax2.set_xlabel("brake coil time constant, ms")
        ax2.set_ylabel("closed-loop bandwidth, Hz")
        ax2.set_title("The lever on servo speed is the coil,\nnot the gains",
                      fontsize=10)
        ax2.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def duty_figure(d: dict, path: str) -> str:
    """
    Left: delivered correction against commanded duty fraction. If the scheme
    worked it would follow the diagonal.

    Right: peak angle of attack against switching period, with the band the
    slow yawing mode occupies over the guided phase. Every period inside that
    band pumps the shell's precession, and the induced drag of the resulting
    yaw is paid for over the whole remaining flight.
    """
    plt = _mpl()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    td = d["task_d"]

    periods = sorted({r["period_s"] for r in td["duty_rows"]})
    colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(periods)))
    for period, col in zip(periods, colours):
        rows = sorted((r for r in td["duty_rows"] if r["period_s"] == period),
                      key=lambda r: r["duty"])
        ax.plot([r["duty"] for r in rows],
                [r["fraction_of_full"] for r in rows], "o-", color=col,
                ms=4, lw=1.4, label=f"period {period:g} s")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="if it were linear")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("commanded duty fraction")
    ax.set_ylabel("delivered correction / full hold")
    ax.set_title("Duty fraction against delivered correction", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    spec = td.get("yaw_spectrum", [])
    if spec:
        by_period = {}
        for s in spec:
            if s.get("period_s"):
                by_period.setdefault(s["period_s"], []).append(
                    s["max_total_aoa_deg"])
        xs = sorted(by_period)
        ax2.plot(xs, [max(by_period[x]) for x in xs], "o-", color="tab:red",
                 ms=5, lw=1.6, label="peak angle of attack")
    ep = d.get("epicyclic")
    if ep:
        lo, hi = 1.0 / ep["slow_hz_max"], 1.0 / ep["slow_hz_min"]
        ax2.axvspan(lo, hi, color="tab:orange", alpha=0.20)
        ax2.annotate("slow yawing mode\n"
                     f"{ep['slow_hz_min']:.2f}-{ep['slow_hz_max']:.2f} Hz",
                     (0.5 * (lo + hi), 0.0), fontsize=8, ha="center",
                     textcoords="offset points", xytext=(0, 14))
    ideal = d.get("servo_cost", {}).get("ideal_max_aoa_deg")
    if ideal and xs:
        ax2.axhline(ideal, color="0.4", ls="--", lw=1.2)
        ax2.annotate(f"ideal kinematic hold, {ideal:.2f} deg",
                     (xs[-1], ideal), fontsize=8, ha="right",
                     textcoords="offset points", xytext=(0, 5))
        ax2.set_ylim(bottom=0.9 * ideal)
    ax2.set_xscale("log")
    ax2.set_xlabel("switching period, s")
    ax2.set_ylabel("peak total angle of attack, deg")
    ax2.set_title("Switching inside the slow-mode band pumps the yaw",
                  fontsize=10)
    ax2.grid(alpha=0.25, which="both")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--servo", default=os.path.join("docs", "roll_servo.json"))
    ap.add_argument("--outdir", default=os.path.join("docs", "figures"))
    args = ap.parse_args(argv)

    with open(args.servo, encoding="utf-8") as f:
        d = json.load(f)

    made = [envelope_figure(d, os.path.join(args.outdir, "roll_servo_envelope.png")),
            response_figure(d, os.path.join(args.outdir, "roll_servo_response.png"))]
    if "task_d" in d:
        made.append(duty_figure(d, os.path.join(args.outdir, "roll_servo_duty.png")))
    for m in made:
        print(f"wrote {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
