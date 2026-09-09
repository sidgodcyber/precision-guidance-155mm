"""
Figures for step 5.

Reads docs/nav_sensors.json, docs/nav_consistency.json and docs/nav_cep.json
and writes docs/figures/nav_*.png. Produces nothing that is not already a
number in one of those files -- a figure here exists to show a SHAPE that a
table cannot, and where a table is enough there is no figure.

Run:  python -m analysis.nav_figures
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

OUT = "docs/figures"
plt.rcParams.update({"figure.dpi": 130, "font.size": 8,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.titlesize": 9, "legend.fontsize": 7})


def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


# ===========================================================================
def figure_saturation(d, path):
    """
    The measurement that decides the layout: what a commodity gyro sees on the
    body and in the despun section.
    """
    s = d["saturation"]
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.1))
    fs = np.array([float(k) for k in s["classes"]])
    body = np.array([s["classes"][k]["body"]["fraction_saturated"]
                     for k in s["classes"]])
    des = np.array([s["classes"][k]["despun"]["fraction_saturated_after_despin"]
                    for k in s["classes"]])
    ax[0].semilogx(fs, 100 * body, "o-", color="#c0392b", label="body-mounted")
    ax[0].semilogx(fs, 100 * des, "s-", color="#27ae60",
                   label="despun, after t_dep + 2.2 s")
    ax[0].set_xlabel("gyro full scale, deg/s")
    ax[0].set_ylabel("% of flight pinned at full scale")
    ax[0].set_title("A body-mounted roll gyro is pinned for the whole flight")
    ax[0].axvspan(2000, 4000, color="#3498db", alpha=0.12)
    ax[0].text(2800, 55, "commodity\nband", ha="center", fontsize=7,
               color="#2471a3")
    ax[0].legend(loc="center left")
    ax[0].set_ylim(-4, 104)
    ax[0].set_xticks(fs)
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].tick_params(axis="x", labelrotation=45, labelsize=7)

    off = np.array([float(k) for k in d["lever"]["offsets"]])
    cb = np.array([d["lever"]["offsets"][k]["centripetal_body_max_g"]
                   for k in d["lever"]["offsets"]])
    cd = np.array([d["lever"]["offsets"][k]["centripetal_despun_rms"] / 9.80665
                   for k in d["lever"]["offsets"]])
    tang = np.array([d["lever"]["offsets"][k]["tangential_rms"] / 9.80665
                     for k in d["lever"]["offsets"]])
    ax[1].loglog(off * 1000, cb, "o-", color="#c0392b",
                 label="body: centripetal, max")
    ax[1].loglog(off * 1000, cd, "s-", color="#27ae60",
                 label="despun: centripetal, rms")
    ax[1].loglog(off * 1000, tang, "^-", color="#e67e22",
                 label="despun: tangential, rms")
    ax[1].axhline(40.0, color="k", ls="--", lw=1)
    ax[1].text(3.4, 46, "+-40 g part range", fontsize=7)
    ax[1].axhline(d["lever"]["flight_specific_force_mean"] / 9.80665,
                  color="#7f8c8d", ls=":", lw=1)
    ax[1].text(3.4, 0.63, "flight signal", fontsize=7, color="#7f8c8d")
    ax[1].set_xlabel("IMU transverse offset from the spin axis, mm")
    ax[1].set_ylabel("specific force, g")
    ax[1].set_title("... and so is a body-mounted accelerometer")
    ax[1].legend(loc="lower right")
    ax[1].set_xticks(off * 1000)
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_observability(d, path):
    """Roll observability round the compass, and what it costs."""
    rows = d["observability"]["rows"]
    az = np.array([r["azimuth_deg"] for r in rows])
    sig = np.array([r["roll_sigma_median_deg"] for r in rows])
    frac = np.array([100 * r["fraction_below_15deg"] for r in rows])
    med = np.array([r["angle_deg"]["median"] for r in rows])

    fig = plt.figure(figsize=(8.2, 3.3))
    axp = fig.add_subplot(1, 2, 1, projection="polar")
    th = np.radians(np.append(az, az[0]))
    axp.plot(th, np.append(sig, sig[0]), "o-", color="#2c3e50")
    axp.fill(th, np.append(sig, sig[0]), color="#3498db", alpha=0.18)
    axp.set_theta_zero_location("N")
    axp.set_theta_direction(-1)
    axp.set_title("roll 1$\\sigma$ from a 1.5 % calibration residual, deg\n"
                  "(by firing azimuth)", pad=14)
    axp.plot([0], [sig[0]], "*", ms=13, color="#c0392b", zorder=5)
    axp.annotate("adopted", xy=(0, sig[0]), xytext=(0.55, sig[0] * 0.72),
                 fontsize=7, color="#c0392b")
    axp.set_rlabel_position(135)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(az, med, "o-", color="#2c3e50", label="median field-to-axis angle")
    ax.set_xlabel("firing azimuth, deg east of true north")
    ax.set_ylabel("degrees", color="#2c3e50")
    ax2 = ax.twinx()
    ax2.bar(az, frac, width=18, color="#c0392b", alpha=0.35)
    ax2.set_ylabel("% of flight below 15 deg", color="#c0392b")
    ax2.grid(False)
    ax.set_title("The adopted engagement is the worst azimuth")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_nees(d, path):
    """
    NEES against its chi-square bounds, over the campaign and per engagement.
    """
    c = d["campaign"]
    fig, ax = plt.subplots(1, 3, figsize=(9.4, 3.1))
    for a, key, dof, name in ((ax[0], "nees_pv", 6, "position + velocity"),
                              (ax[1], "nees_nav", 9, "9 navigation states"),
                              (ax[2], "nees", 15, "all 15 states")):
        vals = np.concatenate([np.asarray(r[key], dtype=float)
                               for r in c["rows"]])
        t = np.concatenate([np.asarray(r["t"], dtype=float) - r["t"][0]
                            for r in c["rows"]])
        a.plot(t, vals, ".", ms=1.0, alpha=0.16, color="#2c3e50")
        r = c[key]
        a.axhline(dof, color="#27ae60", lw=1.4, label=f"expected = {dof}")
        a.axhspan(r["bound_lo"], r["bound_hi"], color="#27ae60", alpha=0.18,
                  label="95 % bound on the mean")
        a.axhline(r["mean"], color="#c0392b", lw=1.4,
                  label=f"measured mean = {r['mean']:.2f}")
        a.set_yscale("log")
        a.set_ylim(0.05, 400)
        a.set_xlabel("s after deployment")
        a.set_title(f"NEES, {name}\n"
                    f"{'PASS' if r['consistent'] else 'FAIL'} "
                    f"(ratio {r['ratio_to_dof']:.2f})")
        a.legend(loc="upper right", framealpha=0.9)
    ax[0].set_ylabel("NEES")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_nis(d, path):
    """NIS per stream, and the whiteness that NIS cannot see."""
    c = d["campaign"]
    streams = [s for s in ("gnss_pos", "gnss_vel", "mag", "cn0", "align")
               if s in c["nis"]]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))

    x = np.arange(len(streams))
    ratio = [c["nis"][s]["ratio_to_dof"] for s in streams]
    colours = ["#27ae60" if c["nis"][s]["consistent"] else "#c0392b"
               for s in streams]
    ax[0].bar(x, ratio, color=colours, alpha=0.8)
    ax[0].axhline(1.0, color="k", lw=1.2)
    ax[0].axhspan(c["nis"][streams[0]]["bound_lo"] / c["nis"][streams[0]]["dof"],
                  c["nis"][streams[0]]["bound_hi"] / c["nis"][streams[0]]["dof"],
                  color="#27ae60", alpha=0.15)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(streams, rotation=20)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("mean NIS / dof")
    ax[0].set_title("NIS: 1.0 is consistent, above is overconfident")
    for xi, r in zip(x, ratio):
        ax[0].text(xi, r * 1.15, f"{r:.2f}", ha="center", fontsize=7)

    for s in streams:
        w = c["whiteness"].get(s)
        if not w:
            continue
        lags = np.arange(1, len(w["acf_mean"]) + 1)
        ax[1].plot(lags, w["acf_mean"], "o-", ms=3, label=s)
    band = np.mean([c["whiteness"][s]["band"] for s in streams
                    if s in c["whiteness"]])
    ax[1].axhspan(-band, band, color="#27ae60", alpha=0.18,
                  label="white, 95 % band")
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set_xlabel("lag")
    ax[1].set_ylabel("innovation autocorrelation")
    ax[1].set_title("Whiteness: the magnetometer and the alignment\n"
                    "reference are NOT white, and neither is noise")
    ax[1].legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_cep(d, path):
    """The navigation contribution, per engagement and per axis."""
    f = d["task_f"]["engagements"]
    labels = list(f)
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2))

    x = np.arange(len(labels))
    tr = [f[k]["truth_cep"]["cep_m"] for k in labels]
    nv_ = [f[k]["nav_cep"]["cep_m"] for k in labels]
    ax[0].bar(x - 0.2, tr, 0.4, label="truth-fed", color="#2c3e50")
    ax[0].bar(x + 0.2, nv_, 0.4, label="navigation-fed", color="#e67e22")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=15)
    ax[0].set_ylabel("CEP, m")
    ax[0].set_title("CEP with and without navigation error")
    ax[0].legend()

    sr = [f[k]["delta"]["range"]["sigma_m"] for k in labels]
    sd = [f[k]["delta"]["deflection"]["sigma_m"] for k in labels]
    br = [f[k]["delta"]["range"]["bias_m"] for k in labels]
    bd = [f[k]["delta"]["deflection"]["bias_m"] for k in labels]
    ax[1].bar(x - 0.2, sr, 0.4, label="range 1$\\sigma$", color="#2980b9")
    ax[1].bar(x + 0.2, sd, 0.4, label="deflection 1$\\sigma$", color="#16a085")
    ax[1].plot(x - 0.2, br, "kv", ms=5, label="range bias")
    ax[1].plot(x + 0.2, bd, "k^", ms=5, label="deflection bias")
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=15)
    ax[1].set_ylabel("m")
    ax[1].set_title("The navigation contribution, paired")
    ax[1].legend(fontsize=6)

    main = "long" if "long" in f else labels[-1]
    p = f[main]["delta"]["pairs"]
    dr = np.array([q["d_range_m"] for q in p])
    dd = np.array([q["d_defl_m"] for q in p])
    ax[2].plot(dd, dr, "o", ms=3, alpha=0.6, color="#8e44ad")
    ax[2].axhline(0, color="k", lw=0.8); ax[2].axvline(0, color="k", lw=0.8)
    th = np.linspace(0, 2 * math.pi, 200)
    ax[2].plot(dd.std(ddof=1) * np.cos(th), dr.std(ddof=1) * np.sin(th),
               "-", color="#c0392b", lw=1.4, label="1$\\sigma$ per axis")
    ax[2].set_xlabel("deflection displacement, m")
    ax[2].set_ylabel("range displacement, m")
    ax[2].set_title(f"Per-round displacement, {main}")
    ax[2].legend()
    ax[2].set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_degradation(d, path):
    """The Task G ladder."""
    g = d["task_g"]["cases"]
    nominal = g["nominal"]["cep"]["cep_m"]
    keys = [k for k in g if k.startswith("outage_")]
    keys.sort(key=lambda k: (g[k]["duration_s"], g[k]["start_fraction"]))
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))

    durs = sorted({g[k]["duration_s"] for k in keys})
    starts = sorted({g[k]["start_fraction"] for k in keys})
    for dur in durs:
        y = [g[f"outage_{dur:.0f}s_at_{int(s * 100)}pct"]["cep"]["cep_m"]
             for s in starts]
        ax[0].plot([100 * s for s in starts], y, "o-", label=f"{dur:.0f} s outage")
    ax[0].axhline(nominal, color="k", ls="--", lw=1.2,
                  label=f"nominal {nominal:.1f} m")
    ax[0].set_xlabel("outage start, % of the guided phase")
    ax[0].set_ylabel("CEP, m")
    ax[0].set_title("GNSS outage: when it happens matters more than how long")
    ax[0].legend(fontsize=7)

    hard = [("nominal", nominal)]
    for k, name in (("cold_start", "cold start"),
                    ("mag_uncalibrated", "magnetometer\nuncalibrated"),
                    ("gnss_denied", "GNSS denied")):
        if k in g:
            hard.append((name, g[k]["cep"]["cep_m"]))
    x = np.arange(len(hard))
    ax[1].bar(x, [h[1] for h in hard],
              color=["#27ae60"] + ["#c0392b"] * (len(hard) - 1), alpha=0.85)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([h[0] for h in hard], rotation=12, fontsize=7)
    ax[1].set_ylabel("CEP, m")
    ax[1].set_title("The hard failures")
    for xi, h in zip(x, hard):
        ax[1].text(xi, h[1] * 1.02, f"{h[1]:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_warm_start(d, path):
    """Task D: does the warm start move the solution inside the deadline?"""
    t = d["task_d"]
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.1))
    rows = t["rows"]
    for warm, colour, name in ((True, "#27ae60", "warm start (gun data)"),
                               (False, "#c0392b", "cold start (GNSS only)")):
        sub = [r for r in rows if r["warm_start"] == warm]
        u = np.array([r["first_usable_s"] if r["first_usable_s"] is not None
                      else np.nan for r in sub])
        u = np.sort(u[np.isfinite(u)])
        ax[0].step(u, np.arange(1, u.size + 1) / max(u.size, 1), where="post",
                   color=colour, lw=1.6, label=name)
    ax[0].axvline(t["deadline_s"], color="k", ls="--", lw=1.2)
    ax[0].text(t["deadline_s"] + 0.2, 0.35,
               "t_dep + 3 s\nthe earliest the law\nmay command", fontsize=7)
    ax[0].set_xlabel("time from muzzle exit to a usable solution, s")
    ax[0].set_ylabel("cumulative fraction of rounds")
    ax[0].set_title("Time to a usable navigation solution")
    ax[0].legend(loc="lower right")

    fix = np.sort(np.array([r["gnss_reacquire_s"] for r in rows]))
    ax[1].step(fix, np.arange(1, fix.size + 1) / fix.size, where="post",
               color="#2c3e50", lw=1.6)
    ax[1].axvline(t["deadline_s"], color="k", ls="--", lw=1.2)
    frac = float(np.mean(fix > t["deadline_s"]))
    ax[1].set_xlabel("GNSS reacquisition after launch, s")
    ax[1].set_ylabel("cumulative fraction")
    ax[1].set_title(f"The LOW-confidence number the answer turns on\n"
                    f"{100 * frac:.0f} % of rounds reacquire after the deadline")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ===========================================================================
def main(argv=None) -> int:
    os.makedirs(OUT, exist_ok=True)
    made = []
    s = _load("docs/nav_sensors.json")
    if s:
        figure_saturation(s, f"{OUT}/nav_saturation.png"); made.append("saturation")
        figure_observability(s, f"{OUT}/nav_observability.png")
        made.append("observability")
    c = _load("docs/nav_consistency.json")
    if c and "campaign" in c:
        figure_nees(c, f"{OUT}/nav_nees.png"); made.append("nees")
        figure_nis(c, f"{OUT}/nav_nis.png"); made.append("nis")
    p = _load("docs/nav_cep.json")
    if p and "task_f" in p:
        figure_cep(p, f"{OUT}/nav_cep.png"); made.append("cep")
    if p and "task_g" in p:
        figure_degradation(p, f"{OUT}/nav_degradation.png")
        made.append("degradation")
    if p and "task_d" in p:
        figure_warm_start(p, f"{OUT}/nav_warm_start.png"); made.append("warm start")
    print("wrote:", ", ".join(made) if made else "nothing (no JSON found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
