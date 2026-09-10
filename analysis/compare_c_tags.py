"""
Compare two Task C campaigns that differ only in the dispersion top-up.

The two campaigns are paired PER ROUND, not merely per campaign: in
`sample_round` the atmosphere, the met message, the fire-control lay, the fuze
setting, the base muzzle-velocity draw, the laying errors, the mass, both
inertias and the fuze phase are all drawn before the top-up is applied, and
the top-up's two draws are the last taken from the round generator. Removing
them shifts nothing else. So round i in the un-inflated campaign is round i in
the headline campaign with two offsets absent.

That licenses a difference-of-differences:

    D_i = (miss_2h - miss_perfect)_uninflated - (miss_2h - miss_perfect)_headline

and a bootstrap that resamples ROUND INDICES JOINTLY across both campaigns,
which is what makes the interval on the difference between the two knowledge
terms far tighter than comparing two independently-quoted sigmas.

Read-only. Writes nothing.
"""

from __future__ import annotations

import json
import math

import numpy as np

SRC = "docs/monte_carlo.json"
BOOT = 20000
SEED = 606


def load(path=SRC):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def arms(campaign, age="2h"):
    """{draw: (d_range, d_defl, terminal_saturated)} for one campaign."""
    rows = campaign["rows"]
    ref = {r["draw"]: r for r in rows["perfect"]}
    out = {}
    for r in rows[age]:
        b = ref.get(r["draw"])
        if b is None:
            continue
        out[r["draw"]] = (r["miss_range_m"] - b["miss_range_m"],
                          r["miss_defl_m"] - b["miss_defl_m"],
                          bool(r.get("g_terminal_saturated")),
                          bool(b.get("g_terminal_saturated")))
    return out


def sigma(v):
    return float(np.std(v, ddof=1))


def boot_sigma(v, n_boot=BOOT, seed=SEED):
    """Percentile interval on one campaign's own 1 sigma."""
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    s = v[idx].std(axis=1, ddof=1)
    return {"sigma_m": sigma(v), "n": int(v.size),
            "lo_m": float(np.percentile(s, 2.5)),
            "hi_m": float(np.percentile(s, 97.5)),
            "se_m": float(s.std(ddof=1))}


def boot_paired_delta(a, b, n_boot=BOOT, seed=SEED):
    """
    Interval on sigma(a) - sigma(b), resampling round indices JOINTLY.

    `a` and `b` are the same rounds under the two campaigns, in the same
    order. Resampling them together keeps the pairing inside every bootstrap
    replicate; resampling independently would throw it away and give the much
    wider interval that comparing two published sigmas gives.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    d = a[idx].std(axis=1, ddof=1) - b[idx].std(axis=1, ddof=1)
    return {"delta_sigma_m": sigma(a) - sigma(b),
            "lo_m": float(np.percentile(d, 2.5)),
            "hi_m": float(np.percentile(d, 97.5)),
            "se_m": float(d.std(ddof=1)),
            "p_two_sided_ge_0": float(min((d >= 0).mean(), (d <= 0).mean()) * 2)}


def report(path=SRC, tag_a="physical", tag_b="headline", age="2h"):
    d = load(path)
    c = d["c"]
    if tag_a not in c:
        raise SystemExit(f"no {tag_a!r} campaign yet; have {sorted(c)}")
    A = arms(c[tag_a]["long"], age)
    B = arms(c[tag_b]["long"], age)
    common = sorted(set(A) & set(B))
    print(f"campaigns: {tag_a} (n={len(A)})  vs  {tag_b} (n={len(B)})")
    print(f"rounds present in both: {len(common)}")
    print()

    ar = np.array([A[i][0] for i in common])
    ad = np.array([A[i][1] for i in common])
    br = np.array([B[i][0] for i in common])
    bd = np.array([B[i][1] for i in common])

    print("THE KNOWLEDGE TERM, each campaign on its own")
    print(f"  {'':<28s} {'sigma':>9s} {'95% CI':>20s} {'n':>5s}")
    for name, v in ((f"{tag_b} (top-up)  range", br),
                    (f"{tag_a} (no top-up) range", ar),
                    (f"{tag_b} (top-up)  defl ", bd),
                    (f"{tag_a} (no top-up) defl ", ad)):
        s = boot_sigma(v)
        print(f"  {name:<28s} {s['sigma_m']:9.2f} "
              f"[{s['lo_m']:8.2f},{s['hi_m']:8.2f}] {s['n']:5d}")
    print()

    print("THE DIFFERENCE, paired (resampling rounds jointly)")
    for axis, a_, b_ in (("range", ar, br), ("deflection", ad, bd)):
        p = boot_paired_delta(a_, b_)
        print(f"  {axis:<10s} sigma(no top-up) - sigma(top-up) = "
              f"{p['delta_sigma_m']:+8.2f} m  "
              f"[{p['lo_m']:+7.2f},{p['hi_m']:+7.2f}]  "
              f"se {p['se_m']:5.2f}  p={p['p_two_sided_ge_0']:.4f}")
    print()

    print("DIFFERENCE OF DIFFERENCES, per round")
    for axis, a_, b_ in (("range", ar, br), ("deflection", ad, bd)):
        D = a_ - b_
        print(f"  {axis:<10s} mean {D.mean():+8.2f} m   sd {sigma(D):8.2f} m   "
              f"median {np.median(D):+8.2f} m   "
              f"|D| p90 {np.percentile(np.abs(D), 90):7.2f} m")
    print()

    print("SPLIT BY TERMINAL SATURATION AT 2h -- the named mechanism")
    print(f"  {'campaign':<12s} {'sat %':>7s} {'sigma|sat':>10s} "
          f"{'n':>4s} {'sigma|unsat':>12s} {'n':>4s}")
    for name, arr, src in ((tag_b, br, B), (tag_a, ar, A)):
        sat = np.array([src[i][2] for i in common])
        f = 100.0 * sat.mean()
        ss = sigma(arr[sat]) if sat.sum() > 2 else float("nan")
        su = sigma(arr[~sat]) if (~sat).sum() > 2 else float("nan")
        print(f"  {name:<12s} {f:6.1f}% {ss:10.2f} {int(sat.sum()):4d} "
              f"{su:12.2f} {int((~sat).sum()):4d}")
    print()
    print("  (perfect-arm terminal saturation, for reference)")
    for name, src in ((tag_b, B), (tag_a, A)):
        satp = np.array([src[i][3] for i in common])
        print(f"    {name:<12s} {100.0 * satp.mean():6.1f}%")
    print()

    print("CONTEXT: the campaigns' own CEP and reachability")
    for name, tg in ((tag_b, tag_b), (tag_a, tag_a)):
        camp = c[tg]["long"]
        for k in ("perfect", age):
            a_ = camp["ages"][k]
            print(f"  {name:<10s} {k:<8s} CEP {a_['cep_m']:7.2f} m  "
                  f"[{a_['cep_lo_m']:6.2f},{a_['cep_hi_m']:7.2f}]  "
                  f"<=30 m {100 * a_['within_30m']:5.1f} %  "
                  f"sd_range {a_['sd_range_m']:7.2f}")


if __name__ == "__main__":
    report()
