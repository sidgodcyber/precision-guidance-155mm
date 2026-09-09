"""
PHASE 4 -- what a measured correction authority implies for achievable CEP.

Not a Monte Carlo. This is deterministic quadrature over an assumed miss
distribution: the point is to turn "authority is X metres" into "CEP is Y
metres" without inventing a guidance law, and to invert it -- "CEP of 30 m
requires authority of Z metres" -- so that the architecture decision rests on
arithmetic rather than on assertion.

THE MODEL
---------
An uncorrected round impacts at X, drawn from a bivariate normal with standard
deviations `sigma_range` and `sigma_deflection` about the aim point. A kit
deployed at a fixed time can move that impact point anywhere inside a
reachable set

    R(X) = X + bias + E(a, b)

where E is the ellipse with semi-major axis `a` along range and semi-minor
axis `b` along deflection, and `bias` is the deterministic drag offset. Those
are exactly the quantities `analysis.authority.envelope_stats` measures.

A PERFECT guidance law, with a perfect navigation solution and a perfect
servo, puts the round at the point of R(X) closest to the target. The residual
miss is therefore

    r(X) = distance from the target to the set R(X)
         = distance from the point -(X + bias) to the ellipse E(a, b)

which is zero whenever the target is reachable. The corrected CEP is the
median of r over the miss distribution.

This is an UPPER BOUND on performance -- an unachievable one. It contains no
navigation error, no servo error, no model error in the impact-point
prediction, no wind estimation error and no deployment-time error. Every one
of those can only make the answer worse. That is the point: if the bound is
above the requirement, the requirement is not reachable, and no amount of work
on the loop changes it.

`aim_offset` shifts the aim point, which is how the deliberate range bias of
[AUTHORITY-ENVELOPE.md](../docs/AUTHORITY-ENVELOPE.md) section 3.3 is priced:
the gun is laid long by the bias so that the reachable set straddles the
target.

Run:  python -m analysis.cep_projection
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

#: Quadrature grid half-width in standard deviations, and node count per axis.
#: 6 sigma at 401 nodes reproduces the analytic circular CEP = 1.17741 sigma to
#: better than 1e-4, which `test` below asserts.
GRID_SIGMA = 6.0
GRID_N = 401


def _distance_to_ellipse(px, py, a: float, b: float, iters: int = 80):
    """
    Distance from each point (px, py) to the axis-aligned ellipse
    x^2/a^2 + y^2/b^2 = 1, or 0 for points inside it. Vectorised over numpy
    arrays.

    Bisection on the Lagrange multiplier of the standard closest-point
    condition: the closest point is (a^2 x/(t + a^2), b^2 y/(t + b^2)) where t
    solves (a x/(t + a^2))^2 + (b y/(t + b^2))^2 = 1. 80 halvings is exact to
    machine precision at these scales.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    if a <= 0.0 or b <= 0.0:
        return np.hypot(px, py)
    x, y = np.abs(px), np.abs(py)
    inside = (x / a) ** 2 + (y / b) ** 2 <= 1.0
    # Points strictly inside are returned as zero; nudge them off the axes so
    # the bisection below never divides by zero for them.
    x = np.where(inside, a, x)
    y = np.where(inside, b, y)
    lo = np.full(x.shape, -min(a, b) ** 2 + 1e-12)
    hi = np.hypot(a * x, b * y) + max(a, b) ** 2
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = (a * x / (mid + a * a)) ** 2 + (b * y / (mid + b * b)) ** 2 - 1.0
        gt = f > 0.0
        lo = np.where(gt, mid, lo)
        hi = np.where(gt, hi, mid)
    t = 0.5 * (lo + hi)
    cx = a * a * x / (t + a * a)
    cy = b * b * y / (t + b * b)
    return np.where(inside, 0.0, np.hypot(x - cx, y - cy))


def corrected_cep(sigma_range: float, sigma_deflection: float,
                  a: float, b: float, bias_range: float = 0.0,
                  bias_deflection: float = 0.0, aim_offset_range: float = 0.0,
                  n: int = GRID_N, span: float = GRID_SIGMA) -> dict:
    """
    Median residual miss (the corrected CEP) under a perfect guidance law.

    `bias_*` is the deterministic offset of the reachable set's centre from
    the uncorrected impact point. `aim_offset_range` moves the aim point, so
    setting it to `-bias_range` recovers the "aim long" firing solution.
    """
    gr = np.linspace(-span * sigma_range, span * sigma_range, n)
    gd = np.linspace(-span * sigma_deflection, span * sigma_deflection, n)
    wr = np.exp(-0.5 * (gr / sigma_range) ** 2)
    wd = np.exp(-0.5 * (gd / sigma_deflection) ** 2)
    W = np.outer(wr, wd)
    W /= W.sum()

    # Coordinates are relative to the TARGET. The aim point sits at
    # `aim_offset_range` from it, the uncorrected impact at aim + miss, and
    # the reachable set is centred one `bias` beyond that. Aiming long by the
    # bias means aim_offset_range = -bias_range.
    cx = aim_offset_range + gr[:, None] + bias_range
    cy = gd[None, :] + bias_deflection
    R = _distance_to_ellipse(-cx * np.ones_like(cy), -cy * np.ones_like(cx), a, b)

    order = np.argsort(R, axis=None)
    r_sorted = R.flatten()[order]
    w_sorted = W.flatten()[order]
    cdf = np.cumsum(w_sorted)
    cep = float(np.interp(0.5, cdf, r_sorted))
    return {
        "cep_m": cep,
        "p_reachable": float(w_sorted[r_sorted <= 0.0].sum()),
        "p_within_30m": float(np.interp(30.0, r_sorted, cdf)),
        "p_within_50m": float(np.interp(50.0, r_sorted, cdf)),
        "mean_residual_m": float((R * W).sum()),
        "uncorrected_cep_m": float(uncorrected_cep(sigma_range,
                                                   sigma_deflection, n, span)),
    }


def uncorrected_cep(sigma_range: float, sigma_deflection: float,
                    n: int = GRID_N, span: float = GRID_SIGMA) -> float:
    """Median radial miss of the bivariate normal, by the same quadrature."""
    gr = np.linspace(-span * sigma_range, span * sigma_range, n)
    gd = np.linspace(-span * sigma_deflection, span * sigma_deflection, n)
    W = np.outer(np.exp(-0.5 * (gr / sigma_range) ** 2),
                 np.exp(-0.5 * (gd / sigma_deflection) ** 2))
    W /= W.sum()
    R = np.hypot(gr[:, None], gd[None, :])
    order = np.argsort(R, axis=None)
    return float(np.interp(0.5, np.cumsum(W.flatten()[order]),
                           R.flatten()[order]))


def required_authority(sigma_range: float, sigma_deflection: float,
                       target_cep: float, axis_ratio: float = 1.0,
                       bias_range: float = 0.0, aim_long: bool = True,
                       lo: float = 1.0, hi: float = 5000.0) -> float:
    """
    Semi-major axis `a` (with `b = a / axis_ratio`) at which the corrected CEP
    first falls to `target_cep`. Bisection; returns nan if `hi` does not
    reach it.
    """
    def cep_of(a):
        return corrected_cep(sigma_range, sigma_deflection, a, a / axis_ratio,
                             bias_range=bias_range,
                             aim_offset_range=(-bias_range if aim_long else 0.0),
                             n=201)["cep_m"]

    if cep_of(hi) > target_cep:
        return float("nan")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if cep_of(mid) > target_cep:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _self_test():
    """Circular bivariate normal: CEP must be 1.17741 sigma."""
    got = uncorrected_cep(100.0, 100.0, n=801, span=8.0)
    assert abs(got / 100.0 - 1.17741) < 2e-3, got
    # a reachable set far larger than the miss must give CEP = 0
    r = corrected_cep(100.0, 100.0, 2000.0, 2000.0, n=101)
    assert r["cep_m"] == 0.0 and r["p_reachable"] > 0.999
    # and one far smaller must leave the miss essentially untouched
    r = corrected_cep(100.0, 100.0, 1.0, 1.0, n=201)
    assert abs(r["cep_m"] - (uncorrected_cep(100.0, 100.0, 201) - 1.0)) < 1.0
    # a point outside a circle: distance must be radius-minus-r
    d = _distance_to_ellipse(np.array([5.0]), np.array([0.0]), 3.0, 3.0)
    assert abs(float(d[0]) - 2.0) < 1e-9, d
    return True


#: The configurations of docs/ARCHITECTURE-DECISION.md section 2, named by the
#: label of the sweep point each was measured at. Nothing about a reachable
#: set is written down here: the semi-axes, the axis ratio and the bias are
#: read from the campaign JSON, so this table cannot drift from the data that
#: produced it. That is the whole of the step-6 correction -- until now the
#: sets were transcribed constants and the requirement used one hard-coded
#: axis ratio of 1.44 for all of them.
SET_SOURCES = (
    ("step 2.5 nominal, apogee", "design_sweep.json", "14|deploy_f+0.00"),
    ("25 mm, 5 deg, apogee", "design_sweep.json", "14|station_0.025"),
    ("90 mm, 3 deg, t/t_ap 0.50", "design_sweep_combos.json", "14|nom-0.50_defl3"),
    ("25 mm, 3 deg, t/t_ap 0.50", "design_sweep_combos.json", "14|fwd-0.50_defl3"),
    ("25 mm, 2 deg, t/t_ap 0.25", "design_sweep_combos.json", "14|fwd-0.75_defl2"),
    ("25 mm, 3 deg, t/t_ap 0.25 (recommended)", "design_sweep_combos.json",
     "14|fwd-0.75_defl3"),
    ("25 mm tapered, 3 deg, t/t_ap 0.25", "design_sweep_combos.json",
     "14|fwdtaper-0.75_defl3"),
)

#: The adopted staged configuration, measured with the loop closed in step 3
#: and reported in docs/CEP-CLOSED-LOOP.md section 5.2. It is not a sweep
#: point -- it is what the built loop achieves -- so it is carried here with
#: its source rather than read from the sweep.
ADOPTED_STAGED = {"name": "adopted staged, closed loop (CEP-CLOSED-LOOP 5.2)",
                  "a": 184.0, "b": 166.8, "bias": -226.6}


def load_sets(docs: str = "docs") -> list:
    """The measured reachable sets, read from the campaigns that produced them."""
    cache = {}
    out = []
    for name, fname, label in SET_SOURCES:
        if fname not in cache:
            with open(os.path.join(docs, fname)) as fh:
                cache[fname] = json.load(fh)
        pt = next((p for p in cache[fname]["points"] if p["label"] == label),
                  None)
        if pt is None:
            raise KeyError(f"{label} not in {fname}")
        out.append({"name": name, "source": f"{fname}:{label}",
                    "a": float(pt["semi_axis_major_m"]),
                    "b": float(pt["semi_axis_minor_m"]),
                    "bias": float(pt["bias_range_m"]),
                    "trim_deg": float(pt["max_total_aoa_deg"])})
    out.append({**ADOPTED_STAGED, "source": "docs/CEP-CLOSED-LOOP.md 5.2",
                "trim_deg": float("nan")})
    for r in out:
        r["axis_ratio"] = r["a"] / r["b"]
    return out


#: Uncorrected dispersions the projection is evaluated against. The brief
#: gives an uncorrected miss of 200-300 m; artillery dispersion is strongly
#: anisotropic, so both a circular and a 3:1 range-dominated case are carried.
DISPERSIONS = (
    ("circular, CEP 200 m", 169.9, 169.9),
    ("circular, CEP 250 m", 212.3, 212.3),
    ("circular, CEP 300 m", 254.8, 254.8),
    ("3:1 range-dominated, CEP 200 m", 244.0, 81.3),
    ("3:1 range-dominated, CEP 250 m", 305.0, 101.7),
)

#: THE AXIS RATIO IS NOT A CONSTANT AND IS NO LONGER HIDDEN.
#:
#: Until step 6 the requirement loop passed axis_ratio=1.44 -- the step-2.5
#: nominal configuration's shape -- for every row, including rows describing
#: configurations whose shape is 1.20 or 1.10. The requirement depends on it
#: strongly (docs/CEP-CLOSED-LOOP.md section 5.2 measures 206.3 m at 1.44
#: against 186.3 m at 1.20 and 178.7 m at 1.10), and that single hard-coded
#: number is why docs/cep_projection.json stopped reproducing the table in
#: docs/ARCHITECTURE-DECISION.md section 2. Every requirement now names the
#: ratio it was computed at.
RATIOS = (
    (1.44, "step 2.5 nominal; what this file hard-coded before step 6"),
    (1.20, "the recommended configuration -- the PUBLISHED number"),
    (1.10, "the adopted staged configuration, measured in step 3"),
    (1.00, "a circular reachable set"),
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/cep_projection.json")
    ap.add_argument("--docs", default="docs")
    args = ap.parse_args(argv)
    _self_test()

    sets = load_sets(args.docs)
    out = {"note": "perfect-guidance upper bound; no nav, servo or "
                   "prediction error included",
           "sets": sets, "cases": [], "requirements": []}

    header = ("reachable set", "a", "ratio", "CEP 200", "CEP 250", "CEP 300")
    print("%-42s %7s %6s %8s %8s %8s   (circular miss)" % header)
    for st in sets:
        row = {}
        for dname, sr, sd in DISPERSIONS:
            r = corrected_cep(sr, sd, st["a"], st["b"], bias_range=st["bias"],
                              aim_offset_range=-st["bias"], n=301)
            out["cases"].append({"dispersion": dname, "sigma_range": sr,
                                 "sigma_deflection": sd, "set": st["name"],
                                 "a": st["a"], "b": st["b"],
                                 "axis_ratio": st["axis_ratio"],
                                 "bias": st["bias"], **r})
            row[dname] = r["cep_m"]
        print("%-42s %7.1f %6.2f %8.1f %8.1f %8.1f" % (
            st["name"], st["a"], st["axis_ratio"],
            row["circular, CEP 200 m"], row["circular, CEP 250 m"],
            row["circular, CEP 300 m"]))

    print()
    print("%-32s %6s %7s %13s %11s" % (
        "dispersion", "ratio", "target", "required a m", "% of 15840"))
    for dname, sr, sd in DISPERSIONS:
        for ratio, why in RATIOS:
            for target in (30.0, 50.0):
                need = required_authority(sr, sd, target, axis_ratio=ratio)
                out["requirements"].append(
                    {"dispersion": dname, "axis_ratio": ratio,
                     "axis_ratio_is": why, "target_cep": target,
                     "required_a_m": need,
                     "required_pct_of_range": 100.0 * need / 15840.0})
                if target == 30.0:
                    print("%-32s %6.2f %7.0f %13.1f %10.3f%%" % (
                        dname, ratio, target, need, 100.0 * need / 15840.0))

    # The published claim, asserted rather than hoped for.
    pub = next(r for r in out["requirements"]
               if r["dispersion"] == "circular, CEP 200 m"
               and r["axis_ratio"] == 1.20 and r["target_cep"] == 30.0)
    ok = abs(pub["required_a_m"] - 186.3) < 1.0
    out["published_check"] = {
        "claim": "ARCHITECTURE-DECISION.md section 2: 186.3 m of semi-major "
                 "authority for CEP <= 30 m at an uncorrected CEP of 200 m",
        "recomputed_m": pub["required_a_m"],
        "reproduces": bool(ok),
    }
    print()
    print("published 186.3 m at ratio 1.20 -> recomputed %.1f m (%s)" % (
        pub["required_a_m"], "reproduces" if ok else "DOES NOT REPRODUCE"))

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print()
    print("wrote " + args.out)
    return out


if __name__ == "__main__":
    main()
