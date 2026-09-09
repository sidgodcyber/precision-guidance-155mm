"""
The analytic inverse map: desired ground correction -> commanded roll angle.

Step 3, Task A. docs/INVERSE-MAP.md.

WHY THIS IS AN INVERSION AND NOT A LOOKUP
-----------------------------------------
Step 4.5 measured the staged configuration's reachable set to be a
first-harmonic ellipse to 1.6 % of its own amplitude
(docs/STAGED-DEPLOYMENT.md section 6.3). That is the whole licence for this
module. The single-stage configuration's misfit was 22 %, which is not an
ellipse with error bars but the wrong description, and a guidance law built on
it would have had to carry a lookup over sampled trajectories. Staging bought
the inversion.

Concretely: hold the despun nose at earth-referenced roll angle `phi` from
deployment until `t_end`, then release. The impact point moves by

    d(phi, t_end) = G(t_end) @ [cos phi, sin phi] + c(t_end)          (1)

against the same round that never held, where `G` is a 2x2 matrix taking the
commanded direction into (range, deflection) metres and `c` is the part of the
displacement that does NOT depend on the commanded angle -- the cost in range
of holding an angle of attack at all. Both are measured, per engagement, by
`analysis.guidance_authority`.

THE INVERSION, IN CLOSED FORM
-----------------------------
Holding on from `t_now` to `t_end` instead of releasing now displaces the
impact by

    Delta(phi) = A @ [cos phi, sin phi] + b,   A = G(t_end) - G(t_now)   (2)

so to move the impact in a desired unit direction `u`:

    v = A^-1 u ;   phi = atan2(v_y, v_x)                                 (3)

and the magnitude delivered in that direction is exactly

    |A w| = 1 / |A^-1 u|,   w = v / |v|                                  (4)

Three lines and one 2x2 inverse. (4) is where the anisotropy lives: the
achievable correction in direction `u` is the reciprocal of the length of the
pre-image, which for the adopted set varies by the axis ratio between the best
and worst directions. A law that treats the miss vector as isotropic either
over-commits in the cheap direction or runs out of authority in the expensive
one.

WHAT IS *NOT* HERE
------------------
The commanded angle is not the correction direction. On the adopted
configuration the two differ by of order 75 degrees, because the Magnus force
dominates the response and rotates it (docs/ARCHITECTURE-DECISION.md section
3). That rotation is inside `G` -- it is not applied as a separate correction,
because `G` is measured and a rotation angle would have to be modelled.

`G` is indexed on TIME TO GO and not on Mach. The task asked for a map "as a
function of Mach"; over the adopted engagement's guided phase Mach runs
1.521 -> 0.864 -> 0.917, so it is double-valued over the last 40 % of the
flight and cannot index anything. Mach is carried alongside each node as a
diagnostic. Section 4 of docs/INVERSE-MAP.md states the measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "AuthorityMap", "MapNode", "fit_harmonic", "ellipse_of",
    "ellipse_contains_fraction", "containment_margin",
]


# ===========================================================================
# Fitting -- the same first harmonic every reachable set in this project uses
# ===========================================================================
def fit_harmonic(angles_deg: Sequence[float], shifts: Sequence[Sequence[float]]):
    """
    Least-squares fit of (1) to measured impact shifts.

    Returns (G, c, residual_fraction) where `G` is 2x2, `c` is length 2 and
    the residual is the rms fit residual divided by the rms amplitude of the
    fitted harmonic -- the 1.6 % of docs/STAGED-DEPLOYMENT.md section 6.3,
    computed identically so the numbers are comparable.
    """
    ang = np.radians(np.asarray(angles_deg, dtype=float))
    xy = np.asarray(shifts, dtype=float)
    basis = np.column_stack([np.cos(ang), np.sin(ang), np.ones_like(ang)])
    coef, *_ = np.linalg.lstsq(basis, xy, rcond=None)
    G = coef[:2, :].T                      # rows: range, deflection
    c = coef[2, :].copy()
    resid = xy - basis @ coef
    amp = math.sqrt(0.5 * float((coef[:2, :] ** 2).sum()))
    rms = math.sqrt(float((resid ** 2).sum() / max(resid.shape[0], 1)))
    return G, c, (rms / amp if amp > 0 else math.inf)


def ellipse_of(G: np.ndarray) -> dict:
    """Semi-axes, tilt and rms amplitude of the ellipse traced by `G w`."""
    u, s, _ = np.linalg.svd(np.asarray(G, dtype=float))
    return {
        "semi_major_m": float(s[0]),
        "semi_minor_m": float(s[1]),
        "axis_ratio": float(s[0] / s[1]) if s[1] > 0 else float("inf"),
        "major_axis_tilt_deg": float(math.degrees(math.atan2(u[1, 0], u[0, 0]))),
        "rms_amplitude_m": float(math.sqrt(0.5 * (s[0] ** 2 + s[1] ** 2))),
    }


# ===========================================================================
# The table the flight computer carries
# ===========================================================================
@dataclass(frozen=True)
class MapNode:
    """
    One row of the map: what a hold from deployment to `t_end` delivers.

    Six floats of payload (`G` and `c`) plus the index. `t_go` rather than
    absolute time so the table is expressed in the variable the guidance law
    actually has.
    """

    t_go: float                 # s from t_end to impact
    G: np.ndarray               # 2x2, m per unit commanded direction
    c: np.ndarray               # 2, m, the angle-independent part
    mach: float = float("nan")  # diagnostic only; see the module docstring
    residual: float = float("nan")   # harmonic misfit of the fit at this node


class AuthorityMap:
    """
    G(t_end) and c(t_end), interpolated, plus the inversion of equation (3).

    Nodes are stored in order of DECREASING `t_go`, i.e. increasing hold
    length, because that is the order the flight computer traverses them.
    """

    def __init__(self, nodes: Sequence[MapNode], impact_time: float,
                 deploy_time: float, label: str = ""):
        ns = sorted(nodes, key=lambda n: -n.t_go)
        self.nodes = list(ns)
        self.impact_time = float(impact_time)
        self.deploy_time = float(deploy_time)
        self.label = label
        self._tgo = np.array([n.t_go for n in ns])
        self._G = np.array([n.G for n in ns])          # (n, 2, 2)
        self._c = np.array([n.c for n in ns])          # (n, 2)
        self._mach = np.array([n.mach for n in ns])
        # Interpolation runs on ASCENDING t_go, so keep a reversed view.
        self._x = self._tgo[::-1].copy()
        self._Gx = self._G[::-1].copy()
        self._cx = self._c[::-1].copy()
        self._machx = self._mach[::-1].copy()

    # -- the table --------------------------------------------------------
    @property
    def size_floats(self) -> int:
        """Payload the flight computer carries: 6 floats per node."""
        return 6 * len(self.nodes)

    def size_bytes(self, dtype_bytes: int = 4) -> int:
        return self.size_floats * dtype_bytes

    def mach_at(self, t_go: float) -> float:
        return float(np.interp(t_go, self._x, self._machx))

    # -- G and c ----------------------------------------------------------
    def delivered(self, t_go: float):
        """
        (G, c) for a hold running from deployment until `t_go` before impact.

        Linear in `t_go` between nodes and held flat outside them. Held flat
        rather than extrapolated on purpose: beyond the last node the map has
        no measurement, and a linear extrapolation of a saturating quantity
        over-promises.
        """
        x = float(min(max(t_go, self._x[0]), self._x[-1]))
        G = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                G[i, j] = np.interp(x, self._x, self._Gx[:, i, j])
        c = np.array([np.interp(x, self._x, self._cx[:, j]) for j in range(2)])
        return G, c

    def increment(self, t_go_now: float, t_go_end: float):
        """
        (A, b) of equation (2): what is gained by holding on from `t_go_now`
        until `t_go_end` instead of releasing now. `t_go_end < t_go_now`.
        """
        G1, c1 = self.delivered(t_go_end)
        G0, c0 = self.delivered(t_go_now)
        return G1 - G0, c1 - c0

    def remaining(self, t_go_now: float):
        """(A, b) for holding all the way to impact."""
        return self.increment(t_go_now, 0.0)

    # -- the inversion ----------------------------------------------------
    @staticmethod
    def invert(A: np.ndarray, direction) -> Optional[float]:
        """
        Equation (3): the commanded roll angle, rad, that moves the impact
        along `direction`. None if `A` is singular -- which is what a stage-2
        failure looks like from here, and the caller must degrade rather than
        command.
        """
        A = np.asarray(A, dtype=float)
        d = np.asarray(direction, dtype=float)
        n = float(math.hypot(float(d[0]), float(d[1])))
        if n <= 0.0:
            return None
        det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
        if not math.isfinite(det) or abs(det) < 1e-9:
            return None
        try:
            v = np.linalg.solve(A, d / n)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(v)) or math.hypot(float(v[0]), float(v[1])) <= 0.0:
            return None
        return math.atan2(float(v[1]), float(v[0]))

    @staticmethod
    def reach_along(A: np.ndarray, direction) -> float:
        """
        Equation (4): metres of correction available along `direction`, using
        the best commanded angle. Zero if `A` is singular.
        """
        A = np.asarray(A, dtype=float)
        d = np.asarray(direction, dtype=float)
        n = float(math.hypot(float(d[0]), float(d[1])))
        if n <= 0.0:
            return 0.0
        det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
        if not math.isfinite(det) or abs(det) < 1e-9:
            return 0.0
        try:
            v = np.linalg.solve(A, d / n)
        except np.linalg.LinAlgError:
            return 0.0
        L = math.hypot(float(v[0]), float(v[1]))
        return 0.0 if L <= 0.0 else 1.0 / L

    def command_for(self, miss, t_go_now: float,
                    t_go_end: float = 0.0) -> Optional[float]:
        """
        Commanded roll angle, rad, to drive the impact point along `miss`
        using the hold from now to `t_go_end`.
        """
        A, _ = self.increment(t_go_now, t_go_end)
        return self.invert(A, miss)

    def hold_end_for(self, miss, t_go_now: float,
                     phi: Optional[float] = None,
                     tol: float = 1e-3) -> tuple:
        """
        The release time that delivers |miss| along `miss`, and the angle to
        fly it at.

        Returns (t_go_end, phi, delivered_m, saturated). `saturated` is True
        when even holding to impact cannot deliver the whole of `miss`, in
        which case `t_go_end` is 0.0 and the law must accept the shortfall.

        Bisection on `t_go_end` rather than an analytic solve: the magnitude
        available is monotone in hold length but not linear in it (the map
        saturates -- docs/CONTROL-CHARACTERISATION.md section 8.4), and 40
        halvings of a 50 s interval is under a microsecond of work on any
        processor that can run the propagation this law also needs.
        """
        m = np.asarray(miss, dtype=float)
        need = math.hypot(float(m[0]), float(m[1]))
        if need <= 0.0:
            return t_go_now, (phi if phi is not None else 0.0), 0.0, False
        u = m / need

        def got(tge: float) -> float:
            A, _ = self.increment(t_go_now, tge)
            if phi is None:
                return self.reach_along(A, u)
            w = np.array([math.cos(phi), math.sin(phi)])
            return float((A @ w) @ u)

        full = got(0.0)
        if full <= need + tol:
            best_phi = phi if phi is not None else self.command_for(u, t_go_now, 0.0)
            return 0.0, (best_phi if best_phi is not None else 0.0), full, True

        lo, hi = 0.0, t_go_now      # lo = hold to impact, hi = release now
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if got(mid) >= need:
                lo = mid
            else:
                hi = mid
        tge = 0.5 * (lo + hi)
        A, _ = self.increment(t_go_now, tge)
        best_phi = phi if phi is not None else self.invert(A, u)
        return tge, (best_phi if best_phi is not None else 0.0), got(tge), False

    # -- serialisation ----------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "deploy_time": self.deploy_time,
            "impact_time": self.impact_time,
            "size_floats": self.size_floats,
            "nodes": [
                {"t_go": n.t_go, "mach": n.mach, "residual": n.residual,
                 "G": [[float(x) for x in row] for row in n.G],
                 "c": [float(x) for x in n.c]}
                for n in self.nodes
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> "AuthorityMap":
        nodes = [MapNode(t_go=float(n["t_go"]), G=np.array(n["G"], dtype=float),
                         c=np.array(n["c"], dtype=float),
                         mach=float(n.get("mach", float("nan"))),
                         residual=float(n.get("residual", float("nan"))))
                 for n in d["nodes"]]
        return AuthorityMap(nodes, float(d["impact_time"]),
                            float(d["deploy_time"]), d.get("label", ""))


# ===========================================================================
# Task G -- containment, the two-dimensional test
# ===========================================================================
def ellipse_contains_fraction(a: float, b: float, tilt_deg: float,
                              sigma_range: float, sigma_deflection: float,
                              bias=(0.0, 0.0), aim_offset=(0.0, 0.0),
                              n: int = 401, span: float = 6.0) -> float:
    """
    The probability that the reachable set contains the target -- the
    two-dimensional replacement for "is the semi-major axis bigger than the
    required authority".

    Both sets are ellipses elongated along range. The scalar test compares one
    semi-axis of one against one number derived from the other, and whether
    that is optimistic or pessimistic depends on their relative eccentricities
    in a way no scalar carries. This computes it.

    `bias` is the reachable set's centre relative to the uncorrected impact;
    `aim_offset` is where the gun is laid relative to the target. Both in
    (range, deflection) metres.
    """
    gr = np.linspace(-span * sigma_range, span * sigma_range, n)
    gd = np.linspace(-span * sigma_deflection, span * sigma_deflection, n)
    W = np.outer(np.exp(-0.5 * (gr / sigma_range) ** 2),
                 np.exp(-0.5 * (gd / sigma_deflection) ** 2))
    W /= W.sum()
    # Vector from the reachable set's centre to the target.
    x = -(aim_offset[0] + gr[:, None] + bias[0]) * np.ones((n, n))
    y = -(aim_offset[1] + gd[None, :] + bias[1]) * np.ones((n, n))
    th = math.radians(tilt_deg)
    ct, stt = math.cos(th), math.sin(th)
    xr = ct * x + stt * y
    yr = -stt * x + ct * y
    inside = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
    return float(W[inside].sum())


def containment_margin(a: float, b: float, tilt_deg: float,
                       sigma_range: float, sigma_deflection: float,
                       target: float = 0.5, **kw) -> float:
    """
    The factor the reachable set must be scaled by for it to contain the miss
    with probability `target`. > 1 means the set is too small.
    """
    lo, hi = 0.05, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p = ellipse_contains_fraction(a * mid, b * mid, tilt_deg,
                                      sigma_range, sigma_deflection, **kw)
        if p < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
