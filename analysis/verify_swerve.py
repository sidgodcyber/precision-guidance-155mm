"""
PHASE 2 -- verification of the 6-DOF against a PUBLISHED configuration whose
control authority is also published.

THE SOURCE
----------
Ollerenshaw, D. and Costello, M., "Simplified Projectile Swerve Solution for
General Control Inputs", Journal of Guidance, Control, and Dynamics, Vol. 31,
No. 5, September-October 2008, pp. 1259-1265.  DOI 10.2514/1.34252.
(The same analysis appears as U.S. Army Research Laboratory report
ARL-CR-0604, "On the Swerve Response of Projectiles to Control Input", 2008.)

Why this source and not a canard paper: it publishes BOTH a complete 155 mm
spin-stabilised configuration (their Table 1: every mass, inertia, station and
aerodynamic coefficient needed) AND the authority that configuration achieves
(swerve magnitude and phase at 5280 ft for a 1 lbf control force, swept over
the force application station), and it validates that closed form against its
own 6-DOF at five stations.  It therefore tests THIS project's airframe
response -- the induced body force, the cancellation, the Magnus term and the
gyroscopic term -- without any of this project's ESTIMATED canard aerodynamics
entering the comparison.  A canard paper would confound the two.

WHAT IS COMPARED
----------------
  * magnitude  R   : Ollerenshaw and Costello Eq. (23)
  * phase      Phi : Ollerenshaw and Costello Eq. (25)
against this repository's `sim.dynamics` 6-DOF, run at their configuration,
with a constant force applied in the no-roll frame at station SL_C -- which is
exactly the "generalised control mechanism" they model.

Their assumptions that must be reproduced:
  * no gravity, no wind          -> taken out by differencing a controlled and
                                    an uncontrolled run from identical initial
                                    conditions (the response is linear, so
                                    gravity is common mode)
  * constant velocity            -> zero-lift drag set to zero
  * constant spin                -> spin damping set to zero
  * constant Mach-dependent aero -> a one-row constant coefficient table
  * pitch damping neglected in
    the simplified form (C_MQ=0) -> reported both with and without

Run:  python -m analysis.verify_swerve
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace

import numpy as np

from sim import aerodata
from sim import canards as cn
from sim import dynamics as dyn
from sim import integrate as ig
from sim import projectile as pr

# ---------------------------------------------------------------------------
# Unit conversions.  The source is in slug-foot-second.
# ---------------------------------------------------------------------------
FT = 0.3048                      # m per ft
SLUG = 14.593902937206364        # kg per slug
SLUG_FT2 = SLUG * FT * FT        # kg m^2 per slug ft^2
LBF = 4.4482216152605            # N per lbf

# ---------------------------------------------------------------------------
# Ollerenshaw and Costello, Table 1, 155 mm spin-stabilised projectile.
# Every number below is theirs; nothing is fitted.
# ---------------------------------------------------------------------------
OC = {
    "V0_fps": 2710.0,
    "p0": 1674.1,                # rad/s
    "rho_slug_ft3": 2.3785e-3,
    "IR_slug_ft2": 0.10857,
    "IP_slug_ft2": 1.3964,
    "m_slug": 2.9465,
    "D_ft": 0.50853,
    "C_NA": 2.6314,
    "C_YPA": -0.9600,
    "C_MQ": -27.700,
    "SLcg_ft": 1.0627,           # from the BASE
    "SLM_ft": -0.52920,          # Magnus force application point minus c.g.
    "SLP_ft": 0.71373,           # centre of pressure minus c.g. (ahead)
}

OC_SI = {
    "V0": OC["V0_fps"] * FT,                 # 825.89 m/s
    "p0": OC["p0"],
    "rho": OC["rho_slug_ft3"] * SLUG / (FT ** 3),
    "IR": OC["IR_slug_ft2"] * SLUG_FT2,
    "IP": OC["IP_slug_ft2"] * SLUG_FT2,
    "m": OC["m_slug"] * SLUG,
    "D": OC["D_ft"] * FT,
    "C_NA": OC["C_NA"],
    "C_YPA": OC["C_YPA"],
    "C_MQ": OC["C_MQ"],
    "SLM": OC["SLM_ft"] * FT,
    "SLP": OC["SLP_ft"] * FT,
}

#: Downrange station at which the source evaluates the swerve.
RANGE_FT = 5280.0
RANGE_M = RANGE_FT * FT

#: The source's control input.
FC_LBF = 1.0
FC_N = FC_LBF * LBF

#: Control-force stations swept by the source, ft ahead of the c.g.
SLC_SWEEP_FT = tuple(np.round(np.arange(-1.0, 1.0001, 0.125), 4))
#: The five stations at which the source ran its own 6-DOF.
SLC_6DOF_FT = (-1.0, -0.5, 0.0, 0.5, 1.0)


def oc_projectile() -> pr.Projectile:
    """A `Projectile` carrying Ollerenshaw and Costello's Table 1 numbers."""
    d = OC_SI["D"]
    # Twist implied by their own V0 and p0: p = 2 pi V / (n d).
    n_cal = 2.0 * math.pi * OC_SI["V0"] / (OC_SI["p0"] * d)
    return pr.Projectile(
        name="155 mm spin-stabilised (Ollerenshaw and Costello 2008, Table 1)",
        mass=OC_SI["m"],
        diameter=d,
        I_axial=OC_SI["IR"],
        I_transverse=OC_SI["IP"],
        # Their stations are measured from the base; only the CP and Magnus
        # offsets RELATIVE to the c.g. enter the physics, so the absolute
        # station only has to be self-consistent with the length.
        x_cg=OC["SLcg_ft"] * FT,
        length=2.0 * OC["SLcg_ft"] * FT,
        twist_calibers=n_cal,
    )


def oc_aero_table(c_ypa_scale: float = 1.0, include_cmq: bool = False) -> aerodata.AeroTable:
    """
    A constant-coefficient table reproducing their Table 1.

    C_Malpha and C_Mpalpha are not tabulated by the source; they are the
    centre-of-pressure and Magnus-force stations expressed as moment slopes in
    this package's convention:

        SL_P = d C_Malpha / C_Nalpha          (CP ahead of the c.g.)
        SL_M = -d C_Mpalpha / |C_Ypalpha|     (Magnus force station)

    Drag and spin damping are zero because the source's linear theory holds V
    and p constant.
    """
    d = OC_SI["D"]
    C_NA = OC_SI["C_NA"]
    C_YPA = OC_SI["C_YPA"] * c_ypa_scale
    C_MA = OC_SI["SLP"] * C_NA / d
    # SL_M = -d C_Mpalpha / |C_Ypalpha|  =>  C_Mpalpha = -SL_M |C_Ypalpha| / d
    C_MPA = -OC_SI["SLM"] * abs(C_YPA) / d if C_YPA != 0.0 else 0.0
    C_MQ = OC_SI["C_MQ"] if include_cmq else 0.0
    #  mach, C_X0, C_X2, C_Nalpha, C_Ypalpha, C_lp, C_Malpha, C_mq, C_Mpalpha
    coeffs = [0.0, 0.0, C_NA, C_YPA, 0.0, C_MA, C_MQ, C_MPA]
    rows = np.array([[0.01] + coeffs, [10.0] + coeffs], dtype=float)
    return aerodata.AeroTable(
        rows,
        name="Ollerenshaw-Costello 2008 Table 1 (155 mm spin-stabilised)",
        source="J. Guidance Control Dynamics 31(5) 2008, Table 1",
    )


class NoRollFrameForce:
    """
    A constant force of fixed magnitude and fixed NO-ROLL-FRAME direction,
    applied at `station` metres ahead of the c.g.

    This is exactly the source's "generalised control mechanism": it is not an
    aerodynamic surface, it has no incidence and no lift-curve slope, so the
    comparison tests the AIRFRAME response and nothing else.

    `phi` is the earth-referenced roll orientation of the force, in the same
    sense as `sim.canards.CanardModel`: phi = 0 points along body -z (up) at
    zero body roll, phi = 90 deg points along body +y.
    """

    def __init__(self, magnitude: float, station: float, phi: float = 0.5 * math.pi,
                 start_time: float = 0.0):
        self.magnitude = float(magnitude)
        self.station = float(station)
        self.phi = float(phi)
        self.start_time = float(start_time)

    def __call__(self, t: float, y, st) -> tuple:
        if t < self.start_time:
            return np.zeros(3), np.zeros(3)
        phi_body = cn.CanardModel.body_roll_angle(y[6:10])
        phi_rel = self.phi - phi_body
        f = self.magnitude
        fy = f * math.sin(phi_rel)
        fz = -f * math.cos(phi_rel)
        force = np.array([0.0, fy, fz])
        a = self.station
        moment = np.array([0.0, -a * fz, a * fy])
        return force, moment


# ===========================================================================
# The source's closed form
# ===========================================================================
def oc_closed_form(slc: float, c_ypa: float | None = None) -> dict:
    """
    Ollerenshaw and Costello Eqs. (23) and (25), in SI.

    Returns the swerve magnitude at RANGE_M and the phase shift, in degrees,
    of the response relative to the applied control force.
    """
    m = OC_SI["m"]; D = OC_SI["D"]; V0 = OC_SI["V0"]; p0 = OC_SI["p0"]
    IR = OC_SI["IR"]; C_NA = OC_SI["C_NA"]
    C_YPA = OC_SI["C_YPA"] if c_ypa is None else c_ypa
    SLM = OC_SI["SLM"]; SLP = OC_SI["SLP"]
    x = RANGE_M
    FC = FC_N

    num = (D * D * p0 * p0 * C_YPA * C_YPA * SLM * SLM
           + 4.0 * V0 * V0 * C_NA * C_NA * (slc - SLP) ** 2)
    den = (p0 * p0 * (2.0 * IR * C_NA + m * D * C_YPA * SLM) ** 2
           + 4.0 * m * m * V0 * V0 * C_NA * C_NA * SLP * SLP)
    R = FC * x * x / (2.0 * V0 * V0) * math.sqrt(num / den)

    phi_num = (2.0 * V0 * p0 * C_NA
               * (2.0 * IR * C_NA * (slc - SLP) + m * D * C_YPA * SLM * slc))
    phi_den = (D * p0 * p0 * C_YPA * SLM * (2.0 * IR * C_NA + m * D * C_YPA * SLM)
               + 4.0 * m * V0 * V0 * C_NA * C_NA * SLP * (slc - SLP))
    phi = math.degrees(math.atan2(phi_num, phi_den))
    return {"R_m": R, "R_ft": R / FT, "phase_deg": phi,
            "accel_m_s2": 2.0 * R * V0 * V0 / (x * x)}


def repo_closed_form(slc: float, c_ypa: float | None = None) -> dict:
    """
    This repository's own quasi-steady closed form, extended to carry the
    Magnus force AND the Magnus moment, in the complex transverse plane
    zeta = y + i z of the no-roll frame:

        A (H_x/V - gamma m) = F_c (i a_c - gamma),
        gamma = i (a_n - i beta) / (1 + i mu)

    with  a_n  the centre-of-pressure arm ahead of the c.g.,
          a_c  the control-force arm ahead of the c.g.,
          mu   = |C_Ypalpha| p_hat / C_Nalpha   the Magnus force ratio,
          beta = d C_Mpalpha p_hat / C_Nalpha   the Magnus moment arm term,
          H_x  = I_axial p, the axial angular momentum.

    Setting mu = beta = 0 recovers the step-2.5 static-trim expression
    F_net = F_c (x_c - x_cp)/(x_cg - x_cp) modified only by the gyroscopic
    term i H_x/V in the denominator.
    """
    m = OC_SI["m"]; D = OC_SI["D"]; V = OC_SI["V0"]; p = OC_SI["p0"]
    IR = OC_SI["IR"]; C_NA = OC_SI["C_NA"]
    C_YPA = OC_SI["C_YPA"] if c_ypa is None else c_ypa
    SLM = OC_SI["SLM"]; a_n = OC_SI["SLP"]
    p_hat = 0.5 * p * D / V
    mu = abs(C_YPA) * p_hat / C_NA
    C_MPA = -SLM * abs(C_YPA) / D if C_YPA != 0.0 else 0.0
    beta = D * C_MPA * p_hat / C_NA
    Hx = IR * p
    gamma = 1j * (a_n - 1j * beta) / (1.0 + 1j * mu)
    A = FC_N * (1j * slc - gamma) / (Hx / V - gamma * m)
    # F_c points along +y of the no-roll frame -> complex +1, so A's argument
    # is already the phase relative to the control force.
    R = 0.5 * abs(A) * (RANGE_M / V) ** 2
    return {"R_m": R, "R_ft": R / FT,
            "phase_deg": math.degrees(math.atan2(A.imag, A.real)),
            "accel_m_s2": abs(A)}


def static_trim_closed_form(slc: float) -> dict:
    """
    The step-2.5 expression, for contrast: net force = F_c (x_c - x_cp) /
    (x_cg - x_cp), anti-parallel to the control force, no gyroscopic term and
    no Magnus term.
    """
    a_n = OC_SI["SLP"]
    A = FC_N * (a_n - slc) / (OC_SI["m"] * a_n)
    R = 0.5 * abs(A) * (RANGE_M / OC_SI["V0"]) ** 2
    return {"R_m": R, "R_ft": R / FT,
            "phase_deg": 0.0 if (a_n - slc) > 0 else 180.0,
            "accel_m_s2": abs(A)}


# ===========================================================================
# The 6-DOF
# ===========================================================================
def _model(c_ypa_scale: float, include_cmq: bool, control=None) -> dyn.FlightModel:
    env = pr.Environment.from_degrees(45.0, include_coriolis=False,
                                      include_inverse_square_gravity=False)
    return dyn.FlightModel(
        projectile=oc_projectile(),
        aero=oc_aero_table(c_ypa_scale, include_cmq),
        environment=env,
        control=control,
    )


def _initial_state(qe_rad: float = 0.0) -> np.ndarray:
    launch = pr.LaunchConditions(muzzle_velocity=OC_SI["V0"],
                                 quadrant_elevation=qe_rad)
    return dyn.initial_state(oc_projectile(), launch)


def swerve_6dof(slc: float, c_ypa_scale: float = 1.0, include_cmq: bool = False,
                dt: float = 2.0e-5, phi_deg: float = 90.0) -> dict:
    """
    Fly controlled and uncontrolled from identical initial conditions and
    difference the transverse position at x = 5280 ft.

    Gravity acts on both runs and is common mode; the difference is the swerve
    response to the control force, which is what the source's Eqs. (23) and
    (25) describe.
    """
    phi = math.radians(phi_deg)
    y0 = _initial_state()
    t_max = 1.6 * RANGE_M / OC_SI["V0"]
    kw = dict(dt=dt, log_every=200, t_max=t_max, stop_on_impact=False)
    free = ig.integrate(y0, _model(c_ypa_scale, include_cmq), **kw)
    ctrl = ig.integrate(
        y0, _model(c_ypa_scale, include_cmq,
                   NoRollFrameForce(FC_N, slc, phi)), **kw)

    def at_range(res):
        tr = res.trajectory
        x = tr.position[:, 0]
        i = int(np.searchsorted(x, RANGE_M))
        i = max(1, min(i, len(x) - 1))
        f = (RANGE_M - x[i - 1]) / (x[i] - x[i - 1])
        return (tr.position[i - 1] + f * (tr.position[i] - tr.position[i - 1]),
                tr.t[i - 1] + f * (tr.t[i] - tr.t[i - 1]))

    p_free, t_free = at_range(free)
    p_ctrl, t_ctrl = at_range(ctrl)
    d = p_ctrl - p_free
    # No-roll frame at zero roll: +y east/right, +z down. The control force at
    # phi = 90 deg points along +y, so the phase is measured from +y toward +z.
    dy, dz = float(d[1]), float(d[2])
    R = math.hypot(dy, dz)
    return {"slc_m": slc, "slc_ft": slc / FT, "R_m": R, "R_ft": R / FT,
            "dy_m": dy, "dz_m": dz,
            "phase_deg": math.degrees(math.atan2(dz, dy)),
            "accel_m_s2": 2.0 * R * (OC_SI["V0"] / RANGE_M) ** 2,
            "t_of_flight_s": float(t_ctrl),
            "uncontrolled_y_m": float(p_free[1]),
            "uncontrolled_z_m": float(p_free[2])}


# ===========================================================================
# Driver
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dt", type=float, default=2.0e-5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="docs/swerve_verification.json")
    args = ap.parse_args(argv)

    stations_ft = SLC_6DOF_FT if args.quick else SLC_SWEEP_FT
    out = {"source": ("Ollerenshaw and Costello, J. Guidance Control Dynamics "
                      "31(5), 2008, pp. 1259-1265, DOI 10.2514/1.34252"),
           "configuration": OC, "configuration_SI": OC_SI,
           "control_force_lbf": FC_LBF, "range_ft": RANGE_FT,
           "dt": args.dt, "rows": [], "magnus_sensitivity": []}

    print(f"Ollerenshaw and Costello 2008, Table 1, 155 mm spin-stabilised")
    print(f"  m = {OC_SI['m']:.3f} kg   d = {OC_SI['D']:.4f} m   "
          f"V0 = {OC_SI['V0']:.1f} m/s   p0 = {OC_SI['p0']:.1f} rad/s")
    print(f"  C_Na = {OC_SI['C_NA']:.4f}  C_Ypa = {OC_SI['C_YPA']:.4f}  "
          f"SL_P = {OC_SI['SLP']:.5f} m   SL_M = {OC_SI['SLM']:.5f} m")
    print(f"  1 lbf control force, swerve at {RANGE_FT:.0f} ft "
          f"({RANGE_M:.1f} m), dt = {args.dt}\n")
    hdr = (f"{'SL_C ft':>8} {'6DOF R ft':>10} {'OC23 R ft':>10} {'ratio':>7} "
           f"{'6DOF phase':>11} {'OC25 phase':>11} {'d deg':>7} "
           f"{'repo R ft':>10} {'trim R ft':>10}")
    print(hdr); print("-" * len(hdr))
    for slc_ft in stations_ft:
        slc = slc_ft * FT
        s = swerve_6dof(slc, dt=args.dt)
        o = oc_closed_form(slc)
        r = repo_closed_form(slc)
        st = static_trim_closed_form(slc)
        # phases are defined modulo the sense of the transverse axes; report
        # the 6-DOF phase folded into (-180, 180]
        dphase = (s["phase_deg"] - o["phase_deg"] + 180.0) % 360.0 - 180.0
        row = {"slc_ft": slc_ft, "slc_m": slc, "sixdof": s, "oc": o,
               "repo": r, "static_trim": st, "phase_error_deg": dphase,
               "magnitude_ratio": s["R_m"] / o["R_m"] if o["R_m"] else float("nan")}
        out["rows"].append(row)
        print(f"{slc_ft:8.3f} {s['R_ft']:10.4f} {o['R_ft']:10.4f} "
              f"{row['magnitude_ratio']:7.3f} {s['phase_deg']:11.2f} "
              f"{o['phase_deg']:11.2f} {dphase:7.2f} "
              f"{r['R_ft']:10.4f} {st['R_ft']:10.4f}")

    # Magnus sensitivity, the source's Fig. 8: C_YPA nominal, half, zero.
    print("\nMagnus sensitivity (source Fig. 8), control force at the CP "
          f"(SL_C = SL_P = {OC_SI['SLP'] / FT:.5f} ft):")
    hdr2 = (f"{'C_YPA':>8} {'6DOF R ft':>10} {'OC23 R ft':>10} {'ratio':>7} "
            f"{'6DOF phase':>11} {'OC25 phase':>11}")
    print(hdr2); print("-" * len(hdr2))
    for scale in (1.0, 0.5, 0.0):
        for slc_ft in (OC_SI["SLP"] / FT, 0.5, -0.5):
            slc = slc_ft * FT
            s = swerve_6dof(slc, c_ypa_scale=scale, dt=args.dt)
            o = oc_closed_form(slc, c_ypa=OC_SI["C_YPA"] * scale)
            out["magnus_sensitivity"].append(
                {"c_ypa_scale": scale, "slc_ft": slc_ft, "sixdof": s, "oc": o})
            if abs(slc_ft - OC_SI["SLP"] / FT) < 1e-9:
                print(f"{OC_SI['C_YPA'] * scale:8.3f} {s['R_ft']:10.4f} "
                      f"{o['R_ft']:10.4f} "
                      f"{s['R_m'] / o['R_m'] if o['R_m'] else float('nan'):7.3f} "
                      f"{s['phase_deg']:11.2f} {o['phase_deg']:11.2f}")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {args.out}")
    return out


if __name__ == "__main__":
    main()
