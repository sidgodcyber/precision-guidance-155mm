"""
Characterisation of the closed-loop roll-angle servo. Tasks A, C and D.

Produces every number in docs/CONTROL-CHARACTERISATION.md and the JSON that
step 3 reads, docs/roll_servo.json.

METHOD, AND WHY THERE ARE TWO MODELS
------------------------------------
The nose roll torque is EXACTLY independent of body attitude, body rates and
angle of attack: the four-panel roll moment sums the flow-induced panel
incidences over four equally spaced azimuths and that sum is identically zero.
Only dynamic pressure, airspeed, Mach and the nose inertial rate survive. That
is not an approximation made here for speed -- it is a property of the plant,
asserted against the full model to 4e-14 relative in
tests/test_roll_control.py.

So the servo can be characterised on a three-state reduced model that runs a
step response in milliseconds instead of the 17 seconds a 6-DOF trajectory
costs, and the sweeps that Task C needs -- four metrics at fifteen flight
times in two directions -- become affordable. The 6-DOF is then flown at
checkpoints to confirm that the reduced model has not quietly diverged, and
Task D is measured in the 6-DOF throughout, because impact displacement is a
trajectory quantity that no reduced nose model can produce.

CONFIGURATION
-------------
The adopted one from docs/ARCHITECTURE-DECISION.md section 1: canards at
25 mm from the nose, steering deflection 3 degrees, nominal panel, deployment
at t/t_apogee = 0.25. Engagement: charge 8, QE 525.3 mils, 15 840 m.

Run:  python -m analysis.roll_servo            full, ~30 min on 8 cores
      python -m analysis.roll_servo --quick    Tasks A and C only, no 6-DOF
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from sim import aerodata, canards as cn
from sim import dynamics as dyn, integrate as ig, projectile as pr
from gnc import roll_control as rc

# ---------------------------------------------------------------------------
# The adopted configuration
# ---------------------------------------------------------------------------
LATITUDE_DEG = 45.0
CHARGE, MUZZLE_VELOCITY, QE_MILS = 8, 684.0, 525.3
DEPLOY_FRACTION_OF_APOGEE = 0.25

GEOMETRY = cn.CanardGeometry(
    station_from_nose=0.025,
    steering_deflection=math.radians(3.0),
)

#: The nominal bearing and brake, unchanged from step 2.5.
NOSE = cn.NoseAssembly()

#: Task A finds the nominal 0.5 N m brake cannot hold the nose at the adopted
#: deployment point. This is the capacity that can, sized in
#: docs/CONTROL-CHARACTERISATION.md section 2.3. It is a HARDWARE requirement
#: reported as a finding, not a controller tuning knob.
BRAKE_MAX_SIZED = 0.75

#: The capacity ADOPTED after Task E. docs/CONTROL-ROBUSTNESS.md section 4.3
#: sizes the brake against the UPPER bound of the +/-30 % canard aerodynamic
#: estimate rather than its nominal, which extends the holdable band from
#: x1.23 to x1.37 of that estimate and covers the uncertainty the aerodynamics
#: actually carries. It supersedes `BRAKE_MAX_SIZED`.
#:
#: `BRAKE_MAX_SIZED` is left in place because every number in
#: docs/CONTROL-CHARACTERISATION.md and docs/roll_servo.json was measured at
#: it and this module must stay able to reproduce them. New work uses
#: `BRAKE_MAX_ADOPTED`; docs/STAGED-DEPLOYMENT.md section 3 reports what the
#: change moves.
BRAKE_MAX_ADOPTED = 0.85

SWEEP_DT = 5.0e-4
LOG_EVERY = 100

#: Flight times at which Task C is measured, as seconds after deployment. The
#: first is 0.5 s, not 0: at deployment the nose is still mechanically locked
#: to the body at 1308 rad/s, so a step response there would be measuring the
#: despin transient. That transient is measured separately, and in the 6-DOF,
#: as `acquisition_s`.
CHAR_OFFSETS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 17.0, 22.0,
                27.0, 32.0, 37.0, 41.0)

#: Step sizes for Task C.1 and C.2, degrees. Positive is FORWARD, the
#: brake-driven direction; negative is the passive return.
#:
#: 179 rather than 180 because a 180 degree command is the antipode and its
#: direction is decided by the wrapping convention, not by the request: the
#: loop always takes the short way round, so 179 is the largest move that can
#: be commanded in a chosen direction and 180 is the largest that exists.
STEP_SIZES_DEG = (90.0, -90.0, 179.0, -179.0)

#: Command frequencies for the Task C.4 bandwidth sweep, Hz. The sweep is run
#: at a subset of the characterisation times because each point costs several
#: command cycles of simulation and the low-frequency end is long.
BANDWIDTH_HZ = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
BANDWIDTH_AMPLITUDES_DEG = (10.0, 45.0)
BANDWIDTH_OFFSETS = (1.0, 3.0, 8.0, 17.0, 27.0, 37.0)
BANDWIDTH_CYCLES = 3.0


# ===========================================================================
# Models
# ===========================================================================
def base_model():
    """The step-1 ballistic model: no nose, no canards, no Coriolis."""
    env = pr.Environment.from_degrees(LATITUDE_DEG, include_coriolis=False)
    return dyn.FlightModel(
        projectile=pr.M107, aero=aerodata.make_m107_table(), environment=env
    )


def make_plant(brake_max: float = None, **nose_overrides) -> rc.NoseRollPlant:
    nose = NOSE
    if brake_max is not None:
        nose = replace(nose, brake_max=brake_max)
    if nose_overrides:
        nose = replace(nose, **nose_overrides)
    return rc.plant_from(GEOMETRY, nose, pr.M107)


def _states_from(tr) -> np.ndarray:
    n = tr.t.size
    y = np.zeros((n, dyn.STATE_SIZE))
    y[:, 0:3] = tr.position
    y[:, 3:6] = tr.velocity
    y[:, 6:10] = tr.quaternion
    y[:, 10:13] = tr.omega
    y[:, 13] = tr.nose_angle
    y[:, 14] = tr.nose_rate
    return y


# ===========================================================================
# Baseline: the uncorrected leg and the free-nose guided leg
# ===========================================================================
def baseline(dt: float = SWEEP_DT) -> dict:
    """
    Fly the engagement twice: uncorrected, to fix the deployment state and the
    impact point every correction is differenced against, and guided with a
    FREE nose, to supply the dynamic-pressure schedule the reduced model runs
    on and the controller schedules against.

    The free-nose leg rather than the uncorrected one, because deploying the
    canards costs 227 m of range and therefore changes the whole downstream
    dynamic-pressure history. Scheduling on the wrong one would mis-set the
    feed-forward by a few per cent for no reason.
    """
    launch = pr.LaunchConditions.from_mils(MUZZLE_VELOCITY, QE_MILS)
    y0 = dyn.initial_state(pr.M107, launch)
    unc = ig.integrate(y0, base_model(), dt=dt, log_every=LOG_EVERY, t_max=200.0)
    tr = unc.trajectory
    alt = -tr.position[:, 2]
    t_apogee = float(tr.t[int(np.argmax(alt))])

    i = int(np.argmin(np.abs(tr.t - DEPLOY_FRACTION_OF_APOGEE * t_apogee)))
    t_dep = float(tr.t[i])
    y_dep = _states_from(tr)[i]

    free_nose = replace(NOSE, deploy_time=t_dep)
    guided = cn.guided_model(base_model(), GEOMETRY, free_nose)
    gr = ig.integrate(y_dep, guided, dt=dt, log_every=LOG_EVERY, t_max=200.0,
                      t_start=t_dep)
    gt = gr.trajectory

    return {
        "dt": dt,
        "charge": CHARGE, "muzzle_velocity": MUZZLE_VELOCITY, "qe_mils": QE_MILS,
        "uncorrected_range": unc.range_m,
        "uncorrected_drift": unc.drift_m,
        "uncorrected_tof": unc.impact_time,
        "apogee_time": t_apogee,
        "deploy_time": t_dep,
        "mach_at_deploy": float(tr.mach[i]),
        "qbar_at_deploy": float(tr.dynamic_pressure[i]),
        "spin_at_deploy": float(tr.omega[i, 0]),
        "y_deploy": y_dep.tolist(),
        # Deployment one, two, three and four logged samples later. The log
        # cadence is 0.05 s, which is ten and a half revolutions of the body,
        # so these are the SAME flight condition at an unrelated body roll
        # phase -- which is what the fuze actually cannot control, and what
        # the engagement-phase ensemble samples.
        "y_deploy_alt": [_states_from(tr)[i + k].tolist() for k in (1, 2, 3, 4)],
        "t_deploy_alt": [float(tr.t[i + k]) for k in (1, 2, 3, 4)],
        "free_range": gr.range_m,
        "free_drift": gr.drift_m,
        "free_tof": gr.impact_time,
        "schedule_t": gt.t.tolist(),
        "schedule_qbar": gt.dynamic_pressure.tolist(),
        "schedule_V": gt.airspeed.tolist(),
        "schedule_mach": gt.mach.tolist(),
        "schedule_spin": gt.omega[:, 0].tolist(),
        "free_nose_spin": gt.nose_spin.tolist(),
        "free_nose_rate": gt.nose_rate.tolist(),
    }


def schedule_from(base: dict) -> rc.ConditionSchedule:
    t = np.asarray(base["schedule_t"])
    p = np.asarray(base["schedule_spin"])
    pdot = np.gradient(p, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(np.abs(p) > 1.0, -pr.M107.I_axial * pdot / p, 0.0)
    return rc.ConditionSchedule(
        time=t,
        qbar=np.asarray(base["schedule_qbar"]),
        airspeed=np.asarray(base["schedule_V"]),
        mach=np.asarray(base["schedule_mach"]),
        spin=p,
        body_damping=k,
    )


# ===========================================================================
# TASK A -- the one-sided actuator envelope
# ===========================================================================
def task_a(base: dict, brake_max_variants=(0.5, BRAKE_MAX_SIZED)) -> dict:
    """
    Maximum forward slew rate at full brake and maximum passive return rate at
    zero brake, as functions of flight time, for each brake capacity.

    Both are steady-state rates of the linearised torque balance; the approach
    to either is first order with the tabulated time constant, and the 6-DOF
    check in `task_a_verify` confirms both to better than a per cent.
    """
    sched = schedule_from(base)
    t_dep = base["deploy_time"]
    t_imp = base["free_tof"]
    times = np.asarray(base["schedule_t"])
    times = times[(times >= t_dep) & (times <= t_imp)]

    out = {"deploy_time": t_dep, "impact_time": t_imp, "variants": {}}
    for u_max in brake_max_variants:
        plant = make_plant(brake_max=u_max)
        rows = []
        for tt in times:
            c = sched.at(float(tt))
            rows.append(plant.envelope(c).as_dict())
        can = np.array([r["can_hold"] for r in rows])
        first = int(np.argmax(can)) if can.any() else -1
        out["variants"][f"{u_max:.2f}"] = {
            "brake_max_Nm": u_max,
            "rows": rows,
            "hold_possible_fraction": float(can.mean()),
            "first_hold_time_s": float(times[first]) if first >= 0 else None,
            "dead_window_s": float(times[first] - t_dep) if first >= 0 else None,
            "min_saturation_margin": float(min(r["saturation_margin"] for r in rows)),
            "max_forward_rate_degs": float(max(r["rate_full_degs"] for r in rows)),
            "max_return_rate_degs": float(min(r["rate_free_degs"] for r in rows)),
        }
    return out


def task_a_verify(args) -> dict:
    """
    Check the closed-form rate limits against the 6-DOF by holding the brake
    at a constant command and letting the nose settle.
    """
    base = args["baseline"]
    u = float(args["brake"])
    t_dep = base["deploy_time"]
    y_dep = np.asarray(base["y_deploy"])
    settle_from = float(args.get("settle_from", 3.0))

    nose = replace(NOSE, deploy_time=t_dep, brake_command=lambda t: u,
                   brake_max=max(u, NOSE.brake_max))
    guided = cn.guided_model(base_model(), GEOMETRY, nose)
    res = ig.integrate(y_dep, guided, dt=base["dt"], log_every=LOG_EVERY,
                       t_max=200.0, t_start=t_dep)
    tr = res.trajectory
    sched = schedule_from(base)
    plant = make_plant(brake_max=max(u, NOSE.brake_max))

    rows = []
    for tt in (t_dep + settle_from, t_dep + 10.0, t_dep + 20.0, t_dep + 30.0):
        if tt > tr.t[-1]:
            continue
        i = int(np.argmin(np.abs(tr.t - tt)))
        c = rc.FlightCondition(float(tr.t[i]), float(tr.dynamic_pressure[i]),
                               float(tr.airspeed[i]), float(tr.mach[i]),
                               float(tr.omega[i, 0]))
        rows.append({
            "time_s": float(tr.t[i]),
            "brake_Nm": u,
            "sixdof_nose_spin_rads": float(tr.nose_spin[i]),
            "reduced_equilibrium_rads": plant.equilibrium_rate(c, u),
        })
    for r in rows:
        a, b = r["sixdof_nose_spin_rads"], r["reduced_equilibrium_rads"]
        r["absolute_error_rads"] = abs(a - b)
        r["relative_error"] = abs(a - b) / max(abs(b), 1e-9)
    return {"brake_Nm": u, "rows": rows,
            "max_absolute_error_rads": max(r["absolute_error_rads"] for r in rows)}


# ===========================================================================
# TASK C -- slew, settling, steady-state error, bandwidth
# ===========================================================================
def _controller(plant, cfg=None, brake_max=None):
    cfg = cfg or rc.ControllerConfig()
    act = rc.BrakeActuator(brake_max=plant.nose.brake_max)
    return rc.RollAngleController(plant, cfg, act)


def _settle(run: rc.ReducedRun, t_step: float, change: float,
            tol_deg: float = 1.0) -> dict:
    """
    Step-response metrics from a reduced-model run. `change` is the SIGNED
    commanded motion, in radians, which the caller knows and the wrapped
    command does not carry.

    Times are measured from `t_step`. `rise` is the first crossing of 90 % of
    the commanded change; `settle` is the last time the error leaves a
    +/- 1 degree band, and is None if the run never stays inside it, which is
    what happens where the brake cannot out-torque the cant and the nose is
    still running backwards at the end of the window. `overshoot` is reported
    only for a run that settles, because on one that does not it measures a
    runaway rather than a transient.
    """
    m = run.t >= t_step
    t = run.t[m] - t_step
    # Unwrap so a near-180 degree move reads as a 180 degree motion rather
    # than wrapping round to the other side.
    phi = np.unwrap(run.phi_nose[m])
    travelled = phi - phi[0]
    tol = math.radians(tol_deg)

    frac = travelled / change
    hit90 = np.nonzero(frac >= 0.9)[0]
    rise = float(t[hit90[0]]) if hit90.size else None

    err = np.abs(change - travelled)
    outside = np.nonzero(err > tol)[0]
    if outside.size == 0:
        settle, settled = 0.0, True
    elif outside[-1] == t.size - 1:
        settle, settled = None, False
    else:
        settle, settled = float(t[outside[-1] + 1]), True

    overshoot = (max(0.0, float(np.max(frac)) - 1.0) * 100.0) if settled else None
    return {"rise_s": rise, "settle_s": settle, "settled": settled,
            "overshoot_pct": overshoot,
            "final_error_deg": float(np.degrees(err[-1])),
            "reached": bool(hit90.size)}


def task_c(base: dict, brake_max: float, cfg: rc.ControllerConfig = None,
           truth_plant=None, offsets=CHAR_OFFSETS) -> dict:
    """
    Tasks C.1-C.4 at every characterisation time, on the reduced model.

    Each point is run as: settle at phi = 0 for `pre` seconds, step the
    command, then hold. The settling window is 4 seconds, which is 8-16 plant
    time constants and 24 outer-loop time constants.
    """
    cfg = cfg or rc.ControllerConfig()
    sched = schedule_from(base)
    t_dep = base["deploy_time"]
    t_imp = base["free_tof"]
    plant = make_plant(brake_max=brake_max)
    truth = truth_plant or plant

    pre, post = 2.0, 4.0
    steps, holds, bands = [], [], []

    for off in offsets:
        t_step = t_dep + off
        t0 = max(t_dep, t_step - pre)
        t1 = min(t_imp, t_step + post)
        if t1 - t_step < 1.0:
            continue
        cond = sched.at(t_step)
        env = truth.envelope(cond)

        # --- C.1 and C.2: steps in both directions ----------------------
        u_hold0 = min(truth.hold_command(sched.at(t0)), brake_max)
        for size_deg in STEP_SIZES_DEG:
            ctl = _controller(plant, cfg)
            ctl.actuator.reset(max(u_hold0, 0.0))
            cmd = rc.step_command(0.0, math.radians(size_deg), t_step)
            run = rc.simulate_reduced(plant, ctl, sched, cmd, t0, t1,
                                      phi0=0.0, p_nose0=0.0, truth=truth)
            met = _settle(run, t_step, math.radians(size_deg))
            met.update({
                "time_s": t_step,
                "since_deploy_s": off,
                "step_deg": size_deg,
                "direction": "forward" if size_deg > 0 else "return",
                "qbar_kPa": 1e-3 * cond.qbar,
                "mach": cond.mach,
                "rate_limit_degs": env.rate_full_degs if size_deg > 0
                else env.rate_free_degs,
                "accel_limit_rads2": env.accel_forward if size_deg > 0
                else env.accel_return,
                "can_hold": env.can_hold,
            })
            steps.append(met)

        # --- C.3: steady-state tracking error ----------------------------
        ctl = _controller(plant, cfg)
        ctl.actuator.reset(max(u_hold0, 0.0))
        cmd = rc.constant_command(0.0)
        t1h = min(t_imp, t_step + 6.0)
        run = rc.simulate_reduced(plant, ctl, sched, cmd, t0, t1h,
                                  phi0=0.0, p_nose0=0.0, truth=truth)
        tail = run.t >= t1h - 1.0
        err = np.abs(run.error[tail])
        holds.append({
            "time_s": t_step,
            "since_deploy_s": off,
            "qbar_kPa": 1e-3 * cond.qbar,
            "can_hold": env.can_hold,
            "mean_abs_error_deg": float(np.degrees(err.mean())),
            "max_abs_error_deg": float(np.degrees(err.max())),
            "rms_error_deg": float(np.degrees(np.sqrt((err ** 2).mean()))),
            "mean_brake_Nm": float(run.brake[tail].mean()),
            "brake_duty_of_capacity": float(run.brake[tail].mean() / brake_max),
            "saturated_fraction": float(run.saturated[tail].mean()),
            "correction_retained": float(np.cos(err).mean()),
        })

        # --- C.4: bandwidth ----------------------------------------------
        for amp_deg in (BANDWIDTH_AMPLITUDES_DEG if off in BANDWIDTH_OFFSETS else ()):
            row = {"time_s": t_step, "since_deploy_s": off,
                   "amplitude_deg": amp_deg, "qbar_kPa": 1e-3 * cond.qbar,
                   "can_hold": env.can_hold, "points": []}
            for f in BANDWIDTH_HZ:
                t1b = t_step + max(1.5, BANDWIDTH_CYCLES / f)
                if t1b > t_imp:
                    continue
                ctl = _controller(plant, cfg)
                cmd = rc.sine_command(0.0, math.radians(amp_deg), f, t_step)
                run = rc.simulate_reduced(plant, ctl, sched, cmd,
                                          max(t_dep, t_step - 1.0), t1b,
                                          phi0=0.0, p_nose0=0.0, truth=truth)
                # Measure over the last two cycles, after the transient.
                m = run.t >= t1b - 2.0 / f
                if m.sum() < 8:
                    continue
                gain, phase = _describe(run.t[m], run.phi_command[m],
                                        run.phi_nose[m], f)
                row["points"].append({
                    "frequency_hz": f, "gain": gain,
                    "gain_dB": 20.0 * math.log10(max(gain, 1e-9)),
                    "phase_deg": phase,
                })
            row["bandwidth_hz"] = _minus3db(row["points"])
            row["phase_90_hz"] = _phase_cross(row["points"], -90.0)
            bands.append(row)

    return {"brake_max_Nm": brake_max, "steps": steps, "holds": holds,
            "bandwidth": bands}


def _describe(t, cmd, out, f):
    """
    Describing-function gain and phase of `out` relative to `cmd` at frequency
    `f`, by least-squares projection onto sine and cosine. Not an FFT, because
    the record is a few cycles long and need not contain a whole number of
    samples per cycle.
    """
    w = 2.0 * math.pi * f
    basis = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    ac, *_ = np.linalg.lstsq(basis, cmd, rcond=None)
    ao, *_ = np.linalg.lstsq(basis, out, rcond=None)
    zc = complex(ac[0], ac[1])
    zo = complex(ao[0], ao[1])
    if abs(zc) < 1e-12:
        return float("nan"), float("nan")
    ratio = zo / zc
    return abs(ratio), math.degrees(math.atan2(ratio.imag, ratio.real))


def _minus3db(points):
    """First frequency at which the gain falls below -3 dB, log-interpolated."""
    pts = [p for p in points if np.isfinite(p["gain_dB"])]
    for a, b in zip(pts, pts[1:]):
        if a["gain_dB"] >= -3.0 > b["gain_dB"]:
            fa, fb = math.log(a["frequency_hz"]), math.log(b["frequency_hz"])
            ga, gb = a["gain_dB"], b["gain_dB"]
            return float(math.exp(fa + (fb - fa) * (-3.0 - ga) / (gb - ga)))
    if pts and pts[-1]["gain_dB"] >= -3.0:
        return float(pts[-1]["frequency_hz"])     # not reached in the sweep
    return None


def _phase_cross(points, target_deg):
    pts = [p for p in points if np.isfinite(p["phase_deg"])]
    for a, b in zip(pts, pts[1:]):
        if a["phase_deg"] >= target_deg > b["phase_deg"]:
            fa, fb = math.log(a["frequency_hz"]), math.log(b["frequency_hz"])
            pa, pb = a["phase_deg"], b["phase_deg"]
            return float(math.exp(fa + (fb - fa) * (target_deg - pa) / (pb - pa)))
    return None


def task_c_acquisition(base: dict, brake_max: float,
                       cfg: rc.ControllerConfig = None,
                       n_angles: int = 12, truth_plant=None) -> dict:
    """
    Time from deployment to the first useful correction.

    At deployment the nose is still mechanically locked to the body at
    1308 rad/s -- stowage carries it up the tube, not the bearing -- and the
    brake has NO authority there at all, because its torque is proportional to
    the relative rate which is exactly zero. The nose must first be despun
    aerodynamically, sweeping several revolutions in the process, before the
    loop can capture any angle.

    The commanded angle is swept round the circle because the body roll angle
    at deployment is not controllable: the shell turns 208 times a second, so
    which angle the nose happens to be passing when the brake becomes useful
    is effectively random. The spread over the sweep is the real uncertainty
    on the acquisition time.
    """
    cfg = cfg or rc.ControllerConfig()
    sched = schedule_from(base)
    plant = make_plant(brake_max=brake_max)
    truth = truth_plant or plant
    t_dep = base["deploy_time"]
    p0 = sched.at(t_dep).body_spin

    rows = []
    for k in range(n_angles):
        angle = 2.0 * math.pi * k / n_angles
        ctl = _controller(plant, cfg)
        run = rc.simulate_reduced(plant, ctl, sched, rc.constant_command(angle),
                                  t_dep, min(t_dep + 12.0, base["free_tof"]),
                                  phi0=0.0, p_nose0=p0, truth=truth)
        err = np.abs(run.error)
        t_acq = _first_dwell(run.t, err, math.radians(5.0), 0.5)
        t_acq1 = _first_dwell(run.t, err, math.radians(1.0), 0.5)
        rows.append({
            "command_deg": math.degrees(angle),
            "acquire_5deg_s": (t_acq - t_dep) if t_acq is not None else None,
            "acquire_1deg_s": (t_acq1 - t_dep) if t_acq1 is not None else None,
            "revolutions_before_capture": float(
                abs(run.phi_nose[np.argmin(np.abs(run.t - (t_acq or run.t[-1])))]
                    - run.phi_nose[0]) / (2.0 * math.pi)),
        })
    got = [r["acquire_5deg_s"] for r in rows if r["acquire_5deg_s"] is not None]
    got1 = [r["acquire_1deg_s"] for r in rows if r["acquire_1deg_s"] is not None]
    return {
        "brake_max_Nm": brake_max,
        "body_spin_at_deploy_rads": p0,
        "rows": rows,
        "acquired": len(got),
        "of": n_angles,
        "min_s": float(min(got)) if got else None,
        "median_s": float(np.median(got)) if got else None,
        "max_s": float(max(got)) if got else None,
        "median_1deg_s": float(np.median(got1)) if got1 else None,
        "max_1deg_s": float(max(got1)) if got1 else None,
    }


def _first_dwell(t, err, tol, dwell):
    inside = err <= tol
    for i in range(t.size):
        if not inside[i]:
            continue
        j = int(np.searchsorted(t, t[i] + dwell))
        if j >= t.size:
            return None
        if inside[i:j].all():
            return float(t[i])
    return None


def task_c_actuator_lag(base: dict, brake_max: float,
                        lags_s=(0.0, 0.003, 0.005, 0.010, 0.020, 0.040),
                        offset: float = 17.0) -> dict:
    """
    What the closed-loop bandwidth would be for a different brake coil.

    The loop is bandwidth-limited by the actuator, not rate-limited by the
    brake -- the steady rate limits are one to two orders of magnitude beyond
    anything the loop asks for. So the lever on servo speed is the electrical
    time constant of the brake, not the control gains, and this prices it.

    Each lag is paired with the bandwidths it supports under the same sizing
    rule used for the nominal design: rate loop at a third of the actuator
    pole, angle loop at a fifth of that.
    """
    sched = schedule_from(base)
    plant = make_plant(brake_max=brake_max)
    t_step = base["deploy_time"] + offset
    cond = sched.at(t_step)
    rows = []
    for tau in lags_s:
        pole = (1.0 / tau) if tau > 0 else 300.0     # sample-rate limited
        cfg = rc.ControllerConfig(rate_bandwidth=pole / 3.0,
                                  angle_bandwidth=pole / 15.0,
                                  integral_zero=pole / 15.0)
        pts = []
        for f in BANDWIDTH_HZ + (8.0, 12.0, 20.0):
            t1 = t_step + max(1.0, BANDWIDTH_CYCLES / f)
            ctl = _controller(plant, cfg)
            ctl.actuator.tau = tau
            ctl.actuator.reset(min(plant.hold_command(cond), brake_max))
            cmd = rc.sine_command(0.0, math.radians(10.0), f, t_step)
            run = rc.simulate_reduced(plant, ctl, sched, cmd,
                                      t_step - 0.5, t1, phi0=0.0, p_nose0=0.0)
            m = run.t >= t1 - 2.0 / f
            if m.sum() < 8:
                continue
            g, ph = _describe(run.t[m], run.phi_command[m], run.phi_nose[m], f)
            pts.append({"frequency_hz": f, "gain": g,
                        "gain_dB": 20.0 * math.log10(max(g, 1e-9)),
                        "phase_deg": ph})
        rows.append({"actuator_tau_s": tau,
                     "rate_bandwidth_rads": cfg.rate_bandwidth,
                     "angle_bandwidth_rads": cfg.angle_bandwidth,
                     "bandwidth_hz": _minus3db(pts), "points": pts})
    return {"offset_s": offset, "qbar_kPa": 1e-3 * cond.qbar, "rows": rows}


# ===========================================================================
# The 6-DOF closed-loop run -- shared by the Task C verification and Task D
# ===========================================================================
def run_closed_loop(args) -> dict:
    """
    One guided trajectory with the servo in the loop, restarted from the
    shared uncorrected leg at the deployment state.

    `command` is described by a small dict rather than a callable so that the
    case can be shipped to a worker process.
    """
    base = args["baseline"]
    dt = args.get("dt", base["dt"])
    k = args.get("deploy_sample", 0)
    if k:
        t_dep = base["t_deploy_alt"][k - 1]
        y_dep = np.asarray(base["y_deploy_alt"][k - 1])
    else:
        t_dep = base["deploy_time"]
        y_dep = np.asarray(base["y_deploy"])
    brake_max = float(args.get("brake_max", NOSE.brake_max))
    spec = args["command"]

    cfg = rc.ControllerConfig(**args.get("config", {}))
    plant = make_plant(brake_max=brake_max, **args.get("nose_overrides", {}))
    truth_nose = replace(plant.nose, deploy_time=t_dep)

    controller = _controller(plant, cfg)
    command = _build_command(spec, t_dep)
    law = rc.BrakeLaw(controller, command, deploy_time=t_dep)

    nose = replace(truth_nose, brake_command=law.brake_command)
    guided = cn.guided_model(base_model(), GEOMETRY, nose)
    res = ig.integrate(y_dep, guided, dt=dt, log_every=LOG_EVERY, t_max=200.0,
                       t_start=t_dep, step_hook=law.sample)
    if law.samples == 0:
        raise RuntimeError("the controller never sampled: step_hook not wired")

    hist = law.history()
    tr = res.trajectory
    held = hist["holding"]
    err = np.abs(hist["error"])
    # Acquisition: the first time the loop is inside 5 degrees and stays there
    # for half a second, measured from deployment.
    acq = _acquisition_time(hist["t"], err, hist["holding"], t_dep)

    settled = held & (hist["t"] > t_dep + (acq if acq is not None else 0.0) + 0.5)
    out = {
        "label": spec.get("label", ""),
        "command": spec,
        "brake_max_Nm": brake_max,
        "deploy_time": t_dep,
        "deploy_sample": k,
        "range_m": res.range_m,
        "drift_m": res.drift_m,
        "tof_s": res.impact_time,
        "d_range_m": res.range_m - base["uncorrected_range"],
        "d_deflection_m": res.drift_m - base["uncorrected_drift"],
        "d_range_vs_free_m": res.range_m - base["free_range"],
        "d_deflection_vs_free_m": res.drift_m - base["free_drift"],
        "max_total_aoa_deg": math.degrees(res.max_total_aoa),
        "samples": law.samples,
        "acquisition_s": acq,
        "engage_s": (controller.state.engage_time - t_dep)
        if controller.state.engage_time is not None else None,
        "engage_gate": cfg.engage_gate,
        "pre_engage_brake": cfg.pre_engage_brake,
        "hold_fraction": float(held.mean()),
        "saturated_fraction_while_holding": float(
            hist["saturated"][held].mean()) if held.any() else 0.0,
        "mean_abs_error_deg_settled": float(np.degrees(err[settled].mean()))
        if settled.any() else None,
        "rms_error_deg_settled": float(
            np.degrees(np.sqrt((err[settled] ** 2).mean()))) if settled.any() else None,
        "p95_abs_error_deg_settled": float(
            np.degrees(np.percentile(err[settled], 95))) if settled.any() else None,
        "correction_retained_settled": float(np.cos(err[settled]).mean())
        if settled.any() else None,
        "mean_brake_Nm_while_holding": float(hist["brake"][held].mean())
        if held.any() else 0.0,
        "min_abs_p_rel_rads": float(np.abs(tr.nose_rate).min()),
        "min_abs_p_rel_after_transient": float(
            np.abs(tr.nose_rate[tr.t > t_dep + 0.5]).min()),
        "bearing_slip_energy_kJ": _slip_energy(tr, nose),
    }
    if args.get("keep_history"):
        out["history"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in hist.items()}
        out["traj_t"] = tr.t.tolist()
        out["traj_nose_spin"] = tr.nose_spin.tolist()
        out["traj_qbar"] = tr.dynamic_pressure.tolist()
        out["traj_q_rate"] = tr.omega[:, 1].tolist()
        out["traj_total_aoa_deg"] = np.degrees(tr.total_aoa).tolist()
    return out


def run_ideal_hold(args) -> dict:
    """
    The step-2.5 idealisation: `phi_nose` pinned by a kinematic constraint
    from deployment, infinite bandwidth, unlimited torque, zero error.

    This is the denominator step 2.5 left for step 4 -- its limitation 3 said
    the envelope had to be discounted by whatever the servo actually achieves,
    and this is the run the closed-loop one is discounted against.
    """
    base = args["baseline"]
    t_dep = base["deploy_time"]
    y_dep = np.asarray(base["y_deploy"])
    angle = math.radians(args["angle_deg"])
    nose = replace(NOSE, deploy_time=t_dep, hold_angle=angle)
    guided = cn.guided_model(base_model(), GEOMETRY, nose)
    res = ig.integrate(y_dep, guided, dt=base["dt"], log_every=LOG_EVERY,
                       t_max=200.0, t_start=t_dep)
    return {
        "label": f"ideal_{args['angle_deg']:.0f}",
        "angle_deg": args["angle_deg"],
        "range_m": res.range_m,
        "drift_m": res.drift_m,
        "d_range_m": res.range_m - base["uncorrected_range"],
        "d_deflection_m": res.drift_m - base["uncorrected_drift"],
        "max_total_aoa_deg": math.degrees(res.max_total_aoa),
    }


def _acquisition_time(t, err, holding, t_dep, tol_deg=5.0, dwell=0.5):
    tol = math.radians(tol_deg)
    inside = (err <= tol) & holding
    for i in range(t.size):
        if not inside[i]:
            continue
        j = np.searchsorted(t, t[i] + dwell)
        if j >= t.size:
            break
        if inside[i:j].all():
            return float(t[i] - t_dep)
    return None


def _slip_energy(tr, nose) -> float:
    """Bearing dissipation over the guided phase, kJ. A hardware finding."""
    p_rel = tr.nose_rate
    torque = np.array([nose.friction_torque(float(p)) for p in p_rel])
    power = np.abs(torque * p_rel)
    return float(np.trapezoid(power, tr.t) / 1e3)


def _build_command(spec: dict, t_dep: float):
    kind = spec["kind"]
    if kind == "hold":
        return rc.constant_command(math.radians(spec["angle_deg"]))
    if kind == "duty":
        return rc.DutyCycleCommander(
            angle=math.radians(spec["angle_deg"]),
            period=spec["period_s"],
            duty=spec["duty"],
            phase=t_dep + spec.get("phase_s", 0.0),
        )
    if kind == "window":
        # Hold the angle only inside [start, start + length); free otherwise.
        angle = math.radians(spec["angle_deg"])
        t0 = t_dep + spec["start_s"]
        t1 = t0 + spec["length_s"]

        def cmd(t: float) -> tuple:
            return angle, (t0 <= t < t1)
        return cmd
    if kind == "step":
        before = math.radians(spec["before_deg"])
        after = math.radians(spec["after_deg"])
        t_step = t_dep + spec["step_s"]

        def cmd(t: float) -> tuple:
            return (after if t >= t_step else before), True
        return cmd
    raise ValueError(f"unknown command kind {kind!r}")


# ===========================================================================
# The frequencies a duty cycle must not excite
# ===========================================================================
def epicyclic_frequencies(base: dict) -> dict:
    """
    The shell's two yawing modes, from the standard closed form.

        phidot_fast,slow = (Ix p / 2 It) [ 1 +/- sqrt(1 - 1/Sg) ]

    Task D found that switching between hold and release costs far more than
    the hold time it gives up, and that the cost grows sharply as the
    switching period shortens. This is why: a duty cycle applies a
    step-like transverse force at its own frequency, and when that frequency
    approaches the SLOW (precession) mode it pumps the yawing motion
    resonantly. The induced drag of the resulting angle of attack is then
    paid for over the whole remaining flight.

    The frequencies are a property of the airframe, not of the servo, and
    step 3 needs them to choose a correction schedule.
    """
    from sim import diagnostics as dg

    proj = pr.M107
    t = np.asarray(base["schedule_t"])
    V = np.asarray(base["schedule_V"])
    mach = np.asarray(base["schedule_mach"])
    p = np.asarray(base["schedule_spin"])
    qbar = np.asarray(base["schedule_qbar"])
    rho = 2.0 * qbar / np.maximum(V, 1e-6) ** 2
    aero = aerodata.make_m107_table()

    rows = []
    for i in range(t.size):
        c = aero.coefficients_at(float(mach[i]))
        Sg = dg.gyroscopic_stability_factor(proj, float(rho[i]), float(V[i]),
                                            float(p[i]), c.C_Malpha)
        if Sg <= 1.0:
            continue
        base_rate = proj.I_axial * p[i] / (2.0 * proj.I_transverse)
        root = math.sqrt(1.0 - 1.0 / Sg)
        fast = float(base_rate * (1.0 + root))
        slow = float(base_rate * (1.0 - root))
        rows.append({
            "time_s": float(t[i]),
            "since_deploy_s": float(t[i] - base["deploy_time"]),
            "mach": float(mach[i]),
            "Sg": float(Sg),
            "fast_rads": fast, "fast_hz": fast / (2.0 * math.pi),
            "slow_rads": slow, "slow_hz": slow / (2.0 * math.pi),
        })
    slow_hz = [r["slow_hz"] for r in rows]
    fast_hz = [r["fast_hz"] for r in rows]
    return {
        "rows": rows,
        "slow_hz_min": min(slow_hz), "slow_hz_max": max(slow_hz),
        "slow_hz_median": float(np.median(slow_hz)),
        "fast_hz_min": min(fast_hz), "fast_hz_max": max(fast_hz),
        "fast_hz_median": float(np.median(fast_hz)),
    }


def measured_yaw_spectrum(base: dict, case: dict) -> dict:
    """
    The yawing motion actually excited in one 6-DOF run, as a spectrum of the
    transverse rate. Confirms that the closed-form slow mode is the one a
    duty cycle is exciting, rather than asserting it.
    """
    t = np.asarray(case["traj_t"])
    q = np.asarray(case.get("traj_q_rate", []))
    if q.size != t.size or t.size < 32:
        return {}
    dt = float(np.median(np.diff(t)))
    win = np.hanning(t.size)
    spec = np.abs(np.fft.rfft((q - q.mean()) * win))
    freq = np.fft.rfftfreq(t.size, dt)
    k = int(np.argmax(spec[1:]) + 1)
    return {"peak_hz": float(freq[k]),
            "nyquist_hz": float(freq[-1]),
            "top": [{"hz": float(freq[j]), "amplitude": float(spec[j])}
                    for j in np.argsort(spec[1:])[::-1][:5] + 1]}


# ===========================================================================
# TASK D -- the duty-cycle mode
# ===========================================================================
def task_d_cases(base: dict, brake_max: float,
                 angles_deg=tuple(float(a) for a in range(0, 360, 45)),
                 duty_angle_deg: float = 180.0,
                 duties=(0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
                 periods=(1.0, 2.0, 4.0, 8.0, 16.0),
                 quantum_starts=(0.5, 5.0, 12.0, 22.0, 32.0),
                 quantum_lengths=(0.25, 0.5, 1.0, 2.0, 4.0),
                 window_starts=(1.0, 12.0),
                 window_lengths=(2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 41.0)) -> list:
    """
    Every 6-DOF case Task D needs, as a list of worker arguments.

    Three families:

    reference   a full hold at each of four commanded angles, which is the
                closed-loop version of the authority ellipse and the
                denominator for everything else;
    duty        one angle at a grid of duty fractions and periods, to test
                whether average delivered correction is linear in duty and to
                price the switching overhead;
    quantum     a single hold window of a given length at a given time, which
                is the minimum correction the kit can command and therefore a
                floor on terminal precision.
    """
    cases = []
    common = {"baseline": base, "brake_max": brake_max}

    for a in angles_deg:
        cases.append({**common, "command": {
            "kind": "hold", "angle_deg": a, "label": f"hold_{a:.0f}"}})

    for period in periods:
        for d in duties:
            if d >= 1.0 and period != periods[0]:
                continue          # a full duty is the same run at any period
            cases.append({**common, "command": {
                "kind": "duty", "angle_deg": duty_angle_deg, "period_s": period,
                "duty": d, "label": f"duty_{duty_angle_deg:.0f}_T{period}_D{d}"}})

    for s in quantum_starts:
        for L in quantum_lengths:
            cases.append({**common, "command": {
                "kind": "window", "angle_deg": duty_angle_deg, "start_s": s,
                "length_s": L, "label": f"window_{s}_{L}"}})
    # One CONTIGUOUS hold of variable length, which is the magnitude-control
    # scheme a duty cycle turns out to be a bad way of implementing.
    for s in window_starts:
        for L in window_lengths:
            if s + L > 42.0:
                continue
            cases.append({**common, "command": {
                "kind": "window", "angle_deg": duty_angle_deg, "start_s": s,
                "length_s": L, "label": f"window_{s}_{L}"}})
    return cases


def _ellipse(points) -> dict:
    """
    Semi-axes and tilt of the reachable set, from four cardinal roll angles.

    The same first-harmonic fit `analysis.authority` uses: with the response
    linear in phi to better than a per cent, four angles determine the
    ellipse, and its semi-axes are the singular values of the 2x2 harmonic
    coefficient matrix.
    """
    ang = np.radians([p[0] for p in points])
    xy = np.array([[p[1], p[2]] for p in points])
    basis = np.column_stack([np.cos(ang), np.sin(ang), np.ones_like(ang)])
    coef, *_ = np.linalg.lstsq(basis, xy, rcond=None)
    harmonic = coef[:2, :].T                  # 2x2: [range; deflection]
    svals = np.linalg.svd(harmonic, compute_uv=False)
    u, s, vt = np.linalg.svd(harmonic)
    return {
        "semi_major_m": float(svals[0]),
        "semi_minor_m": float(svals[1]),
        "axis_ratio": float(svals[0] / svals[1]) if svals[1] > 0 else float("inf"),
        "major_axis_tilt_deg": float(math.degrees(math.atan2(u[1, 0], u[0, 0]))),
        "centre_range_m": float(coef[2, 0]),
        "centre_deflection_m": float(coef[2, 1]),
        "rms_amplitude_m": float(math.sqrt(0.5 * (svals[0] ** 2 + svals[1] ** 2))),
    }


def servo_cost(base: dict, closed: list, ideal: list) -> dict:
    """
    What the servo costs against the ideal kinematic hold, in metres of
    reachable-set size. Step 2.5 limitation 3, closed.
    """
    free_r = base["free_range"] - base["uncorrected_range"]
    free_d = base["free_drift"] - base["uncorrected_drift"]

    def pts(rows, key="angle_deg"):
        return [(r.get(key, r.get("command", {}).get("angle_deg")),
                 r["d_range_m"] - free_r, r["d_deflection_m"] - free_d)
                for r in rows]

    e_closed = _ellipse(pts(closed))
    e_ideal = _ellipse(pts(ideal))
    return {
        "ideal": e_ideal,
        "closed_loop": e_closed,
        "retained_semi_major": e_closed["semi_major_m"] / e_ideal["semi_major_m"],
        "retained_rms": e_closed["rms_amplitude_m"] / e_ideal["rms_amplitude_m"],
        "cost_semi_major_m": e_ideal["semi_major_m"] - e_closed["semi_major_m"],
        "ideal_max_aoa_deg": max(r["max_total_aoa_deg"] for r in ideal),
        "closed_max_aoa_deg": max(r["max_total_aoa_deg"] for r in closed),
        "per_angle": [
            {"angle_deg": c.get("command", {}).get("angle_deg"),
             "closed_range_m": c["d_range_m"] - free_r,
             "closed_deflection_m": c["d_deflection_m"] - free_d,
             "ideal_range_m": i["d_range_m"] - free_r,
             "ideal_deflection_m": i["d_deflection_m"] - free_d,
             "acquisition_s": c["acquisition_s"],
             "tracking_rms_deg": c["rms_error_deg_settled"]}
            for c, i in zip(closed, ideal)
        ],
    }


def task_d_reduce(base: dict, results: list, duty_angle_deg: float = 180.0) -> dict:
    """Turn the raw 6-DOF cases into the Task D answers."""
    by_label = {r["label"]: r for r in results}

    # -- the reference ellipse, closed loop ------------------------------
    holds = [r for r in results if r["command"]["kind"] == "hold"]
    free = {"d_range_m": base["free_range"] - base["uncorrected_range"],
            "d_deflection_m": base["free_drift"] - base["uncorrected_drift"]}
    # Steerable component: the hold shift relative to the free-nose shift,
    # which is the roll-independent drag bias.
    ref = []
    for r in holds:
        dr = r["d_range_m"] - free["d_range_m"]
        dd = r["d_deflection_m"] - free["d_deflection_m"]
        ref.append({"angle_deg": r["command"]["angle_deg"],
                    "steerable_range_m": dr, "steerable_deflection_m": dd,
                    "magnitude_m": math.hypot(dr, dd),
                    "acquisition_s": r["acquisition_s"],
                    "hold_error_deg": r["mean_abs_error_deg_settled"],
                    "saturated_fraction": r["saturated_fraction_while_holding"]})

    full = by_label.get(f"hold_{duty_angle_deg:.0f}")
    full_vec = np.array([full["d_range_m"] - free["d_range_m"],
                         full["d_deflection_m"] - free["d_deflection_m"]])
    full_mag = float(np.linalg.norm(full_vec))
    unit = full_vec / full_mag if full_mag > 0 else np.zeros(2)

    # -- duty fraction against delivered correction -----------------------
    duty_rows = []
    for r in results:
        if r["command"]["kind"] != "duty":
            continue
        v = np.array([r["d_range_m"] - free["d_range_m"],
                      r["d_deflection_m"] - free["d_deflection_m"]])
        duty_rows.append({
            "period_s": r["command"]["period_s"],
            "duty": r["command"]["duty"],
            "steerable_range_m": float(v[0]),
            "steerable_deflection_m": float(v[1]),
            "magnitude_m": float(np.linalg.norm(v)),
            "projected_m": float(v @ unit),
            "fraction_of_full": float(v @ unit) / full_mag if full_mag else 0.0,
            "hold_fraction_commanded": r["hold_fraction"],
            "acquisition_s": r["acquisition_s"],
            "mean_error_deg": r["mean_abs_error_deg_settled"],
        })

    # Linearity: fit fraction_of_full against duty per period and report the
    # departure, plus the effective duty at which delivered correction is half.
    linearity = {}
    for period in sorted({d["period_s"] for d in duty_rows}):
        rows = sorted((d for d in duty_rows if d["period_s"] == period),
                      key=lambda d: d["duty"])
        x = np.array([d["duty"] for d in rows])
        y = np.array([d["fraction_of_full"] for d in rows])
        if x.size < 3:
            continue
        slope, icept = np.polyfit(x, y, 1)
        resid = y - (slope * x + icept)
        linearity[f"{period}"] = {
            "period_s": period,
            "slope": float(slope),
            "intercept": float(icept),
            "max_abs_residual": float(np.max(np.abs(resid))),
            "rms_residual": float(np.sqrt((resid ** 2).mean())),
            "duty": x.tolist(),
            "fraction_of_full": y.tolist(),
            "loss_vs_ideal_pct": [float(100.0 * (xx - yy)) for xx, yy in zip(x, y)],
        }

    # -- the correction quantum -------------------------------------------
    quanta = []
    for r in results:
        if r["command"]["kind"] != "window":
            continue
        v = np.array([r["d_range_m"] - free["d_range_m"],
                      r["d_deflection_m"] - free["d_deflection_m"]])
        quanta.append({
            "start_s": r["command"]["start_s"],
            "length_s": r["command"]["length_s"],
            "steerable_range_m": float(v[0]),
            "steerable_deflection_m": float(v[1]),
            "magnitude_m": float(np.linalg.norm(v)),
            "projected_m": float(v @ unit),
            "per_second_m": float(np.linalg.norm(v)) / r["command"]["length_s"],
            "acquisition_s": r["acquisition_s"],
        })

    return {
        "free_nose_bias": free,
        "reference_holds": ref,
        "reference_magnitude_m": full_mag,
        "reference_angle_deg": duty_angle_deg,
        "duty_rows": duty_rows,
        "linearity": linearity,
        "quanta": quanta,
    }


def _first_sustained(t, inside, dwell):
    """First index from which `inside` holds for `dwell` seconds, or the
    end of the record. A free-running nose sweeps past the commanded angle
    once a revolution, so a single sample inside tolerance is a fly-by and
    not a capture."""
    n = t.size
    for i in range(n):
        if not inside[i]:
            continue
        j = int(np.searchsorted(t, t[i] + dwell))
        if inside[i:min(j, n)].all() and (j >= n or inside[i:j].all()):
            return i
    return None


def hold_arc_stats(hist: dict, tol_deg: float = 5.0, dwell: float = 0.1) -> dict:
    """
    The cost of switching, measured arc by arc.

    Every free-to-hold transition starts with the nose at an essentially
    arbitrary angle -- the free nose sweeps 5 to 13 revolutions in a one-second
    release -- so each hold arc begins with a slew that produces no useful
    correction in the commanded direction. This segments the logged history
    into hold arcs and reports, for each, how long the reacquisition took and
    what fraction of the arc was actually spent on the commanded angle.
    """
    t = np.asarray(hist["t"])
    err = np.abs(np.asarray(hist["error"]))
    hold = np.asarray(hist["holding"], dtype=bool)
    tol = math.radians(tol_deg)

    arcs = []
    i = 0
    n = t.size
    while i < n:
        if not hold[i]:
            i += 1
            continue
        j = i
        while j < n and hold[j]:
            j += 1
        seg_t, seg_e = t[i:j], err[i:j]
        if seg_t.size < 2:
            i = j
            continue
        inside = seg_e <= tol
        k = _first_sustained(seg_t, inside, dwell)
        arcs.append({
            "start_s": float(seg_t[0]),
            "length_s": float(seg_t[-1] - seg_t[0]),
            "reacquire_s": float(seg_t[k] - seg_t[0]) if k is not None else None,
            "inside_fraction": float(inside.mean()),
            "mean_cos_error": float(np.cos(seg_e).mean()),
        })
        i = j

    complete = [a for a in arcs if a["reacquire_s"] is not None]
    return {
        "n_arcs": len(arcs),
        "n_acquired": len(complete),
        "arcs": arcs,
        "median_reacquire_s": float(np.median([a["reacquire_s"] for a in complete]))
        if complete else None,
        "mean_reacquire_s": float(np.mean([a["reacquire_s"] for a in complete]))
        if complete else None,
        "max_reacquire_s": float(np.max([a["reacquire_s"] for a in complete]))
        if complete else None,
        "mean_inside_fraction": float(np.mean([a["inside_fraction"] for a in arcs]))
        if arcs else None,
        "mean_cos_error": float(np.mean([a["mean_cos_error"] for a in arcs]))
        if arcs else None,
        "switching_overhead_pct": float(
            100.0 * (1.0 - np.mean([a["inside_fraction"] for a in arcs]))) if arcs else None,
    }


# ===========================================================================
# Reduced-model against 6-DOF: the fidelity statement
# ===========================================================================
def compare_reduced_to_sixdof(base: dict, sixdof: dict, brake_max: float,
                              cfg: rc.ControllerConfig = None) -> dict:
    """
    Re-run one 6-DOF case on the reduced model and difference the two roll
    histories. This is what licenses every Task C number, all of which are
    measured on the reduced model.
    """
    cfg = cfg or rc.ControllerConfig()
    sched = schedule_from(base)
    plant = make_plant(brake_max=brake_max)
    t_dep = base["deploy_time"]
    spec = sixdof["command"]

    hist = sixdof["history"]
    t6 = np.asarray(hist["t"])
    phi6 = np.asarray(hist["phi_nose"])
    p6 = np.asarray(hist["p_nose"])
    err6 = np.asarray(hist["error"])

    ctl = _controller(plant, cfg)
    command = _build_command(spec, t_dep)
    run = rc.simulate_reduced(plant, ctl, sched, command, t_dep,
                              min(t6[-1], base["free_tof"]),
                              phi0=float(phi6[0]),
                              p_nose0=float(p6[0]))

    m = (t6 >= run.t[0]) & (t6 <= run.t[-1])
    phi_r = np.interp(t6[m], run.t, np.unwrap(run.phi_nose))
    p_r = np.interp(t6[m], run.t, run.p_nose)
    err_r = np.interp(t6[m], run.t, run.error)
    d_phi = np.array([rc.wrap_pi(a - b) for a, b in zip(np.unwrap(phi6[m]), phi_r)])

    settled = t6[m] > t_dep + 3.0
    return {
        "label": spec.get("label", ""),
        "n": int(m.sum()),
        "max_abs_angle_difference_deg": float(np.degrees(np.abs(d_phi).max())),
        "rms_angle_difference_deg": float(np.degrees(np.sqrt((d_phi ** 2).mean()))),
        "rms_angle_difference_settled_deg": float(
            np.degrees(np.sqrt((d_phi[settled] ** 2).mean()))) if settled.any() else None,
        "max_abs_rate_difference_rads": float(np.abs(p6[m] - p_r).max()),
        "sixdof_rms_error_deg": float(np.degrees(np.sqrt((err6[m][settled] ** 2).mean())))
        if settled.any() else None,
        "reduced_rms_error_deg": float(np.degrees(np.sqrt((err_r[settled] ** 2).mean())))
        if settled.any() else None,
    }


def convergence_cases(base: dict, brake_max: float) -> list:
    """
    The same short hold windows at a finer integration step.

    The correction quantum is a DIFFERENCE between two 15.8 km trajectories,
    and the smallest of them are under a metre. A number that size has to be
    shown to be a property of the projectile rather than of the step size
    before it can be quoted as a precision floor.
    """
    fine = {**base, "dt": 2.0e-4}
    # A never-holding reference at the SAME step, because the guided leg's own
    # drag bias moves by 10-15 m between the two step sizes. That is a
    # common-mode shift shared by every corrected run, and differencing it out
    # is what isolates the hold window from the integration.
    out = [{"baseline": fine, "brake_max": brake_max, "dt": 2.0e-4,
            "command": {"kind": "duty", "angle_deg": 180.0, "period_s": 1.0,
                        "duty": 0.0, "label": "fine_free"}}]
    for s_, L in ((12.0, 0.5), (12.0, 2.0), (22.0, 2.0), (32.0, 4.0)):
        out.append({"baseline": fine, "brake_max": brake_max, "dt": 2.0e-4,
                    "command": {"kind": "window", "angle_deg": 180.0,
                                "start_s": s_, "length_s": L,
                                "label": f"fine_{s_}_{L}"}})
    return out


def convergence_reduce(base: dict, coarse: list, fine: list) -> dict:
    """
    Difference the coarse and fine quanta. Reported as an absolute floor in
    metres, because that is what limits the smallest commandable correction.

    The uncorrected leg is flown at the coarse step in both cases, so this
    isolates the step-size sensitivity of the CORRECTED run, which is the one
    carrying the short hold window.
    """
    by = {}
    coarse_free = None
    for r in coarse:
        c = r["command"]
        if c["kind"] == "window":
            by[(c["start_s"], c["length_s"])] = r
        elif c["kind"] == "duty" and c["duty"] == 0.0:
            coarse_free = r
    fine_free = next((r for r in fine
                      if r["command"].get("duty") == 0.0), None)
    if coarse_free is None or fine_free is None:
        return {"rows": [], "worst_difference_m": None,
                "note": "no free-nose reference at one of the two step sizes"}

    rows = []
    for r in fine:
        c = r["command"]
        if c["kind"] != "window":
            continue
        k = (c["start_s"], c["length_s"])
        if k not in by:
            continue
        a, b = by[k], r
        # Steerable component at each step size: the hold window relative to
        # a nose that is never held, flown at the same step.
        ar = a["d_range_m"] - coarse_free["d_range_m"]
        ad = a["d_deflection_m"] - coarse_free["d_deflection_m"]
        br = b["d_range_m"] - fine_free["d_range_m"]
        bd = b["d_deflection_m"] - fine_free["d_deflection_m"]
        rows.append({
            "start_s": c["start_s"], "length_s": c["length_s"],
            "coarse_range_m": ar, "fine_range_m": br,
            "coarse_deflection_m": ad, "fine_deflection_m": bd,
            "coarse_magnitude_m": math.hypot(ar, ad),
            "fine_magnitude_m": math.hypot(br, bd),
            "difference_m": math.hypot(ar - br, ad - bd),
        })
    return {"rows": rows,
            "worst_difference_m": max((r["difference_m"] for r in rows),
                                      default=None)}


def engagement_phase_cases(base: dict, brake_max: float,
                           dwells=(0.0, 0.04, 0.08, 0.12, 0.16),
                           angles=tuple(float(a) for a in range(0, 360, 45))) -> list:
    """
    The reachable set at several engagement dwells.

    The residual acquisition transient is phase-sensitive: the shell turns 208
    times a second, so the body roll angle at the instant the loop engages is
    not a design variable, and how much coning the capture leaves behind
    depends on it. Sixty-four milliseconds of engagement timing moved the
    retained reachable-set rms from 93 % to 81 % between two otherwise
    identical runs, which means a single number from a single phase is luck
    rather than a result.

    `engage_dwell` sweeps that ensemble deliberately. Each dwell is a
    different, equally arbitrary phase, and the SPREAD is the answer.
    """
    return [{"baseline": base, "brake_max": brake_max,
             "config": {"engage_dwell": d},
             "command": {"kind": "hold", "angle_deg": a,
                         "label": f"dwell{d:.2f}_{a:.0f}"}}
            for d in dwells for a in angles]


def engagement_phase_reduce(base: dict, cases: list, ideal: list) -> dict:
    """Servo cost at each engagement dwell, and the spread over them."""
    by_dwell = {}
    for c in cases:
        by_dwell.setdefault(c["command"]["label"].split("_")[0], []).append(c)
    rows = []
    for tag, group in sorted(by_dwell.items()):
        group.sort(key=lambda r: r["command"]["angle_deg"])
        sc = servo_cost(base, group, ideal)
        rows.append({
            "dwell_s": float(tag.replace("dwell", "")),
            "engage_s": group[0]["engage_s"],
            "semi_major_m": sc["closed_loop"]["semi_major_m"],
            "semi_minor_m": sc["closed_loop"]["semi_minor_m"],
            "rms_amplitude_m": sc["closed_loop"]["rms_amplitude_m"],
            "axis_ratio": sc["closed_loop"]["axis_ratio"],
            "retained_semi_major": sc["retained_semi_major"],
            "retained_rms": sc["retained_rms"],
            "max_aoa_deg": sc["closed_max_aoa_deg"],
            "max_acquisition_s": max(r["acquisition_s"] or 0.0 for r in group),
        })
    return {
        "rows": rows,
        "retained_rms_min": min(r["retained_rms"] for r in rows),
        "retained_rms_max": max(r["retained_rms"] for r in rows),
        "retained_rms_median": float(np.median([r["retained_rms"] for r in rows])),
        "retained_semi_major_min": min(r["retained_semi_major"] for r in rows),
        "retained_semi_major_max": max(r["retained_semi_major"] for r in rows),
        "max_aoa_min": min(r["max_aoa_deg"] for r in rows),
        "max_aoa_max": max(r["max_aoa_deg"] for r in rows),
    }


def deployment_phase_cases(base: dict, brake_max: float,
                           samples=(0, 1, 2, 3, 4),
                           angles=tuple(float(a) for a in range(0, 360, 45))) -> list:
    """
    The reachable set at several deployment phases, with the engagement gate on
    and off.

    `deploy_sample` k restarts from the k-th logged sample after the nominal
    deployment time. The log cadence is 0.05 s, which is ten and a half
    revolutions of the body, so each is the SAME flight condition at an
    unrelated body roll phase -- and the body roll phase at capture is exactly
    what the fuze cannot choose.

    This exists because the first, single-phase measurement of the servo's cost
    was not reproducible in the sense that matters: it gave 95.2 % of the ideal
    semi-major axis at one phase and 81.1 % at another 64 ms away, and appeared
    to show the engagement gate worth 24 points when the ensembles overlap.
    Nothing downstream of a capture transient in this plant may be quoted from
    one phase.
    """
    out = []
    for gated in (True, False):
        for k in samples:
            for a in angles:
                out.append({
                    "baseline": base, "brake_max": brake_max,
                    "deploy_sample": k,
                    "config": {} if gated else {"engage_gate": False},
                    "command": {"kind": "hold", "angle_deg": a,
                                "label": f"{'g' if gated else 'u'}{k}_{a:.0f}"},
                })
    return out


def deployment_phase_reduce(base: dict, cases: list, ideal: list) -> dict:
    """Servo cost per deployment phase, gated and ungated, and the spreads."""
    out = {}
    for tag, prefix in (("gated", "g"), ("ungated", "u")):
        runs = [r for r in cases if r["label"].startswith(prefix)]
        rows = []
        for k in sorted({r["deploy_sample"] for r in runs}):
            group = sorted((r for r in runs if r["deploy_sample"] == k),
                           key=lambda r: r["command"]["angle_deg"])
            sc = servo_cost(base, group, ideal)
            rows.append({
                "deploy_sample": k,
                "deploy_time": group[0]["deploy_time"],
                "semi_major_m": sc["closed_loop"]["semi_major_m"],
                "semi_minor_m": sc["closed_loop"]["semi_minor_m"],
                "rms_amplitude_m": sc["closed_loop"]["rms_amplitude_m"],
                "axis_ratio": sc["closed_loop"]["axis_ratio"],
                "retained_semi_major": sc["retained_semi_major"],
                "retained_rms": sc["retained_rms"],
                "max_aoa_deg": sc["closed_max_aoa_deg"],
                "max_acquisition_s": max(r["acquisition_s"] or 0.0 for r in group),
            })
        rr = [r["retained_rms"] for r in rows]
        sm = [r["retained_semi_major"] for r in rows]
        out[tag] = {
            "rows": rows,
            "rms_min": min(rr), "rms_max": max(rr),
            "rms_median": float(np.median(rr)), "rms_mean": float(np.mean(rr)),
            "semi_major_min": min(sm), "semi_major_max": max(sm),
            "semi_major_median": float(np.median(sm)),
            "aoa_max": max(r["max_aoa_deg"] for r in rows),
        }
    return out


# ===========================================================================
# Driver
# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="Tasks A and C only, no 6-DOF trajectories")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join("docs", "roll_servo.json"))
    args = ap.parse_args(argv)

    t_start = time.time()
    print("flying the baseline legs ...", flush=True)
    base = baseline()
    print(f"  uncorrected  range {base['uncorrected_range']:.1f} m  "
          f"drift {base['uncorrected_drift']:.1f} m  tof {base['uncorrected_tof']:.2f} s")
    print(f"  deployment   t = {base['deploy_time']:.3f} s  "
          f"M {base['mach_at_deploy']:.3f}  "
          f"qbar {base['qbar_at_deploy']/1e3:.1f} kPa  "
          f"spin {base['spin_at_deploy']:.0f} rad/s")
    print(f"  free nose    range {base['free_range']:.1f} m  "
          f"bias {base['free_range']-base['uncorrected_range']:.1f} m")

    results = {"configuration": {
        "station_from_nose_m": GEOMETRY.station_from_nose,
        "steering_deflection_deg": math.degrees(GEOMETRY.steering_deflection),
        "cant_angle_deg": math.degrees(GEOMETRY.cant_angle),
        "panel_area_m2": GEOMETRY.panel_area,
        "centroid_radius_m": GEOMETRY.centroid_radius,
        "nose_inertia_kgm2": NOSE.inertia,
        "bearing_viscous_Nms": NOSE.viscous,
        "bearing_coulomb_Nm": NOSE.coulomb,
        "brake_max_nominal_Nm": NOSE.brake_max,
        "brake_max_sized_Nm": BRAKE_MAX_SIZED,
        "charge": CHARGE, "qe_mils": QE_MILS,
        "deploy_fraction_of_apogee": DEPLOY_FRACTION_OF_APOGEE,
        "controller": rc.ControllerConfig().__dict__,
        "dt": SWEEP_DT,
    }, "baseline": {k: v for k, v in base.items()
                    if not k.startswith(("schedule_", "free_nose_", "y_deploy"))}}

    print("Yawing modes the duty cycle has to avoid ...", flush=True)
    results["epicyclic"] = epicyclic_frequencies(base)
    ep = results["epicyclic"]
    print(f"  slow (precession) mode {ep['slow_hz_min']:.2f}-{ep['slow_hz_max']:.2f} Hz, "
          f"median {ep['slow_hz_median']:.2f} Hz")
    print(f"  fast (nutation)  mode {ep['fast_hz_min']:.2f}-{ep['fast_hz_max']:.2f} Hz, "
          f"median {ep['fast_hz_median']:.2f} Hz")

    print("Task A: the one-sided actuator envelope ...", flush=True)
    results["task_a"] = task_a(base)
    for key, v in results["task_a"]["variants"].items():
        print(f"  brake {v['brake_max_Nm']:.2f} N m: "
              f"hold possible over {100*v['hold_possible_fraction']:.1f} % of the "
              f"guided phase, dead window {v['dead_window_s']} s, "
              f"max forward {v['max_forward_rate_degs']:.0f} deg/s, "
              f"max return {v['max_return_rate_degs']:.0f} deg/s")

    print("Task C: slew, settling, tracking error and bandwidth ...", flush=True)
    results["task_c"] = {
        "nominal_brake": task_c(base, NOSE.brake_max),
        "sized_brake": task_c(base, BRAKE_MAX_SIZED),
    }
    for name, tc in results["task_c"].items():
        ok = [s for s in tc["steps"] if s["settled"]]
        print(f"  {name}: {len(ok)}/{len(tc['steps'])} steps settled")

    print("Task C: acquisition from deployment ...", flush=True)
    results["acquisition"] = {
        "nominal_brake": task_c_acquisition(base, NOSE.brake_max),
        "sized_brake": task_c_acquisition(base, BRAKE_MAX_SIZED),
    }
    for name, a in results["acquisition"].items():
        print(f"  {name}: acquired {a['acquired']}/{a['of']}, "
              f"median {a['median_s']} s, max {a['max_s']} s")

    print("Task C: what a faster brake coil would buy ...", flush=True)
    results["actuator_lag"] = task_c_actuator_lag(base, BRAKE_MAX_SIZED)
    for r in results["actuator_lag"]["rows"]:
        print(f"  tau {1e3*r['actuator_tau_s']:5.1f} ms -> "
              f"bandwidth {r['bandwidth_hz']} Hz")

    if args.quick:
        _write(args.out, results)
        print(f"wrote {args.out} in {time.time()-t_start:.1f} s")
        return 0

    # -- 6-DOF work ------------------------------------------------------
    print("Task A verification and Task C verification in the 6-DOF ...",
          flush=True)
    verify_cases = [
        {"baseline": base, "brake": 0.0},
        {"baseline": base, "brake": 0.5},
        {"baseline": base, "brake": 0.75},
    ]
    sixdof_c = [
        {"baseline": base, "brake_max": BRAKE_MAX_SIZED, "keep_history": True,
         "command": {"kind": "hold", "angle_deg": 180.0, "label": "verify_hold_180"}},
        {"baseline": base, "brake_max": BRAKE_MAX_SIZED, "keep_history": True,
         "command": {"kind": "step", "before_deg": 0.0, "after_deg": 90.0,
                     "step_s": 12.0, "label": "verify_step_p90"}},
        {"baseline": base, "brake_max": BRAKE_MAX_SIZED, "keep_history": True,
         "command": {"kind": "step", "before_deg": 0.0, "after_deg": -90.0,
                     "step_s": 12.0, "label": "verify_step_m90"}},
        {"baseline": base, "brake_max": NOSE.brake_max, "keep_history": True,
         "command": {"kind": "hold", "angle_deg": 180.0,
                     "label": "verify_hold_180_nominal_brake"}},
    ]
    print(f"Task D: duty-cycle cases ...", flush=True)
    d_cases = task_d_cases(base, BRAKE_MAX_SIZED)
    for c in d_cases:
        if c["command"]["kind"] == "duty" and c["command"]["duty"] in (0.25, 0.5, 0.75):
            c["keep_history"] = True

    ideal_cases = [{"baseline": base, "angle_deg": float(a)}
                   for a in range(0, 360, 45)]
    # The same holds with the engagement gate removed, so that what the gate
    # is worth is measured rather than asserted.
    ungated_cases = [
        {"baseline": base, "brake_max": BRAKE_MAX_SIZED,
         "config": {"engage_gate": False},
         "command": {"kind": "hold", "angle_deg": float(a),
                     "label": f"ungated_{a}"}}
        for a in range(0, 360, 45)
    ]

    with Pool(processes=args.jobs) as pool:
        av = pool.map(task_a_verify, verify_cases)
        sv = pool.map(run_closed_loop, sixdof_c)
        iv = pool.map(run_ideal_hold, ideal_cases)
        uv = pool.map(run_closed_loop, ungated_cases)
        dv = pool.map(run_closed_loop, d_cases)

    results["task_a_verification"] = av
    worst_abs = max(r["absolute_error_rads"] for v in av for r in v["rows"])
    # The relative figure is quoted only where the rate itself is not near
    # zero: the closed form and the 6-DOF differ by 2.2 rad/s at the point
    # where the equilibrium rate is 6 rad/s, which is 27 % of nothing.
    worst_rel = max(r["relative_error"] for v in av for r in v["rows"]
                    if abs(r["reduced_equilibrium_rads"]) > 20.0)
    print(f"  closed-form rate limits agree with the 6-DOF to "
          f"{worst_abs:.2f} rad/s absolute, {100*worst_rel:.2f} % relative "
          f"where the rate exceeds 20 rad/s")

    results["fidelity"] = [
        compare_reduced_to_sixdof(base, s, s["brake_max_Nm"]) for s in sv
    ]
    for f in results["fidelity"]:
        print(f"  {f['label']}: reduced vs 6-DOF roll angle, "
              f"rms {f['rms_angle_difference_settled_deg']:.4f} deg settled, "
              f"max {f['max_abs_angle_difference_deg']:.4f} deg")

    results["sixdof_checkpoints"] = [
        {k: v for k, v in s.items() if not k.startswith("traj_")
         and k != "history"}
        for s in sv
    ]
    closed_holds = [r for r in dv if r["command"]["kind"] == "hold"]
    results["servo_cost"] = servo_cost(base, closed_holds, iv)
    results["servo_cost_ungated"] = servo_cost(base, uv, iv)
    for tag in ("servo_cost", "servo_cost_ungated"):
        sc = results[tag]
        print(f"  {tag}: ideal semi-major {sc['ideal']['semi_major_m']:.1f} m, "
              f"closed {sc['closed_loop']['semi_major_m']:.1f} m "
              f"({100*sc['retained_semi_major']:.1f} % semi-major, "
              f"{100*sc['retained_rms']:.1f} % rms), axis ratio "
              f"{sc['closed_loop']['axis_ratio']:.2f} against ideal "
              f"{sc['ideal']['axis_ratio']:.2f}, peak AoA "
              f"{sc['closed_max_aoa_deg']:.2f} deg against ideal "
              f"{sc['ideal_max_aoa_deg']:.2f} deg")

    print("engagement phase sensitivity ...", flush=True)
    with Pool(processes=args.jobs) as pool:
        pv = pool.map(run_closed_loop,
                      engagement_phase_cases(base, BRAKE_MAX_SIZED))
    results["engagement_phase"] = engagement_phase_reduce(base, pv, iv)
    ep = results["engagement_phase"]
    for r in ep["rows"]:
        print(f"  dwell {r['dwell_s']:.2f} s (engage {r['engage_s']:.3f}): "
              f"semi-major {r['semi_major_m']:.1f} m "
              f"({100*r['retained_semi_major']:.1f} %), rms "
              f"{100*r['retained_rms']:.1f} %, axis ratio {r['axis_ratio']:.2f}, "
              f"peak AoA {r['max_aoa_deg']:.2f} deg")
    print(f"  over the ensemble: rms retained "
          f"{100*ep['retained_rms_min']:.1f}-{100*ep['retained_rms_max']:.1f} %, "
          f"median {100*ep['retained_rms_median']:.1f} %")

    print("deployment-phase ensemble, gate on and off ...", flush=True)
    with Pool(processes=args.jobs) as pool:
        qv = pool.map(run_closed_loop,
                      deployment_phase_cases(base, BRAKE_MAX_SIZED))
    results["engagement_phase_ensemble"] = deployment_phase_reduce(base, qv, iv)
    for tag, blk in results["engagement_phase_ensemble"].items():
        print(f"  {tag}: semi-major {100*blk['semi_major_min']:.1f}-"
              f"{100*blk['semi_major_max']:.1f} %, rms "
              f"{100*blk['rms_min']:.1f}-{100*blk['rms_max']:.1f} % "
              f"(median {100*blk['rms_median']:.1f}, mean {100*blk['rms_mean']:.1f}), "
              f"peak AoA up to {blk['aoa_max']:.2f} deg")

    print("step-size convergence of the smallest quanta ...", flush=True)
    with Pool(processes=args.jobs) as pool:
        fv = pool.map(run_closed_loop, convergence_cases(base, BRAKE_MAX_SIZED))
    results["quantum_convergence"] = convergence_reduce(base, dv, fv)
    qc = results["quantum_convergence"]
    print(f"  coarse against fine step: worst difference "
          f"{qc['worst_difference_m']:.3f} m")
    for r in qc["rows"]:
        print(f"    start {r['start_s']:.1f} len {r['length_s']:.2f}: "
              f"{r['difference_m']:.3f} m")

    results["task_d"] = task_d_reduce(base, dv)
    results["task_d"]["switching"] = [
        {"label": r["label"], "period_s": r["command"].get("period_s"),
         "duty": r["command"].get("duty"),
         **hold_arc_stats(r["history"])}
        for r in dv if "history" in r
    ]
    for s in results["task_d"]["switching"]:
        s.pop("arcs", None)
        print(f"  {s['label']}: {s['n_arcs']} arcs, median reacquire "
              f"{s['median_reacquire_s']} s, overhead "
              f"{s['switching_overhead_pct']:.1f} %")

    results["task_d"]["yaw_spectrum"] = [
        {"label": r["label"], "period_s": r["command"].get("period_s"),
         "duty": r["command"].get("duty"),
         "max_total_aoa_deg": r["max_total_aoa_deg"],
         **measured_yaw_spectrum(base, r)}
        for r in dv if "traj_q_rate" in r
    ]
    for y in results["task_d"]["yaw_spectrum"]:
        print(f"  {y['label']}: peak yaw rate at {y.get('peak_hz')} Hz, "
              f"peak AoA {y['max_total_aoa_deg']:.2f} deg")

    results["task_d"]["cases"] = [
        {k: v for k, v in r.items() if not k.startswith("traj_")
         and k != "history"}
        for r in dv
    ]

    _write(args.out, results)
    print(f"wrote {args.out} in {time.time()-t_start:.1f} s")
    return 0


def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)


if __name__ == "__main__":
    raise SystemExit(main())
