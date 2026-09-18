"""
Shared harness for step 5: the truth trajectories, the sensor/filter wiring,
and the one closed-loop run function every navigation campaign calls.

Split out of `analysis.nav_sensors` and `analysis.nav_cep` because both need
it and because step 6 will need it too. Nothing here decides anything; it
assembles.

TWO WAYS TO DRIVE THE FILTER, AND WHY BOTH EXIST
------------------------------------------------
`replay` flies ONE 6-DOF trajectory, records truth at the IMU rate, and then
runs the filter over that record as many times as wanted. A consistency
campaign needs 64 sensor seeds against the same truth, and re-flying the 6-DOF
64 times to get them would cost 34 minutes to learn nothing about the
trajectory.

`run_guided_nav` flies the closed loop with the filter IN it, so the servo and
the guidance law read the estimate. That is the only way to measure the
navigation contribution to CEP, because the estimate changes what the round
does and a replay cannot see that.

They share the sensor models and the filter exactly, so a number measured in
one is comparable with a number measured in the other.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np

from sim import aerodata, canards as cn, dynamics as dyn, integrate as ig, projectile as pr
from sim import atmosphere as atm
from gnc import guidance as gd, navigation as nv, roll_control as rc, sensors as sn
from gnc.inverse_map import AuthorityMap
from gnc import scheduler as sch
from analysis import guidance_authority as ga, guidance_cep as gc, roll_servo as rs

#: Truth is recorded at the IMU rate. dt is 5e-4 s, so log every 4th step.
LOG_EVERY_IMU = 4


# ===========================================================================
# Configuration for one navigation fit
# ===========================================================================
def gun_data(base: dict, wind=(0.0, 0.0, 0.0)) -> nv.GunData:
    """The fuze setter's upload for this engagement."""
    return nv.GunData(muzzle_velocity=base["muzzle_velocity"],
                      quadrant_elevation=base["qe_mils"] * pr.MIL_TO_RAD,
                      azimuth=0.0, wind=tuple(wind))


def make_nav(seed: int, base: dict, suite: sn.SuiteConfig = None,
             config: nv.NavConfig = None, warm_start: bool = True,
             log: bool = True, log_nis: bool = False,
             environment=None,
             sensor_suite: sn.SuiteConfig = None,
             gun_wind=(0.0, 0.0, 0.0)) -> nv.NavigationSystem:
    env = environment if environment is not None else rs.base_model().environment
    return nv.NavigationSystem(
        config or nv.NavConfig(), suite or sn.SuiteConfig(), seed=seed,
        projectile=pr.M107, environment=env,
        gun=gun_data(base, wind=gun_wind),
        muzzle_time=0.0, log=log, log_nis=log_nis, warm_start=warm_start,
        sensor_suite=sensor_suite)


# ===========================================================================
# Truth trajectories, flown once and cached
# ===========================================================================
def fly_truth(label: str, mapdata: dict, dmv: float = 0.0, daz: float = 0.0,
              scheduler: str = "proportional") -> dict:
    """
    One closed-loop trajectory with guidance and control on TRUTH, logged at
    the IMU rate.

    This is the reference the replay differences against and it is also the
    truth-fed half of the Task F pair, so the two are the same trajectory by
    construction rather than by intention.
    """
    ctx = gc.engagement_context(mapdata, label)
    base = ctx["base"]
    opts = gc.scheduler_options(mapdata, label)
    dt = base["dt"]

    launch = pr.LaunchConditions.from_mils(base["muzzle_velocity"] + dmv,
                                           base["qe_mils"], azimuth=daz)
    y0 = dyn.initial_state(pr.M107, launch)
    pre = ig.integrate(y0, rs.base_model(), dt=dt, log_every=10 ** 9,
                       t_max=base["deploy_time"], stop_on_impact=False)
    y_dep, t_dep = pre.impact_state.copy(), float(pre.impact_time)

    model = rs.make_plant(brake_max=ga.BRAKE)
    controller = rs._controller(model, rc.ControllerConfig())
    deployment = rc.StagedDeployment(rc.StagedDeploymentConfig(**ga.STAGED),
                                     t_dep, controller=controller, plant=model)
    aim = gc.default_aim_off(base)
    target = (base["uncorrected_range"] - aim, base["uncorrected_drift"])
    mon = {k: v for k, v in ctx["monitor"].items() if not k.startswith("_")}
    gcfg = gd.GuidanceConfig(**mon)
    predictor = gd.ImpactPredictor(pr.M107, rs.base_model().environment,
                                   geometry=rs.GEOMETRY, dt=gcfg.predictor_dt,
                                   iterate_yaw=gcfg.iterate_yaw)
    law_g = gd.GuidanceLaw(gcfg, AuthorityMap.from_dict(ctx["map"]), predictor,
                           sch.make_scheduler(scheduler, **opts[scheduler]),
                           target, t_dep)
    law_b = rc.BrakeLaw(controller, law_g.command, deploy_time=t_dep,
                        deployment=deployment)
    nose = replace(model.nose, deploy_time=t_dep,
                   brake_command=law_b.brake_command,
                   steering_gate=deployment.gate)
    guided = cn.guided_model(rs.base_model(), rs.GEOMETRY, nose)
    res = ig.integrate(y_dep, guided, dt=dt, log_every=LOG_EVERY_IMU,
                       t_max=300.0, t_start=t_dep,
                       step_hook=law_g.chain(law_b.sample))
    tr = res.trajectory
    return {
        "label": label, "t_dep": t_dep, "dmv": dmv, "daz": daz,
        "target": target, "range_m": res.range_m, "drift_m": res.drift_m,
        "t": tr.t, "position": tr.position, "velocity": tr.velocity,
        "quaternion": tr.quaternion, "omega": tr.omega,
        "nose_angle": tr.nose_angle, "nose_rate": tr.nose_rate,
        "y_dep": y_dep, "base": base,
    }


def truth_phi_nose(traj: dict) -> np.ndarray:
    """The controlled variable along a recorded trajectory."""
    return np.array([sn._body_roll_angle(q) for q in traj["quaternion"]]) \
        + traj["nose_angle"]


# ===========================================================================
# Replay: the filter over a recorded trajectory
# ===========================================================================
def replay(traj: dict, seed: int, suite: sn.SuiteConfig = None,
           config: nv.NavConfig = None, warm_start: bool = True,
           log_nis: bool = True, start_at_muzzle: bool = False) -> dict:
    """
    Run the filter over recorded truth. No 6-DOF, so it costs milliseconds.

    NOTE WHAT THIS CANNOT SEE. The trajectory was flown on TRUTH, so the
    filter's errors here do not feed back into what the round does. That is
    exactly right for a consistency test -- NEES asks whether the filter's
    covariance matches its error on a given trajectory -- and exactly wrong
    for a CEP, which is why Task F uses `run_guided_nav` instead.
    """
    model = rs.base_model()
    nav = make_nav(seed, traj["base"], suite=suite, config=config,
                   warm_start=warm_start, log=True, log_nis=log_nis)
    t = traj["t"]
    y = np.zeros(dyn.STATE_SIZE)
    pos, vel, quat = traj["position"], traj["velocity"], traj["quaternion"]
    om, na, nr = traj["omega"], traj["nose_angle"], traj["nose_rate"]

    err_r, err_v, err_psi, err_b = [], [], [], []
    nees, nees_nav, nees_pv, nees_t = [], [], [], []
    for i in range(len(t)):
        y[0:3] = pos[i]; y[3:6] = vel[i]; y[6:10] = quat[i]
        y[10:13] = om[i]; y[13] = na[i]; y[14] = nr[i]
        before = nav.samples
        nav.sample(float(t[i]), y, model)
        if nav.samples == before or not nav.filter.initialised:
            continue
        if i % 25:
            continue
        f = nav.filter
        dr = f.r - pos[i]
        dv = f.v - vel[i]
        dpsi = _attitude_error(f, quat[i], na[i])
        dbg = f.bg - nav.sensors.imu.bias_gyro
        dba = f.ba - nav.sensors.imu.bias_accel
        err_r.append(dr); err_v.append(dv); err_psi.append(dpsi)
        err_b.append(np.concatenate([dbg, dba]))
        nees_t.append(float(t[i]))
        # THREE NEES, because they answer three different questions.
        #
        #   nees      all 15 states, against the sensor model's own true
        #             biases. The strictest test, and the one that exposes a
        #             filter whose bias states have absorbed something that is
        #             not a bias.
        #   nees_nav  the nine states anything downstream actually consumes.
        #             Guidance reads position, velocity and spin; the servo
        #             reads the attitude. Nothing reads a bias.
        #   nees_pv   position and velocity alone, which is what the CEP
        #             budget of Task F is a function of.
        nees.append(_nees(f.P, np.concatenate([dr, dv, dpsi, dbg, dba])))
        nees_nav.append(_nees(f.P[0:9, 0:9], np.concatenate([dr, dv, dpsi])))
        nees_pv.append(_nees(f.P[0:6, 0:6], np.concatenate([dr, dv])))
    return {"nav": nav, "t": np.array(nees_t),
            "position_error": np.array(err_r), "velocity_error": np.array(err_v),
            "attitude_error": np.array(err_psi), "bias_error": np.array(err_b),
            "nees": np.array(nees), "nees_nav": np.array(nees_nav),
            "nees_pv": np.array(nees_pv)}


def _attitude_error(f, q_body, phi_rel) -> np.ndarray:
    """
    Small-angle attitude error in the filter's own parametrisation.

    C_true = (I + [dpsi]x) C_est, so dpsi is the vector part of the earth-frame
    rotation taking the estimate to truth. The truth NOSE attitude is the body
    quaternion turned by `phi_rel` about x.
    """
    from sim import frames
    dq = np.array([math.cos(0.5 * phi_rel), math.sin(0.5 * phi_rel), 0.0, 0.0])
    q_true = frames.quat_normalize(frames.quat_multiply(q_body, dq))
    dR = frames.dcm_from_quat(q_true) @ frames.dcm_from_quat(f.q).T
    return np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                     dR[1, 0] - dR[0, 1]]) * 0.5


def _nees(P: np.ndarray, e: np.ndarray) -> float:
    n = P.shape[0]
    try:
        return float(e @ np.linalg.solve(P + 1e-18 * np.eye(n), e))
    except np.linalg.LinAlgError:
        return float("nan")


# ===========================================================================
# What the actuator spent its time doing
# ===========================================================================
#: A nose more than this far from its command is SLEWING, not holding.
#: `analysis.roll_servo._acquisition_time` uses the same 5 deg to declare a
#: hold acquired, so the two measurements agree on what "on the angle" means.
SLEW_TOL_DEG = 5.0


def _slew_activity(hist: dict, t_arm: float, tol_deg: float = SLEW_TOL_DEG) -> dict:
    """
    The fraction of the ARMED, holding phase the nose spent moving rather than
    on its commanded angle.

    This is the servo-side half of the chatter question of
    docs/NAV-CHATTER.md. `GuidanceLaw.command_activity` counts how often the
    command moved; this counts what that cost, and the two are independent
    measurements of the same thing.
    """
    t, err, hold = hist["t"], np.abs(hist["error"]), hist["holding"]
    m = hold & (t >= t_arm)
    if not m.any():
        return {"slew_fraction": float("nan"), "slew_time_s": 0.0,
                "holding_time_s": 0.0}
    tm, em = t[m], err[m]
    dt = np.gradient(tm) if tm.size > 1 else np.array([0.0])
    off = em > math.radians(tol_deg)
    total = float(dt.sum())
    return {"slew_fraction": float(dt[off].sum() / total) if total > 0 else 0.0,
            "slew_time_s": float(dt[off].sum()),
            "holding_time_s": total}


# ===========================================================================
# STEP 6 -- the per-round world
# ===========================================================================
# Everything below builds ONE ROUND'S physics from the draw. It is separated
# out because step 6 is the first study in which the world itself varies round
# to round: until now every campaign flew the same projectile through the same
# still ISA and varied only the launch and the sensor seed.
#
# All of it defaults to exactly what steps 1 to 5.5 flew, so `run_guided_nav`
# called without any of these keys is bit-for-bit the function it was.
def round_projectile(args) -> pr.Projectile:
    """This round's shell. Mass and the two inertias, or the nominal M107."""
    ov = args.get("projectile_overrides")
    return pr.M107.perturbed(**ov) if ov else pr.M107


def round_aero(args):
    """
    This round's aerodynamic deck, with any EPISTEMIC coefficient scaling
    applied.

    `aero_scales` is {coefficient name: factor} and multiplies whole columns
    of the table -- `{"C_Ypalpha": 3.3}` is the factor-3.3 Magnus uncertainty
    docs/COEFFICIENTS.md grades LOW. It is a lot property and not a round
    property, so a campaign that varies it is drawing a BAND and not a CEP.
    """
    base = aerodata.make_m107_table()
    sc = args.get("aero_scales")
    if not sc:
        return base
    rows = np.column_stack([base.mach, base.values])
    for name, f in sc.items():
        rows[:, 1 + aerodata.COEFFICIENT_NAMES.index(name)] *= float(f)
    return aerodata.AeroTable(rows, name=base.name + " (scaled)",
                              source=base.source + f"; scaled {sc}")


def round_geometry(args) -> cn.CanardGeometry:
    """This round's canards. `cla_scale` is the +-30 % epistemic knob."""
    g = rs.GEOMETRY
    if args.get("cla_scale") is not None:
        g = replace(g, cla_scale=float(args["cla_scale"]))
    return g


def round_base_model(args, met=None):
    """
    The step-1 ballistic model this round flies: its own shell, its own
    aerodynamic deck, and the atmosphere it actually flies through.

    `met` is a `sim.atmosphere.MetProfile` and it is THE AIR, not the fuze's
    belief about it. The standard profile is passed as None so the derivative
    keeps calling `isa_scalars` directly and the hot path is unchanged.
    """
    m = dyn.FlightModel(projectile=round_projectile(args), aero=round_aero(args),
                        environment=rs.base_model().environment)
    if met is not None and not met.is_standard:
        m = replace(m, wind=met.wind, atmosphere=met.scalars)
    return m


def round_met(args):
    """(the air, the met message the fuze was set with). Both may be None."""
    return args.get("met"), args.get("met_message")


def round_plant(args, projectile):
    """This round's nose plant, with the epistemic bearing knobs applied."""
    ov = dict(args.get("nose_overrides") or {})
    nose = rs.NOSE
    if ov:
        nose = replace(nose, **ov)
    nose = replace(nose, brake_max=ga.BRAKE)
    return rc.plant_from(round_geometry(args), nose, projectile)


def round_controller(args, plant):
    cfg = rc.ControllerConfig(**(args.get("controller_config") or {}))
    act = rc.BrakeActuator(brake_max=plant.nose.brake_max,
                           tau=float(args.get("brake_tau", 0.010)))
    return rc.RollAngleController(plant, cfg, act)


# ===========================================================================
# The closed loop, with navigation in it
# ===========================================================================
def run_guided_nav(args) -> dict:
    """
    Fly one round with guidance and control fed the ESTIMATE.

    Deliberately the same shape as `analysis.guidance_cep.run_guided`, and it
    differs from it in exactly three lines: a `NavigationSystem` is built, it
    is passed to `BrakeLaw` and `GuidanceLaw`, and it runs first in the step
    hook. Everything else -- the perturbation, the aim-off, the scheduler, the
    staged deployment, the authority monitor -- is identical, so a paired
    difference between the two IS the navigation contribution.
    """
    base = args["baseline"]
    amap = AuthorityMap.from_dict(args["map"])
    dt = args.get("dt", base["dt"])
    mode = args.get("mode", "full")
    use_nav = bool(args.get("use_nav", True))

    # -- STEP 6: this round's world. Every one of these defaults to what
    # -- steps 1 to 5.5 flew, so a call without them is unchanged.
    met, met_msg = round_met(args)
    proj = round_projectile(args)
    world = round_base_model(args, met)
    geom = round_geometry(args)
    #: The fuze is set before launch, so the deployment time is the fire
    #: control solution's, not this round's own. It varies round to round
    #: because the SETTING is quantised and the fuze's clock is not perfect,
    #: and docs/STAGED-DEPLOYMENT.md section 5 showed the phase it lands on
    #: drives a 35-point spread that the fuze cannot choose.
    t_dep = float(args.get("deploy_time", base["deploy_time"]))

    launch = pr.LaunchConditions.from_mils(
        base["muzzle_velocity"] + args.get("dmv", 0.0),
        base["qe_mils"] + args.get("dqe_mils", 0.0),
        azimuth=args.get("daz", 0.0))
    y0 = dyn.initial_state(proj, launch)

    nav = None
    if use_nav:
        suite = sn.SuiteConfig(
            gnss_outages=tuple(args.get("gnss_outages", ())),
            gnss_denied=bool(args.get("gnss_denied", False)),
            gnss_reacquire=args.get("gnss_reacquire"),
            mag_calibrated=bool(args.get("mag_calibrated", True)),
            mag_residual_fraction=args.get("mag_residual_fraction", 0.015),
            despun_imu=bool(args.get("despun_imu", True)),
        )
        # The antenna's TRANSVERSE phase-centre offset is the term the spin
        # modulates, so it is the one `analysis.nav_antenna` sweeps. The axial
        # offset is unchanged: it is what the position lever-arm correction
        # uses and it does not spin.
        if args.get("antenna_transverse") is not None:
            suite = replace(suite, antenna_lever=(
                suite.antenna_lever[0], float(args["antenna_transverse"]),
                suite.antenna_lever[2]))
        # -- per-source ablation, for analysis.nav_ablation ----------------
        # Both are {part: {field: value}} and replace fields on the frozen
        # part spec, so ONE error term moves without touching any other draw:
        # `_misalignment_matrix` and the bias draws still consume the same rng
        # in the same order, so the ablation is a change to one number rather
        # than a reshuffle of every sensor's noise.
        #
        # THE TWO ARE NOT INTERCHANGEABLE. `spec_overrides` moves the world
        # AND the filter's belief about it, which is what a GEOMETRY change
        # is. `truth_spec_overrides` moves only the world and leaves the
        # filter tuned as it was, which is what removing an ERROR SOURCE is --
        # and using the wrong one measured 425 m of contribution from a
        # perfect sensor suite, because the filter's process noise is derived
        # from the same numbers. See `NavigationSystem.sensor_config`.
        truth_suite = suite
        for part, fields in (args.get("spec_overrides") or {}).items():
            suite = replace(suite,
                            **{part: replace(getattr(suite, part), **fields)})
        if args.get("suite_overrides"):
            suite = replace(suite, **dict(args["suite_overrides"]))
        truth_suite = suite
        for part, fields in (args.get("truth_spec_overrides") or {}).items():
            truth_suite = replace(
                truth_suite,
                **{part: replace(getattr(truth_suite, part), **fields)})
        cfg = nv.NavConfig(**args.get("nav_config", {}))
        nav = make_nav(args.get("seed", 0), base, suite=suite, config=cfg,
                       warm_start=bool(args.get("warm_start", True)),
                       log=bool(args.get("keep_nav_log", False)), log_nis=False,
                       sensor_suite=(truth_suite if truth_suite is not suite
                                     else None),
                       environment=world.environment,
                       # The warm start propagates from GUN DATA, so the wind
                       # it uses is the MET MESSAGE's and never the air's.
                       gun_wind=(met_msg.ballistic_wind()
                                 if met_msg is not None else (0.0, 0.0, 0.0)))

    # -- the pre-deployment leg. Navigation runs through it, because the
    # -- receiver reacquires and the warm start is running during it.
    pre = ig.integrate(y0, world, dt=dt, log_every=10 ** 9,
                       t_max=t_dep, stop_on_impact=False,
                       step_hook=(nav.sample if nav is not None else None))
    y_dep = pre.impact_state.copy()
    t_dep_actual = float(pre.impact_time)

    model = round_plant(args, proj)
    controller = round_controller(args, model)
    stage_cfg = rc.StagedDeploymentConfig(**dict(ga.STAGED,
                                                 **(args.get("staging") or {})))
    deployment = rc.StagedDeployment(stage_cfg, t_dep_actual,
                                     controller=controller, plant=model)
    aim_off = float(args.get("aim_off", gc.default_aim_off(base)))
    target = (base["uncorrected_range"] - aim_off,
              base["uncorrected_drift"] - float(args.get("aim_off_deflection", 0.0)))

    gcfg = gd.GuidanceConfig(
        rate_hz=args.get("rate_hz", 1.0),
        arm_delay_s=args.get("arm_delay_s", 3.0), mode=mode,
        pred_filter_tau=args.get("pred_filter_tau"),
        nav_fail_after_s=args.get("nav_fail_after_s"),
        inhibit_threshold_m=args.get("inhibit_threshold_m"),
        authority_monitor=args.get("authority_monitor", True),
        monitor_window_s=args.get("monitor_window_s", 6.0),
        monitor_fraction=args.get("monitor_fraction", 0.30),
        monitor_floor_m=args.get("monitor_floor_m", 20.0),
        monitor_start_s=args.get("monitor_start_s", 0.0),
        monitor_shadow=bool(args.get("monitor_shadow", False)))
    # THE ONBOARD MODEL. Its projectile, its aerodynamic deck and its
    # canard geometry are the FUZE'S BELIEFS and are deliberately the nominal
    # ones: a real flight computer carries the firing table's shell and the
    # design's canards, not this round's. Only the met message is variable,
    # because that one really is uploaded per engagement.
    predictor = gd.ImpactPredictor(pr.M107, world.environment,
                                   geometry=rs.GEOMETRY, dt=gcfg.predictor_dt,
                                   iterate_yaw=gcfg.iterate_yaw,
                                   # A message that IS the standard
                                   # atmosphere is passed as no message at
                                   # all, so the predictor's hot path stays
                                   # bit-identical to every step before 6
                                   # rather than looking up a zero wind.
                                   wind=(met_msg.wind
                                         if met_msg is not None
                                         and not met_msg.is_standard
                                         else None),
                                   atmosphere=(met_msg.scalars
                                               if met_msg is not None
                                               and not met_msg.is_standard
                                               else None))
    scheduler = sch.make_scheduler(args.get("scheduler", "proportional"),
                                   **args.get("scheduler_opts", {}))
    law_g = gd.GuidanceLaw(gcfg, amap, predictor, scheduler, target,
                           t_dep_actual, nav=nav)
    law_b = rc.BrakeLaw(controller, law_g.command, deploy_time=t_dep_actual,
                        deployment=deployment, nav=nav)
    gate = gc._never if mode == "stage2_fail" else deployment.gate
    nose = replace(model.nose, deploy_time=t_dep_actual,
                   brake_command=law_b.brake_command, steering_gate=gate)
    guided = cn.guided_model(world, geom, nose)
    hook = law_g.chain(law_b.sample)
    if nav is not None:
        hook = nav.chain(hook)
    res = ig.integrate(y_dep, guided, dt=dt, log_every=rs.LOG_EVERY,
                       t_max=300.0, t_start=t_dep_actual, step_hook=hook)
    if law_b.samples == 0:
        raise RuntimeError("the controller never sampled: step_hook not wired")

    miss_r = res.range_m - target[0]
    miss_d = res.drift_m - target[1]
    hist = law_b.history()
    out = {
        "engagement": base["label"], "mode": mode, "use_nav": use_nav,
        "seed": args.get("seed"), "draw": args.get("draw"),
        "dmv": args.get("dmv", 0.0), "daz": args.get("daz", 0.0),
        "dqe_mils": args.get("dqe_mils", 0.0),
        "deploy_time": t_dep, "t_dep_actual": t_dep_actual,
        "aim_off": aim_off, "target_range": target[0], "target_defl": target[1],
        "range_m": res.range_m, "drift_m": res.drift_m,
        "tof_s": res.impact_time,
        "miss_range_m": miss_r, "miss_defl_m": miss_d,
        "miss_m": math.hypot(miss_r, miss_d),
        "max_total_aoa_deg": math.degrees(res.max_total_aoa),
        "stage2_delay_s": deployment.delay,
        "acquisition_s": rs._acquisition_time(hist["t"], np.abs(hist["error"]),
                                              hist["holding"], t_dep_actual),
        "servo_error_deg": float(np.degrees(np.abs(hist["error"]).mean())),
        **_slew_activity(hist, t_dep_actual + gcfg.arm_delay_s),
        **{f"g_{k}": v for k, v in law_g.summary().items()},
    }
    if args.get("monitor_shadow") or args.get("keep_monitor_log"):
        out["monitor_log"] = law_g.monitor_log
    if args.get("keep_guidance_log"):
        # The per-cycle predicted impact point, which is what the scheduler
        # actually decided on. `analysis.nav_ablation` differences it between
        # the truth-fed and navigation-fed halves of a pair to get the
        # PREDICTION error directly, rather than inferring it from the miss.
        L = law_g.log
        out["g_log"] = {k: list(L[k]) for k in
                        ("t", "t_go", "pred_range", "pred_defl", "miss_m",
                         "phi_deg", "holding", "armed", "note")}
    if nav is not None:
        out.update({f"n_{k}": v for k, v in nav.summary().items()})
        tr = res.trajectory
        e = nav.errors_against(tr.t, tr.position, tr.velocity,
                               np.array([sn._body_roll_angle(q)
                                         for q in tr.quaternion]) + tr.nose_angle,
                               truth_p_body=tr.omega[:, 0])
        if e and args.get("keep_error_log"):
            # Decimated to the guidance grid, which is the grid the
            # sensitivities of docs/NAV-ERROR-DECOMPOSITION.md are evaluated
            # on. Sampling the 500 Hz log at 1 Hz is not a loss: the
            # prediction is only made once a second, so the error at any
            # other instant never reaches the impact point.
            step = max(int(round(gcfg.period * nav.config.imu_rate)), 1)
            k = slice(None, None, step)
            out["e_log"] = {
                "t": [float(v) for v in e["t"][k]],
                "position_error": e["position_error"][k].tolist(),
                "velocity_error": e["velocity_error"][k].tolist(),
                "roll_error": [float(v) for v in e["roll_error"][k]],
                "spin_error": [float(v) for v in e["spin_error"][k]],
            }
        if e:
            m = e["t"] >= t_dep_actual + 3.0
            if m.any():
                pe, ve = e["position_error"][m], e["velocity_error"][m]
                re = e["roll_error"][m]
                out["n_pos_rms_m"] = [float(v) for v in
                                      np.sqrt((pe ** 2).mean(axis=0))]
                out["n_vel_rms_ms"] = [float(v) for v in
                                       np.sqrt((ve ** 2).mean(axis=0))]
                out["n_roll_bias_deg"] = float(np.degrees(re.mean()))
                out["n_roll_sd_deg"] = float(np.degrees(re.std()))
    if args.get("keep_state_trajectory"):
        # The GUIDED-PHASE trajectory only (deployment to impact) --
        # `res.trajectory` is already computed above by the integration
        # this function runs regardless of this flag; this block only
        # serialises it, the same shape of addition as keep_guidance_log/
        # keep_error_log above. The PRE-deployment leg (launch to
        # deployment) is logged too coarsely to extract
        # (`log_every=10**9` on the `pre` integration above) and this flag
        # does not change that -- exposing an already-computed structure,
        # not altering what gets computed, is the point.
        tr = res.trajectory
        out["state_trajectory"] = {
            "t": tr.t.tolist(),
            "position": tr.position.tolist(),
            "velocity": tr.velocity.tolist(),
            "quaternion": tr.quaternion.tolist(),
            "omega": tr.omega.tolist(),
            "mach": tr.mach.tolist(),
            "airspeed": tr.airspeed.tolist(),
            "total_aoa": tr.total_aoa.tolist(),
            "alpha": tr.alpha.tolist(),
            "beta": tr.beta.tolist(),
            "density": tr.density.tolist(),
            "dynamic_pressure": tr.dynamic_pressure.tolist(),
            "nose_angle": tr.nose_angle.tolist() if tr.nose_angle is not None else None,
            "nose_rate": tr.nose_rate.tolist() if tr.nose_rate is not None else None,
        }
    return out
