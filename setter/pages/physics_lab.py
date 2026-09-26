"""
L3 -- Physics Lab (Control Room spec Part B5, Step 8 first half).

Six concept cards, each a plain-language statement, one control, a live
visual, and a citation of the repository function it calls. Every card is
its own `@st.fragment` so dragging one slider never recomputes the other
five.

Cards 1-4 use a single evaluation of already-public `sim.canards`/
`gnc.roll_control` functions at ONE fixed flight condition -- the adopted
configuration's own deployment point, read from `docs/roll_servo.json` via
`setter.physics_lab_data.adopted_deployment_condition()` rather than calling
`analysis.roll_servo.baseline()` live (measured at ~33 s: two full 6-DOF
flights to impact -- far too slow for a slider). Card 5 reads
`docs/authority_results.json`'s stored deploy-time sweep; Card 6 reads
`docs/monte_carlo.json`'s Task C scatter via `setter.campaign`. Nothing
here runs a new campaign or a new 6-DOF trajectory.

Registered in `setter/app.py` as a FILE-based `st.Page`, unconditionally
(like Error Budget/Archive): no fired round is needed.
"""

from __future__ import annotations

import math
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from analysis import roll_servo as rs
from gnc import roll_control as rc
from sim import aerodata, canards as cn, dynamics as dyn, frames, projectile as pr

from setter import campaign, physics_lab_data as pld
from setter.config import MET_AGE_BUCKETS
from setter.plotting import (authority_budget_figure, brake_power_figure,
                              cep_circle_figure, fixed_fin_figure,
                              nose_force_and_correction_figure,
                              roll_decoupling_figure)

#: One evaluation of any card below costs microseconds to low milliseconds
#: (a handful of closed-form calls, no integration) -- this guards against a
#: future edit accidentally calling a campaign-class function (e.g.
#: analysis.roll_servo.baseline(), ~33 s) from inside a card instead of
#: reading its stored output.
_MAX_PLAUSIBLE_PAGE_LOAD_S = 15.0


def _adopted_condition_for_roll(brake_max_nm: float) -> tuple:
    """The adopted-configuration plant (25 mm/3 deg canards, the nominal
    bearing) with its brake capacity overridden to `brake_max_nm` -- the
    SAME override mechanism `analysis.roll_servo.make_plant(brake_max=...)`
    uses to compare the nominal/sized/adopted capacities, not a new plant
    design. Returns (plant, flight_condition)."""
    nose = replace(rs.NOSE, brake_max=brake_max_nm)
    plant = rc.plant_from(rs.GEOMETRY, nose, pr.M107)
    c = pld.adopted_deployment_condition()
    fc = rc.FlightCondition(time=c.time_s, qbar=c.qbar_pa, airspeed=c.airspeed_ms,
                             mach=c.mach, body_spin=c.spin_rad_s)
    return plant, fc


def render() -> None:
    st.title("Physics Lab")
    st.caption(
        "Six control-authority concepts, each with one slider and one live "
        "visual -- every number below comes from a single evaluation of an "
        "already-public repository function at the adopted configuration's "
        "own deployment condition, not a new simulation or campaign.")

    condition = pld.adopted_deployment_condition()
    st.caption(
        f"Flight condition used throughout (unless a card says otherwise): "
        f"Mach {condition.mach:.2f}, q-bar {condition.qbar_pa/1000:.0f} kPa, "
        f"altitude {condition.altitude_m:.0f} m -- the 'long' engagement's "
        f"adopted-configuration deployment point, read from "
        f"`{condition.source}`.")

    # =======================================================================
    # Card 1 -- why a fixed fin cannot steer a spinning shell
    # =======================================================================
    @st.fragment
    def card_fixed_fin():
        with st.container(border=True):
            st.subheader("1. Why a fixed fin cannot steer a spinning shell")
            st.markdown(
                "A canard bolted rigidly to the body pushes in a fixed "
                "direction RELATIVE TO THE BODY. As the body spins, that "
                "push sweeps around the compass -- averaged over more than "
                "a revolution, it cancels to (almost) nothing.")
            spin = st.slider("Body spin rate, rad/s", 0.0, 1500.0,
                              value=float(round(condition.spin_rad_s)), step=25.0,
                              key="pl_spin")

            model = rs.base_model()
            canard = cn.CanardModel(geometry=cn.NOMINAL_GEOMETRY, nose=cn.NOMINAL_NOSE,
                                     projectile=pr.M107)
            r = np.array([0.0, 0.0, -condition.altitude_m])
            v0 = np.array([condition.airspeed_ms, 0.0, 0.0])

            # The static locus radius: the transverse force magnitude at
            # phi_body = 0 (any other clocking gives the same magnitude,
            # rotated -- see fixed_fin_figure's own docstring).
            q0 = frames.quat_from_euler(0.0, 0.0, 0.0)
            v = frames.dcm_from_quat(q0) @ v0
            y0 = dyn.pack(r, v, q0, np.array([spin, 0.0, 0.0]), nose=(0.0, 0.0))
            st0 = dyn.aero_state(0.0, y0, model)
            f0, _m0, _rt0 = canard(0.0, y0, st0)
            circle_radius_n = math.hypot(f0[1], f0[2])

            # Time-averaged earth-frame force over a fixed 1 s window,
            # sweeping phi_body = spin * t. f_body (computed once above, at
            # phi_body = 0) does not itself depend on phi_body -- rotating
            # about the body's own x axis, which is aligned with velocity
            # here, changes neither the relative-velocity components nor the
            # canard incidences it drives -- so the earth-frame force over
            # the window is exactly f_body rotated by R(spin*t), and its
            # average has the closed form below rather than needing many
            # samples (which aliases badly once one revolution per sample
            # is no longer resolved -- confirmed against a 20000-sample
            # numerical check at a low spin rate before adopting this form).
            window_s = 1.0
            f_complex = complex(f0[1], f0[2])
            if spin == 0.0:
                factor = 1.0 + 0.0j
            else:
                theta = spin * window_s
                factor = (complex(math.cos(theta), math.sin(theta)) - 1.0) / complex(0.0, theta)
            avg_c = f_complex * factor
            avg_vector = (avg_c.real, avg_c.imag)

            fig = fixed_fin_figure(circle_radius_n, avg_vector, spin)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "1-second time average, closed form (no sampling -- see this "
                "card's source for why). At 0 rad/s the average sits on the "
                "circle (the fin never moves); by the adopted deployment "
                f"spin it has collapsed to {math.hypot(*avg_vector):.2f} N -- "
                f"{100 * (1 - math.hypot(*avg_vector) / circle_radius_n):.1f}% "
                "cancelled. (The average does not fall off monotonically with "
                "spin -- it has exact nulls at whole numbers of revolutions "
                "per window and partial revivals between them, the same "
                "sinc-shaped pattern as any periodic signal averaged over a "
                "fixed window.)")
            st.caption("`sim/canards.py:CanardModel.__call__`, `sim/dynamics.py:aero_state`, "
                       "`sim/frames.py:dcm_from_quat` -- single evaluations, no integration.")

    card_fixed_fin()

    # =======================================================================
    # Card 2 -- roll decoupling
    # =======================================================================
    @st.fragment
    def card_roll_decoupling():
        with st.container(border=True):
            st.subheader("2. Roll decoupling")
            st.markdown(
                "The despun nose rides on a bearing: a brake can hold it "
                "away from the body's own fast roll, decoupling the "
                "canards' orientation from the shell's spin.")
            brake = st.slider("Commanded brake torque, N·m", 0.0, 1.0,
                               value=0.5, step=0.05, key="pl_brake_2")

            plant, fc = _adopted_condition_for_roll(1.0)  # full 0-1 N.m range holdable
            hold_cmd = plant.hold_command(fc)
            cant = plant.cant_torque(fc)
            p_eq = plant.equilibrium_rate(fc, brake)
            tau = plant.time_constant(fc)

            t = np.linspace(0.0, 5.0 * tau, 120)
            p0 = fc.body_spin  # nose locked to body at the instant of deployment
            p_of_t = p_eq + (p0 - p_eq) * np.exp(-t / tau)
            phi_of_t = np.cumsum(np.concatenate(([0.0], np.diff(t) * p_of_t[:-1])))
            phi_deg = np.degrees(phi_of_t) % 360.0

            fig = roll_decoupling_figure(t, phi_deg, hold_cmd, brake, cant, 1.0)
            st.pyplot(fig)
            plt.close(fig)
            can_hold = hold_cmd <= 1.0
            st.caption(
                f"Steady nose rate at this command: {p_eq:.1f} rad/s "
                f"(time constant {tau:.2f} s). Holding the nose still needs "
                f"{hold_cmd:.2f} N·m; the out-of-box nominal brake is "
                "0.5 N·m, which CONTROL-CHARACTERISATION.md finds cannot "
                "hold this condition (0.75-0.85 N·m does, per "
                "docs/ARCHITECTURE-DECISION.md).")
            st.caption("`gnc/roll_control.py:NoseRollPlant.equilibrium_rate`/"
                       "`.time_constant`/`.hold_command` -- closed form, no integrator.")

    card_roll_decoupling()

    # =======================================================================
    # Card 3 -- one variable, two axes
    # =======================================================================
    @st.fragment
    def card_one_variable_two_axes():
        with st.container(border=True):
            st.subheader("3. One variable, two axes")
            st.markdown(
                "The nose roll angle phi is the ONE thing the flight "
                "computer commands. It sets a direction in the nose frame; "
                "the airframe converts that into a correction in range and "
                "deflection on the ground.")
            phi_deg = st.slider("Nose roll angle, phi (deg)", 0.0, 360.0,
                                 value=0.0, step=1.0, key="pl_phi")

            model = rs.base_model()
            canard = cn.CanardModel(geometry=cn.NOMINAL_GEOMETRY, nose=cn.NOMINAL_NOSE,
                                     projectile=pr.M107)
            r = np.array([0.0, 0.0, -condition.altitude_m])
            q = frames.quat_from_euler(0.0, 0.0, 0.0)
            v = frames.dcm_from_quat(q) @ np.array([condition.airspeed_ms, 0.0, 0.0])
            phi_rel = math.radians(phi_deg)
            y = dyn.pack(r, v, q, np.array([condition.spin_rad_s, 0.0, 0.0]),
                         nose=(phi_rel, 0.0))
            st_ = dyn.aero_state(0.0, y, model)
            fb, _mb, _rt = canard(0.0, y, st_)
            force_yz = (fb[1], fb[2])

            tbl = aerodata.make_m107_table()
            C_X0, C_X2, C_Nalpha, C_Ypalpha, C_lp, C_Malpha, C_mq, C_Mpalpha = tbl.lookup(condition.mach)
            cla = cn.canard_lift_curve_slope(condition.mach, cn.NOMINAL_GEOMETRY.aspect_ratio_effective)
            resp = cn.steering_response(
                cn.NOMINAL_GEOMETRY, pr.M107, C_Nalpha, C_Malpha, C_Ypalpha, C_Mpalpha,
                cla, condition.spin_rad_s, condition.airspeed_ms)
            mag = resp["net_over_direct_magnitude"]
            phase = math.radians(resp["net_over_direct_phase_deg"])
            # The ratio is independent of phi (steering_response takes no phi
            # argument at all) -- phi only rotates the direct force, and the
            # ground-plane correction rotates and scales with it.
            f_dir_angle = math.atan2(fb[2], fb[1])
            corr_angle = f_dir_angle + phase
            f_dir_mag = math.hypot(fb[1], fb[2])
            corr_mag = f_dir_mag * mag
            correction = (corr_mag * math.cos(corr_angle), corr_mag * math.sin(corr_angle))

            stored = pld.authority_phi_sweep("c8_qe525.3")
            fig = nose_force_and_correction_figure(phi_deg, force_yz, correction, stored)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Right panel's live line uses `sim.canards.steering_response`'s "
                "net-over-direct ratio, which does not depend on phi -- only "
                "the direct force's own rotation with phi does. The stored "
                "overlay is `docs/authority_results.json`'s own deployment "
                "condition, not this card's slider condition -- shown for "
                "scale and geometry cross-check, not claimed identical.")
            st.caption("`sim/canards.py:CanardModel.__call__`, `sim/canards.py:steering_response` "
                       "-- both at `sim.canards.NOMINAL_GEOMETRY`, matching the stored overlay's geometry.")

    card_one_variable_two_axes()

    # =======================================================================
    # Card 4 -- the brake is the alternator
    # =======================================================================
    @st.fragment
    def card_brake_is_alternator():
        with st.container(border=True):
            st.subheader("4. The brake is the alternator")
            st.markdown(
                "A friction brake dissipates power exactly where a slip-ring "
                "alternator would generate it: torque times the relative "
                "slip rate between nose and body.")
            brake = st.slider("Commanded brake torque, N·m", 0.0, 1.0,
                               value=0.5, step=0.05, key="pl_brake_4")

            plant, fc = _adopted_condition_for_roll(1.0)
            u_curve = np.linspace(0.0, 1.0, 60)
            power_curve = []
            for u in u_curve:
                p_eq = plant.equilibrium_rate(fc, float(u))
                p_rel = p_eq - fc.body_spin
                torque = abs(plant.brake_torque(float(u), p_rel))
                power_curve.append(torque * abs(p_rel))
            p_eq_now = plant.equilibrium_rate(fc, brake)
            p_rel_now = p_eq_now - fc.body_spin
            power_now = abs(plant.brake_torque(brake, p_rel_now)) * abs(p_rel_now)

            fig = brake_power_figure(list(u_curve), power_curve, brake, power_now, 1.0)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                f"At {brake:.2f} N·m the nose slips at {abs(p_rel_now):.0f} rad/s "
                f"relative to the body, dissipating {power_now:.1f} W -- power a "
                "slip-ring generator on the same shaft could instead harvest.")
            st.caption("Composed from `gnc/roll_control.py:NoseRollPlant.brake_torque`/"
                       "`.equilibrium_rate` (P = T*omega); no upstream power function exists.")

    card_brake_is_alternator()

    # =======================================================================
    # Card 5 -- authority is a budget
    # =======================================================================
    @st.fragment
    def card_authority_budget():
        with st.container(border=True):
            st.subheader("5. Authority is a budget")
            st.markdown(
                "Deploying earlier buys more time for the canards to act, "
                "but also more time flying uncorrected beforehand -- the "
                "achievable correction is not free to maximise by deploying "
                "as early as possible.")
            deploy_sweep = pld.authority_deploy_sweep()
            tag_labels = {
                "c4_qe97.2": "short (charge 4, QE 97.2)",
                "c6_qe258.4": "an intermediate case, not one of the five named "
                              "engagements (charge 6, QE 258.4)",
                "c8_qe525.3": "long (charge 8, QE 525.3)",
            }
            tag = st.selectbox("Engagement", list(tag_labels.keys()),
                                format_func=lambda t: tag_labels[t], index=2, key="pl_authority_tag")

            # Each stored entry's own rows[0]["deploy_fraction"] is the
            # authoritative fraction for that entry (the key's "_f..."
            # suffix carries the same value, but the row field is read here
            # rather than parsed back out of a string).
            entries = [(entry["rows"][0]["deploy_fraction"], entry)
                       for key, entry in deploy_sweep.items() if key.startswith(tag + "_f")]
            entries.sort(key=lambda fe: fe[0])
            fracs = [f for f, _ in entries]
            majors = [entry["stats"]["semi_axis_major_m"] for _, entry in entries]

            current = st.select_slider("Deployment time, fraction offset from the "
                                        "reference deploy point",
                                        options=fracs, value=0.0 if 0.0 in fracs else fracs[len(fracs) // 2],
                                        format_func=lambda f: f"{f:+.0%}", key="pl_deploy_frac")

            fig = authority_budget_figure(fracs, majors, current)
            st.pyplot(fig)
            plt.close(fig)
            i = fracs.index(current)
            st.caption(
                f"At {current:+.0%}: achievable correction (semi-major axis) "
                f"{majors[i]:.0f} m -- stored point, not interpolated.")
            st.caption("`analysis/authority.py:envelope_stats`, `docs/authority_results.json['deploy']` "
                       "-- 6 stored points per engagement, no live 6-DOF call.")

    card_authority_budget()

    # =======================================================================
    # Card 6 -- why the weather dominates
    # =======================================================================
    @st.fragment
    def card_weather_dominates():
        with st.container(border=True):
            st.subheader("6. Why the weather dominates")
            st.markdown(
                "Canard authority corrects a KNOWN error. An unknown "
                "atmosphere is not correctable at all -- and the campaign's "
                "own stored impact scatter widens visibly as the met "
                "message gets older.")
            age = st.select_slider("Meteorological message age", options=MET_AGE_BUCKETS,
                                    value="2h", key="pl_met_age")

            scatter = campaign.task_c_scatter(age, tag="headline", engagement="long")
            point = campaign.task_c_point(age, tag="headline", engagement="long")
            axis_limit_m = campaign.task_c_max_miss_m(tag="headline", engagement="long") * 1.08

            if point.available:
                fig = cep_circle_figure(scatter, point.cep_m, axis_limit_m,
                                         "long (Task C, truth-fed)", age)
                st.pyplot(fig)
                plt.close(fig)
                st.caption(f"CEP {point.cep_m:.1f} m, n={point.n} rounds, met age '{age}' -- "
                           "stored Task C campaign data, navigation excluded so this isolates "
                           "the atmospheric knowledge term alone.")
            else:
                st.info(point.reason)
            st.caption("`setter/campaign.py:task_c_scatter` (new), `docs/monte_carlo.json` -- "
                       "no live simulation.")

    card_weather_dominates()


render()
