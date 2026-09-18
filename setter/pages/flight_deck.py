"""
L3 -- Flight Deck: time-scrubbed replay of the last fired round.

Registered in `setter/app.py` as a FILE-based `st.Page` -- but, unlike
Overview/Mission Control/Archive, only ADDED to `st.navigation`'s page list
when `st.session_state["fired_round"]` exists: "an empty Flight Deck is not
useful" (Control Room spec). See `setter/app.py`'s own note on how that
conditional registration works.

DATA SOURCE. Everything here reads `st.session_state["fired_round"]`,
written by Mission Control's FIRE fragment (`setter/pages/mission_control.py`)
via `setter/fire_worker.py`. The state trajectory is REAL 6-DOF data from
the actual fired round (`analysis.nav_common.run_guided_nav`'s
`keep_state_trajectory` flag -- see that module and the FIRE build step),
not the reduced-order visualisation Mission Control's own "Simulation
trajectory" section uses. It covers the GUIDED PHASE ONLY: deployment to
impact. The pre-deployment leg (launch to deployment) was never captured
(see fire_worker.py's docstring) -- the scrubber therefore starts at
deployment, and the phase strip says so plainly rather than implying a
launch-to-impact replay that isn't there.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from setter import fuze_replay
from setter.palette import AMBER
from setter.plotting import flight_deck_figure


def _nearest_index(ts: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(ts, t))
    idx = max(0, min(idx, len(ts) - 1))
    if idx > 0 and abs(ts[idx - 1] - t) < abs(ts[idx] - t):
        idx -= 1
    return idx


def _fixed_axis_limits(position: np.ndarray, g_log: dict, target_range: float,
                       target_defl: float) -> dict:
    """Computed ONCE from the whole trajectory (+ the whole prediction
    trail + the target), never refit per scrubber frame -- see
    `setter.plotting.flight_deck_figure`'s own docstring for why."""
    downrange = position[:, 0]
    crossrange = position[:, 1]
    altitude = -position[:, 2]

    dr_candidates = list(downrange) + [target_range]
    cr_candidates = list(crossrange) + [target_defl]
    if g_log:
        dr_candidates += list(g_log.get("pred_range", []))
        cr_candidates += list(g_log.get("pred_defl", []))

    dr_lo, dr_hi = min(dr_candidates), max(dr_candidates)
    cr_lo, cr_hi = min(cr_candidates), max(cr_candidates)
    dr_pad = max((dr_hi - dr_lo) * 0.08, 50.0)
    cr_pad = max((cr_hi - cr_lo) * 0.15, 20.0)
    alt_pad = max((altitude.max() - altitude.min()) * 0.10, 20.0)

    return {
        "downrange": (dr_lo - dr_pad, dr_hi + dr_pad),
        "crossrange": (cr_lo - cr_pad, cr_hi + cr_pad),
        "altitude": (min(0.0, altitude.min()) - alt_pad, altitude.max() + alt_pad),
    }


def _phase_boundaries(t_dep: float, t_impact: float, g_log: dict) -> dict:
    """DEPLOYMENT ends and CORRECTION begins at the guidance log's first
    `armed=True` sample -- a real, stored transition, not a guess.
    TERMINAL begins at the first sample with `t_go <= 3.0 s`, a small fixed
    terminal-window fixture (the spec's own out-of-scope note: "No
    threshold values for the event engine -- those stay fixtures" -- this
    isn't the fuze engine, but the same discipline applies: don't invent a
    precise number where a round one will do). Falls back to sane defaults
    if `g_log` is missing or never arms."""
    correction_start = t_dep
    terminal_start = max(t_dep, t_impact - 3.0)
    if g_log and g_log.get("t"):
        ts, armed, t_go = g_log["t"], g_log.get("armed", []), g_log.get("t_go", [])
        for i, a in enumerate(armed):
            if a:
                correction_start = ts[i]
                break
        for i, tg in enumerate(t_go):
            if tg <= 3.0:
                terminal_start = ts[i]
                break
    return {"deployment": t_dep, "correction": correction_start, "terminal": terminal_start}


def render() -> None:
    st.title("Flight Deck")

    fired = st.session_state.get("fired_round")
    if not fired or not fired.get("ok"):
        st.info("No fired round yet. Fire one from Mission Control first.")
        return
    if not fired.get("state_trajectory_available") or not fired.get("state_trajectory"):
        st.warning("This fired round has no stored state trajectory to replay.")
        return

    tr = fired["state_trajectory"]
    ts = np.asarray(tr["t"])
    position = np.asarray(tr["position"])
    velocity = np.asarray(tr["velocity"])
    mach = np.asarray(tr["mach"])
    omega = np.asarray(tr["omega"])
    nose_angle = np.asarray(tr["nose_angle"]) if tr.get("nose_angle") is not None else None

    t_dep = float(fired["t_dep_actual"])
    t_impact = float(ts[-1])
    g_log = fired.get("g_log")

    st.caption(
        f"**{fired['engagement']}** · met age **{fired['met_age_bucket']}** · "
        f"fuze mode {fired['fuze_mode_context_only']} · seed {fired['seed']} · "
        f"miss {fired['miss_m']:.1f} m")
    st.caption(
        f"Replay covers the GUIDED PHASE ONLY: deployment (t={t_dep:.2f} s) to "
        f"impact (t={t_impact:.2f} s). The pre-deployment leg (launch to "
        f"deployment) was not captured by the fired round -- see "
        f"setter/fire_worker.py.")

    axis_limits = _fixed_axis_limits(position, g_log, fired["target_range"], fired["target_defl"])
    phases = _phase_boundaries(t_dep, t_impact, g_log)

    # =======================================================================
    # The scrubber -- drives everything below it
    # =======================================================================
    scrubber_t = st.slider(
        "Time", min_value=t_dep, max_value=t_impact, value=t_dep,
        step=max((t_impact - t_dep) / 200.0, 0.01), format="%.2f s")

    # -- phase strip -------------------------------------------------------
    total = t_impact - t_dep if t_impact > t_dep else 1.0
    ballistic_frac = 0.12  # a fixed, honest sliver representing "before this replay's data starts"
    dep_frac = 0.03
    corr_frac = max(0.02, (phases["terminal"] - phases["correction"]) / total * (1 - ballistic_frac - dep_frac))
    term_frac = max(0.02, 1 - ballistic_frac - dep_frac - corr_frac)
    strip_cols = st.columns([ballistic_frac, dep_frac, corr_frac, term_frac])
    current_phase = ("BALLISTIC" if scrubber_t < t_dep else
                     "DEPLOYMENT" if scrubber_t < phases["correction"] else
                     "TERMINAL" if scrubber_t >= phases["terminal"] else "CORRECTION")
    labels = [("BALLISTIC", "no data — not captured"), ("DEPLOYMENT", "kit deploys"),
              ("CORRECTION", "guidance armed"), ("TERMINAL", "final approach")]
    for col, (label, sub) in zip(strip_cols, labels):
        with col:
            if label == current_phase:
                st.markdown(f"**[ {label} ]**")
            else:
                st.caption(label)
            st.caption(sub)

    idx = _nearest_index(ts, scrubber_t)

    # =======================================================================
    # Guidance's predicted-impact trail, up to the scrubber
    # =======================================================================
    pred_trail_range, pred_trail_defl = [], []
    current_pred_miss = None
    if g_log and g_log.get("t"):
        for i, gt in enumerate(g_log["t"]):
            if gt > scrubber_t:
                break
            pred_trail_range.append(g_log["pred_range"][i])
            pred_trail_defl.append(g_log["pred_defl"][i])
            current_pred_miss = g_log["miss_m"][i]

    # =======================================================================
    # Ground track + altitude (fixed axes) -- Centre
    # =======================================================================
    col_track, col_state = st.columns([2, 1])
    with col_track:
        fig = flight_deck_figure(
            position[:, 0], position[:, 1], -position[:, 2], idx,
            fired["target_range"], fired["target_defl"],
            pred_trail_range, pred_trail_defl, axis_limits, fired["engagement"])
        st.pyplot(fig)
        plt.close(fig)
        if current_pred_miss is not None:
            st.caption(
                f"Guidance's current predicted miss: **{current_pred_miss:.1f} m** "
                f"(converges toward 0 as the round nears impact -- the actual "
                f"final miss was {fired['miss_m']:.1f} m).")

    # =======================================================================
    # State panel -- Right
    # =======================================================================
    with col_state:
        st.subheader("State")
        st.metric("Time of flight", f"{scrubber_t:.2f} s")
        st.metric("Downrange", f"{position[idx, 0]:.0f} m")
        st.metric("Altitude", f"{-position[idx, 2]:.0f} m")
        st.metric("Mach", f"{mach[idx]:.2f}")
        st.metric("Body spin rate", f"{np.degrees(omega[idx, 0]):.0f} °/s")
        if nose_angle is not None:
            st.metric("Nose roll angle φ", f"{np.degrees(nose_angle[idx]) % 360:.0f}°")

        st.subheader("Authority")
        spent = fired.get("g_saturated_fraction")
        if spent is not None:
            st.progress(min(1.0, max(0.0, spent)),
                       text=f"{spent * 100:.0f}% of the flight saturated")
            if fired.get("g_authority_ok") is False:
                st.markdown(f"<span style='color:{AMBER}'>authority monitor flagged "
                           f"this round</span>", unsafe_allow_html=True)
            else:
                st.caption("authority monitor OK")
            st.caption("Whole-flight summary -- not scrubber-dependent.")
        else:
            st.caption("No authority summary stored for this round.")

    # =======================================================================
    # Fuze state -- rendered state machine, not a simulation
    # =======================================================================
    st.subheader("Fuze state")
    fz = fuze_replay.replay_fuze_state(
        tr, t_dep, fired["fuze_mode_context_only"],
        fired["fuze_event_time_s"], fired["fuze_motion_threshold"],
        fired["fuze_proximity_trigger_m"], scrubber_t)
    states = ["SAFE", "ARMING", "ARMED", "FUNCTION_EVENT"]
    fz_cols = st.columns(len(states))
    for col, s in zip(fz_cols, states):
        with col:
            label = "FUNCTION" if s == "FUNCTION_EVENT" else s
            if s == fz.state:
                st.markdown(f"**[ {label} ]**")
            else:
                st.caption(label)
    st.caption(fz.reason)


render()
