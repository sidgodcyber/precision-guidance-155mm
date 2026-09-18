"""
L2 -- Mission Control, plus transitional L3-ish reference content that has
not yet moved to its own screen.

This is the content that used to be the whole of `setter/app.py` before
Step 4 introduced `st.navigation` (see that module) so the Archive screen
(`setter/pages/archive.py`) could exist alongside it. Nothing about the
Mission Control fragment's behaviour changed in that move -- only where its
code lives.

Registered in `setter/app.py` as a FILE-based `st.Page` (not a callable):
`AppTest.switch_page()` only resolves file-based pages -- confirmed against
the installed Streamlit, where a callable-based `st.Page` raises "Could not
find a navigation page" from `switch_page` -- and Step 6 onward needs page
tests. `render()` stays a plain function (so it can still be unit-tested by
calling it directly), and the bottom of this file also calls it
unconditionally, unguarded by `if __name__ == "__main__"`: Streamlit execs
a file-based page's module top to bottom exactly like the main script, on
every rerun where this page is active.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from fuze.config import EVENT_KINDS
from setter import campaign, message_codec, simulation_adapter as sim_adapter
from setter.config import MET_AGE_BUCKETS, REPO_ROOT, SUPPORTED_ENGAGEMENTS
from setter.plotting import (cep_circle_figure, fire_result_figure,
                              ground_track_figure, knowledge_term_figure)
from setter.schemas import (EventConfiguration, MetProfileMessage, Position,
                             SetterMessage, SimulationConfig)
from setter.validation import SetterValidationError, validate_message_dict

#: A hard ceiling on ONE fired round -- measured cost is ~28-31 s
#: (analysis.nav_common.run_guided_nav, single call, this machine; see
#: setter/fire_worker.py's own docstring for how that was traced and
#: measured, not assumed). 4x margin for a slower machine or a harder
#: round to converge, same reasoning as setter.runner.DEFAULT_TIMEOUT_S.
FIRE_TIMEOUT_S = 120


def render() -> None:
    st.title("SIMULATION SETTER")
    st.caption("Software-only mission-planning dashboard over a simulation study. "
               "No hardware, no embedded target, no operational control interface.")

    # =======================================================================
    # Engagement -- the one control that stays outside every fragment.
    #
    # Changing it is a full script rerun, which is correct: it is upstream
    # of Mission Control below AND of the legacy reference sections further
    # down the page, so both must see the new value in the same rerun. A
    # widget declared inside a fragment could not do that -- see the note
    # above `mission_control_fragment`.
    #
    # `key="engagement"` (not the default auto-generated key) so the L1
    # Overview page can read the current mission back out of
    # `st.session_state["engagement"]` without recomputing anything.
    # =======================================================================
    engagement = st.selectbox("Engagement", SUPPORTED_ENGAGEMENTS,
                               index=len(SUPPORTED_ENGAGEMENTS) - 1, key="engagement")
    base = sim_adapter.baseline_for(engagement)

    st.divider()

    # =======================================================================
    # L2 -- MISSION CONTROL, as one fragment.
    #
    # Left / Centre / Right per the Control Room spec Part C. All three
    # columns, and every widget that feeds them (target offset, fuze
    # mode/parameters, the met-age slider), live INSIDE this one fragment --
    # not split across several -- because they are one coupled screen:
    # dragging met-age must update the predicted CEP AND the fire-control
    # solution AND the live setter message together. Streamlit fragments
    # only isolate cost at their own boundary, so the only way for ONE
    # slider to drive all three columns without going stale is for all
    # three to be inside its same fragment. Splitting them (as the Step 2
    # draft did, with the CEP panel as its own fragment and a duplicate
    # top-level slider for the legacy sections) meant two independent
    # controls doing the same job -- consolidated here into one.
    #
    # `engagement` is a parameter (from the top-level selectbox above, not
    # re-declared here) for the same reason it isn't re-declared in Step
    # 1's fragment: changing it is already a full rerun.
    #
    # The legacy reference sections further down this page (Task A/C table,
    # raw trajectory) are OUTSIDE this fragment and read the met-age bucket
    # back out of `st.session_state["met_age_bucket"]`, which this fragment
    # writes on every run. On a real fragment-scoped rerun (dragging the
    # slider without touching anything else) those sections do NOT
    # re-render -- they keep showing whatever they last rendered on a full
    # rerun. That is expected fragment behaviour, not a bug: those sections
    # are transitional L3-ish content, not part of L2, and Step 6 gives
    # them their own screen.
    # =======================================================================
    @st.fragment
    def mission_control_fragment(engagement: str, base: dict):
        col_left, col_centre, col_right = st.columns([1, 1.3, 1.1])

        # -------------------------------------------------------------
        # Left -- the mission
        # -------------------------------------------------------------
        with col_left:
            st.subheader("Mission")
            st.caption(
                f"**{engagement}** · charge {base['charge']} · "
                f"QE {base['qe_mils']:.1f} mils · muzzle velocity "
                f"{base['muzzle_velocity']:.0f} m/s · uncorrected range "
                f"{base['uncorrected_range']:.0f} m")

            st.markdown("**Target offset**")
            off_cols = st.columns(2)
            range_offset_m = off_cols[0].number_input(
                "Range offset, m", value=0.0, step=10.0, key="mc_target_range_offset",
                help="Added to the gun's own uncorrected impact point to place the target.")
            defl_offset_m = off_cols[1].number_input(
                "Deflection offset, m", value=0.0, step=10.0, key="mc_target_defl_offset")

            st.markdown("**Fuze mode**")
            mode = st.selectbox(
                "Mode", ["time", "motion", "proximity", "combined"],
                format_func=str.capitalize, key="mc_fuze_mode")
            event_time_s = st.number_input(
                "Function time, s", value=30.0, min_value=0.0, key="mc_function_time",
                help="Internally time_event.event_time_s.")
            proximity_trigger = st.number_input(
                "Burst height, m", value=5.0, min_value=0.0, key="mc_burst_height",
                help="Internally proximity_event.trigger_value.")
            motion_threshold = st.number_input(
                "Impact sensitivity", value=50.0, min_value=0.1, key="mc_impact_sensitivity",
                help="Internally motion_event.threshold.")

            st.markdown("**Meteorological message age**")
            age_bucket = st.select_slider(
                "Meteorological message age", options=MET_AGE_BUCKETS, value="2h",
                key="mc_met_age", label_visibility="collapsed",
                help="Looks up a stored Task A campaign point for this engagement "
                     "and age, and drives the live fire-control solve and setter "
                     "message below. Ages the campaign did not fly for this "
                     "engagement report unavailable rather than an interpolated "
                     "number. It does not launch a new Monte Carlo campaign.")
            st.session_state["met_age_bucket"] = age_bucket

        met_profile = sim_adapter.met_profile_for_age(age_bucket, seed=hash(engagement) & 0xFFFF)
        lay = sim_adapter.lay_gun(engagement, met_profile)
        deploy_time = sim_adapter.fuze_setting(engagement, met_profile, lay["dqe_mils"])

        hours = sim_adapter.age_bucket_to_hours(age_bucket)
        if hours is None:
            point = campaign.CampaignPoint(
                available=False, engagement=engagement, tag="", met_age_hours=None,
                reason=f"Task A has no stored campaign point for met-age bucket "
                       f"{age_bucket!r} (it carries no message-age hours to look up).")
        else:
            point = campaign.task_a_by_age(engagement, hours)

        # The L1 Overview page's "predicted accuracy" tile reads this back
        # rather than recomputing it -- same CampaignPoint, not a second
        # lookup with room to disagree.
        st.session_state["predicted_cep"] = {
            "engagement": engagement, "age_bucket": age_bucket,
            "available": point.available,
            "cep_m": point.cep_m if point.available else None,
            "n": point.n if point.available else None,
            "reason": point.reason if not point.available else None,
        }

        # -------------------------------------------------------------
        # Centre -- the prediction (the hero of the application)
        # -------------------------------------------------------------
        with col_centre:
            st.subheader("Predicted accuracy")
            if point.available:
                st.metric("Predicted CEP", f"{point.cep_m:.1f} m")
                st.caption(
                    f"**{engagement}** · met age **{age_bucket}** · "
                    f"n={point.n} guided rounds"
                    + (f" · SE {point.cep_se_m:.1f} m" if point.cep_se_m is not None else ""))
            else:
                st.metric("Predicted CEP", "unavailable")
                st.caption(point.reason)

            axis_limit_m = campaign.task_a_max_miss_m() * 1.08
            if point.available:
                scatter = campaign.task_a_scatter(engagement, point.tag)
                fig = cep_circle_figure(scatter, point.cep_m, axis_limit_m, engagement, age_bucket)
                st.pyplot(fig)
                plt.close(fig)  # this fragment redraws on every slider drag
                                 # within one session -- an unclosed Figure leaks.
            else:
                st.info(f"No stored campaign scatter for {engagement!r} at met age {age_bucket!r}.")

            st.caption(
                "Stored Task A campaign statistic (navigation-in-loop) -- not a "
                "live simulation result. Headline finding (CLAUDE.md): 106.2 m "
                "at a two-hour-old met message vs 27.5 m with met uploaded at "
                "fuze setting, long engagement -- not claimed as compliance "
                "against the 30 m requirement.")

            st.markdown("**Fire-control solution**")
            # Stacked, not a 3-way st.columns() split: col_centre is already
            # one of three top-level columns, and squeezing "Quadrant
            # elevation" / "Azimuth correction" into a further three-way
            # split left both label and value too narrow for their
            # container, which Streamlit renders as a truncating ellipsis
            # rather than wrapping. Fixed decimal counts throughout (no
            # bare/variable-length floats).
            st.metric("Quadrant elevation", f"{base['qe_mils'] + lay['dqe_mils']:.1f} mils")
            st.metric("Azimuth correction", f"{lay['daz']:+.2f} mils")
            st.metric("Deployment time", f"{deploy_time:.2f} s")
            st.caption("Attributed to `analysis.monte_carlo.lay_gun` / `.fuze_setting` "
                       "against the current meteorological message.")

        # -------------------------------------------------------------
        # Build + validate the message (feeds the Right column)
        # -------------------------------------------------------------
        target_position = Position(
            x_m=base["uncorrected_range"] + range_offset_m,
            y_m=base["uncorrected_drift"] + defl_offset_m, z_m=0.0)
        event_configuration = EventConfiguration(
            mode=mode,
            parameters={
                "event_time_s": event_time_s,
                "motion_threshold": motion_threshold,
                "proximity_trigger_value": proximity_trigger,
                "precedence": list(EVENT_KINDS),
            },
        )
        message = SetterMessage(
            scenario_id=engagement,
            reference_position=Position(x_m=0.0, y_m=0.0, z_m=0.0),
            target_position=target_position,
            met_profile=MetProfileMessage.from_met_profile(met_profile, profile_id=f"{engagement}-{age_bucket}"),
            met_age_hours=hours,
            simulation_config=SimulationConfig(
                scenario_id=engagement, charge=base["charge"], qe_mils=base["qe_mils"],
                muzzle_velocity_ms=base["muzzle_velocity"], dqe_mils=lay["dqe_mils"],
                daz_mils=lay["daz"], deploy_time_s=deploy_time,
                guided_phase_s=base["guided_phase_s"]),
            event_configuration=event_configuration,
        )

        try:
            validate_message_dict(json.loads(message.model_dump_json(by_alias=True)))
            validation_ok = True
            validation_error = None
        except SetterValidationError as exc:
            validation_ok = False
            validation_error = str(exc)

        blob, checksum = message_codec.encode(message)

        # -------------------------------------------------------------
        # Right -- the setter message
        # -------------------------------------------------------------
        with col_right:
            st.subheader("Setter message")
            st.caption(
                "The data crossing the inductive interface before firing, "
                "including the full meteorological profile.")

            if validation_ok:
                st.success("Configuration message is valid.")
            else:
                st.error(f"Invalid configuration: {validation_error}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Message size", f"{len(blob)} bytes")
            m2.metric("CRC32", checksum)
            m3.metric("Round-trip check",
                      "OK" if message_codec.verify_checksum(blob) else "FAILED")

            st.download_button(
                "Export configuration (.json)", data=blob,
                file_name=f"setter_{engagement}_{age_bucket}.json", mime="application/json",
                disabled=not validation_ok, key="mc_export")

            with st.expander("Full message JSON"):
                st.code(blob.decode("utf-8"), language="json")

        st.session_state["mission_message"] = {
            "engagement": engagement, "age_bucket": age_bucket,
            "validation_ok": validation_ok, "blob": blob,
        }

    mission_control_fragment(engagement, base)

    st.divider()

    # =======================================================================
    # Bottom -- FIRE (Control Room spec Part C / B3).
    #
    # One real round through the ACTUAL 6-DOF engine, navigation in the
    # loop -- not the reduced-order model the "Simulation trajectory"
    # section below (and the live ground track) uses. Traced, not assumed:
    # `analysis.monte_carlo.task_a` (the campaign function behind this
    # project's headline CEP) flies each round via
    # `analysis.nav_common.run_guided_nav`, called here exactly as the
    # campaign calls it -- see `setter/fire_worker.py`'s docstring for the
    # full trace, the measured ~28-31 s cost, and what it does and does not
    # return (no full state trajectory; `g_log`, the guidance law's own
    # predicted-impact history, is the closest available substitute).
    #
    # `run_every="1s"`, per B1: only this fragment reruns while a round is
    # in flight, so dragging Mission Control's other controls stays fully
    # responsive during a 30-second FIRE. Subprocess per B3/B4 -- launched
    # via `python -m setter.fire_worker`, never imported and called inline
    # -- with a hard timeout (`FIRE_TIMEOUT_S`) and a Cancel button that
    # kills it.
    #
    # NEVER RUN FROM OVERVIEW OR ARCHIVE: this fragment exists only here.
    # =======================================================================
    def _launch_fire(mission: dict) -> dict:
        out_path = Path(tempfile.gettempdir()) / f"setter_fire_{int(time.time() * 1000)}.json"
        seed = int.from_bytes(os.urandom(4), "big")
        proc = subprocess.Popen(
            [sys.executable, "-m", "setter.fire_worker",
             "--engagement", mission["engagement"], "--met-age", mission["met_age_bucket"],
             "--range-offset", str(mission["range_offset_m"]),
             "--defl-offset", str(mission["defl_offset_m"]),
             "--fuze-mode", mission["fuze_mode"],
             "--fuze-event-time-s", str(mission["fuze_event_time_s"]),
             "--fuze-motion-threshold", str(mission["fuze_motion_threshold"]),
             "--fuze-proximity-trigger-m", str(mission["fuze_proximity_trigger_m"]),
             "--seed", str(seed), "--out", str(out_path)],
            cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return {"proc": proc, "out_path": out_path, "start": time.time(),
                "mission": mission, "seed": seed}

    def _current_mission_for_fire() -> dict:
        # The three fuze-event parameters travel with the fired round (not
        # just the mode) so Flight Deck can replay the SAME fuze
        # configuration that was live at fire time, even if the operator
        # changes Mission Control's sliders afterward.
        return {
            "engagement": st.session_state.get("engagement", engagement),
            "met_age_bucket": st.session_state.get("met_age_bucket", "2h"),
            "range_offset_m": st.session_state.get("mc_target_range_offset", 0.0),
            "defl_offset_m": st.session_state.get("mc_target_defl_offset", 0.0),
            "fuze_mode": st.session_state.get("mc_fuze_mode", "time"),
            "fuze_event_time_s": st.session_state.get("mc_function_time", 30.0),
            "fuze_motion_threshold": st.session_state.get("mc_impact_sensitivity", 50.0),
            "fuze_proximity_trigger_m": st.session_state.get("mc_burst_height", 5.0),
        }

    def _render_fired_result(result: dict) -> None:
        st.subheader("Last FIRE result")
        m = result
        st.caption(
            f"**{m['engagement']}** · met age **{m['met_age_bucket']}** · "
            f"fuze mode {m['fuze_mode_context_only']} · seed {m['seed']} · "
            f"real 6-DOF round, {m['elapsed_s']:.1f} s")
        hours = sim_adapter.age_bucket_to_hours(m["met_age_bucket"])
        ref_point = (campaign.task_a_by_age(m["engagement"], hours)
                    if hours is not None else None)
        cep_ref = ref_point.cep_m if (ref_point and ref_point.available) else campaign.task_a_max_miss_m() * 0.3
        axis_limit_m = campaign.task_a_max_miss_m() * 1.08
        col_num, col_fig = st.columns([1, 2])
        with col_num:
            st.metric("Miss distance", f"{m['miss_m']:.1f} m")
            st.caption(f"range miss {m['miss_range_m']:+.1f} m · "
                      f"deflection miss {m['miss_defl_m']:+.1f} m")
            st.caption(f"time of flight {m['tof_s']:.1f} s · "
                      f"deployment at {m['t_dep_actual']:.2f} s")
            if m.get("state_trajectory_available"):
                n_samples = len(m["state_trajectory"]["t"]) if m.get("state_trajectory") else 0
                st.caption(
                    f"Full 6-DOF state trajectory stored ({n_samples} samples, "
                    f"guided phase only -- deployment to impact).")
                st.page_link("pages/flight_deck.py", label="Open Flight Deck")
            else:
                st.caption(
                    "Full 6-DOF state history is not available for this "
                    "result -- see setter/fire_worker.py.")
        with col_fig:
            fig = fire_result_figure(m["miss_range_m"], m["miss_defl_m"], cep_ref,
                                     axis_limit_m, m["engagement"], m["met_age_bucket"])
            st.pyplot(fig)
            plt.close(fig)

    @st.fragment(run_every="1s")
    def fire_fragment():
        # No explicit st.rerun() anywhere in this function, deliberately:
        # every path that changes state below (launching, cancelling,
        # timing out, completing) does so INSIDE a rerun that a widget
        # click or the run_every timer already triggered -- Streamlit
        # reruns a fragment automatically on either, so forcing another
        # one is both unnecessary and, for scope="fragment" specifically,
        # invalid outside a genuine fragment-scoped rerun (confirmed: it
        # raises StreamlitInvalidLayoutContextError under AppTest's
        # full-script reruns, which is exactly the kind of rerun a real
        # browser also performs on e.g. the initial page load). Instead,
        # each branch that changes `fire_job`/`fired_round` just falls
        # through to render the new state in the SAME pass.
        st.header("FIRE")
        job = st.session_state.get("fire_job")

        if job is None:
            mission = _current_mission_for_fire()
            st.caption(
                "Runs ONE real round through the 6-DOF engine with navigation "
                "in the loop, at the current mission above -- not the "
                "reduced-order model the ground track further down this page "
                "uses. Measured cost: ~28-31 s for one round on this machine. "
                "Runs as a subprocess so the rest of Mission Control stays "
                "responsive, and can be cancelled.")
            if st.button("FIRE", type="primary", key="fire_button"):
                job = _launch_fire(mission)
                st.session_state["fire_job"] = job
                # falls through below to show its status immediately,
                # rather than waiting up to 1 s for the next timer tick.
            if job is None:
                fired = st.session_state.get("fired_round")
                if fired is not None:
                    _render_fired_result(fired)
                return

        proc = job["proc"]
        elapsed = time.time() - job["start"]
        if proc.poll() is None:
            if elapsed > FIRE_TIMEOUT_S:
                proc.kill()
                st.session_state.pop("fire_job", None)
                st.error(f"FIRE timed out after {FIRE_TIMEOUT_S} s and was stopped.")
                return
            with st.status(f"Firing… {elapsed:.0f} s elapsed (typically ~28-31 s)",
                           expanded=True, state="running"):
                st.caption(
                    f"{job['mission']['engagement']} · met age "
                    f"{job['mission']['met_age_bucket']} · seed {job['seed']}")
                if st.button("Cancel", key="fire_cancel"):
                    proc.kill()
                    st.session_state.pop("fire_job", None)
                    st.warning("Cancelled.")
            return

        out_path = job["out_path"]
        try:
            with open(out_path, encoding="utf-8") as fh:
                result = json.load(fh)
        except Exception as exc:
            result = {"ok": False, "error": f"could not read FIRE output: {exc}"}
        out_path.unlink(missing_ok=True)
        st.session_state.pop("fire_job", None)
        if result.get("ok"):
            st.session_state["fired_round"] = result
            # A PLAIN (unscoped) rerun here, deliberately -- not the
            # scope="fragment" rerun this function otherwise avoids
            # entirely (see the note above). Flight Deck is only added to
            # `st.navigation`'s page list (setter/app.py) on a FULL rerun
            # of the top-level script; this fragment-scoped completion
            # pass is not one, so `st.page_link("pages/flight_deck.py")`
            # below would raise StreamlitPageNotFoundError if called right
            # here. One full rerun, exactly once per completed FIRE, is a
            # small, deliberate exception to "no explicit rerun in this
            # function" -- the alternative (never registering Flight Deck
            # until some UNRELATED full rerun happens to occur) is worse.
            st.rerun()
        else:
            st.error(f"FIRE failed: {result.get('error')}")

    fire_fragment()

    st.divider()

    # =======================================================================
    # Campaign accuracy / reference data (transitional L3-ish content -- see
    # module docstring; reads the met-age bucket Mission Control last set).
    #
    # Collapsed by default: with Mission Control now covering the same
    # ground (predicted CEP, its own impact scatter) this section is pure
    # duplication unless you go looking for it, and left expanded it read
    # as two different apps stitched together. Undo (or remove outright)
    # once Step 6 gives this content its own Archive screen.
    # =======================================================================
    met_age_bucket = st.session_state.get("met_age_bucket", "2h")

    with st.expander("Campaign accuracy / reference data", expanded=False):
        st.subheader("Navigation-in-loop campaign reference (Task A)")

        headline_table = campaign.task_a_headline_table()
        rows = [
            {"engagement": e, "cep_m": round(p.cep_m, 2) if p.available else None, "n": p.n}
            for e, p in headline_table.items()
        ]
        st.table(rows)
        st.warning(
            "The 2-hour headline CEP does not order monotonically with range "
            "(mid2 exceeds both middle and long). This is reported as an unexplained "
            "campaign variation, not smoothed or fitted -- see CLAUDE.md.")

        fresh = campaign.task_a_point("long", "fresh_met")
        headline_long = campaign.task_a_point("long", "headline")
        physical = campaign.task_a_point("long", "physical")
        c1, c2, c3 = st.columns(3)
        c1.metric("Fresh met (long, upload-at-fuze-setting)",
                  f"{fresh.cep_m:.2f} m" if fresh.available else "unavailable")
        c2.metric("2h met (long, headline)",
                  f"{headline_long.cep_m:.2f} m" if headline_long.available else "unavailable")
        c3.metric("Physical control (long, no dispersion top-up)",
                  f"{physical.cep_m:.2f} m" if physical.available else "unavailable")
        if engagement != "long":
            st.caption(
                f"Task A stores fresh_met/physical points only for 'long'. "
                f"No campaign-quality fresh-met or physical figure is stored for "
                f"'{engagement}'.")

        st.subheader("Atmospheric knowledge term -- truth-fed, navigation excluded (Task C)")
        curve = campaign.task_c_curve()
        fig_c = knowledge_term_figure(curve, highlight_age=met_age_bucket)
        st.pyplot(fig_c)

        selected_c = campaign.task_c_point(met_age_bucket)
        if selected_c.available:
            st.write(
                f"At **{met_age_bucket}**: CEP {selected_c.cep_m:.2f} m (n={selected_c.n})"
                + (f", knowledge-term CEP contribution {selected_c.cep_contribution_m:.2f} m"
                   if selected_c.cep_contribution_m is not None else ""))
        else:
            st.info(selected_c.reason)

        st.page_link("pages/error_budget.py",
                    label="Open Error Budget -- ranked contributions and the paired staleness result")

    st.divider()

    # =======================================================================
    # Simulation trajectory (transitional L3-ish content -- see module docstring)
    # =======================================================================
    st.header("Simulation trajectory")
    st.caption("One lightweight reduced-order simulation trajectory for visualization -- "
               "not a new campaign result.")

    met_profile = sim_adapter.met_profile_for_age(met_age_bucket, seed=hash(engagement) & 0xFFFF)
    trajectory = sim_adapter.single_trajectory(engagement, met=met_profile)
    st.pyplot(ground_track_figure(trajectory, engagement))
    st.caption(
        f"time of flight {trajectory.duration_s:.1f} s, range {trajectory.range_m:.0f} m, "
        f"met profile: {met_profile.label}")


render()
