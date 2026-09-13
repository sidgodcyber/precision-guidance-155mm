"""
Simulation setter / mission-planning dashboard.

Run with:  streamlit run setter/app.py

This is a software-only mission-planning UI over the existing simulation
engine. Campaign accuracy figures are read from `docs/monte_carlo.json`
(precomputed -- no Monte Carlo campaign is ever run from a callback here);
the ground-track plot is one lightweight reduced-order trajectory generated
live via `setter.simulation_adapter`. See that module and
`setter/campaign.py` for the boundary against the frozen simulation engine.
"""


from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run setter/app.py` execs this file directly and only puts its
# OWN directory (`setter/`) on sys.path (see
# `streamlit.web.bootstrap._fix_sys_path`) -- never the repository root. That
# leaves `import setter` and `import fuze` unresolvable even though this file
# lives inside the `setter` package, because the package's PARENT directory
# was never added. pytest/AppTest never hit this: pytest puts the repo root
# on sys.path for the whole test process, which masked it there. Put the
# repo root on sys.path before importing anything from `setter`/`fuze`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json

import streamlit as st

from fuze.config import EVENT_KINDS
from setter import campaign, message_codec, simulation_adapter as sim_adapter
from setter.config import MET_AGE_BUCKETS, SUPPORTED_ENGAGEMENTS
from setter.plotting import ground_track_figure, knowledge_term_figure
from setter.schemas import (EventConfiguration, MetProfileMessage, Position,
                             SetterMessage, SimulationConfig)
from setter.validation import SetterValidationError, validate_message_dict

st.set_page_config(page_title="Simulation Setter", layout="wide")

st.title("SIMULATION SETTER")
st.caption("Software-only mission-planning dashboard over a simulation study. "
           "No hardware, no embedded target, no operational control interface.")

# ===========================================================================
# Engagement + met-age controls
# ===========================================================================
col_engagement, col_met_age = st.columns(2)
with col_engagement:
    engagement = st.selectbox("Engagement", SUPPORTED_ENGAGEMENTS, index=len(SUPPORTED_ENGAGEMENTS) - 1)
with col_met_age:
    met_age_bucket = st.select_slider(
        "Meteorological message age", options=MET_AGE_BUCKETS, value="2h",
        help="Selects a precomputed Task C reference point and a demonstration "
             "met profile for the trajectory view below. It does not launch a "
             "new Monte Carlo campaign.")

base = sim_adapter.baseline_for(engagement)
st.write(
    f"**{engagement}** -- charge {base['charge']}, "
    f"QE {base['qe_mils']:.1f} mils, muzzle velocity {base['muzzle_velocity']:.0f} m/s, "
    f"uncorrected range {base['uncorrected_range']:.0f} m")

st.divider()

# ===========================================================================
# Campaign accuracy / reference data
# ===========================================================================
st.header("Campaign accuracy / reference data")
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

st.divider()

# ===========================================================================
# Simulation trajectory
# ===========================================================================
st.header("Simulation trajectory")
st.caption("One lightweight reduced-order simulation trajectory for visualization -- "
           "not a new campaign result.")

met_profile = sim_adapter.met_profile_for_age(met_age_bucket, seed=hash(engagement) & 0xFFFF)
trajectory = sim_adapter.single_trajectory(engagement, met=met_profile)
st.pyplot(ground_track_figure(trajectory, engagement))
st.caption(
    f"time of flight {trajectory.duration_s:.1f} s, range {trajectory.range_m:.0f} m, "
    f"met profile: {met_profile.label}")

st.divider()

# ===========================================================================
# Simulation configuration + message build/validate/export -- one fragment.
#
# Everything below is downstream of `engagement`/`met_age_bucket` (set above,
# outside any fragment -- changing either is a full rerun, correctly, since
# almost the whole page depends on them) but its OWN widgets -- the event
# mode and its three parameters -- feed nothing upstream. Before fragments,
# nudging an event parameter re-ran the whole script: two matplotlib
# renders, a fresh reduced-order trajectory, and the lay_gun/fuze_setting
# solve, none of which the event mode touches. Isolating this section in an
# `@st.fragment` means that interaction now reruns only this: JSON
# assembly, pydantic validation, and CRC32 -- see Step 1 of the Control Room
# build order for the measured before/after.
#
# `base`/`met_profile`/`lay`/`deploy_time` are read fresh on every full
# rerun (i.e. whenever engagement/met-age change) and simply persist in
# closure/session state across the fragment's own reruns, since this
# function is called once per full script execution and Streamlit reruns
# only its body -- not the enclosing script -- on its own widgets.
# ===========================================================================
lay = sim_adapter.lay_gun(engagement, met_profile)
deploy_time = sim_adapter.fuze_setting(engagement, met_profile, lay["dqe_mils"])


@st.fragment
def configuration_and_message_fragment(engagement, met_age_bucket, base, met_profile, lay, deploy_time):
    st.header("Simulation configuration")

    st.subheader("Scenario")
    st.json({
        "scenario_id": engagement,
        "charge": base["charge"],
        "qe_mils": base["qe_mils"],
        "muzzle_velocity_ms": base["muzzle_velocity"],
        "dqe_mils": lay["dqe_mils"],
        "daz_mils": lay["daz"],
        "deploy_time_s": deploy_time,
        "guided_phase_s": base["guided_phase_s"],
    }, expanded=False)

    st.subheader("Environment")
    st.json(met_profile.summary(), expanded=False)

    st.subheader("Mode identifier / parameters")
    mode = st.selectbox("Event engine demonstration mode", ["time", "motion", "proximity", "combined"])
    param_cols = st.columns(3)
    event_time_s = param_cols[0].number_input("time_event.event_time_s", value=30.0, min_value=0.0)
    motion_threshold = param_cols[1].number_input("motion_event.threshold", value=50.0, min_value=0.1)
    proximity_trigger = param_cols[2].number_input("proximity_event.trigger_value", value=5.0, min_value=0.0)

    event_configuration = EventConfiguration(
        mode=mode,
        parameters={
            "event_time_s": event_time_s,
            "motion_threshold": motion_threshold,
            "proximity_trigger_value": proximity_trigger,
            "precedence": list(EVENT_KINDS),
        },
    )

    # -----------------------------------------------------------------
    # Build + validate the message
    # -----------------------------------------------------------------
    message = SetterMessage(
        scenario_id=engagement,
        reference_position=Position(x_m=0.0, y_m=0.0, z_m=0.0),
        target_position=Position(x_m=base["uncorrected_range"], y_m=base["uncorrected_drift"], z_m=0.0),
        met_profile=MetProfileMessage.from_met_profile(met_profile, profile_id=f"{engagement}-{met_age_bucket}"),
        met_age_hours=sim_adapter.age_bucket_to_hours(met_age_bucket),
        simulation_config=SimulationConfig(
            scenario_id=engagement, charge=base["charge"], qe_mils=base["qe_mils"],
            muzzle_velocity_ms=base["muzzle_velocity"], dqe_mils=lay["dqe_mils"],
            daz_mils=lay["daz"], deploy_time_s=deploy_time,
            guided_phase_s=base["guided_phase_s"]),
        event_configuration=event_configuration,
    )

    st.subheader("Validation status")
    try:
        validate_message_dict(json.loads(message.model_dump_json(by_alias=True)))
        st.success("Configuration message is valid.")
        validation_ok = True
    except SetterValidationError as exc:
        st.error(f"Invalid configuration: {exc}")
        validation_ok = False

    st.divider()

    # -----------------------------------------------------------------
    # Serialized configuration
    # -----------------------------------------------------------------
    st.header("Serialized configuration")

    blob, checksum = message_codec.encode(message)
    st.code(blob.decode("utf-8"), language="json")

    m1, m2, m3 = st.columns(3)
    m1.metric("Message size", f"{len(blob)} bytes")
    m2.metric("CRC32", checksum)
    m3.metric("Round-trip check",
              "OK" if message_codec.verify_checksum(blob) else "FAILED")

    st.download_button(
        "Export configuration (.json)", data=blob,
        file_name=f"setter_{engagement}_{met_age_bucket}.json", mime="application/json",
        disabled=not validation_ok)


configuration_and_message_fragment(engagement, met_age_bucket, base, met_profile, lay, deploy_time)
