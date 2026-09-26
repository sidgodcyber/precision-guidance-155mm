"""
L3 -- The Engine (Control Room spec Part B6, Step 8 second half).

The validation chain as four linked stages: real guns -> published firing
tables -> 6-DOF ground-truth model -> reduced-order flight model -> flight
computer. Two of the four links are measured comparisons stored in
`docs/`; the other two are given (a citation) or true by construction (the
flight computer runs a copy of the reduced-order model, so there is nothing
downstream of it to compare against). A model-comparison section flies one
engagement through both models live and shows the divergence that measures
the flight model's own simplification -- the same quantity
`docs/MODEL-ERROR.md` recommends as a budget term, and explicitly NOT the
same "model error" label `setter.campaign.error_budget` already uses for an
unrelated Task U proxy.

Registered in `setter/app.py` as a FILE-based `st.Page`, unconditionally
(like Error Budget/Archive): no fired round is needed.
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

from setter import validation_data as vd
from setter.config import REPO_ROOT, SUPPORTED_ENGAGEMENTS
from setter.palette import ACCENT
from setter.plotting import (firing_table_error_figure, model_comparison_figure,
                              mpmm_divergence_bar_figure)

#: The live comparison's 6-DOF side costs ~4-29 s depending on engagement
#: (measured: 'short' ~2.5 s, 'long' up to ~29 s on this machine, from the
#: firing-table step counts) -- same reasoning and margin as
#: setter/pages/mission_control.py's FIRE_TIMEOUT_S.
ENGINE_TIMEOUT_S = 120

_FLIGHT_SIDE_FILES = (
    "models/mpmm.py", "gnc/guidance.py", "gnc/inverse_map.py",
    "gnc/navigation.py", "gnc/roll_control.py", "gnc/scheduler.py",
)
_GROUND_ONLY_DIRS = ("sim", "analysis")


@st.cache_data(ttl=300)
def _line_counts() -> dict:
    """Non-blank line counts for the flight-side files and a total for the
    ground-only directories -- computed live from the files on disk (the
    same `.py`s CLAUDE.md names), not repeated as a hand-maintained figure
    that can drift from them."""
    flight = {}
    for rel in _FLIGHT_SIDE_FILES:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        flight[rel] = sum(1 for ln in text.splitlines() if ln.strip())
    ground = {}
    for d in _GROUND_ONLY_DIRS:
        total = 0
        for p in (REPO_ROOT / d).rglob("*.py"):
            total += sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
        ground[d] = total
    return {"flight": flight, "flight_total": sum(flight.values()), "ground": ground}


def render() -> None:
    st.title("The Engine")
    st.caption(
        "The validation chain behind every number this dashboard shows: "
        "each link states what is measured, what is given, and what is "
        "true by construction rather than compared at all.")

    # =======================================================================
    # The four-link chain
    # =======================================================================
    st.header("Real guns → firing tables → 6-DOF → reduced-order → flight computer")

    with st.expander("1. Real guns → published firing tables", expanded=False):
        st.markdown(
            "**Given, not measured in this repository.** The firing-table "
            "values this project validates against are published data, not "
            "a comparison this codebase performs.")
        st.caption(
            "FT 155-AM-2 via Lim, NPS thesis 2016 (AD1029824), Tables 15-19 "
            "(the 15-case envelope in `analysis/mpmm_compare.py:FIRING_TABLE`); "
            "ASAT-13 §4.3 for the max-range/max-charge case "
            "(`docs/validation_results.json['rung5b']`).")

    with st.expander("2. Firing tables → 6-DOF ground-truth model", expanded=True):
        rows = vd.firing_table_rows()
        rms = vd.firing_table_range_rms_pct()
        st.metric("Range RMS error vs firing table, 15 cases",
                  f"{rms:.2f} %")
        st.markdown(
            f"<span style='color:{ACCENT}'>The 6-DOF model reproduces the "
            f"published firing table to {rms:.2f}% range RMS.</span>",
            unsafe_allow_html=True)
        fig = firing_table_error_figure(rows)
        st.pyplot(fig)
        plt.close(fig)
        r5b = vd.rung5b()
        st.caption(
            f"Also checked against ASAT-13's separate max-range case: model "
            f"range {r5b['range_m']:.0f} m vs spec {r5b['spec_range_m']:.0f} m, "
            f"model time of flight {r5b['tof_s']:.1f} s vs spec "
            f"{r5b['spec_tof_s']:.2f} s.")
        st.caption("`run_validation.py` (root-level, not `setter/registry.py`), "
                   "`docs/validation_results.json`. RMS computed here from the "
                   "15 stored per-case `range_err_pct` values, not stored as a scalar.")

    with st.expander("3. 6-DOF → reduced-order flight model (model error)", expanded=False):
        points = vd.mpmm_all_points()
        summary = vd.model_error_summary()
        fig = mpmm_divergence_bar_figure(points)
        st.pyplot(fig)
        plt.close(fig)
        c1, c2 = st.columns(2)
        c1.metric("Range model error, RMS (computed)",
                  f"{summary['range']['rms_m']:.2f} m")
        c2.metric("Deflection model error, RMS (computed)",
                  f"{summary['deflection']['rms_m']:.2f} m")
        st.caption(
            f"Computed here from `docs/mpmm_results.json`'s stored "
            f"`ap_it_d_range`/`ap_it_d_drift` across all 15 engagements "
            f"(apogee-initialised, yaw-of-repose iterated on). "
            f"`docs/MODEL-ERROR.md` rounds this to a recommended budget of "
            f"{summary['recommended_budget_m']['range_1sigma']:.1f} m 1σ range / "
            f"{summary['recommended_budget_m']['deflection_1sigma']:.1f} m 1σ "
            "deflection, and states it is 'no longer a meaningful contributor to CEP.'")
        st.warning(
            "This is a DIFFERENT quantity from Error Budget's 'model error "
            "(shell/lot dispersion proxy)' bar. That bar is Task U's "
            "projectile mass/inertia lot-dispersion proxy from "
            "`docs/monte_carlo.json` -- a different, unrelated quantity, "
            "labelled as a proxy by `setter.campaign.error_budget`'s own "
            "docstring. This number is the actual flight-model-vs-6-DOF "
            "validation error, and it is not currently wired into the CEP "
            "budget at all.")
        st.caption("`analysis/mpmm_compare.py:run_case`, `docs/mpmm_results.json`.")

    with st.expander("4. Reduced-order flight model → flight computer", expanded=False):
        st.markdown(
            "**Validated by construction, not by comparison.** The flight "
            "computer would run the reduced-order model itself (or a direct "
            "C port of it) -- there is no separate, independent "
            "implementation downstream to compare against. The boundary "
            "that actually matters is one hop back: link 3 above.")
        st.caption("See the module diagram below for what would run on the "
                   "flight computer versus what only ever runs on the ground.")

    st.divider()

    # =======================================================================
    # Model-comparison: same shot, both models, overlaid
    # =======================================================================
    st.header("Model comparison: the same shot, both models")
    st.markdown(
        "The reduced-order flight model is **deliberately simplified** -- "
        "no full 6-DOF rigid-body integration, no epicyclic yaw motion. The "
        "difference between the two models flying the identical shot IS the "
        "model error above, and it is a term in the error budget, not an "
        "oversight.")

    engagement = st.selectbox("Engagement", SUPPORTED_ENGAGEMENTS,
                               index=len(SUPPORTED_ENGAGEMENTS) - 1, key="engine_engagement")
    stored = vd.mpmm_point(engagement)
    if stored.available:
        st.caption(
            f"Precomputed (`docs/mpmm_results.json`): peak divergence "
            f"{stored.max_divergence_m:.2f} m at t={stored.t_of_max_divergence:.1f} s.")

    @st.fragment(run_every="1s")
    def live_comparison_fragment():
        job = st.session_state.get("engine_job")
        if job is None:
            st.caption(
                "Flies this engagement from launch through BOTH models live "
                "-- the actual 6-DOF engine and the reduced-order flight "
                "model -- for the full divergence-over-time curve the "
                "precomputed summary above does not carry. Measured cost: "
                "~2-30 s depending on engagement (the 6-DOF side; the "
                "reduced-order side is near-instant). Runs as a subprocess "
                "so the rest of the page stays responsive, and can be "
                "cancelled.")
            if st.button("Run live now", key="engine_run_live"):
                out_path = Path(tempfile.gettempdir()) / f"setter_engine_{int(time.time() * 1000)}.json"
                proc = subprocess.Popen(
                    [sys.executable, "-m", "setter.engine_worker",
                     "--engagement", engagement, "--out", str(out_path)],
                    cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                st.session_state["engine_job"] = {
                    "proc": proc, "out_path": out_path, "start": time.time(),
                    "engagement": engagement}
                job = st.session_state["engine_job"]
            if job is None:
                cached = st.session_state.get("engine_result")
                if cached and cached.get("engagement") == engagement:
                    _render_comparison(cached)
                return

        proc = job["proc"]
        elapsed = time.time() - job["start"]
        if proc.poll() is None:
            if elapsed > ENGINE_TIMEOUT_S:
                proc.kill()
                st.session_state.pop("engine_job", None)
                st.error(f"Live comparison timed out after {ENGINE_TIMEOUT_S} s and was stopped.")
                return
            with st.status(f"Running both models… {elapsed:.0f} s elapsed",
                           expanded=True, state="running"):
                st.caption(f"{job['engagement']}")
                if st.button("Cancel", key="engine_cancel"):
                    proc.kill()
                    st.session_state.pop("engine_job", None)
                    st.warning("Cancelled.")
            return

        out_path = job["out_path"]
        try:
            with open(out_path, encoding="utf-8") as fh:
                result = json.load(fh)
        except Exception as exc:
            result = {"ok": False, "error": f"could not read engine_worker output: {exc}"}
        out_path.unlink(missing_ok=True)
        st.session_state.pop("engine_job", None)
        if result.get("ok"):
            st.session_state["engine_result"] = result
            _render_comparison(result)
        else:
            st.error(f"Live comparison failed: {result.get('error')}")

    def _render_comparison(result: dict) -> None:
        st.caption(
            f"Live run, {result['elapsed_s']:.1f} s. Peak divergence "
            f"{result['max_divergence_m']:.2f} m at t={result['t_of_max_divergence']:.1f} s.")
        fig = model_comparison_figure(
            result["six_dof"]["downrange_m"], result["six_dof"]["crossrange_m"],
            result["mpmm"]["downrange_m"], result["mpmm"]["crossrange_m"],
            result["divergence"]["t"], result["divergence"]["distance_m"],
            result["engagement"])
        st.pyplot(fig)
        plt.close(fig)

    live_comparison_fragment()

    st.divider()

    # =======================================================================
    # Two-column module diagram
    # =======================================================================
    st.header("What runs where")
    counts = _line_counts()
    col_flight, col_ground = st.columns(2)
    with col_flight:
        st.subheader("Flight computer")
        st.caption("Would port; this is the code the guidance kit actually flies.")
        for rel, n in counts["flight"].items():
            st.write(f"`{rel}` -- {n:,} lines")
        st.metric("Total", f"{counts['flight_total']:,} lines")
    with col_ground:
        st.subheader("Ground only")
        st.caption("Never ports; this is the verification bench.")
        for d, n in counts["ground"].items():
            st.write(f"`{d}/` -- {n:,} lines")
        st.caption("`gnc/sensors.py` sits between the two: it models sensors, "
                   "so it is replaced by real parts on hardware.")

    st.info(
        "**The boundary, in the code's own words** (`gnc/sensors.py`): "
        "module docstring -- \"Step 5 sensor models: what the flight "
        "computer actually gets to see.\" `SensorTruth`'s own docstring -- "
        "\"`truth_at` is the boundary between the simulator and the sensor "
        "models, and it is the only place they touch.\"")


render()
