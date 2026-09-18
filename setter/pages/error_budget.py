"""
L3 -- Error Budget (Control Room spec Part C, Step 6 second half).

Entirely precomputed campaign data. Nothing on this page runs a new
simulation, a new campaign, or a new bootstrap -- every number is read
from `docs/monte_carlo.json` (via `setter.campaign`, which already reads
it) or from `docs/setter_validation/paired_age_comparison.json` (the
already-computed Part 2 paired result -- see
`setter/paired_age_analysis.py`, which produced it).

Registered in `setter/app.py` as a FILE-based `st.Page`, unconditionally
(unlike Flight Deck): this page needs no fired round, only the campaign
data that already ships in the repository.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import streamlit as st

from setter import campaign
from setter.config import REPO_ROOT
from setter.palette import AMBER
from setter.plotting import error_budget_figure, knowledge_term_figure

PAIRED_COMPARISON_PATH = REPO_ROOT / "docs" / "setter_validation" / "paired_age_comparison.json"


@st.cache_data(ttl=300)
def _load_paired_comparison() -> dict:
    with open(PAIRED_COMPARISON_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _paired_row(comps: dict, key: str, label: str) -> None:
    c = comps[key]
    cep = c["cep_difference"]
    excludes_zero = c["interpretation"]["cep_interval_excludes_zero"]
    decomp = c["variance_bias_decomposition"]

    col_verdict, col_detail = st.columns([1, 2.2])
    with col_verdict:
        st.markdown(f"**{label}**")
        if excludes_zero:
            st.markdown("real, dispersion-driven" if decomp["category"] == "dispersion"
                        else "real (CI excludes zero)")
        else:
            st.markdown(f"<span style='color:{AMBER}'>not established</span>",
                       unsafe_allow_html=True)
    with col_detail:
        st.caption(
            f"paired CEP difference {cep['delta_m']:+.1f} m "
            f"[{cep['lo_m']:+.1f}, {cep['hi_m']:+.1f}] m · n={cep['n']} shared "
            f"rounds · {cep['n_boot']} bootstrap replicates, seed {cep['seed']}")
        st.caption(c["interpretation"]["statement"])
        if excludes_zero:
            st.caption(
                f"range-axis decomposition: {decomp['category']} "
                f"(σ delta {decomp['range_sigma_delta_m']:+.1f} m, "
                f"bias delta {decomp['range_bias_delta_m']:+.1f} m)")


def render() -> None:
    st.title("Error Budget")
    st.caption(
        "Entirely precomputed campaign data -- nothing on this page runs a "
        "new simulation, a new campaign, or a new bootstrap.")

    # =======================================================================
    # Ranked contributions
    # =======================================================================
    st.header("Ranked contributions to range dispersion")
    terms = campaign.error_budget()
    if terms:
        fig = error_budget_figure(terms)
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("What each bar measures, and its condition"):
            for t in terms:
                framing = "a guided round's own residual" if t.framing == "guided_residual" \
                    else "what an UNGUIDED round would scatter by (this source alone)"
                st.markdown(f"**{t.name}** — {t.sigma_range_m:.1f} m range 1σ, n={t.n}")
                st.caption(f"{t.note} ({framing}.)")
    else:
        st.info("No error-budget terms available in the stored campaign data.")

    st.divider()

    # =======================================================================
    # Staleness curve
    # =======================================================================
    st.header("Staleness: CEP vs meteorological message age")
    curve = campaign.task_c_curve()
    fig2 = knowledge_term_figure(curve)
    st.pyplot(fig2)
    plt.close(fig2)
    available = [p for p in curve if p.available]
    st.caption(
        "Every point is Task C's own stored bootstrap interval (truth-fed, "
        "navigation excluded, 'long' engagement, headline tag; n="
        f"{available[0].n if available else '?'} per point). CEP rises with "
        "message age out to 6h, then drops at 'none' (no message at all) -- "
        "visible above, and labelled here as UNEXPLAINED: not smoothed, not "
        "fitted, not attributed to a mechanism. See the paired result below "
        "for what part of this curve is and is not established.")

    st.divider()

    # =======================================================================
    # Part 2 -- the paired result
    # =======================================================================
    st.header("The paired result")
    try:
        data = _load_paired_comparison()
    except FileNotFoundError:
        st.warning(
            "docs/setter_validation/paired_age_comparison.json not found -- "
            "run `python -m setter.paired_age_analysis` to generate it.")
    else:
        comps = data["comparisons"]
        _paired_row(comps, "3h_vs_6h", "3h → 6h")
        st.divider()
        _paired_row(comps, "6h_vs_none", "6h → none")
        prov = data.get("provenance", {})
        commit = prov.get("commit")
        st.caption(
            f"Source: docs/setter_validation/paired_age_comparison.json, "
            f"generated {prov.get('timestamp', 'unknown time')}, commit "
            f"{commit[:8] if commit else 'unknown'}. Paired by round index "
            f"within Task C's stored data -- not an independent-sample "
            f"comparison; see setter/paired_age_analysis.py.")

    st.divider()

    # =======================================================================
    # Plain-language summary
    # =======================================================================
    st.header("Summary")
    st.markdown(
        "**This design is limited by how well the gun knows the air, not by "
        "how hard the canards push.** Meteorological knowledge is the "
        "largest single contribution to range dispersion in this budget -- "
        "well above navigation and muzzle velocity, the next two terms -- "
        "and the 3-hour-to-6-hour degradation in that knowledge is a real, "
        "dispersion-driven effect, not sampling noise. The practical "
        "implication is direct: **upload a fresh meteorological message at "
        "fuze setting**, not one that has been sitting for hours.")


render()
