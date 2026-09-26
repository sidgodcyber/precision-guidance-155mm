"""
Matplotlib figures for the setter dashboard.

Two distinct figures, matching the two distinct data sources they draw from
(see setter/campaign.py's module docstring):

  `ground_track_figure`     -- a single lightweight simulation trajectory
                                (`fuze.trajectory_adapter.TrajectorySample`),
                                a visualization, not a campaign result.
  `knowledge_term_figure`   -- the precomputed Task C atmospheric-knowledge
                                curve across met-message age.

Neither function runs any simulation; figures are built from data already
computed elsewhere (setter.simulation_adapter, setter.campaign).
"""

from __future__ import annotations

import math
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fuze.trajectory_adapter import TrajectorySample
from setter.campaign import BudgetTerm, KnowledgeTermPoint
from setter.palette import ACCENT as _ACCENT
from setter.palette import AMBER as _AMBER
from setter.palette import BG as _BG
from setter.palette import GRID as _GRID
from setter.palette import INK as _INK
from setter.palette import TEXT as _TEXT

__all__ = ["ground_track_figure", "knowledge_term_figure", "cep_circle_figure",
           "fire_result_figure", "flight_deck_figure", "error_budget_figure",
           "fixed_fin_figure", "roll_decoupling_figure",
           "nose_force_and_correction_figure", "brake_power_figure",
           "authority_budget_figure", "firing_table_error_figure",
           "mpmm_divergence_bar_figure", "model_comparison_figure"]


def _style_axes(*axes) -> None:
    """Shared dark-theme axis styling (grid/spine/tick colours) -- every
    figure in this module applies this, so Physics Lab / The Engine's new
    figures below factor it out rather than repeating six lines each."""
    for ax in axes:
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_INK)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.grid(alpha=0.3, color=_GRID)

#: ISA-101 colour discipline (see CLAUDE.md / Control Room spec Part A2),
#: from `setter.palette` -- the app's one colour vocabulary, shared with
#: every other screen. `_ACCENT` marks the single most important number on
#: a given screen -- here, the predicted CEP itself. Scatter points and the
#: target are neutral; amber is reserved for an unresolved/unavailable
#: condition elsewhere in the UI, never used decoratively in this figure.


def ground_track_figure(sample: TrajectorySample, label: str) -> plt.Figure:
    fig, (ax_plan, ax_profile) = plt.subplots(1, 2, figsize=(9, 4))

    ax_plan.plot(sample.downrange / 1000.0, sample.crossrange, color="#2b6cb0")
    ax_plan.scatter([sample.downrange[-1] / 1000.0], [sample.crossrange[-1]],
                     color="#c53030", zorder=5, label="impact")
    ax_plan.set_xlabel("downrange, km")
    ax_plan.set_ylabel("crossrange, m")
    ax_plan.set_title(f"ground track -- {label}")
    ax_plan.legend(loc="best", fontsize=8)
    ax_plan.grid(alpha=0.3)

    ax_profile.plot(sample.downrange / 1000.0, sample.altitude, color="#2f855a")
    ax_profile.set_xlabel("downrange, km")
    ax_profile.set_ylabel("altitude, m")
    ax_profile.set_title("altitude profile")
    ax_profile.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def knowledge_term_figure(points: List[KnowledgeTermPoint],
                           highlight_age: str = None) -> plt.Figure:
    """The Task C staleness curve: CEP vs met-message age across all seven
    buckets, with each available point's stored bootstrap interval drawn
    as an error bar (`KnowledgeTermPoint.cep_lo_m`/`cep_hi_m` -- see
    `setter.campaign.task_c_point`). Shared by Mission Control's own
    campaign-accuracy section and the Error Budget page (Part C L3): one
    figure, one set of honesty rules, not two that could drift apart.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor(_BG)

    ages = [p.age for p in points]
    ceps = [p.cep_m if p.available else float("nan") for p in points]
    los = [p.cep_m - p.cep_lo_m if (p.available and p.cep_lo_m is not None) else 0.0 for p in points]
    his = [p.cep_hi_m - p.cep_m if (p.available and p.cep_hi_m is not None) else 0.0 for p in points]
    colors = [_ACCENT if p.age == highlight_age else _INK for p in points]

    ax.plot(ages, ceps, color=_INK, linewidth=1, zorder=1)
    ax.errorbar(ages, ceps, yerr=[los, his], fmt="none", ecolor=_INK,
               elinewidth=1, capsize=3, alpha=0.7, zorder=2)
    ax.scatter(ages, ceps, c=colors, zorder=3)
    for p, cep in zip(points, ceps):
        if not p.available:
            continue
        ax.annotate(f"{cep:.1f} m", (p.age, cep), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=_TEXT)

    ax.set_xlabel("met message age", color=_INK)
    ax.set_ylabel("CEP, m (truth-fed, navigation excluded)", color=_INK)
    ax.set_title("atmospheric knowledge term -- Task C", color=_TEXT)
    ax.tick_params(colors=_INK)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(alpha=0.3, color=_GRID)
    fig.tight_layout()
    return fig


def cep_circle_figure(scatter_m: List[tuple], cep_m: float, axis_limit_m: float,
                       engagement: str, age_label: str) -> plt.Figure:
    """A to-scale top-down view: the target at the origin, a circle at the
    stored campaign CEP, and that campaign's own per-round impact scatter
    (range-miss, deflection-miss), all in metres from the target.

    `axis_limit_m` is a FIXED reference (see
    `setter.campaign.task_a_max_miss_m`), not fit to `scatter_m` -- so the
    circle's size change between met ages is to scale, not a relabelled
    graphic on a rescaling frame. Dark, transparent background per the
    Control Room spec's dark-theme rule.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor(_BG)

    if scatter_m:
        xs, ys = zip(*scatter_m)
        ax.scatter(xs, ys, s=12, c=_INK, alpha=0.55, linewidths=0, zorder=2,
                   label=f"campaign impacts, stored (n={len(scatter_m)})")

    circle = plt.Circle((0, 0), cep_m, fill=False, edgecolor=_ACCENT, linewidth=2.2, zorder=3)
    ax.add_patch(circle)
    ax.scatter([0], [0], marker="+", s=160, c=_TEXT, linewidths=2, zorder=4, label="target")

    ax.set_xlim(-axis_limit_m, axis_limit_m)
    ax.set_ylim(-axis_limit_m, axis_limit_m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("range miss, m", color=_INK)
    ax.set_ylabel("deflection miss, m", color=_INK)
    ax.set_title(f"{engagement} -- met age {age_label}", color=_TEXT)
    ax.tick_params(colors=_INK)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(alpha=0.3, color=_GRID)
    # Placed BELOW the axes, not in a corner: the scatter is real campaign
    # data, and which corner it spreads toward varies by engagement/met age
    # (e.g. mid2/middle both reach into the upper right at their headline
    # point) -- a fixed corner will eventually sit on top of points for some
    # combination. Below the frame never collides, regardless of the data.
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
                       fontsize=7, facecolor=_BG, edgecolor=_GRID, frameon=True)
    for text in legend.get_texts():
        text.set_color(_TEXT)
    fig.tight_layout()
    return fig


def fire_result_figure(miss_range_m: float, miss_defl_m: float, cep_m_for_reference: float,
                       axis_limit_m: float, engagement: str, age_label: str) -> plt.Figure:
    """A to-scale top-down view for one FIRE result: target, the STORED
    predicted-CEP circle for context (neutral -- it is the number already
    shown elsewhere, not new), and this round's own actual impact point
    (accent-coloured -- the one new, most important thing this view adds),
    joined by a line labelled with the miss distance.

    Same fixed `axis_limit_m` as `cep_circle_figure` (see
    `setter.campaign.task_a_max_miss_m`), so a FIRE result and the stored
    campaign scatter are always visually comparable on the same frame.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor(_BG)

    circle = plt.Circle((0, 0), cep_m_for_reference, fill=False, edgecolor=_INK,
                        linewidth=1.4, linestyle="--", zorder=2,
                        label=f"predicted CEP, stored ({cep_m_for_reference:.1f} m)")
    ax.add_patch(circle)
    ax.scatter([0], [0], marker="+", s=160, c=_TEXT, linewidths=2, zorder=4, label="target")

    ax.plot([0, miss_range_m], [0, miss_defl_m], color=_ACCENT, linewidth=1.2,
           linestyle=":", zorder=3)
    miss_m = (miss_range_m ** 2 + miss_defl_m ** 2) ** 0.5
    ax.scatter([miss_range_m], [miss_defl_m], marker="o", s=90, c=_ACCENT,
              edgecolors=_TEXT, linewidths=0.8, zorder=5,
              label=f"actual impact, this round ({miss_m:.1f} m miss)")

    ax.set_xlim(-axis_limit_m, axis_limit_m)
    ax.set_ylim(-axis_limit_m, axis_limit_m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("range miss, m", color=_INK)
    ax.set_ylabel("deflection miss, m", color=_INK)
    ax.set_title(f"{engagement} -- met age {age_label} -- FIRE result", color=_TEXT)
    ax.tick_params(colors=_INK)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(alpha=0.3, color=_GRID)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1,
                       fontsize=7, facecolor=_BG, edgecolor=_GRID, frameon=True)
    for text in legend.get_texts():
        text.set_color(_TEXT)
    fig.tight_layout()
    return fig


def flight_deck_figure(downrange_m, crossrange_m, altitude_m, idx: int,
                       target_range_m: float, target_defl_m: float,
                       pred_trail_range_m, pred_trail_defl_m,
                       axis_limits: dict, engagement: str) -> plt.Figure:
    """Flight Deck's two-panel replay view: ground track (downrange vs
    crossrange) and altitude profile, current position marked on both,
    the guidance law's predicted-impact trail and current prediction drawn
    on the ground track with a miss vector to the target.

    `axis_limits` is a dict of FIXED bounds -- `{"downrange": (lo, hi),
    "crossrange": (lo, hi), "altitude": (lo, hi)}` -- computed ONCE from
    the whole flown trajectory (and the whole prediction trail) by the
    caller, not refit per frame: the spec requires the view not rescale as
    the scrubber moves, and a circle/line that rescales with the data looks
    like it's growing or shrinking even when nothing has changed.

    The predicted-impact point is the ONE accent-coloured element (Part A2:
    "watching it converge on the target... is the whole demo" -- the single
    most important thing on this screen); current position is a neutral
    marker, since it is simply where the round is, not new information.
    """
    fig, (ax_track, ax_alt) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax in (ax_track, ax_alt):
        ax.set_facecolor(_BG)
    fig.patch.set_alpha(0.0)

    dr_km = [d / 1000.0 for d in downrange_m]

    # -- ground track --------------------------------------------------
    ax_track.plot(dr_km, crossrange_m, color=_INK, linewidth=1.2, zorder=2,
                 label="flown (guided phase)")
    ax_track.scatter([target_range_m / 1000.0], [target_defl_m], marker="+",
                     s=160, c=_TEXT, linewidths=2, zorder=4, label="target")
    if pred_trail_range_m:
        trail_km = [r / 1000.0 for r in pred_trail_range_m]
        ax_track.plot(trail_km, pred_trail_defl_m, color=_ACCENT, linewidth=0.9,
                     linestyle=":", alpha=0.6, zorder=3)
        ax_track.plot([pred_trail_range_m[-1] / 1000.0, target_range_m / 1000.0],
                     [pred_trail_defl_m[-1], target_defl_m], color=_ACCENT,
                     linewidth=1.0, linestyle="--", zorder=3)
        ax_track.scatter([pred_trail_range_m[-1] / 1000.0], [pred_trail_defl_m[-1]],
                         marker="D", s=70, c=_ACCENT, edgecolors=_TEXT, linewidths=0.6,
                         zorder=5, label="guidance's current predicted impact")
    ax_track.scatter([dr_km[idx]], [crossrange_m[idx]], marker="o", s=60, c=_TEXT,
                     edgecolors=_BG, linewidths=0.8, zorder=6, label="current position")
    ax_track.set_xlim(axis_limits["downrange"][0] / 1000.0, axis_limits["downrange"][1] / 1000.0)
    ax_track.set_ylim(*axis_limits["crossrange"])
    ax_track.set_xlabel("downrange, km", color=_INK)
    ax_track.set_ylabel("crossrange, m", color=_INK)
    ax_track.set_title("ground track", color=_TEXT)

    # -- altitude profile ------------------------------------------------
    ax_alt.plot(dr_km, altitude_m, color=_INK, linewidth=1.2, zorder=2)
    ax_alt.scatter([dr_km[idx]], [altitude_m[idx]], marker="o", s=60, c=_TEXT,
                   edgecolors=_BG, linewidths=0.8, zorder=6)
    ax_alt.set_xlim(axis_limits["downrange"][0] / 1000.0, axis_limits["downrange"][1] / 1000.0)
    ax_alt.set_ylim(*axis_limits["altitude"])
    ax_alt.set_xlabel("downrange, km", color=_INK)
    ax_alt.set_ylabel("altitude, m", color=_INK)
    ax_alt.set_title("altitude profile", color=_TEXT)

    for ax in (ax_track, ax_alt):
        ax.tick_params(colors=_INK)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.grid(alpha=0.3, color=_GRID)

    legend = ax_track.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                             fontsize=7, facecolor=_BG, edgecolor=_GRID, frameon=True)
    for text in legend.get_texts():
        text.set_color(_TEXT)

    fig.suptitle(f"{engagement} -- Flight Deck replay", color=_TEXT, fontsize=10)
    fig.tight_layout()
    return fig


def error_budget_figure(terms: List[BudgetTerm]) -> plt.Figure:
    """The Error Budget's ranked-contributions chart: horizontal bars,
    sorted descending (the caller does the sorting -- see
    `setter.campaign.error_budget`), dominant term obvious at a glance.

    The single largest bar is the one accent-coloured element, per Part
    A2 ("one accent colour for the single most important number on
    screen"); everything else neutral. A navigation-style bar (one with a
    stored interval) gets an error bar; the others don't, because the
    underlying data doesn't have one -- no interval is invented to make
    the chart look uniform.
    """
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(terms) + 1.2))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor(_BG)

    names = [t.name for t in terms]
    values = [t.sigma_range_m for t in terms]
    y = list(range(len(terms)))[::-1]
    colors = [_ACCENT if i == 0 else _INK for i in range(len(terms))]

    ax.barh(y, values, color=colors, height=0.55, zorder=2)
    for yi, t in zip(y, terms):
        label_x = t.sigma_range_m
        if t.interval_range_m and None not in t.interval_range_m:
            lo, hi = t.interval_range_m
            ax.errorbar([t.sigma_range_m], [yi], xerr=[[t.sigma_range_m - lo], [hi - t.sigma_range_m]],
                       fmt="none", ecolor=_TEXT, elinewidth=1.2, capsize=4, zorder=3)
            label_x = hi
        ax.annotate(f"{t.sigma_range_m:.1f} m  (n={t.n})", (label_x, yi),
                   textcoords="offset points", xytext=(10, 0), va="center",
                   fontsize=8, color=_TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels(names, color=_TEXT)
    ax.set_xlabel("range 1σ, m", color=_INK)
    ax.set_title("error budget -- ranked contributions to range dispersion",
                color=_TEXT, fontsize=10)
    ax.tick_params(colors=_INK)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(alpha=0.3, color=_GRID, axis="x")
    ax.set_xlim(0, max(values) * 1.45)
    fig.tight_layout()
    return fig


# ===========================================================================
# Physics Lab (Step 8, Part B5) -- six concept cards over the frozen
# sim.canards / gnc.roll_control functions. Each figure takes ALREADY
# COMPUTED numbers; no physics lives in this module (see
# setter/pages/physics_lab.py for where every number here comes from).
# ===========================================================================
def fixed_fin_figure(circle_radius_n: float, avg_vector_yz: tuple,
                      spin_rad_s: float) -> plt.Figure:
    """Card 1 -- why a fixed fin cannot steer a spinning shell.

    The dashed circle is every direction a canard BOLTED RIGIDLY to the body
    (no bearing) could push, depending only on where it happens to be
    clocked -- its radius is the one fixed force magnitude the geometry
    produces, independent of spin. The single accent point is that same
    force, time-averaged over a fixed one-second window at the CURRENT spin
    rate: at zero spin it sits on the circle (the fin never moves, so the
    average is just wherever it started); as spin rises the point collapses
    toward the origin, because the body -- and the bolted fin with it --
    sweeps through every direction many times within the averaging window.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_alpha(0.0)
    _style_axes(ax)

    theta = np.linspace(0, 2 * math.pi, 200)
    ax.plot(circle_radius_n * np.cos(theta), circle_radius_n * np.sin(theta),
            color=_INK, linewidth=1.2, linestyle="--", alpha=0.7,
            label="fixed-fin force, any clocking (radius = force magnitude)")

    ay, az = avg_vector_yz
    ax.annotate("", xy=(ay, az), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=_ACCENT, linewidth=2.2))
    ax.scatter([ay], [az], s=70, c=_ACCENT, zorder=5,
               label=f"1 s time-average at {spin_rad_s:.0f} rad/s "
                     f"({math.hypot(ay, az):.1f} N)")
    ax.scatter([0], [0], marker="+", s=90, c=_TEXT, linewidths=1.6, zorder=4)

    lim = circle_radius_n * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("transverse force, right, N", color=_INK)
    ax.set_ylabel("transverse force, up, N", color=_INK)
    ax.set_title("force direction, earth frame", color=_TEXT)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                        fontsize=7, facecolor=_BG, edgecolor=_GRID, frameon=True)
    for text in legend.get_texts():
        text.set_color(_TEXT)
    fig.tight_layout()
    return fig


def roll_decoupling_figure(t_s, phi_nose_deg, hold_command_nm: float,
                            brake_torque_nm: float, cant_torque_nm: float,
                            brake_max_nm: float) -> plt.Figure:
    """Card 2 -- roll decoupling: the nose settling toward the commanded
    brake torque's steady angle, and the two torques it balances against.

    Left: phi_nose(t), closed form (`NoseRollPlant.equilibrium_rate`/
    `.time_constant`), with the steady-state line marked. Right: the
    aerodynamic despin (cant) torque and the commanded brake torque as
    opposing bars, with the unclipped hold command and the brake's own
    capacity marked so it is visible whether this command could actually
    hold the nose still.
    """
    fig, (ax_t, ax_torque) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    fig.patch.set_alpha(0.0)
    _style_axes(ax_t, ax_torque)

    ax_t.plot(t_s, phi_nose_deg, color=_ACCENT, linewidth=1.6)
    ax_t.axhline(phi_nose_deg[-1], color=_INK, linewidth=1, linestyle=":",
                 label=f"asymptote {phi_nose_deg[-1]:.1f} deg")
    ax_t.set_xlabel("time since brake command, s", color=_INK)
    ax_t.set_ylabel("nose roll angle, deg", color=_INK)
    ax_t.set_title("nose angle settling", color=_TEXT)
    leg1 = ax_t.legend(loc="lower right", fontsize=7, facecolor=_BG,
                        edgecolor=_GRID, frameon=True)
    for text in leg1.get_texts():
        text.set_color(_TEXT)

    bars_y = [1, 0]
    bars_v = [cant_torque_nm, brake_torque_nm]
    colors = [_INK, _ACCENT]
    ax_torque.barh(bars_y, bars_v, color=colors, height=0.5)
    ax_torque.axvline(hold_command_nm, color=_AMBER, linewidth=1.4, linestyle="--",
                       label=f"hold command needed, {hold_command_nm:.2f} N·m")
    ax_torque.axvline(brake_max_nm, color=_TEXT, linewidth=1.2, linestyle=":",
                       label=f"brake capacity, {brake_max_nm:.2f} N·m")
    ax_torque.set_yticks(bars_y)
    ax_torque.set_yticklabels(["aerodynamic despin (cant)", "commanded brake"], color=_TEXT)
    ax_torque.set_xlabel("torque, N·m", color=_INK)
    ax_torque.set_title("what the brake is fighting", color=_TEXT)
    leg2 = ax_torque.legend(loc="lower right", fontsize=7, facecolor=_BG,
                             edgecolor=_GRID, frameon=True)
    for text in leg2.get_texts():
        text.set_color(_TEXT)

    fig.tight_layout()
    return fig


def nose_force_and_correction_figure(phi_deg: float, force_yz: tuple,
                                      correction_range_defl: tuple,
                                      stored_points: list) -> plt.Figure:
    """Card 3 -- one variable (phi), two axes: the nose-frame force it
    points, and the ground-plane (range, deflection) correction it produces.

    Left: the live canard force in the nose frame -- a single vector that
    simply rotates with phi, magnitude fixed. Right: the resulting
    ground-plane correction, live (continuous in phi) alongside the 12
    stored 30-degree-step points from `docs/authority_results.json` for
    scale and cross-check -- not claimed to be the same flight condition,
    only the same geometry.
    """
    fig, (ax_nose, ax_ground) = plt.subplots(1, 2, figsize=(9.5, 4.6))
    fig.patch.set_alpha(0.0)
    _style_axes(ax_nose, ax_ground)

    fy, fz = force_yz
    fmag = math.hypot(fy, fz)
    ax_nose.annotate("", xy=(fy, fz), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color=_ACCENT, linewidth=2.2))
    lim = max(fmag * 1.3, 1.0)
    ax_nose.set_xlim(-lim, lim)
    ax_nose.set_ylim(-lim, lim)
    ax_nose.set_aspect("equal", adjustable="box")
    ax_nose.scatter([0], [0], marker="+", s=90, c=_TEXT, linewidths=1.6)
    ax_nose.set_xlabel("nose-frame force, right, N", color=_INK)
    ax_nose.set_ylabel("nose-frame force, up, N", color=_INK)
    ax_nose.set_title(f"canard force, phi = {phi_deg:.0f} deg", color=_TEXT)

    if stored_points:
        sx = [p[1] for p in stored_points]
        sy = [p[2] for p in stored_points]
        ax_ground.scatter(sx, sy, s=24, c=_INK, alpha=0.6, zorder=2,
                           label="stored, 30 deg steps (docs/authority_results.json)")
    dr, dd = correction_range_defl
    ax_ground.annotate("", xy=(dr, dd), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=_ACCENT, linewidth=2.2))
    ax_ground.scatter([dr], [dd], s=60, c=_ACCENT, zorder=5,
                       label=f"live, closed form ({dr:+.1f}, {dd:+.1f}) m")
    ax_ground.scatter([0], [0], marker="+", s=90, c=_TEXT, linewidths=1.6, zorder=4)
    ax_ground.set_xlabel("range correction, m", color=_INK)
    ax_ground.set_ylabel("deflection correction, m", color=_INK)
    ax_ground.set_title("ground-plane correction", color=_TEXT)
    leg = ax_ground.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                            fontsize=7, facecolor=_BG, edgecolor=_GRID, frameon=True)
    for text in leg.get_texts():
        text.set_color(_TEXT)

    fig.tight_layout()
    return fig


def brake_power_figure(u_curve, power_curve, u_current: float,
                        power_current: float, brake_max_nm: float) -> plt.Figure:
    """Card 4 -- the brake is the alternator: dissipated power (torque x
    relative slip rate) as a function of commanded brake torque, the
    generated-power curve a slip-power alternator would see. No upstream
    function computes this; composed here from
    `NoseRollPlant.brake_torque`/`.equilibrium_rate`, the same P = T*omega
    pattern `analysis.authority._slip_power` uses for the bearing.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    fig.patch.set_alpha(0.0)
    _style_axes(ax)

    ax.plot(u_curve, power_curve, color=_INK, linewidth=1.6)
    ax.axvline(brake_max_nm, color=_TEXT, linewidth=1, linestyle=":",
               label=f"brake capacity, {brake_max_nm:.2f} N·m")
    ax.scatter([u_current], [power_current], s=80, c=_ACCENT, zorder=5,
               label=f"{power_current:.1f} W at {u_current:.2f} N·m")
    ax.set_xlabel("commanded brake torque, N·m", color=_INK)
    ax.set_ylabel("dissipated power, W", color=_INK)
    ax.set_title("brake torque vs generated power", color=_TEXT)
    legend = ax.legend(loc="best", fontsize=7, facecolor=_BG, edgecolor=_GRID, frameon=True)
    for text in legend.get_texts():
        text.set_color(_TEXT)
    fig.tight_layout()
    return fig


def authority_budget_figure(deploy_fractions, semi_axis_major_m,
                             current_fraction: float) -> plt.Figure:
    """Card 5 -- authority is a budget: achievable correction (the
    envelope's semi-major axis, `analysis.authority.envelope_stats`) against
    deployment time, at the 6 stored `deploy_fraction` points only -- no
    interpolation between points the campaign never flew.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    fig.patch.set_alpha(0.0)
    _style_axes(ax)

    ax.plot(deploy_fractions, semi_axis_major_m, color=_INK, linewidth=1.2,
            marker="o", markersize=5, zorder=2)
    if current_fraction in deploy_fractions:
        i = deploy_fractions.index(current_fraction)
        ax.scatter([deploy_fractions[i]], [semi_axis_major_m[i]], s=90,
                   c=_ACCENT, zorder=5,
                   label=f"{semi_axis_major_m[i]:.0f} m at {current_fraction:+.0%}")
        legend = ax.legend(loc="best", fontsize=8, facecolor=_BG,
                            edgecolor=_GRID, frameon=True)
        for text in legend.get_texts():
            text.set_color(_TEXT)
    ax.set_xlabel("deployment time, fraction of apogee offset", color=_INK)
    ax.set_ylabel("achievable correction, semi-major axis, m", color=_INK)
    ax.set_title("authority vs deployment time (stored points only)", color=_TEXT)
    fig.tight_layout()
    return fig


# ===========================================================================
# The Engine (Step 8, Part B6) -- the validation chain and model comparison.
# ===========================================================================
def firing_table_error_figure(rows: list) -> plt.Figure:
    """Link 2 of The Engine's chain: 6-DOF range error against each of the
    15 stored firing-table cases, with the RMS the page quotes marked."""
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.4))
    fig.patch.set_alpha(0.0)
    _style_axes(ax)

    labels = [f"c{r['charge']} QE{r['qe_mils']:.1f}" for r in rows]
    values = [r["range_err_pct"] for r in rows]
    y = list(range(len(rows)))[::-1]
    rms = math.sqrt(sum(v * v for v in values) / len(values))

    ax.barh(y, values, color=_INK, height=0.55, zorder=2)
    ax.axvline(rms, color=_ACCENT, linewidth=1.6, linestyle="--",
               label=f"RMS {rms:.2f} %")
    ax.axvline(-rms, color=_ACCENT, linewidth=1.6, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=_TEXT, fontsize=7)
    ax.set_xlabel("range error vs firing table, %", color=_INK)
    ax.set_title("6-DOF vs FT 155-AM-2, all 15 cases", color=_TEXT, fontsize=10)
    legend = ax.legend(loc="best", fontsize=8, facecolor=_BG, edgecolor=_GRID, frameon=True)
    for text in legend.get_texts():
        text.set_color(_TEXT)
    fig.tight_layout()
    return fig


def mpmm_divergence_bar_figure(points: dict) -> plt.Figure:
    """Link 3 of The Engine's chain: stored MPMM-vs-6-DOF peak position
    divergence, one bar per named engagement (`setter.validation_data.MpmmPoint`)."""
    items = [(e, p) for e, p in points.items() if p.available]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(items) + 1.4))
    fig.patch.set_alpha(0.0)
    _style_axes(ax)

    names = [e for e, _ in items]
    values = [p.max_divergence_m for _, p in items]
    y = list(range(len(items)))[::-1]
    ax.barh(y, values, color=_INK, height=0.55, zorder=2)
    for yi, (e, p) in zip(y, items):
        ax.annotate(f"{p.max_divergence_m:.1f} m @ t={p.t_of_max_divergence:.1f} s",
                    (p.max_divergence_m, yi), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=8, color=_TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=_TEXT)
    ax.set_xlabel("peak position divergence, m", color=_INK)
    ax.set_title("MPMM vs 6-DOF, from launch, peak divergence", color=_TEXT, fontsize=10)
    ax.set_xlim(0, max(values) * 1.6 if values else 1.0)
    fig.tight_layout()
    return fig


def model_comparison_figure(six_downrange_m, six_crossrange_m,
                             mpmm_downrange_m, mpmm_crossrange_m,
                             divergence_t_s, divergence_m,
                             engagement: str) -> plt.Figure:
    """The Engine's live model-comparison view: the same shot flown through
    both models, ground tracks overlaid (6-DOF is the accent-coloured
    reference; MPMM is the simplified model being checked against it), and
    their position divergence over time beneath.
    """
    fig, (ax_track, ax_div) = plt.subplots(2, 1, figsize=(7.5, 7.5))
    fig.patch.set_alpha(0.0)
    _style_axes(ax_track, ax_div)

    ax_track.plot(np.asarray(six_downrange_m) / 1000.0, six_crossrange_m,
                  color=_ACCENT, linewidth=1.6, label="6-DOF (reference)")
    ax_track.plot(np.asarray(mpmm_downrange_m) / 1000.0, mpmm_crossrange_m,
                  color=_INK, linewidth=1.4, linestyle="--", label="MPMM (flight model)")
    ax_track.set_xlabel("downrange, km", color=_INK)
    ax_track.set_ylabel("crossrange, m", color=_INK)
    ax_track.set_title(f"{engagement} -- ground track overlay", color=_TEXT)
    leg = ax_track.legend(loc="best", fontsize=8, facecolor=_BG, edgecolor=_GRID, frameon=True)
    for text in leg.get_texts():
        text.set_color(_TEXT)

    ax_div.plot(divergence_t_s, divergence_m, color=_AMBER, linewidth=1.4)
    ax_div.set_xlabel("time, s", color=_INK)
    ax_div.set_ylabel("position divergence, m", color=_INK)
    ax_div.set_title("MPMM minus 6-DOF, position difference", color=_TEXT)

    fig.tight_layout()
    return fig
