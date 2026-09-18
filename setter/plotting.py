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

from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fuze.trajectory_adapter import TrajectorySample
from setter.campaign import BudgetTerm, KnowledgeTermPoint
from setter.palette import ACCENT as _ACCENT
from setter.palette import BG as _BG
from setter.palette import GRID as _GRID
from setter.palette import INK as _INK
from setter.palette import TEXT as _TEXT

__all__ = ["ground_track_figure", "knowledge_term_figure", "cep_circle_figure",
           "fire_result_figure", "flight_deck_figure", "error_budget_figure"]

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
