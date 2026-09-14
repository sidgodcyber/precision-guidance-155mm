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
from setter.campaign import KnowledgeTermPoint

__all__ = ["ground_track_figure", "knowledge_term_figure", "cep_circle_figure"]

#: ISA-101 colour discipline (see CLAUDE.md / Control Room spec Part A2):
#: a neutral base, with colour reserved for states that need attention.
#: `_ACCENT` marks the single most important number on a given screen --
#: here, the predicted CEP itself. Scatter points and the target are
#: neutral; amber is reserved for an unresolved/unavailable condition
#: elsewhere in the UI, never used decoratively in this figure.
_BG = "#111827"
_GRID = "#374151"
_INK = "#9aa5b1"
_TEXT = "#e5e7eb"
_ACCENT = "#38bdf8"


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
    fig, ax = plt.subplots(figsize=(6, 4))
    ages = [p.age for p in points]
    ceps = [p.cep_m if p.available else float("nan") for p in points]
    colors = ["#c53030" if p.age == highlight_age else "#2b6cb0" for p in points]

    ax.plot(ages, ceps, color="#2b6cb0", linewidth=1, zorder=1)
    ax.scatter(ages, ceps, c=colors, zorder=2)
    for p, cep in zip(points, ceps):
        if not p.available:
            continue
        ax.annotate(f"{cep:.1f} m", (p.age, cep), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)

    ax.set_xlabel("met message age")
    ax.set_ylabel("CEP, m (truth-fed, navigation excluded)")
    ax.set_title("atmospheric knowledge term -- Task C")
    ax.grid(alpha=0.3)
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
