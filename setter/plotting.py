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

__all__ = ["ground_track_figure", "knowledge_term_figure"]


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
