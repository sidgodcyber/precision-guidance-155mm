"""
Read-only access to `docs/validation_results.json` (produced by the
root-level `run_validation.py`, step 1 -- NOT `setter/registry.py`) and
`docs/mpmm_results.json` (produced by `analysis.mpmm_compare`, step 2) --
the validation-chain artifacts The Engine reads.

Kept separate from `setter/campaign.py`, whose own docstring scopes it to
`docs/monte_carlo.json` only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from setter.config import MPMM_RESULTS_PATH, SUPPORTED_ENGAGEMENTS, VALIDATION_RESULTS_PATH

__all__ = [
    "ENGAGEMENT_TO_FIRING_TABLE_INDEX",
    "firing_table_rows", "firing_table_range_rms_pct", "rung5b",
    "detail_run_history", "MpmmPoint", "mpmm_point", "mpmm_all_points",
    "model_error_summary",
]

#: Named engagement -> index into BOTH docs/validation_results.json's
#: "firing_table" list and docs/mpmm_results.json's flat list -- both are in
#: analysis.mpmm_compare.FIRING_TABLE order. Not stored anywhere as an
#: explicit mapping; confirmed here by matching (charge, qe_mils) against
#: FIRING_TABLE for all five engagements: index 0 = charge 4/QE 97.2
#: ("short"), 2 = charge 4/QE 211.6 ("short2"), 7 = charge 6/QE 378.6
#: ("middle"), 11 = charge 7/QE 520.7 ("mid2"), 14 = charge 8/QE 525.3
#: ("long").
ENGAGEMENT_TO_FIRING_TABLE_INDEX = {
    "short": 0, "short2": 2, "middle": 7, "mid2": 11, "long": 14,
}


@lru_cache(maxsize=1)
def _validation() -> dict:
    with open(VALIDATION_RESULTS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _mpmm() -> list:
    with open(MPMM_RESULTS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def firing_table_rows() -> list:
    """The 15 stored firing-table validation cases verbatim -- each already
    carries its own `range_err_pct`/`tof_err_pct`/`drift_err_pct`/
    `vimp_err_pct`/`maxord_err_pct` (computed by `run_validation.py`, not
    here)."""
    return list(_validation()["firing_table"])


def firing_table_range_rms_pct() -> float:
    """sqrt(mean(range_err_pct**2)) over all 15 stored rows -- display-time
    arithmetic on already-stored per-case data (the same discipline
    `setter.campaign.error_budget` uses), not a new simulation. Matches
    CLAUDE.md's headline "range RMS 0.48%"."""
    rows = firing_table_rows()
    return math.sqrt(sum(r["range_err_pct"] ** 2 for r in rows) / len(rows))


def rung5b() -> dict:
    """`docs/validation_results.json["rung5b"]` verbatim: one paired
    model-vs-spec case (max-range/max-charge), not an error-percent table
    like `firing_table`."""
    return dict(_validation()["rung5b"])


def detail_run_history() -> dict:
    """`docs/validation_results.json["detail_run"]["history"]` -- one
    charge-8/QE-525.3 case's full time history (~400 samples), for an
    optional trace plot."""
    return dict(_validation()["detail_run"]["history"])


@dataclass(frozen=True)
class MpmmPoint:
    """One engagement's stored MPMM-vs-6-DOF divergence, or an explicit
    report that none is stored."""

    engagement: str
    available: bool
    max_divergence_m: Optional[float] = None
    t_of_max_divergence: Optional[float] = None
    ap_d_range_m: Optional[float] = None
    ap_d_drift_m: Optional[float] = None
    reason: Optional[str] = None


def mpmm_point(engagement: str) -> MpmmPoint:
    """The stored MPMM-vs-6-DOF comparison for one named engagement, read
    from `docs/mpmm_results.json` by way of `ENGAGEMENT_TO_FIRING_TABLE_INDEX`."""
    if engagement not in SUPPORTED_ENGAGEMENTS:
        raise ValueError(f"unsupported engagement {engagement!r}; supported: {SUPPORTED_ENGAGEMENTS}")
    idx = ENGAGEMENT_TO_FIRING_TABLE_INDEX[engagement]
    rows = _mpmm()
    if idx >= len(rows):
        return MpmmPoint(engagement=engagement, available=False,
                          reason=f"docs/mpmm_results.json has no row at index {idx}")
    r = rows[idx]
    return MpmmPoint(
        engagement=engagement, available=True,
        max_divergence_m=r["max_divergence_m"], t_of_max_divergence=r["t_of_max_divergence"],
        ap_d_range_m=r["ap_d_range"], ap_d_drift_m=r["ap_d_drift"])


def mpmm_all_points() -> dict:
    """`mpmm_point` for all five named engagements, keyed by engagement."""
    return {e: mpmm_point(e) for e in SUPPORTED_ENGAGEMENTS}


def model_error_summary() -> dict:
    """Mean / sample-1sigma / RMS(bias(+)sigma) of `ap_it_d_range` and
    `ap_it_d_drift` across all 15 stored engagements -- the apogee-
    initialised, `iterate_yaw=True` configuration `docs/MODEL-ERROR.md`
    recommends. Reproduces that document's own numbers from the same
    stored per-case data (not by parsing the markdown)."""
    rows = _mpmm()
    n = len(rows)

    def _stats(key: str) -> dict:
        vals = [r[key] for r in rows]
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        sigma = math.sqrt(var)
        rms = math.sqrt(sum(v * v for v in vals) / n)
        return {"mean_m": mean, "sigma_m": sigma, "rms_m": rms, "n": n}

    return {
        "range": _stats("ap_it_d_range"),
        "deflection": _stats("ap_it_d_drift"),
        "recommended_budget_m": {"range_1sigma": 0.7, "deflection_1sigma": 0.1},
        "source": "docs/mpmm_results.json (ap_it_d_range/ap_it_d_drift, all 15 "
                   "engagements); rounded recommendation from docs/MODEL-ERROR.md",
    }
