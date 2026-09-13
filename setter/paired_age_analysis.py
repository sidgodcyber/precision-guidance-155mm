"""
Paired statistical analysis of already-stored Task C per-round data.

Phase 2 characterization flagged a secondary, previously-unremarked
non-monotonicity in the Task C headline curve: the "none" (no met message)
age has a SMALLER CEP than "6h" (a 6-hour-old message) --
`docs/setter_validation/campaign_summary.json`. The two ages' INDEPENDENTLY
reported bootstrap intervals overlap substantially:

    6h   CEP 126.97 m  [95.07, 170.44]
    none CEP 118.23 m  [ 87.44, 149.54]

Those intervals are computed by `analysis.monte_carlo.bootstrap_cep`
resampling each age's 128 rows INDEPENDENTLY, which throws away the fact
that "6h" and "none" are the SAME 128 underlying rounds -- Task C's design
draws one atmosphere/gun/mass/sensor realization per round index and reuses
it across every met-message age, varying only how stale the message given
to fire control is (see `sim.atmosphere.MetProfile.message` and
`analysis/compare_c_tags.py`, which relies on the identical guarantee to
pair two TAGS at one age). `docs/monte_carlo.json`'s
`c.headline.long.rows[age]` carries that per-round data keyed by `draw`, and
this module follows `analysis/compare_c_tags.py`'s existing pattern --
matching rows by `draw`, then bootstrapping by resampling round indices
JOINTLY into both arms -- but applied across AGES within one tag instead of
across TAGS at one age.

Read-only against `docs/monte_carlo.json`: this module runs no new
campaign, does not call into any frozen module, and writes only to
`docs/setter_validation/paired_age_comparison.json`.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from setter.config import MONTE_CARLO_PATH, REPO_ROOT

ANALYSIS_ID = "paired-age-comparison"
ANALYSIS_VERSION = "1.0.0"
OUT_PATH = REPO_ROOT / "docs" / "setter_validation" / "paired_age_comparison.json"

#: Matches `analysis/compare_c_tags.py`'s BOOT/SEED -- same bootstrap
#: replicate count and seed, for consistency with the existing pattern this
#: module follows.
N_BOOT = 20000
SEED = 606

TAG = "headline"
ENGAGEMENT = "long"

__all__ = [
    "load_rows_by_draw", "paired_arrays", "boot_paired_median_delta",
    "boot_paired_mean_delta", "boot_paired_sigma_delta", "per_round_differences",
    "compare_ages", "decompose_variance_bias", "run", "write_artifact",
]


def _load() -> dict:
    with open(MONTE_CARLO_PATH) as fh:
        return json.load(fh)


def load_rows_by_draw(age: str, tag: str = TAG, engagement: str = ENGAGEMENT) -> dict:
    """`{draw_index: row_dict}` for Task C's stored per-round rows at one
    met-message age. Raises KeyError with the available ages if `age` was
    never stored -- no fabrication of a missing age."""
    campaign = _load()["c"][tag][engagement]
    rows = campaign["rows"]
    if age not in rows:
        raise KeyError(f"Task C {tag!r}/{engagement!r} has no stored rows for age {age!r}; "
                        f"available: {sorted(rows)}")
    return {r["draw"]: r for r in rows[age]}


def paired_arrays(age_a: str, age_b: str, tag: str = TAG, engagement: str = ENGAGEMENT) -> dict:
    """Per-round arrays for the draws common to both ages, aligned by draw
    index (sorted ascending, so both arms use the identical round order)."""
    a = load_rows_by_draw(age_a, tag, engagement)
    b = load_rows_by_draw(age_b, tag, engagement)
    common = sorted(set(a) & set(b))
    return {
        "draws": common,
        "n": len(common),
        "n_a": len(a),
        "n_b": len(b),
        "miss_m_a": np.array([a[i]["miss_m"] for i in common], dtype=float),
        "miss_m_b": np.array([b[i]["miss_m"] for i in common], dtype=float),
        "range_a": np.array([a[i]["miss_range_m"] for i in common], dtype=float),
        "range_b": np.array([b[i]["miss_range_m"] for i in common], dtype=float),
        "defl_a": np.array([a[i]["miss_defl_m"] for i in common], dtype=float),
        "defl_b": np.array([b[i]["miss_defl_m"] for i in common], dtype=float),
    }


def _p_two_sided(d: np.ndarray) -> float:
    return float(min((d >= 0).mean(), (d <= 0).mean()) * 2)


def boot_paired_median_delta(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
                              seed: int = SEED) -> dict:
    """CI on median(a) - median(b) ("the difference in CEP"), resampling
    round indices JOINTLY into both arms every replicate -- the same
    mechanic as `compare_c_tags.boot_paired_delta`, with the statistic
    swapped from std to median."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    d = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return {
        "delta_m": float(np.median(a) - np.median(b)),
        "lo_m": float(np.percentile(d, 2.5)), "hi_m": float(np.percentile(d, 97.5)),
        "se_m": float(d.std(ddof=1)), "p_two_sided_ge_0": _p_two_sided(d),
        "n": int(a.size), "n_boot": n_boot, "seed": seed,
    }


def boot_paired_mean_delta(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
                            seed: int = SEED) -> dict:
    """CI on mean(a) - mean(b) (the paired BIAS difference), jointly resampled."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {
        "delta_m": float(a.mean() - b.mean()),
        "lo_m": float(np.percentile(d, 2.5)), "hi_m": float(np.percentile(d, 97.5)),
        "se_m": float(d.std(ddof=1)), "p_two_sided_ge_0": _p_two_sided(d),
        "n": int(a.size), "n_boot": n_boot, "seed": seed,
    }


def boot_paired_sigma_delta(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
                             seed: int = SEED) -> dict:
    """CI on std(a) - std(b) (the paired DISPERSION difference), jointly
    resampled -- identical in form to
    `analysis.compare_c_tags.boot_paired_delta`."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    d = a[idx].std(axis=1, ddof=1) - b[idx].std(axis=1, ddof=1)
    return {
        "delta_m": float(a.std(ddof=1) - b.std(ddof=1)),
        "lo_m": float(np.percentile(d, 2.5)), "hi_m": float(np.percentile(d, 97.5)),
        "se_m": float(d.std(ddof=1)), "p_two_sided_ge_0": _p_two_sided(d),
        "n": int(a.size), "n_boot": n_boot, "seed": seed,
    }


def per_round_differences(a: np.ndarray, b: np.ndarray) -> dict:
    """Raw paired per-round differences D_i = a_i - b_i, summarised --
    `analysis.compare_c_tags.report`'s "DIFFERENCE OF DIFFERENCES" block,
    without the "of differences" (there is no third "perfect" reference arm
    in this comparison: `a`/`b` are the two ages' own outcomes, already
    paired by round)."""
    D = np.asarray(a, float) - np.asarray(b, float)
    return {
        "mean_m": float(D.mean()), "sd_m": float(D.std(ddof=1)),
        "median_m": float(np.median(D)),
        "abs_p90_m": float(np.percentile(np.abs(D), 90)),
        "n": int(D.size),
    }


def compare_ages(age_a: str, age_b: str) -> dict:
    """The full paired comparison between two Task C met-message ages at
    the stored engagement/tag: per-axis paired differences, the paired CEP
    (median) difference with its jointly-resampled bootstrap interval, and
    the same for the range/deflection paired means and standard deviations
    (feeding `decompose_variance_bias`)."""
    arr = paired_arrays(age_a, age_b)
    result = {
        "comparison": f"{age_a} vs {age_b}",
        "age_a": age_a, "age_b": age_b,
        "n_common_rounds": arr["n"], "n_a_total": arr["n_a"], "n_b_total": arr["n_b"],
        "cep_difference": boot_paired_median_delta(arr["miss_m_a"], arr["miss_m_b"]),
        "per_round_differences": {
            "radial_miss": per_round_differences(arr["miss_m_a"], arr["miss_m_b"]),
            "range": per_round_differences(arr["range_a"], arr["range_b"]),
            "deflection": per_round_differences(arr["defl_a"], arr["defl_b"]),
        },
        "range_axis": {
            "bias_difference": boot_paired_mean_delta(arr["range_a"], arr["range_b"]),
            "sigma_difference": boot_paired_sigma_delta(arr["range_a"], arr["range_b"]),
        },
        "deflection_axis": {
            "bias_difference": boot_paired_mean_delta(arr["defl_a"], arr["defl_b"]),
            "sigma_difference": boot_paired_sigma_delta(arr["defl_a"], arr["defl_b"]),
        },
    }
    result["variance_bias_decomposition"] = decompose_variance_bias(result)
    result["interpretation"] = _interpret(result)
    return result


def decompose_variance_bias(comparison: dict) -> dict:
    """Categorizes the RANGE-axis paired difference as dispersion, bias,
    both, or undetermined -- a comparison of two summary MOMENTS (mean,
    std), not a decomposition of the CEP shift itself (CEP is the median of
    a nonlinear combination of both axes, and no exact split of a median
    shift into "bias part" and "spread part" is attempted here). This is
    the OBSERVED comparison; `interpretation`/the STOP report keeps it
    separate from any proposed real-world mechanism, per instruction.
    """
    sigma = comparison["range_axis"]["sigma_difference"]
    bias = comparison["range_axis"]["bias_difference"]
    sigma_excludes_zero = sigma["lo_m"] > 0.0 or sigma["hi_m"] < 0.0
    bias_excludes_zero = bias["lo_m"] > 0.0 or bias["hi_m"] < 0.0
    if sigma_excludes_zero and bias_excludes_zero:
        category = "both"
    elif sigma_excludes_zero:
        category = "dispersion"
    elif bias_excludes_zero:
        category = "bias"
    else:
        category = "insufficient evidence to distinguish"
    return {
        "range_sigma_ci_excludes_zero": sigma_excludes_zero,
        "range_bias_ci_excludes_zero": bias_excludes_zero,
        "range_sigma_delta_m": sigma["delta_m"],
        "range_bias_delta_m": bias["delta_m"],
        "category": category,
    }


def _interpret(result: dict) -> dict:
    cep = result["cep_difference"]
    excludes_zero = cep["lo_m"] > 0.0 or cep["hi_m"] < 0.0
    return {
        "cep_interval_excludes_zero": excludes_zero,
        "statement": (
            f"the paired CEP difference [{cep['lo_m']:+.2f}, {cep['hi_m']:+.2f}] m "
            + ("excludes zero: the stored simulation experiment establishes a "
               f"non-zero paired difference over these {cep['n']} shared rounds "
               f"at the '{result['age_a']}'/'{result['age_b']}' met-message ages, "
               "'long' engagement only."
               if excludes_zero else
               "includes zero: the available paired sample does not establish "
               f"a real difference between '{result['age_a']}' and "
               f"'{result['age_b']}' at this engagement.")
        ),
    }


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def run() -> dict:
    return {
        "provenance": {
            "commit": _git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_data_file": "docs/monte_carlo.json",
            "analysis_id": ANALYSIS_ID,
            "analysis_version": ANALYSIS_VERSION,
            "tag": TAG, "engagement": ENGAGEMENT,
            "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "method": "percentile, jointly resampled round indices"},
        },
        "scope_note": (
            "Offline analysis of already-stored simulation data; docs/monte_carlo.json "
            "was not modified and no campaign was re-run. Simulation-only: this "
            "characterizes the stored experiment at the 'long' engagement across "
            "met-message ages, and does not generalize to other engagements, ages "
            "not represented in the data, or real-world munition performance."),
        "comparisons": {
            "6h_vs_none": compare_ages("6h", "none"),
            "3h_vs_6h": compare_ages("3h", "6h"),
        },
    }


def write_artifact(out_path: Path = OUT_PATH) -> dict:
    result = run()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1)
    return result


if __name__ == "__main__":
    write_artifact()
