"""
Unit tests for `setter.paired_age_analysis` -- the paired Task C met-age
comparison built on already-stored `docs/monte_carlo.json` per-round data.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from setter import paired_age_analysis as paa

pytestmark = pytest.mark.skipif(
    not paa.MONTE_CARLO_PATH.exists(),
    reason="docs/monte_carlo.json not present")


# ===========================================================================
# Row loading / pairing
# ===========================================================================
def test_load_rows_by_draw_unsupported_age_raises():
    with pytest.raises(KeyError):
        paa.load_rows_by_draw("4h")


def test_load_rows_by_draw_has_128_rounds_for_stored_ages():
    for age in ("3h", "6h", "none"):
        rows = paa.load_rows_by_draw(age)
        assert len(rows) == 128
        assert set(rows) == set(range(128))


def test_paired_arrays_uses_full_common_set():
    arr = paa.paired_arrays("6h", "none")
    assert arr["n"] == 128
    assert arr["n_a"] == 128 and arr["n_b"] == 128
    assert arr["draws"] == list(range(128))
    assert arr["miss_m_a"].shape == (128,)


def test_paired_arrays_matches_stored_cep_via_median():
    arr = paa.paired_arrays("6h", "none")
    assert float(np.median(arr["miss_m_a"])) == pytest.approx(126.9664616440906)
    assert float(np.median(arr["miss_m_b"])) == pytest.approx(118.22760347750562)


# ===========================================================================
# Bootstrap mechanics: joint resampling, determinism, point estimates
# ===========================================================================
def test_boot_paired_median_delta_point_estimate_matches_plain_medians():
    a = np.array([1.0, 2.0, 3.0, 100.0])
    b = np.array([0.0, 1.0, 2.0, 4.0])
    result = paa.boot_paired_median_delta(a, b, n_boot=500, seed=1)
    assert result["delta_m"] == pytest.approx(np.median(a) - np.median(b))
    assert result["n"] == 4


def test_boot_paired_median_delta_is_deterministic_for_same_seed():
    a = np.array([1.0, 5.0, 3.0, 8.0, 2.0])
    b = np.array([2.0, 4.0, 1.0, 9.0, 3.0])
    r1 = paa.boot_paired_median_delta(a, b, n_boot=1000, seed=42)
    r2 = paa.boot_paired_median_delta(a, b, n_boot=1000, seed=42)
    assert r1 == r2


def test_boot_paired_delta_zero_when_arms_identical():
    """Resampling is JOINT: if a and b are the same array, every bootstrap
    replicate compares a resampled set to itself, so the interval must
    collapse to exactly zero width -- this would NOT hold under independent
    resampling of the two arms."""
    a = np.array([1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0])
    r = paa.boot_paired_median_delta(a, a, n_boot=2000, seed=7)
    assert r["delta_m"] == 0.0
    assert r["lo_m"] == 0.0
    assert r["hi_m"] == 0.0
    assert r["se_m"] == 0.0


def test_boot_paired_sigma_delta_zero_when_arms_identical():
    a = np.array([1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0])
    r = paa.boot_paired_sigma_delta(a, a, n_boot=2000, seed=7)
    assert r["delta_m"] == 0.0
    assert r["lo_m"] == 0.0 and r["hi_m"] == 0.0


def test_boot_paired_mean_delta_matches_plain_means():
    a = np.array([10.0, 20.0, 30.0])
    b = np.array([5.0, 5.0, 5.0])
    r = paa.boot_paired_mean_delta(a, b, n_boot=500, seed=3)
    assert r["delta_m"] == pytest.approx(a.mean() - b.mean())


# ===========================================================================
# per_round_differences
# ===========================================================================
def test_per_round_differences_basic():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])
    d = paa.per_round_differences(a, b)
    assert d["mean_m"] == pytest.approx(2.5)
    assert d["median_m"] == pytest.approx(2.5)
    assert d["n"] == 4


# ===========================================================================
# Variance/bias decomposition categorization
# ===========================================================================
def test_decompose_variance_bias_dispersion_category():
    fake = {
        "range_axis": {
            "sigma_difference": {"delta_m": 40.0, "lo_m": 2.0, "hi_m": 75.0},
            "bias_difference": {"delta_m": 4.0, "lo_m": -29.0, "hi_m": 38.0},
        }
    }
    d = paa.decompose_variance_bias(fake)
    assert d["category"] == "dispersion"
    assert d["range_sigma_ci_excludes_zero"] is True
    assert d["range_bias_ci_excludes_zero"] is False


def test_decompose_variance_bias_insufficient_evidence_category():
    fake = {
        "range_axis": {
            "sigma_difference": {"delta_m": 1.0, "lo_m": -5.0, "hi_m": 6.0},
            "bias_difference": {"delta_m": 1.0, "lo_m": -5.0, "hi_m": 6.0},
        }
    }
    assert paa.decompose_variance_bias(fake)["category"] == "insufficient evidence to distinguish"


def test_decompose_variance_bias_both_category():
    fake = {
        "range_axis": {
            "sigma_difference": {"delta_m": 40.0, "lo_m": 2.0, "hi_m": 75.0},
            "bias_difference": {"delta_m": 20.0, "lo_m": 5.0, "hi_m": 35.0},
        }
    }
    assert paa.decompose_variance_bias(fake)["category"] == "both"


def test_decompose_variance_bias_bias_category():
    fake = {
        "range_axis": {
            "sigma_difference": {"delta_m": 1.0, "lo_m": -5.0, "hi_m": 6.0},
            "bias_difference": {"delta_m": 20.0, "lo_m": 5.0, "hi_m": 35.0},
        }
    }
    assert paa.decompose_variance_bias(fake)["category"] == "bias"


# ===========================================================================
# End-to-end compare_ages against the known, checked-in numbers
# ===========================================================================
def test_compare_ages_6h_vs_none_matches_known_figures():
    result = paa.compare_ages("6h", "none")
    assert result["n_common_rounds"] == 128
    cep = result["cep_difference"]
    assert cep["delta_m"] == pytest.approx(126.9664616440906 - 118.22760347750562)
    dec = result["variance_bias_decomposition"]
    assert dec["range_sigma_delta_m"] == pytest.approx(249.8540349209293 - 209.20432275415374, abs=0.01)
    assert dec["range_bias_delta_m"] == pytest.approx(7.888027905868341 - 3.4572341885646267, abs=0.01)


def test_compare_ages_6h_vs_none_cep_interval_includes_zero():
    """The headline finding of this analysis: the naive 6h/none inversion
    does not survive a properly paired bootstrap."""
    result = paa.compare_ages("6h", "none")
    interp = result["interpretation"]
    assert interp["cep_interval_excludes_zero"] is False


def test_compare_ages_3h_vs_6h_cep_interval_excludes_zero():
    result = paa.compare_ages("3h", "6h")
    interp = result["interpretation"]
    assert interp["cep_interval_excludes_zero"] is True
    assert result["cep_difference"]["delta_m"] < 0.0  # 3h has a smaller CEP than 6h


def test_compare_ages_unsupported_age_raises():
    with pytest.raises(KeyError):
        paa.compare_ages("6h", "4h")


# ===========================================================================
# Artifact generation
# ===========================================================================
def test_run_produces_expected_top_level_structure():
    result = paa.run()
    assert set(result) == {"provenance", "scope_note", "comparisons"}
    assert set(result["comparisons"]) == {"6h_vs_none", "3h_vs_6h"}
    prov = result["provenance"]
    assert prov["source_data_file"] == "docs/monte_carlo.json"
    assert prov["analysis_id"] == paa.ANALYSIS_ID
    assert "commit" in prov and "timestamp" in prov


def test_write_artifact_round_trips_as_json(tmp_path):
    out = tmp_path / "paired_age_comparison.json"
    result = paa.write_artifact(out)
    assert out.exists()
    with open(out) as fh:
        reloaded = json.load(fh)
    assert reloaded["comparisons"]["6h_vs_none"]["n_common_rounds"] == \
        result["comparisons"]["6h_vs_none"]["n_common_rounds"]


def test_docs_monte_carlo_json_is_not_modified_by_running_the_analysis():
    before = paa.MONTE_CARLO_PATH.read_bytes()
    paa.run()
    after = paa.MONTE_CARLO_PATH.read_bytes()
    assert before == after
