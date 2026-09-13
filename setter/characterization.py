"""
Phase 2 characterization: builds the demonstration artifacts under
`docs/setter_validation/` from data and code that already exist -- this
module runs no new Monte Carlo campaign and changes no engine behavior. It
only reads `docs/monte_carlo.json` through `setter.campaign`, benchmarks the
already-existing single-trajectory path through `setter.simulation_adapter`,
and exercises message construction/validation through `setter.schemas` /
`setter.message_codec` / `setter.validation`.

Run:  python -m setter.characterization
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Optional

from setter import campaign, message_codec, simulation_adapter as sa
from setter.config import MET_AGE_BUCKETS, REPO_ROOT, SUPPORTED_ENGAGEMENTS, TASK_A_TAGS
from setter.schemas import EventConfiguration, MetProfileMessage, Position, SetterMessage, SimulationConfig
from setter.validation import SetterValidationError, validate_message_dict

OUT_DIR = REPO_ROOT / "docs" / "setter_validation"

__all__ = [
    "provenance", "campaign_summary", "runtime_summary", "validation_summary", "write_all",
]


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def provenance(**extra) -> dict:
    p = {
        "commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_data_file": "docs/monte_carlo.json",
    }
    p.update(extra)
    return p


# ===========================================================================
# Section 3/4/5 -- campaign-data, engagement, and met-age characterization
# ===========================================================================
def _point_dict(p) -> dict:
    d = {"available": p.available}
    if p.available:
        d["cep_m"] = p.cep_m
        d["n"] = p.n
        if getattr(p, "cep_se_m", None) is not None:
            d["cep_se_m"] = p.cep_se_m
        if getattr(p, "met_age_hours", None) is not None:
            d["met_age_hours"] = p.met_age_hours
        if getattr(p, "cep_contribution_m", None) is not None:
            d["cep_contribution_m"] = p.cep_contribution_m
    else:
        d["reason"] = p.reason
    return d


def campaign_summary() -> dict:
    """Task A (navigation-in-loop) and Task C (truth-fed, atmospheric
    knowledge) kept strictly separate, with every engagement/tag and
    every met-age bucket enumerated -- unsupported combinations included
    explicitly as `available: false`, never omitted or fabricated."""
    task_a = {
        tag: {e: _point_dict(campaign.task_a_point(e, tag)) for e in SUPPORTED_ENGAGEMENTS}
        for tag in TASK_A_TAGS
    }
    headline_cep = {e: task_a["headline"][e]["cep_m"] for e in SUPPORTED_ENGAGEMENTS}
    task_c = {
        tag: {age: _point_dict(campaign.task_c_point(age, tag)) for age in MET_AGE_BUCKETS}
        for tag in ("headline", "physical")
    }
    return {
        "provenance": provenance(),
        "task_a": {
            "description": "navigation-in-loop campaign result",
            "points": task_a,
            "anomaly": {
                "description": (
                    "2h headline CEP does not order monotonically with engagement "
                    "range -- reported as measured, not smoothed, fitted, or explained."),
                "headline_2h_cep_m_by_engagement": headline_cep,
            },
        },
        "task_c": {
            "description": "truth-fed atmospheric-knowledge term, navigation excluded",
            "points": task_c,
        },
    }


# ===========================================================================
# Section 6 -- live simulation (single-trajectory) characterization
# ===========================================================================
def runtime_summary(n_samples: int = 5) -> dict:
    """Wall-clock timing of the setter's live, non-campaign operations:
    the single reduced-order trajectory per engagement, and message
    validation/serialization. `n_samples` is a small deterministic
    benchmark repeat count, not a Monte Carlo sample size."""
    met = sa.met_profile_for_age("2h", seed=1)
    trajectories = {}
    for engagement in SUPPORTED_ENGAGEMENTS:
        durations = []
        sample = None
        for _ in range(n_samples):
            t0 = time.perf_counter()
            sample = sa.single_trajectory(engagement, met=met)
            durations.append(time.perf_counter() - t0)
        trajectories[engagement] = {
            "median_s": median(durations),
            "max_s": max(durations),
            "min_s": min(durations),
            "n_samples": n_samples,
            "n_logged_points": int(sample.t.size),
            "range_m": sample.range_m,
            "duration_s": sample.duration_s,
        }

    # engagement_context cache characterization: first call for a cold label
    # vs a repeated (cached) call.
    sa.engagement_context.clear()
    t0 = time.perf_counter()
    sa.engagement_context("mid2")
    cold_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    sa.engagement_context("mid2")
    warm_s = time.perf_counter() - t0

    msg = _representative_message("long", "2h")
    t0 = time.perf_counter()
    for _ in range(n_samples):
        blob, _ = message_codec.encode(msg)
    encode_s = (time.perf_counter() - t0) / n_samples
    t0 = time.perf_counter()
    for _ in range(n_samples):
        validate_message_dict(json.loads(blob))
    validate_s = (time.perf_counter() - t0) / n_samples

    return {
        "provenance": provenance(),
        "single_trajectory": trajectories,
        "engagement_context_cache": {"cold_s": cold_s, "warm_s": warm_s},
        "message_serialization": {"encode_s_per_call": encode_s, "message_size_bytes": len(blob)},
        "message_validation": {"validate_s_per_call": validate_s},
    }


# ===========================================================================
# Section 9 -- configuration/message characterization per engagement
# ===========================================================================
def _representative_message(engagement: str, age_bucket: str, seed: int = 17) -> SetterMessage:
    base = sa.baseline_for(engagement)
    met = sa.met_profile_for_age(age_bucket, seed=seed)
    lay = sa.lay_gun(engagement, met)
    deploy = sa.fuze_setting(engagement, met, lay["dqe_mils"])
    return SetterMessage(
        scenario_id=engagement,
        reference_position=Position(),
        target_position=Position(x_m=base["uncorrected_range"], y_m=base["uncorrected_drift"]),
        met_profile=MetProfileMessage.from_met_profile(met, profile_id=f"{engagement}-{age_bucket}"),
        met_age_hours=sa.age_bucket_to_hours(age_bucket),
        simulation_config=SimulationConfig(
            scenario_id=engagement, charge=base["charge"], qe_mils=base["qe_mils"],
            muzzle_velocity_ms=base["muzzle_velocity"], dqe_mils=lay["dqe_mils"],
            daz_mils=lay["daz"], deploy_time_s=deploy, guided_phase_s=base["guided_phase_s"]),
        event_configuration=EventConfiguration(mode="time", parameters={"event_time_s": 30.0}),
    )


def validation_summary() -> dict:
    per_engagement = {}
    for engagement in SUPPORTED_ENGAGEMENTS:
        msg = _representative_message(engagement, "2h")
        blob, checksum = message_codec.encode(msg)
        round_trip_ok = message_codec.decode(blob).scenario_id == engagement
        checksum_ok = message_codec.verify_checksum(blob)
        met_fields_ok = len(msg.met_profile.grid_m) == len(msg.met_profile.wind_north_ms) \
            == len(msg.met_profile.wind_east_ms) == len(msg.met_profile.density_ratio) \
            == len(msg.met_profile.temperature_ratio) > 0
        try:
            validate_message_dict(json.loads(blob))
            valid = True
        except SetterValidationError:
            valid = False
        per_engagement[engagement] = {
            "validation_status": "valid" if valid else "invalid",
            "message_size_bytes": len(blob),
            "checksum": checksum,
            "round_trip_ok": round_trip_ok,
            "checksum_verified": checksum_ok,
            "met_profile_fields_complete": met_fields_ok,
            "met_profile_grid_points": len(msg.met_profile.grid_m),
        }

    # Field-change -> checksum-change sensitivity.
    baseline = _representative_message("long", "2h")
    changed = baseline.model_copy(
        update={"simulation_config": baseline.simulation_config.model_copy(
            update={"qe_mils": baseline.simulation_config.qe_mils + 1.0})})
    sensitivity = {
        "single_field_change_changes_checksum":
            message_codec.compute_checksum(baseline) != message_codec.compute_checksum(changed),
    }

    # Malformed/corrupted message rejection.
    blob, _ = message_codec.encode(baseline)
    data = json.loads(blob)
    tampered = dict(data)
    tampered["checksum"] = "00000000"
    tampered_blob = json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    missing_field = {k: v for k, v in data.items() if k != "simulation_config"}
    rejection_checks = {
        "tampered_checksum_rejected": not message_codec.verify_checksum(tampered_blob),
        "missing_required_field_rejected": _rejects(missing_field),
        "unsupported_engagement_rejected": _rejects({**data, "scenario_id": "nonexistent"}),
        "malformed_json_rejected": not message_codec.verify_checksum(b"{not json"),
    }

    return {
        "provenance": provenance(),
        "per_engagement": per_engagement,
        "sensitivity": sensitivity,
        "rejection_checks": rejection_checks,
    }


def _rejects(raw: dict) -> bool:
    try:
        validate_message_dict(raw)
        return False
    except SetterValidationError:
        return True


# ===========================================================================
# Entry point
# ===========================================================================
def write_all(out_dir: Path = OUT_DIR, n_runtime_samples: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "campaign_summary.json": campaign_summary(),
        "runtime_summary.json": runtime_summary(n_runtime_samples),
        "validation_summary.json": validation_summary(),
    }
    for name, content in artifacts.items():
        with open(out_dir / name, "w") as fh:
            json.dump(content, fh, indent=1)


if __name__ == "__main__":
    write_all()
