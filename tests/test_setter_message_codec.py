"""
Unit tests for `setter.message_codec` -- canonical JSON encoding and CRC32
checksumming of `setter.schemas.SetterMessage`.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import json
import zlib

import pytest

from setter import message_codec
from setter.schemas import EventConfiguration, MetProfileMessage, Position, SetterMessage, SimulationConfig
from sim.atmosphere import MetProfile


def _message() -> SetterMessage:
    return SetterMessage(
        scenario_id="long",
        reference_position=Position(),
        target_position=Position(x_m=15839.5, y_m=330.9),
        met_profile=MetProfileMessage.from_met_profile(MetProfile.standard(), profile_id="std"),
        met_age_hours=None,
        simulation_config=SimulationConfig(
            scenario_id="long", charge=8, qe_mils=525.3, muzzle_velocity_ms=684.0,
            deploy_time_s=5.55, guided_phase_s=42.69),
        event_configuration=EventConfiguration(mode="time", parameters={"event_time_s": 30.0}),
    )


# ===========================================================================
# Deterministic serialization
# ===========================================================================
def test_canonical_body_is_deterministic():
    msg = _message()
    assert message_codec.canonical_body(msg) == message_codec.canonical_body(msg)


def test_canonical_body_is_compact_json():
    """Compact separators and sorted keys: re-dumping the parsed body the
    same way reproduces it byte-for-byte."""
    body = message_codec.canonical_body(_message())
    reparsed = json.loads(body)
    redumped = json.dumps(reparsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert body == redumped


def test_canonical_body_excludes_checksum_field():
    msg = _message().model_copy(update={"checksum": "deadbeef"})
    body = json.loads(message_codec.canonical_body(msg))
    assert "checksum" not in body


# ===========================================================================
# CRC32 calculation
# ===========================================================================
def test_compute_checksum_matches_zlib_crc32():
    msg = _message()
    expected = format(zlib.crc32(message_codec.canonical_body(msg)), "08x")
    assert message_codec.compute_checksum(msg) == expected


def test_checksum_changes_when_body_changes():
    msg_a = _message()
    msg_b = _message().model_copy(update={"scenario_id": "short",
                                           "simulation_config": SimulationConfig(
                                               scenario_id="short", charge=6, qe_mils=100.0,
                                               muzzle_velocity_ms=500.0, deploy_time_s=1.0,
                                               guided_phase_s=1.0)})
    assert message_codec.compute_checksum(msg_a) != message_codec.compute_checksum(msg_b)


# ===========================================================================
# JSON round trip / checksum verification / message size
# ===========================================================================
def test_encode_decode_round_trip_preserves_content():
    msg = _message()
    blob, checksum = message_codec.encode(msg)
    decoded = message_codec.decode(blob)
    assert decoded.scenario_id == msg.scenario_id
    assert decoded.checksum == checksum


def test_verify_checksum_true_for_untampered_message():
    blob, _ = message_codec.encode(_message())
    assert message_codec.verify_checksum(blob) is True


def test_verify_checksum_false_for_tampered_message():
    blob, _ = message_codec.encode(_message())
    data = json.loads(blob)
    data["simulation_config"]["qe_mils"] = 999.0
    tampered = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert message_codec.verify_checksum(tampered) is False


def test_verify_checksum_false_for_missing_checksum_field():
    blob, _ = message_codec.encode(_message())
    data = json.loads(blob)
    del data["checksum"]
    stripped = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert message_codec.verify_checksum(stripped) is False


def test_verify_checksum_false_for_malformed_json():
    assert message_codec.verify_checksum(b"{not valid json") is False


def test_message_size_is_reported_in_bytes():
    blob, _ = message_codec.encode(_message())
    assert isinstance(len(blob), int)
    assert len(blob) > 0


def test_schema_id_and_version_present_in_encoded_message():
    blob, _ = message_codec.encode(_message())
    data = json.loads(blob)
    assert data["schema"] == "SIM-SETTER-V1"
    assert data["schema_version"] == "1.0.0"
