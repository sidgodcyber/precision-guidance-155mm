"""
Unit tests for `setter.schemas` -- the pydantic setter configuration message.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from setter.schemas import (EventConfiguration, MetProfileMessage, Position,
                             SetterMessage, SimulationConfig)
from sim.atmosphere import MetProfile


def _met_message() -> MetProfileMessage:
    return MetProfileMessage.from_met_profile(MetProfile.standard(), profile_id="std")


def _sim_config(scenario_id: str = "long") -> SimulationConfig:
    return SimulationConfig(scenario_id=scenario_id, charge=8, qe_mils=525.3,
                             muzzle_velocity_ms=684.0, deploy_time_s=5.55,
                             guided_phase_s=42.69)


def _valid_message(**overrides) -> dict:
    msg = SetterMessage(
        scenario_id="long",
        reference_position=Position(),
        target_position=Position(x_m=15839.5, y_m=330.9),
        met_profile=_met_message(),
        met_age_hours=2.0,
        simulation_config=_sim_config(),
        event_configuration=EventConfiguration(mode="time", parameters={"event_time_s": 30.0}),
    )
    data = msg.model_dump(by_alias=True)
    data.update(overrides)
    return data


# ===========================================================================
# Valid schema
# ===========================================================================
def test_valid_message_round_trips():
    data = _valid_message()
    msg = SetterMessage.model_validate(data)
    assert msg.scenario_id == "long"
    assert msg.schema_name == "SIM-SETTER-V1"


def test_met_profile_message_carries_full_profile_not_just_an_id():
    met = MetProfile.sample(__import__("numpy").random.default_rng(1), label="test")
    m = MetProfileMessage.from_met_profile(met, profile_id="p1")
    assert m.profile_id == "p1"
    assert len(m.wind_north_ms) == len(met.grid)
    assert m.grid_m == list(met.grid)


# ===========================================================================
# Malformed schema / missing required fields / invalid types
# ===========================================================================
def test_missing_required_field_rejected():
    data = _valid_message()
    del data["simulation_config"]
    with pytest.raises(ValidationError):
        SetterMessage.model_validate(data)


def test_wrong_type_rejected():
    data = _valid_message()
    data["met_age_hours"] = "two hours"
    with pytest.raises(ValidationError):
        SetterMessage.model_validate(data)


def test_unknown_field_rejected():
    data = _valid_message()
    data["extra_unspecified_field"] = 1
    with pytest.raises(ValidationError):
        SetterMessage.model_validate(data)


# ===========================================================================
# Range / vocabulary validation
# ===========================================================================
def test_unsupported_engagement_rejected():
    with pytest.raises(ValidationError):
        SimulationConfig(scenario_id="nonexistent", charge=8, qe_mils=1.0,
                          muzzle_velocity_ms=1.0, deploy_time_s=1.0, guided_phase_s=1.0)


def test_unsupported_scenario_id_rejected_on_message():
    data = _valid_message(scenario_id="nonexistent")
    with pytest.raises(ValidationError):
        SetterMessage.model_validate(data)


def test_unknown_event_mode_rejected():
    with pytest.raises(ValidationError):
        EventConfiguration(mode="detonate", parameters={})


def test_event_configuration_stays_generic_and_type_checked():
    cfg = EventConfiguration(mode="motion", parameters={"threshold": 50.0})
    assert cfg.mode == "motion"
    assert cfg.parameters["threshold"] == 50.0
