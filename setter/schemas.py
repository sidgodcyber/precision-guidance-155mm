"""
Pydantic schema for the setter's serialized simulation configuration message.

This is a SIMULATION configuration message ("SIM-SETTER-V1"): it carries
engagement selection, a meteorological simulation state
(`sim.atmosphere.MetProfile`), and generic event-engine configuration
consumed by `fuze/`. It is not a real-world hardware or weapon-control
protocol, and `event_configuration` here is deliberately generic (mode +
parameters) -- the structured, range-checked engine configuration lives in
`fuze.config.FuzeEngineConfig` and is validated separately.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from setter.config import SCHEMA_ID, SCHEMA_VERSION, SUPPORTED_ENGAGEMENTS

__all__ = [
    "Position", "MetProfileMessage", "SimulationConfig",
    "EventConfiguration", "SetterMessage", "now_iso",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Position(BaseModel):
    """Display/schema information only. Coordinates here describe the
    selected engagement for reference; they are not the computational basis
    of any simulation run -- the engagement selector is (see
    setter/campaign.py, setter/simulation_adapter.py)."""

    model_config = ConfigDict(extra="forbid")

    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    frame: str = "range-table NED, muzzle-plane origin"


class MetProfileMessage(BaseModel):
    """The complete simulation representation of one `MetProfile`, so the
    message carries actual environmental simulation state and not merely an
    identifier. `profile_id` may still be carried as metadata alongside it."""

    model_config = ConfigDict(extra="forbid")

    label: str
    grid_m: list
    wind_north_ms: list
    wind_east_ms: list
    density_ratio: list
    temperature_ratio: list
    is_standard: bool
    profile_id: Optional[str] = None

    @field_validator("grid_m", "wind_north_ms", "wind_east_ms",
                      "density_ratio", "temperature_ratio")
    @classmethod
    def _same_length(cls, v: list, info) -> list:
        if not v:
            raise ValueError(f"{info.field_name} must be non-empty")
        return v

    @classmethod
    def from_met_profile(cls, met, profile_id: Optional[str] = None) -> "MetProfileMessage":
        return cls(
            label=met.label,
            grid_m=[float(x) for x in met.grid],
            wind_north_ms=[float(x) for x in met.wind_north],
            wind_east_ms=[float(x) for x in met.wind_east],
            density_ratio=[float(x) for x in met.density_ratio],
            temperature_ratio=[float(x) for x in met.temperature_ratio],
            is_standard=bool(met.is_standard),
            profile_id=profile_id,
        )


class SimulationConfig(BaseModel):
    """Reference solution values for the selected engagement -- what fire
    control would order, per `setter.simulation_adapter`. Not a new campaign
    result."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    charge: int
    qe_mils: float
    muzzle_velocity_ms: float
    dqe_mils: float = 0.0
    daz_mils: float = 0.0
    deploy_time_s: float
    guided_phase_s: float

    @field_validator("scenario_id")
    @classmethod
    def _known_engagement(cls, v: str) -> str:
        if v not in SUPPORTED_ENGAGEMENTS:
            raise ValueError(f"unsupported engagement {v!r}; supported: {SUPPORTED_ENGAGEMENTS}")
        return v


class EventConfiguration(BaseModel):
    """Generic configuration data for the simulation event engine
    (`fuze/`). Deliberately loose: `mode` names a demonstration mode and
    `parameters` carries whatever that mode needs. This layer validates
    types only -- it must not implement or encode physical effects."""

    model_config = ConfigDict(extra="forbid")

    mode: str
    parameters: dict = Field(default_factory=dict)

    @field_validator("mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        allowed = {"time", "motion", "proximity", "combined"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}, got {v!r}")
        return v


class SetterMessage(BaseModel):
    """The full serialized simulation configuration message."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_name: str = Field(default=SCHEMA_ID, alias="schema")
    schema_version: str = SCHEMA_VERSION
    generated_at: str = Field(default_factory=now_iso)
    scenario_id: str
    reference_position: Position
    target_position: Position
    met_profile: MetProfileMessage
    met_age_hours: Optional[float] = None
    simulation_config: SimulationConfig
    event_configuration: EventConfiguration
    units: dict = Field(default_factory=lambda: {
        "length": "m", "angle": "mils", "time": "s", "velocity": "m/s",
    })
    checksum: Optional[str] = None

    @field_validator("scenario_id")
    @classmethod
    def _known_engagement(cls, v: str) -> str:
        if v not in SUPPORTED_ENGAGEMENTS:
            raise ValueError(f"unsupported engagement {v!r}; supported: {SUPPORTED_ENGAGEMENTS}")
        return v
