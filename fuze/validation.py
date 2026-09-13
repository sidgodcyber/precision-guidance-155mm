"""
Validation helpers for `fuze.config`.

`fuze.config` already enforces field types and ranges through pydantic at
construction. This module (a) turns that into one clean exception type so
callers don't need to import pydantic's `ValidationError`, and (b) rejects
malformed raw input explicitly rather than letting it clamp or coerce
silently.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from fuze.config import EVENT_KINDS, FuzeEngineConfig

__all__ = ["FuzeConfigError", "build_engine_config", "validate_reading_kind"]


class FuzeConfigError(ValueError):
    """A fuze engine configuration was malformed or out of range."""


def build_engine_config(raw: Any) -> FuzeEngineConfig:
    """Validate a raw (e.g. JSON-decoded) mapping into a `FuzeEngineConfig`.

    Rejects anything that isn't a mapping, any field of the wrong type, any
    out-of-range value, and any unknown field -- there is no silent
    clamping or coercion.
    """
    if not isinstance(raw, dict):
        raise FuzeConfigError(f"engine configuration must be a mapping, got {type(raw).__name__}")
    try:
        return FuzeEngineConfig.model_validate(raw)
    except ValidationError as exc:
        raise FuzeConfigError(str(exc)) from exc


def validate_reading_kind(kind: str) -> None:
    if kind not in EVENT_KINDS:
        raise FuzeConfigError(f"unknown event kind {kind!r}; expected one of {EVENT_KINDS}")
