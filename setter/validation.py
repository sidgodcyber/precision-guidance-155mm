"""
Explicit validation for setter inputs and messages.

Nothing here clamps or silently substitutes a value: an unsupported
engagement, an unsupported met age, or a malformed/tampered message is
rejected with a `SetterValidationError`, never coerced into something
"close enough".
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from setter import message_codec
from setter.config import MET_AGE_BUCKETS, SUPPORTED_ENGAGEMENTS
from setter.schemas import SetterMessage

__all__ = [
    "SetterValidationError",
    "validate_engagement", "validate_met_age_bucket",
    "validate_message_dict", "validate_checksum",
]


class SetterValidationError(ValueError):
    """A setter input, configuration, or message failed validation."""


def validate_engagement(engagement: str) -> None:
    if engagement not in SUPPORTED_ENGAGEMENTS:
        raise SetterValidationError(
            f"unsupported engagement {engagement!r}; supported: {SUPPORTED_ENGAGEMENTS}")


def validate_met_age_bucket(age: str) -> None:
    if age not in MET_AGE_BUCKETS:
        raise SetterValidationError(
            f"unsupported met age {age!r}; supported: {MET_AGE_BUCKETS}")


def validate_message_dict(raw: Any) -> SetterMessage:
    """Validate a raw (e.g. JSON-decoded) mapping into a `SetterMessage`."""
    if not isinstance(raw, dict):
        raise SetterValidationError(f"message must be a mapping, got {type(raw).__name__}")
    try:
        return SetterMessage.model_validate(raw)
    except ValidationError as exc:
        raise SetterValidationError(str(exc)) from exc


def validate_checksum(blob: bytes) -> None:
    if not message_codec.verify_checksum(blob):
        raise SetterValidationError("checksum mismatch or malformed message")
