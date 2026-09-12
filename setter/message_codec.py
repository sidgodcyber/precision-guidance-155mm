"""
Deterministic canonical-JSON encoding and CRC32 checksumming for
`setter.schemas.SetterMessage`.

Format v1 is JSON, compact separators, keys sorted -- so two calls encoding
the same message CONTENT always produce byte-identical output and the same
checksum. The checksum covers the canonical body WITHOUT `checksum` itself (a
field can't authenticate its own value) and WITHOUT `generated_at`: that
field is generation provenance (wall-clock time the message was built), not
configuration content, and two messages built a second apart from an
otherwise identical engagement/met-profile/simulation/event configuration
must produce the SAME checksum -- that is the reproducibility property Phase
2 characterization checks for (`tests/test_setter_reproducibility.py`). An
earlier version of this module included `generated_at` in the checksummed
body, which broke exactly that property; kept out here deliberately.
"""

from __future__ import annotations

import json
import zlib
from typing import Tuple

from setter.schemas import SetterMessage

__all__ = [
    "canonical_body", "compute_checksum", "encode", "decode", "verify_checksum",
]

#: Fields excluded from the checksummed canonical body: `checksum` because a
#: field cannot authenticate its own value, `generated_at` because it is
#: generation provenance, not configuration content (see module docstring).
_CHECKSUM_EXCLUDED_FIELDS = {"checksum", "generated_at"}


def canonical_body(message: SetterMessage) -> bytes:
    """The canonical JSON body used for checksumming: alias field names,
    `_CHECKSUM_EXCLUDED_FIELDS` excluded, keys sorted, compact separators."""
    body = message.model_dump(by_alias=True, exclude=_CHECKSUM_EXCLUDED_FIELDS)
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_checksum(message: SetterMessage) -> str:
    """8-hex-digit CRC32 of `canonical_body(message)`."""
    crc = zlib.crc32(canonical_body(message))
    return format(crc, "08x")


def encode(message: SetterMessage) -> Tuple[bytes, str]:
    """Returns `(canonical_json_bytes, checksum_hex)` for `message`, with the
    checksum computed and embedded."""
    checksum = compute_checksum(message)
    stamped = message.model_copy(update={"checksum": checksum})
    body = stamped.model_dump(by_alias=True)
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return blob, checksum


def decode(blob: bytes) -> SetterMessage:
    """Parses `blob` back into a `SetterMessage`, validating it against the
    schema. Raises `pydantic.ValidationError` on a malformed message."""
    data = json.loads(blob.decode("utf-8"))
    return SetterMessage.model_validate(data)


def verify_checksum(blob: bytes) -> bool:
    """True iff `blob` parses AND its embedded checksum matches a freshly
    computed CRC32 of its own body."""
    try:
        data = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    claimed = data.get("checksum")
    if claimed is None:
        return False
    try:
        message = SetterMessage.model_validate(data)
    except Exception:
        return False
    return claimed == compute_checksum(message)
