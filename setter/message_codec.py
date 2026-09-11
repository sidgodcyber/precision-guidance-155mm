"""
Deterministic canonical-JSON encoding and CRC32 checksumming for
`setter.schemas.SetterMessage`.

Format v1 is JSON, compact separators, keys sorted -- so two calls encoding
the same message content always produce byte-identical output. The checksum
covers the canonical body WITHOUT the `checksum` field itself (a field can't
authenticate its own value), then the field is filled in and the message is
re-serialized, still canonically, for transport.
"""

from __future__ import annotations

import json
import zlib
from typing import Tuple

from setter.schemas import SetterMessage

__all__ = [
    "canonical_body", "compute_checksum", "encode", "decode", "verify_checksum",
]


def canonical_body(message: SetterMessage) -> bytes:
    """The canonical JSON body used for checksumming: alias field names,
    `checksum` excluded, keys sorted, compact separators."""
    body = message.model_dump(by_alias=True, exclude={"checksum"})
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
