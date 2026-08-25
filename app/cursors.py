"""Integrity-protected cursors for deterministic canonical-block scans."""

import base64
import binascii
import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import Any

from app.models import (
    COLLECTION_ID_PATTERN,
    MAX_EXTERNAL_ID_CHARACTERS,
    SECTION_REFERENCE_PATTERN,
)

_CURSOR_PREFIX = "scan1_"
_COLLECTION_PATTERN = re.compile(COLLECTION_ID_PATTERN)
_SECTION_REFERENCE_PATTERN = re.compile(SECTION_REFERENCE_PATTERN)
_SOURCE_MARKER_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_KEYS = frozenset({"collection", "external", "source", "section", "after"})


class InvalidScanCursorError(ValueError):
    """Raised when a scan cursor is malformed or fails integrity checks."""


@dataclass(frozen=True, slots=True)
class ScanCursor:
    collection_id: str
    external_id: str
    source_marker: str
    section_ref: str | None
    after_ordinal: int


class ScanCursorCodec:
    """Encode and validate opaque scan continuation state."""

    def __init__(self, secret: str) -> None:
        self._key = hashlib.sha256(
            b"seshat-scan-cursor\0" + secret.encode("utf-8")
        ).digest()

    def encode(self, cursor: ScanCursor) -> str:
        payload = {
            "after": cursor.after_ordinal,
            "collection": cursor.collection_id,
            "external": cursor.external_id,
            "section": cursor.section_ref,
            "source": cursor.source_marker,
        }
        encoded = _encode_bytes(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        signature = _encode_bytes(self._signature(encoded))
        return f"{_CURSOR_PREFIX}{encoded}.{signature}"

    def decode(self, token: str) -> ScanCursor:
        try:
            encoded, separator, signature = token.removeprefix(
                _CURSOR_PREFIX
            ).partition(".")
            if not token.startswith(_CURSOR_PREFIX) or not separator:
                raise InvalidScanCursorError
            expected = _encode_bytes(self._signature(encoded))
            if not hmac.compare_digest(signature, expected):
                raise InvalidScanCursorError
            raw = json.loads(_decode_bytes(encoded))
            return _validated_cursor_payload(raw)
        except (
            binascii.Error,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            raise InvalidScanCursorError("The scan cursor is invalid.") from None

    def _signature(self, encoded_payload: str) -> bytes:
        message = f"{_CURSOR_PREFIX}{encoded_payload}".encode("ascii")
        return hmac.new(self._key, message, hashlib.sha256).digest()


def _validated_cursor_payload(raw: Any) -> ScanCursor:
    if not isinstance(raw, dict) or set(raw) != _PAYLOAD_KEYS:
        raise InvalidScanCursorError
    collection_id = raw["collection"]
    external_id = raw["external"]
    source_marker = raw["source"]
    section_ref = raw["section"]
    after_ordinal = raw["after"]
    if not isinstance(collection_id, str) or not _COLLECTION_PATTERN.fullmatch(
        collection_id
    ):
        raise InvalidScanCursorError
    if (
        not isinstance(external_id, str)
        or not external_id
        or len(external_id) > MAX_EXTERNAL_ID_CHARACTERS
        or external_id != external_id.strip()
    ):
        raise InvalidScanCursorError
    if not isinstance(source_marker, str) or not _SOURCE_MARKER_PATTERN.fullmatch(
        source_marker
    ):
        raise InvalidScanCursorError
    if section_ref is not None and (
        not isinstance(section_ref, str)
        or not _SECTION_REFERENCE_PATTERN.fullmatch(section_ref)
    ):
        raise InvalidScanCursorError
    if (
        not isinstance(after_ordinal, int)
        or isinstance(after_ordinal, bool)
        or after_ordinal < 0
    ):
        raise InvalidScanCursorError
    return ScanCursor(
        collection_id=collection_id,
        external_id=external_id,
        source_marker=source_marker,
        section_ref=section_ref,
        after_ordinal=after_ordinal,
    )


def _encode_bytes(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_bytes(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True)
