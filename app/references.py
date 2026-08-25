"""Opaque public markers for current source representations."""

import hashlib
from base64 import urlsafe_b64encode
from uuid import UUID


def section_reference(
    collection_id: str,
    external_id: str,
    revision_id: UUID,
    section_id: UUID,
) -> str:
    """Return a stable handle for one section of one current revision."""
    digest = _digest_components(
        "section", collection_id, external_id, str(revision_id), str(section_id)
    )
    token = urlsafe_b64encode(digest[:18]).decode("ascii").rstrip("=")
    return f"sec_{token}"


def source_revision_marker(checksum: str, revision_id: UUID) -> str:
    """Bind a cursor to both source bytes and the active block representation."""
    return _digest_components("source", checksum, str(revision_id)).hex()


def _digest_components(*components: str) -> bytes:
    digest = hashlib.sha256()
    for component in components:
        encoded = component.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.digest()
