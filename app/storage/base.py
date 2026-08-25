"""The source-object storage contract."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import UUID

# Which representation of one source a key addresses. "original" is the bytes the
# tester uploaded; "preview" is the normalized text the service indexed, kept so
# a format that cannot be shown inline still has something safe to display.
Variant = str
ORIGINAL_VARIANT: Variant = "original"
PREVIEW_VARIANT: Variant = "preview"
VARIANTS: tuple[Variant, ...] = (ORIGINAL_VARIANT, PREVIEW_VARIANT)


class StorageError(RuntimeError):
    """Raised when durable source storage cannot serve a request."""


class StoredObjectMissingError(StorageError):
    """Raised when a key that the database references is absent from storage."""


@dataclass(frozen=True, slots=True)
class StoredObject:
    """What a backend guarantees about one stored object."""

    key: str
    byte_size: int
    checksum: str


def storage_key(identity_id: UUID, variant: Variant = ORIGINAL_VARIANT) -> str:
    """Build the storage key for one private source representation."""
    if variant not in VARIANTS:  # pragma: no cover - defensive
        raise ValueError(f"Unknown storage variant: {variant!r}")
    identity = identity_id.hex
    return f"{identity[:2]}/{identity}/{variant}"


class SourceObjectStore(ABC):
    """Durable, service-keyed storage for source bytes."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Short backend name, persisted alongside the key for diagnostics."""

    @abstractmethod
    async def put(self, key: str, content: bytes) -> StoredObject:
        """Store *content* at *key* atomically, replacing any previous object.

        A reader must never observe a partially written object, and a crash
        mid-write must leave either the old object or no object at all.
        """

    @abstractmethod
    async def stat(self, key: str) -> StoredObject | None:
        """Return what is stored at *key*, or ``None`` when it is absent."""

    @abstractmethod
    def read(
        self, key: str, *, offset: int = 0, length: int | None = None
    ) -> AsyncIterator[bytes]:
        """Stream at most *length* bytes from *offset*."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove *key* if it exists. Absence is not an error."""
