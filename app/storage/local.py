"""Local-filesystem source storage, backed by a private Docker volume."""

import asyncio
import hashlib
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from app.log import logger
from app.storage.base import (
    SourceObjectStore,
    StorageError,
    StoredObject,
    StoredObjectMissingError,
)

_READ_CHUNK_BYTES = 256 * 1024
_KEY_SEGMENT_CHARACTERS = frozenset("0123456789abcdefghijklmnopqrstuvwxyz-_")
# In-progress writes carry this suffix, which no key can produce, so a partial file
# is never mistaken for an object and a crash leaves nothing readable behind.
_PARTIAL_SUFFIX = ".partial"


class LocalFileSourceStore(SourceObjectStore):
    """Stores each object as one file under *root*."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    @property
    def backend(self) -> str:
        return "local"

    @property
    def root(self) -> Path:
        return self._root

    async def prepare(self) -> None:
        """Create and probe the storage root so a bad mount fails at startup."""
        await asyncio.to_thread(self._prepare)
        logger.info("Source storage ready at %s (backend=local)", self._root)

    def _prepare(self) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            probe = self._root / ".writable"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as exc:
            raise StorageError(
                f"Source storage directory {self._root} is not writable: {exc.strerror}"
            ) from exc

    def _path(self, key: str) -> Path:
        """Resolve *key* to a path, refusing anything that is not a plain key."""
        segments = key.split("/")
        if len(segments) < 2 or any(
            not segment or not set(segment) <= _KEY_SEGMENT_CHARACTERS
            for segment in segments
        ):
            raise StorageError("Refusing a source storage key that is not a plain key.")
        return self._root.joinpath(*segments)

    async def put(self, key: str, content: bytes) -> StoredObject:
        return await asyncio.to_thread(self._put, key, content)

    def _put(self, key: str, content: bytes) -> StoredObject:
        target = self._path(key)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # An exclusive temporary file in the target directory preserves atomic
            # replacement when concurrent writers address the same key.
            descriptor, temporary_name = tempfile.mkstemp(
                dir=target.parent, prefix=f"{target.name}.", suffix=_PARTIAL_SUFFIX
            )
        except OSError as exc:
            raise StorageError(
                f"Could not store the source file: {exc.strerror}"
            ) from exc

        temporary = Path(temporary_name)
        try:
            # fdopen owns the descriptor and closes it with the writer.
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            self._sync_directory(target.parent)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise StorageError(
                f"Could not store the source file: {exc.strerror}"
            ) from exc
        return StoredObject(
            key=key,
            byte_size=len(content),
            checksum=hashlib.sha256(content).hexdigest(),
        )

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        """Persist the rename itself, so a host crash cannot lose the new name."""
        try:
            handle = os.open(directory, os.O_RDONLY)
        except OSError:  # pragma: no cover - platform dependent
            return
        try:
            os.fsync(handle)
        except OSError:  # pragma: no cover - platform dependent
            pass
        finally:
            os.close(handle)

    async def stat(self, key: str) -> StoredObject | None:
        return await asyncio.to_thread(self._stat, key)

    def _stat(self, key: str) -> StoredObject | None:
        path = self._path(key)
        try:
            info = path.stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StorageError(
                f"Could not read the source file: {exc.strerror}"
            ) from exc
        if not path.is_file():  # pragma: no cover - defensive
            return None
        # The checksum is authoritative in PostgreSQL; re-hashing on every stat
        # would read the whole file for each metadata request.
        return StoredObject(key=key, byte_size=info.st_size, checksum="")

    async def read(
        self, key: str, *, offset: int = 0, length: int | None = None
    ) -> AsyncIterator[bytes]:
        path = self._path(key)
        try:
            handle = await asyncio.to_thread(open, path, "rb")
        except FileNotFoundError as exc:
            raise StoredObjectMissingError(
                "The stored source file is no longer available."
            ) from exc
        except OSError as exc:
            raise StorageError(
                f"Could not open the source file: {exc.strerror}"
            ) from exc
        try:
            if offset:
                await asyncio.to_thread(handle.seek, offset)
            remaining = length
            while remaining is None or remaining > 0:
                want = (
                    _READ_CHUNK_BYTES
                    if remaining is None
                    else min(_READ_CHUNK_BYTES, remaining)
                )
                chunk = await asyncio.to_thread(handle.read, want)
                if not chunk:
                    return
                if remaining is not None:
                    remaining -= len(chunk)
                yield chunk
        finally:
            await asyncio.to_thread(handle.close)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._delete, key)

    def _delete(self, key: str) -> None:
        path = self._path(key)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - defensive
            raise StorageError(
                f"Could not remove the source file: {exc.strerror}"
            ) from exc
