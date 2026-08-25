"""The local source-object backend and the range rules layered above it."""

import asyncio
import errno
import os
import threading
from pathlib import Path
from uuid import UUID

import pytest

import app.storage.local as local_storage
from app.serving import (
    DOWNLOAD_MEDIA_TYPE,
    UnsatisfiableRangeError,
    content_disposition,
    parse_range,
    resolve_media_type,
)
from app.storage import (
    ORIGINAL_VARIANT,
    PREVIEW_VARIANT,
    LocalFileSourceStore,
    StorageError,
    StoredObjectMissingError,
    storage_key,
)

DOCUMENT_ID = UUID("1b4e28ba-2fa1-11d2-883f-0016d3cca427")


async def collect(store: LocalFileSourceStore, key: str, **kwargs: object) -> bytes:
    chunks = [chunk async for chunk in store.read(key, **kwargs)]  # type: ignore[arg-type]
    return b"".join(chunks)


def test_keys_come_from_document_identity_only() -> None:
    key = storage_key(DOCUMENT_ID, ORIGINAL_VARIANT)

    assert key == "1b/1b4e28ba2fa111d2883f0016d3cca427/original"
    assert storage_key(DOCUMENT_ID, PREVIEW_VARIANT).endswith("/preview")
    # Stable across calls, so a re-upload of identical content replaces one object.
    assert storage_key(DOCUMENT_ID) == key


def test_an_unknown_variant_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        storage_key(DOCUMENT_ID, "../../etc/passwd")


async def test_stored_bytes_survive_a_new_store_over_the_same_root(tmp_path: Path) -> None:
    key = storage_key(DOCUMENT_ID)
    first = LocalFileSourceStore(tmp_path / "sources")
    await first.prepare()

    stored = await first.put(key, b"%PDF-1.7 body")

    assert stored.byte_size == 13
    assert len(stored.checksum) == 64
    # A restart is a fresh store object over the same directory.
    second = LocalFileSourceStore(tmp_path / "sources")
    await second.prepare()
    assert await collect(second, key) == b"%PDF-1.7 body"
    assert (await second.stat(key)).byte_size == 13


async def test_writing_the_same_key_replaces_it_without_leaving_partials(tmp_path: Path) -> None:
    store = LocalFileSourceStore(tmp_path)
    key = storage_key(DOCUMENT_ID)

    await store.put(key, b"first version")
    await store.put(key, b"second")

    assert await collect(store, key) == b"second"
    directory = tmp_path.joinpath(*key.split("/")[:-1])
    assert sorted(entry.name for entry in directory.iterdir()) == ["original"]


async def test_concurrent_writers_of_one_key_all_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two uploads of the same content race, and neither may fail.

    Concurrency is forced rather than hoped for: every writer blocks on a barrier
    inside fsync, so all of them hold an open partial file before any of them
    renames. A temp name derived from the key made them share one file, and every
    rename after the first failed with ENOENT.
    """
    # Keep this below the smallest executor size used by supported Python
    # runtimes. Every worker blocks in fsync until all writers arrive; asking
    # for more writers than the executor can run would deadlock the harness.
    writers = 4
    store = LocalFileSourceStore(tmp_path)
    key = storage_key(DOCUMENT_ID)
    content = b"%PDF-1.7 " + bytes(5 * 1024 * 1024)

    barrier = threading.Barrier(writers, timeout=30)
    real_fsync = os.fsync
    real_replace = os.replace
    renamed: list[str] = []
    waited: set[int] = set()
    guard = threading.Lock()

    def synchronised_fsync(descriptor: int) -> None:
        real_fsync(descriptor)
        # Once per writer only: the directory fsync after the rename must not
        # need a second round, or a failing run would block until the timeout
        # instead of reporting the failure.
        with guard:
            first = waited.isdisjoint({threading.get_ident()})
            waited.add(threading.get_ident())
        if first:
            barrier.wait()

    def recording_replace(source, destination):  # type: ignore[no-untyped-def]
        renamed.append(str(source))
        return real_replace(source, destination)

    monkeypatch.setattr(local_storage.os, "fsync", synchronised_fsync)
    monkeypatch.setattr(local_storage.os, "replace", recording_replace)

    results = await asyncio.gather(
        *(store.put(key, content) for _ in range(writers)), return_exceptions=True
    )

    failures = [result for result in results if isinstance(result, BaseException)]
    assert not failures, failures
    # Each writer renamed its own partial file, which is what makes that true.
    assert len(set(renamed)) == writers
    assert all(name.endswith(".partial") for name in renamed)

    monkeypatch.undo()
    assert await collect(store, key) == content
    assert (await store.stat(key)).byte_size == len(content)
    # The directory holds the object and nothing else: no partial survives.
    directory = tmp_path.joinpath(*key.split("/")[:-1])
    assert sorted(entry.name for entry in directory.iterdir()) == ["original"]


async def test_a_failed_write_leaves_no_partial_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalFileSourceStore(tmp_path)
    key = storage_key(DOCUMENT_ID)
    await store.put(key, b"first version")

    def failing_replace(source, destination):  # type: ignore[no-untyped-def]
        raise OSError(errno.EIO, "input/output error")

    monkeypatch.setattr(local_storage.os, "replace", failing_replace)
    with pytest.raises(StorageError):
        await store.put(key, b"second version")
    monkeypatch.undo()

    directory = tmp_path.joinpath(*key.split("/")[:-1])
    assert sorted(entry.name for entry in directory.iterdir()) == ["original"]
    # A failed replace leaves the previous object intact rather than a partial one.
    assert await collect(store, key) == b"first version"


async def test_range_reads_return_only_the_requested_window(tmp_path: Path) -> None:
    store = LocalFileSourceStore(tmp_path)
    key = storage_key(DOCUMENT_ID)
    await store.put(key, bytes(range(64)))

    assert await collect(store, key, offset=10, length=4) == bytes(range(10, 14))
    assert await collect(store, key, offset=60) == bytes(range(60, 64))
    # Asking beyond the end yields what exists rather than padding.
    assert await collect(store, key, offset=62, length=100) == bytes(range(62, 64))


async def test_a_missing_object_is_distinguishable_from_a_broken_one(tmp_path: Path) -> None:
    store = LocalFileSourceStore(tmp_path)
    key = storage_key(DOCUMENT_ID)

    assert await store.stat(key) is None
    with pytest.raises(StoredObjectMissingError):
        await collect(store, key)
    # Deleting something absent is not an error, so cleanup stays idempotent.
    await store.delete(key)


@pytest.mark.parametrize(
    "key",
    ["../escape", "ab/../../etc/passwd", "onlyonesegment", "ab//original", "AB/x/original", "ab/x/orig inal"],
)
async def test_keys_that_are_not_plain_keys_are_refused(tmp_path: Path, key: str) -> None:
    store = LocalFileSourceStore(tmp_path)

    with pytest.raises(StorageError):
        await store.put(key, b"payload")


async def test_an_unwritable_root_fails_at_startup(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.mkdir(mode=0o500)
    store = LocalFileSourceStore(blocked / "sources")

    with pytest.raises(StorageError, match="not writable"):
        await store.prepare()


@pytest.mark.parametrize(
    ("media_type", "expected", "inline"),
    [
        ("application/pdf", "application/pdf", True),
        ("text/plain", "text/plain; charset=utf-8", True),
        ("text/markdown", "text/markdown; charset=utf-8", True),
        # User HTML must never render in this origin.
        ("text/html", DOWNLOAD_MEDIA_TYPE, False),
        ("image/svg+xml", DOWNLOAD_MEDIA_TYPE, False),
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            DOWNLOAD_MEDIA_TYPE,
            False,
        ),
    ],
)
def test_only_safe_types_are_served_inline(media_type: str, expected: str, inline: bool) -> None:
    assert resolve_media_type(media_type, variant=ORIGINAL_VARIANT) == (expected, inline)


def test_a_preview_is_always_plain_text() -> None:
    assert resolve_media_type("text/html", variant=PREVIEW_VARIANT) == (
        "text/plain; charset=utf-8",
        True,
    )


@pytest.mark.parametrize(
    "filename",
    ['re"port.pdf', "report\r\nX-Injected: 1.pdf", "rapport-å-ä-ö.pdf", "\x00\x01.pdf", "   "],
)
def test_content_disposition_cannot_inject_a_header(filename: str) -> None:
    value = content_disposition(filename, inline=True)

    assert "\r" not in value and "\n" not in value
    quoted = value.split('filename="', 1)[1].split('"', 1)[0]
    assert '"' not in quoted and all(0x20 <= ord(character) < 0x7F for character in quoted)
    assert value.startswith("inline; ")


def test_content_disposition_marks_downloads_as_attachments() -> None:
    value = content_disposition("report.docx", inline=False)

    assert value.startswith("attachment; ")
    assert 'filename="report.docx"' in value
    assert "filename*=UTF-8''report.docx" in value


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("bytes=0-9", (0, 10)),
        ("bytes=5-", (5, 95)),
        ("bytes=-10", (90, 10)),
        ("bytes=90-1000", (90, 10)),
        ("bytes=-1000", (0, 100)),
    ],
)
def test_satisfiable_ranges_resolve_against_the_object(header: str, expected: tuple) -> None:
    resolved = parse_range(header, 100)

    assert resolved is not None
    assert (resolved.start, resolved.length) == expected


@pytest.mark.parametrize(
    "header",
    [None, "", "items=0-1", "bytes=0-1, 5-6", "bytes=abc", "bytes=", "bytes=9-3", "bytes=x-1"],
)
def test_unsupported_ranges_fall_back_to_the_whole_object(header: str | None) -> None:
    assert parse_range(header, 100) is None


@pytest.mark.parametrize("header", ["bytes=100-200", "bytes=-0", "bytes=200-"])
def test_ranges_outside_the_object_are_unsatisfiable(header: str) -> None:
    with pytest.raises(UnsatisfiableRangeError):
        parse_range(header, 100)


def test_a_range_against_an_empty_object_is_ignored() -> None:
    assert parse_range("bytes=0-9", 0) is None
