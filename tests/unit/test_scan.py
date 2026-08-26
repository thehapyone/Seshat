"""Canonical scans: scope, pagination, bounds, and cursor invalidation."""

import hashlib
from typing import Any
from uuid import uuid4

import pytest
from httpx import AsyncClient, Response

from app.cursors import InvalidScanCursorError, ScanCursor, ScanCursorCodec
from app.embeddings.chunking import CanonicalBlock
from app.config import Settings
from app.models import ScanRequest
from app.references import source_revision_marker
from app.repository import DocumentRecord
from app.representation import build_document_structure
from tests.conftest import auth_headers
from tests.fakes import InMemoryRepository

COLLECTION = "example-collection"
EXTERNAL_ID = "service-manual"


def seed_source(
    repository: InMemoryRepository,
    blocks: tuple[CanonicalBlock, ...],
    *,
    collection_id: str = COLLECTION,
    external_id: str = EXTERNAL_ID,
    checksum: str | None = None,
) -> DocumentRecord:
    key = (collection_id, external_id)
    existing = repository.documents.get(key)
    document_id = existing.document_id if existing is not None else uuid4()
    revision_id = uuid4()
    structure = build_document_structure(document_id, revision_id, blocks)
    source_bytes = "\n".join(block.rendered_text for block in blocks).encode("utf-8")
    source_checksum = checksum or hashlib.sha256(source_bytes).hexdigest()
    document = DocumentRecord(
        document_id=document_id,
        collection_id=collection_id,
        external_id=external_id,
        title="Service manual",
        source_type="manual",
        source_uri="",
        checksum=source_checksum,
        normalized_checksum=source_checksum,
        current_revision_id=revision_id,
        version="",
        page=None,
        section="",
        page_count=None,
        recognized_section_count=sum(
            not section.is_root for section in structure.sections
        ),
        recognized_table_count=None,
        recognized_figure_count=0,
        chunk_count=len(blocks),
    )
    repository.documents[key] = document
    repository.sections = {
        key: section
        for key, section in repository.sections.items()
        if key[0] != document_id
    }
    repository.sections.update(
        {
            (section.document_id, section.ordinal): section
            for section in structure.sections
        }
    )
    repository.blocks = {
        key: block
        for key, block in repository.blocks.items()
        if key[0] != document_id
    }
    repository.blocks.update(
        {(block.document_id, block.ordinal): block for block in structure.blocks}
    )
    return document


async def scan(client: AsyncClient, body: dict[str, object]) -> Response:
    return await client.post("/v1/scan", json=body, headers=auth_headers())


async def outline(client: AsyncClient) -> dict[str, Any]:
    response = await client.get(
        "/v1/documents/source/outline",
        params={"collection_id": COLLECTION, "external_id": EXTERNAL_ID},
        headers=auth_headers(),
    )
    response.raise_for_status()
    return response.json()


def test_scan_cursor_requires_the_same_server_signing_key() -> None:
    cursor = ScanCursorCodec("first-signing-key-0123456789012345").encode(
        ScanCursor(
            collection_id=COLLECTION,
            external_id=EXTERNAL_ID,
            source_marker="a" * 64,
            section_ref=None,
            after_ordinal=7,
        )
    )

    with pytest.raises(InvalidScanCursorError, match="invalid"):
        ScanCursorCodec("second-signing-key-012345678901234").decode(cursor)


def test_scan_cursor_accepts_the_largest_encoded_external_id() -> None:
    external_id = "a" + "\x01" * 254 + "b"
    codec = ScanCursorCodec("cursor-signing-key-0123456789012345")
    cursor = codec.encode(
        ScanCursor(
            collection_id=COLLECTION,
            external_id=external_id,
            source_marker="a" * 64,
            section_ref=None,
            after_ordinal=7,
        )
    )

    request = ScanRequest(
        collection_id=COLLECTION,
        external_id=external_id,
        cursor=cursor,
    )

    assert len(cursor) > 2_048
    assert codec.decode(request.cursor or "").external_id == external_id


async def test_whole_source_scan_returns_all_content_once_in_packed_passages(
    client: AsyncClient,
    repository: InMemoryRepository,
) -> None:
    texts = [f"Canonical block {ordinal}" for ordinal in range(5)]
    seed_source(
        repository,
        tuple(
            CanonicalBlock(ordinal=ordinal, kind="text", text=text)
            for ordinal, text in enumerate(texts)
        ),
    )

    cursor: str | None = None
    returned: list[str] = []
    while True:
        response = await scan(
            client,
            {
                "collection_id": COLLECTION,
                "external_id": EXTERNAL_ID,
                "limit": 2,
                "cursor": cursor,
            },
        )
        response.raise_for_status()
        body = response.json()
        returned.extend(item["text"] for item in body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
        assert cursor.startswith("scan1_")
        assert EXTERNAL_ID not in cursor

    assert "\n\n".join(returned) == "\n\n".join(texts)


async def test_section_scan_includes_its_nested_subsections(
    client: AsyncClient,
    repository: InMemoryRepository,
) -> None:
    seed_source(
        repository,
        (
            CanonicalBlock(ordinal=0, kind="text", text="Preface"),
            CanonicalBlock(
                ordinal=1,
                kind="text",
                text="General maintenance",
                section_path=("Maintenance",),
                page_start=2,
                page_end=2,
            ),
            CanonicalBlock(
                ordinal=2,
                kind="text",
                text="Battery procedure",
                section_path=("Maintenance", "Battery"),
                page_start=3,
                page_end=4,
            ),
            CanonicalBlock(
                ordinal=3,
                kind="text",
                text="Appendix material",
                section_path=("Appendix",),
                page_start=8,
                page_end=8,
            ),
        ),
    )
    source_outline = await outline(client)
    maintenance = next(
        item
        for item in source_outline["sections"]
        if item["section_path"] == ["Maintenance"]
    )

    response = await scan(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "section_ref": maintenance["section_ref"],
            "limit": 10,
        },
    )

    response.raise_for_status()
    body = response.json()
    assert [item["text"] for item in body["items"]] == [
        "General maintenance\n\nBattery procedure"
    ]
    assert "section_ref" not in body["items"][0]
    assert "section_path" not in body["items"][0]
    assert body["items"][0]["page_start"] == 2
    assert body["items"][0]["page_end"] == 4
    assert body["next_cursor"] is None


async def test_scan_renders_table_context_without_exposing_table_fields(
    client: AsyncClient,
    repository: InMemoryRepository,
) -> None:
    seed_source(
        repository,
        (
            CanonicalBlock(
                ordinal=0,
                kind="table_part",
                text="| 310 | Low voltage |",
                section_path=("Errors",),
                table_ref="table-7",
                table_caption="Power module diagnostics",
                table_header="| Code | Meaning |\n| --- | --- |",
                part=1,
                parts=1,
            ),
        ),
    )

    response = await scan(
        client,
        {"collection_id": COLLECTION, "external_id": EXTERNAL_ID},
    )

    response.raise_for_status()
    item = response.json()["items"][0]
    assert item["text"] == (
        "Power module diagnostics\n| Code | Meaning |\n| --- | --- |\n| 310 | Low voltage |"
    )
    assert set(item) == {"text", "section_ref", "section_path"}


async def test_scan_packs_sparse_converter_fragments_with_their_table(
    client: AsyncClient,
    repository: InMemoryRepository,
) -> None:
    seed_source(
        repository,
        (
            CanonicalBlock(
                ordinal=0, kind="text", text="21", page_start=149, page_end=149
            ),
            CanonicalBlock(
                ordinal=1, kind="text", text="3", page_start=149, page_end=149
            ),
            CanonicalBlock(
                ordinal=2,
                kind="text",
                text="Technical error code",
                section_path=("Technical error code",),
                page_start=149,
                page_end=149,
            ),
            CanonicalBlock(
                ordinal=3,
                kind="table_part",
                text="| 81 | Replace the flow sensor. |",
                section_path=("Technical error code",),
                table_header="| Code | Action |\n| --- | --- |",
                page_start=149,
                page_end=149,
            ),
            CanonicalBlock(
                ordinal=4,
                kind="text",
                text="Run the complete checkout procedure after replacement.",
                section_path=("Verification",),
                page_start=150,
                page_end=150,
            ),
        ),
    )

    response = await scan(
        client,
        {"collection_id": COLLECTION, "external_id": EXTERNAL_ID, "limit": 20},
    )

    response.raise_for_status()
    assert response.json()["items"] == [
        {
            "text": (
                "21\n\n3\n\nTechnical error code\n\n"
                "| Code | Action |\n| --- | --- |\n"
                "| 81 | Replace the flow sensor. |\n\n"
                "Run the complete checkout procedure after replacement."
            ),
            "page_start": 149,
            "page_end": 150,
        }
    ]


async def test_scan_payload_bound_defers_whole_blocks(
    client: AsyncClient,
    repository: InMemoryRepository,
) -> None:
    block_texts = [str(index) * 60_000 for index in range(5)]
    seed_source(
        repository,
        tuple(
            CanonicalBlock(ordinal=index, kind="text", text=text)
            for index, text in enumerate(block_texts)
        ),
    )

    first = await scan(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "limit": 100,
        },
    )

    first.raise_for_status()
    body = first.json()
    assert len(body["items"]) == 4
    assert sum(len(item["text"].encode("utf-8")) for item in body["items"]) <= 256_000
    assert body["items"][-1]["text"] == block_texts[3]
    assert body["next_cursor"] is not None

    second = await scan(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "limit": 100,
            "cursor": body["next_cursor"],
        },
    )
    second.raise_for_status()
    assert [item["text"] for item in second.json()["items"]] == [block_texts[4]]
    assert second.json()["next_cursor"] is None


async def test_cursor_tampering_and_scope_reuse_fail_closed(
    client: AsyncClient,
    repository: InMemoryRepository,
) -> None:
    seed_source(
        repository,
        (
            CanonicalBlock(ordinal=0, kind="text", text="First"),
            CanonicalBlock(ordinal=1, kind="text", text="Second"),
        ),
    )
    first = await scan(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "limit": 1,
        },
    )
    cursor = first.json()["next_cursor"]
    assert cursor is not None
    tampered = f"{cursor[:-1]}{'A' if cursor[-1] != 'A' else 'B'}"

    invalid = await scan(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "cursor": tampered,
        },
    )
    crossed = await scan(
        client,
        {
            "collection_id": "other-collection",
            "external_id": EXTERNAL_ID,
            "cursor": cursor,
        },
    )

    assert invalid.status_code == 400
    assert invalid.json()["detail"]["code"] == "invalid_cursor"
    assert crossed.status_code == 400
    assert crossed.json()["detail"]["code"] == "invalid_cursor"


async def test_section_cursor_requires_the_same_section_scope(
    client: AsyncClient,
    repository: InMemoryRepository,
) -> None:
    seed_source(
        repository,
        (
            CanonicalBlock(
                ordinal=0,
                kind="text",
                text="First",
                section_path=("Maintenance",),
            ),
            CanonicalBlock(
                ordinal=1,
                kind="text",
                text="Second",
                section_path=("Maintenance",),
            ),
        ),
    )
    section_ref = (await outline(client))["sections"][0]["section_ref"]
    first = await scan(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "section_ref": section_ref,
            "limit": 1,
        },
    )

    mismatched = await scan(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "cursor": first.json()["next_cursor"],
        },
    )

    assert mismatched.status_code == 400
    assert mismatched.json()["detail"]["code"] == "invalid_cursor"


async def test_signed_cursor_position_must_remain_inside_its_scope(
    client: AsyncClient,
    repository: InMemoryRepository,
    settings: Settings,
) -> None:
    document = seed_source(
        repository,
        (
            CanonicalBlock(ordinal=0, kind="text", text="First"),
            CanonicalBlock(ordinal=1, kind="text", text="Second"),
        ),
    )
    assert document.current_revision_id is not None
    cursor = ScanCursorCodec(settings.cursor_signing_key).encode(
        ScanCursor(
            collection_id=COLLECTION,
            external_id=EXTERNAL_ID,
            source_marker=source_revision_marker(
                document.checksum, document.current_revision_id
            ),
            section_ref=None,
            after_ordinal=99,
        )
    )

    response = await scan(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "cursor": cursor,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_cursor"


async def test_replacement_invalidates_cursor_even_when_source_bytes_match(
    client: AsyncClient,
    repository: InMemoryRepository,
) -> None:
    checksum = "a" * 64
    seed_source(
        repository,
        (
            CanonicalBlock(ordinal=0, kind="text", text="First"),
            CanonicalBlock(ordinal=1, kind="text", text="Second"),
        ),
        checksum=checksum,
    )
    first = await scan(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "limit": 1,
        },
    )
    cursor = first.json()["next_cursor"]

    seed_source(
        repository,
        (
            CanonicalBlock(ordinal=0, kind="text", text="Rebuilt first"),
            CanonicalBlock(ordinal=1, kind="text", text="Rebuilt second"),
        ),
        checksum=checksum,
    )
    continued = await scan(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "limit": 1,
            "cursor": cursor,
        },
    )

    assert continued.status_code == 409
    assert continued.json()["detail"]["code"] == "source_changed"


async def test_replaced_section_reference_is_not_resolved(
    client: AsyncClient,
    repository: InMemoryRepository,
) -> None:
    seed_source(
        repository,
        (
            CanonicalBlock(
                ordinal=0,
                kind="text",
                text="Old section content",
                section_path=("Maintenance",),
            ),
        ),
    )
    old_reference = (await outline(client))["sections"][0]["section_ref"]
    seed_source(
        repository,
        (
            CanonicalBlock(
                ordinal=0,
                kind="text",
                text="Replacement content",
                section_path=("Maintenance",),
            ),
        ),
    )

    response = await scan(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "section_ref": old_reference,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "section_not_found"


async def test_scan_reports_missing_sources_and_configured_limit_errors(
    client: AsyncClient,
) -> None:
    missing = await scan(
        client,
        {"collection_id": COLLECTION, "external_id": EXTERNAL_ID},
    )
    too_many = await scan(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "limit": 101,
        },
    )

    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "source_not_found"
    assert too_many.status_code == 422
    assert too_many.json()["detail"]["code"] == "invalid_limit"
