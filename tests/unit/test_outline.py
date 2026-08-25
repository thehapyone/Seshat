"""Current-source outline: structural facts, opaque references, and isolation."""

import re
from dataclasses import replace

from httpx import AsyncClient, Response

from app.parsing import DocumentSegment
from tests.conftest import auth_headers, ingest
from tests.fakes import InMemoryRepository, RecordingConverter

COLLECTION = "example-collection"
EXTERNAL_ID = "service-manual"


async def get_outline(
    client: AsyncClient,
    *,
    collection_id: str = COLLECTION,
    external_id: str = EXTERNAL_ID,
) -> Response:
    return await client.get(
        "/v1/documents/source/outline",
        params={"collection_id": collection_id, "external_id": external_id},
        headers=auth_headers(),
    )


async def upload_pdf(
    client: AsyncClient,
    content: bytes,
    *,
    collection_id: str = COLLECTION,
    external_id: str = EXTERNAL_ID,
) -> None:
    accepted = await client.post(
        "/v1/documents/file",
        files={"file": ("service.pdf", content, "application/pdf")},
        data={"collection_id": collection_id, "external_id": external_id},
        headers=auth_headers(),
    )
    accepted.raise_for_status()
    ingestion = client.app.state.runtime.ingestion  # type: ignore[attr-defined]
    await ingestion.wait_for_pending()
    job = await client.get(
        f"/v1/jobs/{accepted.json()['job_id']}", headers=auth_headers()
    )
    job.raise_for_status()
    assert job.json()["status"] == "completed"


async def test_outline_returns_persisted_counts_and_recognized_sections(
    client: AsyncClient,
    converter: RecordingConverter,
    repository: InMemoryRepository,
) -> None:
    converter.segments = [
        DocumentSegment(
            text="Battery maintenance interval.",
            page=2,
            section_path=("Maintenance", "Battery"),
            section_ref="#/groups/battery",
        ),
        DocumentSegment(
            text="Terms and abbreviations.",
            section_path=("Appendix",),
            section_ref="#/groups/appendix",
        ),
    ]
    await upload_pdf(client, b"%PDF-structured-outline")
    document = repository.documents[(COLLECTION, EXTERNAL_ID)]
    repository.documents[(COLLECTION, EXTERNAL_ID)] = replace(
        document,
        page_count=12,
        recognized_section_count=3,
        recognized_table_count=1,
        recognized_figure_count=0,
    )

    response = await get_outline(client)

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "collection_id": COLLECTION,
        "external_id": EXTERNAL_ID,
        "page_count": 12,
        "recognized_section_count": 3,
        "recognized_table_count": 1,
        "recognized_figure_count": 0,
        "sections": body["sections"],
    }
    assert [item["section_path"] for item in body["sections"]] == [
        ["Maintenance"],
        ["Maintenance", "Battery"],
        ["Appendix"],
    ]
    assert body["sections"][0]["page_start"] == 2
    assert body["sections"][0]["page_end"] == 2
    assert "page_start" not in body["sections"][2]
    assert "page_end" not in body["sections"][2]
    assert all(
        re.fullmatch(r"sec_[A-Za-z0-9_-]{24}", item["section_ref"])
        for item in body["sections"]
    )
    serialized = response.text
    assert str(document.document_id) not in serialized
    assert str(document.current_revision_id) not in serialized
    assert all(
        str(section.section_id) not in serialized
        for section in repository.sections.values()
    )


async def test_unstructured_source_reports_unknown_counts_and_a_reason(
    client: AsyncClient,
) -> None:
    await ingest(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "text": "Unstructured field notes.",
        },
    )

    response = await get_outline(client)

    assert response.status_code == 200
    assert response.json() == {
        "collection_id": COLLECTION,
        "external_id": EXTERNAL_ID,
        "page_count": None,
        "recognized_section_count": None,
        "recognized_table_count": None,
        "recognized_figure_count": None,
        "sections": [],
        "reason": "No reliable section structure was extracted.",
    }


async def test_confirmed_empty_outline_is_distinct_from_unsupported_extraction(
    client: AsyncClient,
    repository: InMemoryRepository,
) -> None:
    await ingest(
        client,
        {
            "collection_id": COLLECTION,
            "external_id": EXTERNAL_ID,
            "text": "A converter confirmed that this source has no sections.",
        },
    )
    document = repository.documents[(COLLECTION, EXTERNAL_ID)]
    repository.documents[(COLLECTION, EXTERNAL_ID)] = replace(
        document,
        page_count=1,
        recognized_section_count=0,
        recognized_table_count=0,
        recognized_figure_count=0,
    )

    response = await get_outline(client)

    assert response.status_code == 200
    assert response.json() == {
        "collection_id": COLLECTION,
        "external_id": EXTERNAL_ID,
        "page_count": 1,
        "recognized_section_count": 0,
        "recognized_table_count": 0,
        "recognized_figure_count": 0,
        "sections": [],
        "reason": "No sections were recognized in this source.",
    }


async def test_replacing_a_source_changes_its_section_references(
    client: AsyncClient,
    converter: RecordingConverter,
) -> None:
    converter.segments = [
        DocumentSegment(
            text="Old battery guidance.",
            page=2,
            section_path=("Maintenance", "Battery"),
        )
    ]
    await upload_pdf(client, b"%PDF-old-revision")
    first = (await get_outline(client)).json()

    converter.segments = [
        DocumentSegment(
            text="New battery guidance.",
            page=3,
            section_path=("Maintenance", "Battery"),
        )
    ]
    await upload_pdf(client, b"%PDF-new-revision")
    second = (await get_outline(client)).json()

    assert [item["section_path"] for item in first["sections"]] == [
        item["section_path"] for item in second["sections"]
    ]
    assert {item["section_ref"] for item in first["sections"]}.isdisjoint(
        item["section_ref"] for item in second["sections"]
    )


async def test_outline_is_scoped_to_the_requested_collection(
    client: AsyncClient,
) -> None:
    await ingest(
        client,
        {
            "collection_id": "other-collection",
            "external_id": EXTERNAL_ID,
            "text": "Other collection content.",
        },
    )

    unavailable = await get_outline(client)
    available = await get_outline(client, collection_id="other-collection")

    assert unavailable.status_code == 404
    assert unavailable.json() == {
        "detail": {
            "code": "source_not_found",
            "message": "Current source not found.",
        }
    }
    assert available.status_code == 200
    assert available.json()["collection_id"] == "other-collection"
