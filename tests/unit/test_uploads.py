"""The authenticated file-upload and source-listing endpoints."""

import pytest
from httpx import AsyncClient

from app.parsing import ConversionUnavailableError, DocumentSegment
from tests.conftest import API_TOKEN, auth_headers
from tests.fakes import InMemoryRepository, RecordingConverter

COLLECTION = "example-collection"


async def upload(
    client: AsyncClient,
    *,
    filename: str = "notes.txt",
    content: bytes = b"The outlet valve needs calibration every year.",
    media_type: str = "text/plain",
    collection_id: str = COLLECTION,
    external_id: str = "source-1",
    title: str | None = None,
    token: str = API_TOKEN,
):
    data = {"collection_id": collection_id, "external_id": external_id}
    if title is not None:
        data["title"] = title
    return await client.post(
        "/v1/documents/file",
        files={"file": (filename, content, media_type)},
        data=data,
        headers=auth_headers(token),
    )


async def settle(client: AsyncClient, job_id: str) -> dict:
    await client.app.state.runtime.ingestion.wait_for_pending()  # type: ignore[attr-defined]
    job = await client.get(f"/v1/jobs/{job_id}", headers=auth_headers())
    job.raise_for_status()
    return job.json()


async def sources(client: AsyncClient, collection_id: str = COLLECTION) -> list[dict]:
    response = await client.get(
        "/v1/documents", params={"collection_id": collection_id}, headers=auth_headers()
    )
    response.raise_for_status()
    return response.json()["items"]


async def test_a_text_upload_becomes_a_ready_selectable_source(
    client: AsyncClient,
) -> None:
    content = b"The outlet valve needs calibration every year."
    accepted = await upload(client, filename="valve-notes.txt", content=content)

    assert accepted.status_code == 202
    body = accepted.json()
    assert body["status"] == "accepted"
    assert body["collection_id"] == COLLECTION
    assert body["external_id"] == "source-1"

    job = await settle(client, body["job_id"])
    assert job["status"] == "completed"
    assert job["chunk_count"] >= 1

    listed = await sources(client)
    assert len(listed) == 1
    assert listed[0]["external_id"] == body["external_id"]
    assert listed[0]["status"] == "ready"
    assert listed[0]["title"] == "valve-notes.txt"
    assert listed[0]["source_type"] == "text"
    assert listed[0]["filename"] == "valve-notes.txt"
    assert listed[0]["media_type"] == "text/plain"
    assert listed[0]["chunk_count"] >= 1
    assert listed[0]["detail"] is None
    assert listed[0]["created_at"] and listed[0]["updated_at"]


async def test_uploading_identical_content_is_idempotent(client: AsyncClient) -> None:
    content = b"Battery modules are replaced every three years."

    first = await settle(
        client, (await upload(client, content=content)).json()["job_id"]
    )
    second = await settle(
        client, (await upload(client, content=content)).json()["job_id"]
    )

    assert first["unchanged"] is False
    assert second["unchanged"] is True
    assert second["external_id"] == first["external_id"]
    assert len(await sources(client)) == 1


async def test_same_bytes_in_a_different_format_replace_the_indexed_content(
    client: AsyncClient,
    converter: RecordingConverter,
    repository: InMemoryRepository,
) -> None:
    content = b"<p>Original upload bytes.</p>"
    first = await settle(
        client,
        (await upload(client, filename="notes.txt", content=content)).json()["job_id"],
    )
    converter.markdown = "# Converted HTML\n\nBattery alarm guidance."
    second = await settle(
        client,
        (
            await upload(
                client,
                filename="notes.html",
                content=content,
                media_type="text/html",
            )
        ).json()["job_id"],
    )

    document = repository.documents[(COLLECTION, "source-1")]
    blocks = await repository.list_document_blocks(COLLECTION, "source-1")
    assert first["unchanged"] is False
    assert second["unchanged"] is False
    assert str(document.current_revision_id) == second["job_id"]
    assert document.source_type == "html"
    assert [block.text for block in blocks] == [converter.markdown]


async def test_an_explicit_title_wins_over_the_filename(client: AsyncClient) -> None:
    accepted = await upload(client, filename="raw.md", title="  Field notes  ")
    await settle(client, accepted.json()["job_id"])

    listed = await sources(client)
    assert listed[0]["title"] == "Field notes"
    assert listed[0]["filename"] == "raw.md"
    assert listed[0]["source_type"] == "markdown"


async def test_markdown_uses_local_normalization_with_unknown_structural_counts(
    client: AsyncClient,
    converter: RecordingConverter,
    repository: InMemoryRepository,
) -> None:
    accepted = await upload(
        client,
        filename="guide.md",
        content=b"# Heading\n\nBody text.",
        media_type="text/markdown",
    )
    await settle(client, accepted.json()["job_id"])

    document = repository.documents[(COLLECTION, "source-1")]
    assert converter.calls == []
    assert document.page_count is None
    assert document.recognized_section_count is None
    assert document.recognized_table_count is None
    assert document.recognized_figure_count is None
    sections = await repository.list_document_sections(COLLECTION, "source-1")
    assert len(sections) == 1 and sections[0].is_root is True


async def test_a_path_like_filename_is_stored_as_a_bare_name(
    client: AsyncClient,
) -> None:
    accepted = await upload(client, filename="../../etc/passwd.txt")
    await settle(client, accepted.json()["job_id"])

    listed = await sources(client)
    assert listed[0]["filename"] == "passwd.txt"
    assert "/" not in listed[0]["title"]


async def test_uploads_require_the_service_token(client: AsyncClient) -> None:
    assert (await upload(client, token="wrong-token")).status_code == 401
    unauthenticated = await client.get(
        "/v1/documents", params={"collection_id": COLLECTION}
    )
    assert unauthenticated.status_code == 401


async def test_unsupported_and_empty_uploads_are_rejected(client: AsyncClient) -> None:
    assert (await upload(client, filename="payload.zip")).status_code == 415
    assert (await upload(client, filename="payload")).status_code == 415
    mismatched = await upload(client, filename="report.pdf", media_type="text/html")
    assert mismatched.status_code == 415
    assert (await upload(client, content=b"")).status_code == 422
    assert (await upload(client, content=b"\xff\xfe\x00\x01")).status_code == 202


async def test_oversized_uploads_are_rejected(client: AsyncClient, settings) -> None:
    oversized = b"x" * (settings.max_upload_bytes + 1)

    assert (await upload(client, content=oversized)).status_code == 413


async def test_an_invalid_collection_id_is_rejected(client: AsyncClient) -> None:
    assert (await upload(client, collection_id="Bad Collection")).status_code == 422
    rejected = await client.get(
        "/v1/documents", params={"collection_id": "../other"}, headers=auth_headers()
    )
    assert rejected.status_code == 422


async def test_file_upload_requires_a_caller_owned_external_id(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/v1/documents/file",
        files={"file": ("notes.txt", b"Body", "text/plain")},
        data={"collection_id": COLLECTION},
        headers=auth_headers(),
    )

    assert response.status_code == 422


async def test_a_failed_upload_is_listed_with_its_reason(client: AsyncClient) -> None:
    accepted = await upload(client, content=b"\xff\xfe\x00\x01\x02\x03")
    job = await settle(client, accepted.json()["job_id"])

    assert job["status"] == "failed"
    listed = await sources(client)
    assert listed[0]["status"] == "failed"
    assert listed[0]["title"] == "notes.txt"
    assert listed[0]["filename"] == "notes.txt"
    assert listed[0]["source_type"] == "text"
    assert "UTF-8" in listed[0]["detail"]
    assert listed[0]["chunk_count"] == 0


async def test_a_pdf_is_converted_through_the_adapter(
    client: AsyncClient, converter: RecordingConverter
) -> None:
    accepted = await upload(
        client,
        filename="service.pdf",
        content=b"%PDF-1.7 binary",
        media_type="application/pdf",
    )
    job = await settle(client, accepted.json()["job_id"])

    assert job["status"] == "completed"
    assert converter.calls == [
        {
            "filename": "service.pdf",
            "media_type": "application/pdf",
            "content": b"%PDF-1.7 binary",
        }
    ]
    listed = await sources(client)
    assert listed[0]["status"] == "ready"
    assert listed[0]["source_type"] == "pdf"
    assert listed[0]["media_type"] == "application/pdf"


async def test_a_caption_only_table_from_a_converter_is_indexed(
    client: AsyncClient, converter: RecordingConverter
) -> None:
    caption = "Table 4-2 Diagnostic codes"
    converter.segments = [
        DocumentSegment(text=caption, is_table=True, caption=caption)
    ]

    accepted = await upload(
        client,
        filename="service.pdf",
        content=b"%PDF-1.7 caption-only-table",
        media_type="application/pdf",
    )
    job = await settle(client, accepted.json()["job_id"])

    assert job["status"] == "completed"
    assert job["chunk_count"] == 1


@pytest.mark.parametrize("conversion_enabled", [False], indirect=True)
async def test_a_pdf_fails_clearly_when_no_converter_is_configured(
    client: AsyncClient, converter: RecordingConverter
) -> None:
    accepted = await upload(
        client, filename="service.pdf", media_type="application/pdf"
    )
    job = await settle(client, accepted.json()["job_id"])

    assert job["status"] == "failed"
    assert "not configured" in job["detail"]
    assert converter.calls == []


@pytest.mark.parametrize("conversion_enabled", [False], indirect=True)
async def test_text_uploads_keep_working_when_conversion_is_unavailable(
    client: AsyncClient,
) -> None:
    job = await settle(client, (await upload(client)).json()["job_id"])

    assert job["status"] == "completed"


async def test_a_converter_outage_fails_only_that_job(
    client: AsyncClient, converter: RecordingConverter
) -> None:
    converter.error = ConversionUnavailableError(
        "The document converter is unavailable."
    )

    failed = await settle(
        client,
        (
            await upload(
                client,
                filename="a.pdf",
                media_type="application/pdf",
                external_id="failed-pdf",
            )
        ).json()["job_id"],
    )
    converter.error = None
    indexed = await settle(
        client,
        (
            await upload(
                client,
                filename="b.txt",
                content=b"Different text entirely.",
                external_id="working-text",
            )
        ).json()["job_id"],
    )

    assert failed["status"] == "failed"
    assert failed["detail"] == "The document converter is unavailable."
    assert indexed["status"] == "completed"

    listed = {item["status"] for item in await sources(client)}
    assert listed == {"failed", "ready"}


async def test_the_manual_and_uploads_are_listed_side_by_side(
    client: AsyncClient,
) -> None:
    manual = await client.post(
        "/v1/documents/text",
        json={
            "collection_id": COLLECTION,
            "external_id": "example-manual",
            "title": "Example service manual",
            "source_type": "manual",
            "text": "The controller alarm list starts here.",
        },
        headers=auth_headers(),
    )
    await settle(client, manual.json()["job_id"])
    await settle(client, (await upload(client)).json()["job_id"])

    listed = {item["external_id"]: item for item in await sources(client)}

    assert set(listed) == {
        "example-manual",
        "source-1",
    }
    assert all(item["status"] == "ready" for item in listed.values())
    assert listed["example-manual"]["source_type"] == "manual"
    assert listed["example-manual"]["filename"] is None


async def test_sources_are_scoped_to_the_requested_collection(
    client: AsyncClient,
) -> None:
    await settle(
        client,
        (await upload(client, filename="one.txt", external_id="one")).json()["job_id"],
    )
    await settle(
        client,
        (
            await upload(
                client,
                filename="two.txt",
                content=b"Other text",
                collection_id="other-collection",
                external_id="two",
            )
        ).json()["job_id"],
    )

    assert [item["filename"] for item in await sources(client)] == ["one.txt"]
    assert [item["filename"] for item in await sources(client, "other-collection")] == [
        "two.txt"
    ]
