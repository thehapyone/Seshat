"""Ingestion: idempotency, collection scoping, and validation."""

from typing import Any
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.ingest import document_uuid, text_checksum
from tests.conftest import auth_headers, ingest
from tests.fakes import InMemoryRepository, RecordingVectorStore

MANUAL_TEXT = (
    "The inlet pressure sensor must be calibrated during preventive maintenance. "
    "Replace the outlet valve when the alarm persists after calibration."
)


def payload(**overrides: object) -> dict:
    base = {
        "collection_id": "example-collection",
        "external_id": "example-manual",
        "title": "Example service manual",
        "source_type": "manual",
        "source_uri": "file:///data/example-manual.txt",
        "version": "rev-19",
        "text": MANUAL_TEXT,
    }
    base.update(overrides)
    return base


async def test_ingest_indexes_chunks_and_records_document_identity(
    client: AsyncClient,
    repository: InMemoryRepository,
    vector_store: RecordingVectorStore,
) -> None:
    job = await ingest(client, payload())

    assert job["status"] == "completed"
    assert job["unchanged"] is False
    assert job["chunk_count"] >= 1
    assert vector_store.nodes

    document = repository.documents[("example-collection", "example-manual")]
    assert document.document_id == document_uuid("example-collection", "example-manual")
    assert document.checksum == text_checksum(MANUAL_TEXT)
    assert document.title == "Example service manual"
    assert document.version == "rev-19"
    assert document.chunk_count == job["chunk_count"]
    assert "example-collection" in repository.collections
    sections = await repository.list_document_sections(
        "example-collection", "example-manual"
    )
    blocks = await repository.list_document_blocks(
        "example-collection", "example-manual"
    )
    assert len(sections) == 1 and sections[0].is_root is True
    assert [block.ordinal for block in blocks] == list(range(len(blocks)))
    assert {block.revision_id for block in blocks} == {document.current_revision_id}
    assert document.page_count is None
    assert document.recognized_section_count is None
    assert document.recognized_table_count is None
    assert document.recognized_figure_count is None


async def test_long_document_chunks_without_external_tokenizer_data(
    client: AsyncClient, vector_store: RecordingVectorStore
) -> None:
    long_text = "\n".join(f"Section {index}. {MANUAL_TEXT}" for index in range(20))

    job = await ingest(client, payload(text=long_text))

    assert job["status"] == "completed"
    assert job["chunk_count"] > 1
    assert len(vector_store.nodes) == job["chunk_count"]


async def test_canonical_rows_are_collection_scoped(
    client: AsyncClient, repository: InMemoryRepository
) -> None:
    await ingest(client, payload(collection_id="first", text="First collection text."))
    await ingest(
        client, payload(collection_id="second", text="Second collection text.")
    )

    first_sections = await repository.list_document_sections("first", "example-manual")
    second_sections = await repository.list_document_sections(
        "second", "example-manual"
    )
    first_blocks = await repository.list_document_blocks("first", "example-manual")
    second_blocks = await repository.list_document_blocks("second", "example-manual")

    assert first_sections and second_sections
    assert first_sections[0].document_id != second_sections[0].document_id
    assert [block.text for block in first_blocks] == ["First collection text."]
    assert [block.text for block in second_blocks] == ["Second collection text."]
    assert await repository.list_document_blocks("third", "example-manual") == []


async def test_repeated_ingest_of_unchanged_content_is_idempotent(
    client: AsyncClient, vector_store: RecordingVectorStore
) -> None:
    first = await ingest(client, payload())
    node_ids = set(vector_store.nodes)

    second = await ingest(client, payload())

    assert second["unchanged"] is True
    assert second["status"] == "completed"
    assert second["chunk_count"] == first["chunk_count"]
    assert "document_id" not in first
    assert "document_id" not in second
    assert set(vector_store.nodes) == node_ids
    assert vector_store.deleted_refs == []
    assert vector_store.clear_count == 0


async def test_changed_content_replaces_only_that_document(
    client: AsyncClient,
    repository: InMemoryRepository,
    vector_store: RecordingVectorStore,
) -> None:
    original = await ingest(client, payload())
    await ingest(client, payload(collection_id="other-manual"))
    other_nodes = {
        node_id
        for node_id, node in vector_store.nodes.items()
        if node.metadata["collection_id"] == "other-manual"
    }

    updated = await ingest(
        client, payload(text=f"{MANUAL_TEXT} Replace the supply sensor.")
    )

    assert updated["unchanged"] is False
    assert vector_store.deleted_refs == [original["job_id"]]
    assert vector_store.clear_count == 0
    assert other_nodes <= set(vector_store.nodes)
    current = repository.documents[("example-collection", "example-manual")]
    blocks = await repository.list_document_blocks(
        "example-collection", "example-manual"
    )
    assert str(current.current_revision_id) == updated["job_id"]
    assert {str(block.revision_id) for block in blocks} == {updated["job_id"]}


async def test_force_reindex_reembeds_unchanged_content(
    client: AsyncClient, vector_store: RecordingVectorStore
) -> None:
    original = await ingest(client, payload())

    job = await ingest(client, payload(force_reindex=True))

    assert job["unchanged"] is False
    assert vector_store.deleted_refs == [original["job_id"]]


async def test_failed_reindex_leaves_the_current_revision_unchanged(
    client: AsyncClient,
    repository: InMemoryRepository,
    vector_store: RecordingVectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = await ingest(client, payload())
    original_nodes = dict(vector_store.nodes)
    original_sections = dict(repository.sections)
    original_blocks = dict(repository.blocks)
    runtime = client.app.state.runtime  # type: ignore[attr-defined]
    pipeline_type = type(runtime.ingestion._pipeline)  # noqa: SLF001
    original_arun = pipeline_type.arun

    async def fail_embedding(
        _pipeline: object, *_args: object, **_kwargs: object
    ) -> None:
        raise RuntimeError("embedding endpoint unreachable")

    monkeypatch.setattr(pipeline_type, "arun", fail_embedding)
    failed = await ingest(client, payload(force_reindex=True))

    assert failed["status"] == "failed"
    current = repository.documents[("example-collection", "example-manual")]
    assert str(current.current_revision_id) == original["job_id"]
    assert current.chunk_count == original["chunk_count"]
    assert vector_store.nodes == original_nodes
    assert repository.sections == original_sections
    assert repository.blocks == original_blocks

    monkeypatch.setattr(pipeline_type, "arun", original_arun)
    recovered = await ingest(client, payload())

    assert recovered["status"] == "completed"
    assert recovered["unchanged"] is True
    assert recovered["chunk_count"] > 0
    assert vector_store.nodes == original_nodes


async def test_failed_initial_ingest_publishes_no_document(
    client: AsyncClient,
    repository: InMemoryRepository,
    vector_store: RecordingVectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = client.app.state.runtime  # type: ignore[attr-defined]
    pipeline_type = type(runtime.ingestion._pipeline)  # noqa: SLF001
    original_arun = pipeline_type.arun

    async def fail_embedding(
        _pipeline: object, *_args: object, **_kwargs: object
    ) -> None:
        raise RuntimeError("embedding endpoint unreachable")

    monkeypatch.setattr(pipeline_type, "arun", fail_embedding)
    failed = await ingest(client, payload())

    assert failed["status"] == "failed"
    assert ("example-collection", "example-manual") not in repository.documents
    assert vector_store.nodes == {}

    monkeypatch.setattr(pipeline_type, "arun", original_arun)
    recovered = await ingest(client, payload())

    assert recovered["status"] == "completed"
    assert recovered["unchanged"] is False
    assert recovered["chunk_count"] > 0
    assert vector_store.nodes


async def test_invalid_search_section_provenance_prevents_activation(
    client: AsyncClient,
    repository: InMemoryRepository,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = client.app.state.runtime  # type: ignore[attr-defined]
    pipeline_type = type(runtime.ingestion._pipeline)  # noqa: SLF001
    original_arun = runtime.ingestion._pipeline.arun  # noqa: SLF001

    async def corrupt_section_id(
        _pipeline: object, *args: Any, **kwargs: Any
    ) -> list[Any]:
        nodes = await original_arun(*args, **kwargs)
        nodes[0].metadata["section_id"] = str(uuid4())
        return nodes

    monkeypatch.setattr(pipeline_type, "arun", corrupt_section_id)
    failed = await ingest(client, payload())

    assert failed["status"] == "failed"
    assert "canonical block section" in caplog.text
    assert ("example-collection", "example-manual") not in repository.documents


async def test_callers_cannot_override_content_change_detection(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/v1/documents/text",
        json=payload(checksum="caller-controlled"),
        headers=auth_headers(),
    )

    assert response.status_code == 422


async def test_page_and_section_metadata_are_preserved(
    client: AsyncClient,
    repository: InMemoryRepository,
    vector_store: RecordingVectorStore,
) -> None:
    await ingest(client, payload(page=151, section="6.5.2 Pre-use check"))

    document = repository.documents[("example-collection", "example-manual")]
    assert document.page == 151
    assert document.section == "6.5.2 Pre-use check"
    node = next(iter(vector_store.nodes.values()))
    assert node.metadata["page"] == 151
    assert node.metadata["section"] == "6.5.2 Pre-use check"


async def test_arbitrary_metadata_round_trips(
    client: AsyncClient,
    repository: InMemoryRepository,
    vector_store: RecordingVectorStore,
) -> None:
    await ingest(client, payload(metadata={"language": "en", "revision": 19}))

    assert repository.documents[("example-collection", "example-manual")].metadata == {
        "language": "en",
        "revision": 19,
    }
    node = next(iter(vector_store.nodes.values()))
    assert node.metadata["language"] == "en"


@pytest.mark.parametrize(
    "reserved_key", ["collection_id", "section_ref", "_node_content"]
)
async def test_metadata_may_not_overwrite_service_owned_keys(
    client: AsyncClient, reserved_key: str
) -> None:
    response = await client.post(
        "/v1/documents/text",
        json=payload(metadata={reserved_key: "caller-controlled"}),
        headers=auth_headers(),
    )
    assert response.status_code == 422
    assert reserved_key in response.json()["detail"]


async def test_invalid_collection_identifiers_are_rejected(client: AsyncClient) -> None:
    for collection_id in ("", "Example Manual", "../escape", "x" * 64):
        response = await client.post(
            "/v1/documents/text",
            json=payload(collection_id=collection_id),
            headers=auth_headers(),
        )
        assert response.status_code == 422, collection_id


async def test_oversized_documents_are_rejected(client: AsyncClient) -> None:
    limit = client.app.state.settings.max_document_bytes
    response = await client.post(
        "/v1/documents/text",
        json=payload(text="x" * (limit + 1)),
        headers=auth_headers(),
    )
    assert response.status_code == 413


async def test_unknown_fields_are_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/documents/text",
        json=payload(collection="example-collection"),
        headers=auth_headers(),
    )
    assert response.status_code == 422
