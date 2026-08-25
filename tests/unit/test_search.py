"""Search: collection isolation, filtering, citations, and bounds."""

import json
from uuid import uuid4

from httpx import AsyncClient
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import FilterOperator

from app.parsing import DocumentSegment
from tests.conftest import auth_headers, ingest
from tests.fakes import (
    DeterministicEmbedding,
    InMemoryRepository,
    RecordingConverter,
    RecordingVectorStore,
)

EXAMPLE_TEXT = (
    "Calibrate the inlet pressure sensor during preventive maintenance. "
    "A persistent alarm means the outlet valve needs replacement."
)
OTHER_TEXT = (
    "Calibrate the supply pressure sensor on the backup unit. "
    "A persistent alarm means the battery pack needs replacement."
)


async def seed_two_collections(client: AsyncClient) -> None:
    await ingest(
        client,
        {
            "collection_id": "example-collection",
            "external_id": "example-manual",
            "title": "Example service manual",
            "source_type": "manual",
            "source_uri": "file:///data/example-manual.txt",
            "section": "6.5.2 Pre-use check",
            "page": 151,
            "text": EXAMPLE_TEXT,
        },
    )
    await ingest(
        client,
        {
            "collection_id": "other-manual",
            "external_id": "backup-unit-manual",
            "title": "Backup unit manual",
            "source_type": "manual",
            "text": OTHER_TEXT,
        },
    )


async def search(client: AsyncClient, body: dict) -> dict:
    response = await client.post("/v1/search", json=body, headers=auth_headers())
    response.raise_for_status()
    return response.json()


async def upload_pdf(client: AsyncClient, *, external_id: str) -> None:
    accepted = await client.post(
        "/v1/documents/file",
        files={"file": ("service.pdf", b"%PDF-search-provenance", "application/pdf")},
        data={"collection_id": "example-collection", "external_id": external_id},
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


async def test_search_never_returns_another_collection(
    client: AsyncClient, vector_store: RecordingVectorStore
) -> None:
    await seed_two_collections(client)

    result = await search(
        client,
        {"query": "pressure sensor alarm", "collection_ids": ["example-collection"]},
    )

    assert result["items"], "the requested collection should match"
    assert {item["collection_id"] for item in result["items"]} == {"example-collection"}
    assert all(
        "backup unit" not in item["text"].lower() for item in result["items"]
    )
    # The fake store ignores metadata filters, so isolation here is the service's
    # own post-retrieval guard rather than backend cooperation.
    assert len(vector_store.nodes) > len(result["items"])


async def test_search_can_span_several_named_collections(client: AsyncClient) -> None:
    await seed_two_collections(client)

    result = await search(
        client,
        {
            "query": "pressure sensor alarm",
            "collection_ids": ["example-collection", "other-manual"],
        },
    )

    assert {item["collection_id"] for item in result["items"]} == {
        "example-collection",
        "other-manual",
    }


async def test_search_sends_a_collection_filter_to_the_backend(
    client: AsyncClient, vector_store: RecordingVectorStore
) -> None:
    await seed_two_collections(client)
    await search(
        client, {"query": "pressure", "collection_ids": ["example-collection"]}
    )

    applied = vector_store.queries[-1].filters
    assert applied is not None
    collection_filter = next(f for f in applied.filters if f.key == "collection_id")
    assert collection_filter.operator == FilterOperator.IN
    assert collection_filter.value == ["example-collection"]
    current_filter = next(f for f in applied.filters if f.key == "projection_state")
    assert current_filter.operator == FilterOperator.EQ
    assert current_filter.value == "current"


async def test_search_drops_a_staged_revision_even_if_the_backend_returns_it(
    client: AsyncClient,
    vector_store: RecordingVectorStore,
    embed_model: DeterministicEmbedding,
) -> None:
    await seed_two_collections(client)
    staged = TextNode(
        text="battery battery battery private replacement",
        metadata={
            "collection_id": "example-collection",
            "external_id": "example-manual",
            "revision_id": str(uuid4()),
            "projection_state": "staged",
        },
        embedding=embed_model.get_text_embedding("battery battery battery"),
    )
    vector_store.add([staged])

    result = await search(
        client, {"query": "battery", "collection_ids": ["example-collection"]}
    )

    assert all("private replacement" not in item["text"] for item in result["items"])


async def test_unchanged_content_refreshes_search_metadata_without_reembedding(
    client: AsyncClient,
    repository: InMemoryRepository,
    vector_store: RecordingVectorStore,
) -> None:
    await ingest(
        client,
        {
            "collection_id": "example-collection",
            "external_id": "example-manual",
            "title": "Old title",
            "source_uri": "https://example.test/old",
            "page": 4,
            "section": "Old section",
            "metadata": {"audience": "old"},
            "text": EXAMPLE_TEXT,
        },
    )
    node_ids = set(vector_store.nodes)

    repeated = await ingest(
        client,
        {
            "collection_id": "example-collection",
            "external_id": "example-manual",
            "title": "Current title",
            "source_uri": "https://example.test/current",
            "page": 7,
            "section": "Current section",
            "metadata": {"audience": "current"},
            "text": EXAMPLE_TEXT,
        },
    )
    result = await search(
        client, {"query": "pressure sensor", "collection_ids": ["example-collection"]}
    )

    assert repeated["unchanged"] is True
    assert set(vector_store.nodes) == node_ids
    assert result["items"][0]["title"] == "Current title"
    assert result["items"][0]["source_uri"] == "https://example.test/current"
    assert result["items"][0]["page"] == 7
    assert result["items"][0]["section"] == "Current section"
    assert result["items"][0]["citations"][0]["page"] == 7
    assert result["items"][0]["citations"][0]["section"] == "Current section"
    assert result["items"][0]["metadata"]["audience"] == "current"
    assert result["items"][0]["text"] == (
        "[Current title > Current section - page 7]\n" + EXAMPLE_TEXT
    )
    assert "Old title" not in result["items"][0]["text"]
    blocks = await repository.list_document_blocks(
        "example-collection", "example-manual"
    )
    assert {(block.page_start, block.page_end) for block in blocks} == {(7, 7)}


async def test_search_requires_at_least_one_collection(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/search",
        json={"query": "pressure", "collection_ids": []},
        headers=auth_headers(),
    )
    assert response.status_code == 422

    response = await client.post(
        "/v1/search", json={"query": "pressure"}, headers=auth_headers()
    )
    assert response.status_code == 422


async def test_results_carry_document_metadata_and_citations(
    client: AsyncClient,
) -> None:
    await seed_two_collections(client)

    result = await search(
        client,
        {"query": "pressure sensor alarm", "collection_ids": ["example-collection"]},
    )
    item = result["items"][0]

    assert item["external_id"] == "example-manual"
    assert item["title"] == "Example service manual"
    assert item["source_type"] == "manual"
    assert item["source_uri"] == "file:///data/example-manual.txt"
    assert item["page"] == 151
    assert item["section"] == "6.5.2 Pre-use check"
    assert "document_id" not in item
    assert "chunk_id" not in item
    assert item["section_ref"] is None
    assert item["section_path"] is None
    assert item["updated_at"]
    citation = item["citations"][0]
    assert citation["label"] == "Example service manual"
    assert citation["locator"] == "section:6.5.2 Pre-use check"
    assert citation["page"] == 151


async def test_unlocated_result_cites_only_its_source(client: AsyncClient) -> None:
    await ingest(
        client,
        {
            "collection_id": "example-collection",
            "external_id": "field-notes",
            "text": "Battery pressure observations.",
        },
    )

    result = await search(
        client,
        {"query": "battery pressure", "collection_ids": ["example-collection"]},
    )

    item = result["items"][0]
    citation = item["citations"][0]
    assert citation["label"] == "field-notes"
    assert citation["locator"] is None
    assert citation["page"] is None
    assert citation["section"] is None
    assert citation["section_ref"] is None
    assert citation["section_path"] is None
    assert "document_id" not in item
    assert "chunk_id" not in item


async def test_search_section_reference_opens_the_same_scan_scope(
    client: AsyncClient,
    converter: RecordingConverter,
    repository: InMemoryRepository,
    vector_store: RecordingVectorStore,
) -> None:
    converter.segments = [
        DocumentSegment(
            text="Battery replacement procedure.",
            page=4,
            page_end=5,
            section_path=("Maintenance", "Battery"),
            section_ref="#/groups/battery",
        ),
        DocumentSegment(
            text="Valve inspection procedure.",
            page=8,
            section_path=("Maintenance", "Valve"),
            section_ref="#/groups/valve",
        ),
    ]
    await upload_pdf(client, external_id="structured-manual")

    result = await search(
        client,
        {
            "query": "battery",
            "collection_ids": ["example-collection"],
            "filters": {"external_id": ["structured-manual"]},
        },
    )
    item = next(
        result_item
        for result_item in result["items"]
        if "Battery replacement" in result_item["text"]
    )
    citation = item["citations"][0]

    assert item["section_path"] == ["Maintenance", "Battery"]
    assert item["page"] == 4
    assert item["page_end"] == 5
    assert citation["section_ref"] == item["section_ref"]
    assert citation["section_path"] == item["section_path"]
    assert citation["page_end"] == 5

    serialized = json.dumps(result)
    document = repository.documents[("example-collection", "structured-manual")]
    assert str(document.document_id) not in serialized
    assert str(document.current_revision_id) not in serialized
    assert all(
        str(section.section_id) not in serialized
        for section in repository.sections.values()
    )
    assert all(node_id not in serialized for node_id in vector_store.nodes)

    scanned = await client.post(
        "/v1/scan",
        json={
            "collection_id": item["collection_id"],
            "external_id": item["external_id"],
            "section_ref": item["section_ref"],
        },
        headers=auth_headers(),
    )
    scanned.raise_for_status()
    assert [block["text"] for block in scanned.json()["items"]] == [
        "Battery replacement procedure."
    ]


async def test_filters_narrow_results_within_a_collection(client: AsyncClient) -> None:
    await seed_two_collections(client)
    await ingest(
        client,
        {
            "collection_id": "example-collection",
            "external_id": "release-note",
            "source_type": "note",
            "text": "The pressure sensor alarm threshold changed in this release.",
        },
    )

    everything = await search(
        client,
        {"query": "pressure sensor alarm", "collection_ids": ["example-collection"]},
    )
    assert {item["source_type"] for item in everything["items"]} == {"manual", "note"}

    notes = await search(
        client,
        {
            "query": "pressure sensor alarm",
            "collection_ids": ["example-collection"],
            "filters": {"source_type": ["note"]},
        },
    )
    assert {item["source_type"] for item in notes["items"]} == {"note"}

    by_document = await search(
        client,
        {
            "query": "pressure sensor alarm",
            "collection_ids": ["example-collection"],
            "filters": {"external_id": ["release-note"]},
        },
    )
    assert {item["external_id"] for item in by_document["items"]} == {"release-note"}

    without_legacy = await search(
        client,
        {
            "query": "pressure sensor alarm",
            "collection_ids": ["example-collection"],
            "filters": {"exclude_external_id": ["example-manual"]},
        },
    )
    assert {item["external_id"] for item in without_legacy["items"]} == {"release-note"}


async def test_empty_results_report_a_warning(client: AsyncClient) -> None:
    result = await search(
        client, {"query": "pressure sensor", "collection_ids": ["example-collection"]}
    )

    assert result["items"] == []
    assert result["warnings"] == ["no_results"]
    assert result["stats"]["returned"] == 0


async def test_top_k_is_bounded_by_configuration(client: AsyncClient) -> None:
    await seed_two_collections(client)

    result = await search(
        client,
        {
            "query": "pressure sensor alarm",
            "collection_ids": ["example-collection"],
            "top_k": 1,
        },
    )
    assert len(result["items"]) == 1

    response = await client.post(
        "/v1/search",
        json={
            "query": "pressure",
            "collection_ids": ["example-collection"],
            "top_k": 500,
        },
        headers=auth_headers(),
    )
    assert response.status_code == 422
