"""Job lifecycle, durability across restarts, and readiness behaviour."""

import asyncio
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from app.config import Settings
from app.main import EmbeddingModelMismatchError, create_app
from app.models import TextDocumentRequest
from app.repository import EmbeddingState, JobRecord
from tests.conftest import auth_headers, ingest
from tests.fakes import InMemoryRepository, RecordingVectorStore

DOCUMENT = {
    "collection_id": "example-collection",
    "external_id": "example-manual",
    "text": "Calibrate the inlet pressure sensor.",
}


async def test_unknown_job_returns_404(client: AsyncClient) -> None:
    response = await client.get(f"/v1/jobs/{uuid4()}", headers=auth_headers())
    assert response.status_code == 404


async def test_job_transitions_are_persisted(
    client: AsyncClient, repository: InMemoryRepository
) -> None:
    job = await ingest(client, DOCUMENT)

    stored = repository.jobs[UUID(job["job_id"])]
    assert stored.status == "completed"
    assert stored.chunk_count == job["chunk_count"]
    assert job["created_at"] and job["updated_at"]


async def test_concurrent_text_submissions_reuse_the_active_job(
    client: AsyncClient,
    repository: InMemoryRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestion = client.app.state.runtime.ingestion  # type: ignore[attr-defined]
    scheduled: list[UUID] = []
    monkeypatch.setattr(
        ingestion,
        "_spawn",
        lambda job_id, _start: scheduled.append(job_id),
    )
    request = TextDocumentRequest(**DOCUMENT)

    first, second = await asyncio.gather(
        ingestion.submit(request), ingestion.submit(request)
    )

    assert first.job_id == second.job_id
    assert scheduled == [first.job_id]
    assert list(repository.jobs) == [first.job_id]


async def test_job_status_survives_a_restart(
    settings: Settings, runtime_factory, repository: InMemoryRepository
) -> None:
    app = create_app(settings, runtime_factory=runtime_factory)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://seshat"
        ) as http_client:
            http_client.app = app  # type: ignore[attr-defined]
            job = await ingest(http_client, DOCUMENT)

    # A second process starting against the same durable state.
    restarted = create_app(settings, runtime_factory=runtime_factory)
    async with restarted.router.lifespan_context(restarted):
        transport = ASGITransport(app=restarted)
        async with AsyncClient(
            transport=transport, base_url="http://seshat"
        ) as http_client:
            response = await http_client.get(
                f"/v1/jobs/{job['job_id']}", headers=auth_headers()
            )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["chunk_count"] == job["chunk_count"]


async def test_interrupted_jobs_are_reconciled_on_startup(
    settings: Settings, runtime_factory, repository: InMemoryRepository
) -> None:
    interrupted = JobRecord(
        job_id=uuid4(),
        collection_id="example-collection",
        external_id="example-manual",
        status="processing",
    )
    await repository.create_job(interrupted)

    app = create_app(settings, runtime_factory=runtime_factory)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://seshat"
        ) as http_client:
            response = await http_client.get(
                f"/v1/jobs/{interrupted.job_id}", headers=auth_headers()
            )

    body = response.json()
    assert body["status"] == "failed"
    assert "restart" in body["detail"]


async def test_interrupted_text_projection_is_discarded_on_startup(
    settings: Settings,
    runtime_factory,
    repository: InMemoryRepository,
    vector_store: RecordingVectorStore,
) -> None:
    interrupted = JobRecord(
        job_id=uuid4(),
        collection_id="example-collection",
        external_id="example-manual",
        status="processing",
    )
    await repository.create_job(interrupted)
    staged = TextNode(text="private staged text")
    staged.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
        node_id=str(interrupted.job_id)
    )
    vector_store.add([staged])

    app = create_app(settings, runtime_factory=runtime_factory)
    async with app.router.lifespan_context(app):
        pass

    assert vector_store.nodes == {}
    assert vector_store.deleted_refs == [str(interrupted.job_id)]


async def test_failed_ingestion_is_reported_on_the_job(
    client: AsyncClient, repository: InMemoryRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = client.app.state.runtime  # type: ignore[attr-defined]

    async def explode(*_args, **_kwargs):
        raise RuntimeError("embedding endpoint unreachable")

    monkeypatch.setattr(runtime.ingestion, "_index", explode)

    response = await client.post(
        "/v1/documents/text", json=DOCUMENT, headers=auth_headers()
    )
    job_id = response.json()["job_id"]
    await runtime.ingestion.wait_for_pending()

    job = (await client.get(f"/v1/jobs/{job_id}", headers=auth_headers())).json()
    assert job["status"] == "failed"
    assert "failed unexpectedly" in job["detail"]
    assert "embedding endpoint unreachable" not in job["detail"]


async def test_late_failure_cannot_remove_an_already_current_revision(
    client: AsyncClient,
    repository: InMemoryRepository,
    vector_store: RecordingVectorStore,
) -> None:
    completed = await ingest(client, DOCUMENT)
    job_id = UUID(completed["job_id"])
    before = dict(vector_store.nodes)
    ingestion = client.app.state.runtime.ingestion  # type: ignore[attr-defined]

    await ingestion._record_failure(  # noqa: SLF001 - lifecycle race regression
        job_id,
        DOCUMENT["collection_id"],
        DOCUMENT["external_id"],
        RuntimeError("late worker failure"),
    )

    assert repository.jobs[job_id].status == "completed"
    assert vector_store.nodes == before
    assert str(job_id) not in vector_store.deleted_refs


async def test_readiness_reports_database_failures(
    client: AsyncClient, repository: InMemoryRepository
) -> None:
    assert (await client.get("/health/ready")).json()["status"] == "ready"

    repository.probe_error = RuntimeError("connection refused")
    response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert (await client.get("/health/live")).status_code == 200


async def test_embedding_state_is_recorded_on_first_start(
    client: AsyncClient, repository: InMemoryRepository, settings: Settings
) -> None:
    assert repository.embedding_state == EmbeddingState(
        model_name=settings.embedding_model, model_dim=settings.embedding_dimension
    )


async def test_startup_fails_when_the_indexed_embedding_model_changed(
    settings: Settings, runtime_factory, repository: InMemoryRepository
) -> None:
    await repository.set_embedding_state(
        EmbeddingState(model_name="other-model", model_dim=12)
    )

    app = create_app(settings, runtime_factory=runtime_factory)
    with pytest.raises(EmbeddingModelMismatchError, match="does not match"):
        async with app.router.lifespan_context(app):
            pass
