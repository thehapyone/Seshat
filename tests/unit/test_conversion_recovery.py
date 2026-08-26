"""Durable asynchronous conversion: one submission per file, resumable across restarts.

Seshat hands an uploaded file to Docling as a task and then polls it. What has to
hold is that the task is submitted exactly once and recorded before the first
poll, so a duplicate upload or a Seshat restart continues the same remote
conversion instead of paying for a second one.
"""

import asyncio
import hashlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from app.config import Settings
from app.ingest import CONVERTER_CHANGED_DETAIL, document_uuid
from app.main import create_app
from app.parsing import (
    ConversionDeadlineExceededError,
    ConversionSubmissionError,
    ConversionTaskLostError,
    DocumentSegment,
)
from app.repository import JobRecord, StagedSourceObjectRecord
from app.storage import ORIGINAL_VARIANT, storage_key
from tests.conftest import auth_headers
from tests.fakes import (
    InMemoryRepository,
    RecordingAsyncConverter,
    RecordingVectorStore,
)
from tests.unit.test_uploads import COLLECTION, settle, sources, upload

pytestmark = pytest.mark.parametrize("converter_mode", ["async"], indirect=True)

PDF = {
    "filename": "service.pdf",
    "content": b"%PDF-1.7 binary",
    "media_type": "application/pdf",
}
PDF_EXTERNAL_ID = "source-1"
LOCATED = [
    DocumentSegment(
        text="Inspect the controller module every three years.",
        page=4,
        section="1 Preventive maintenance",
    ),
    DocumentSegment(
        text="The outlet valve alarm persists.", page=17, section="2 Alarms"
    ),
]


@asynccontextmanager
async def held(converter: RecordingAsyncConverter):
    """Keep the converter's conversion in flight for the body of the block.

    Released in a ``finally`` so a failing assertion cannot leave the background
    job -- and with it the application's shutdown -- waiting forever.
    """
    converter.released = asyncio.Event()
    try:
        yield
    finally:
        converter.released.set()


async def reach(condition, *, ticks: int = 200) -> None:
    """Yield to the event loop until *condition* holds, or fail the test.

    Used instead of a sleep so a test never depends on how many awaits the
    background job happens to take before it blocks on the converter.
    """
    for _ in range(ticks):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("The background conversion never reached the expected state.")


def pending_job(**overrides) -> JobRecord:
    """A job row as a crash mid-conversion would have left it behind."""
    base = {
        "job_id": uuid4(),
        "collection_id": COLLECTION,
        "external_id": PDF_EXTERNAL_ID,
        "status": "processing",
        "title": "service.pdf",
        "source_type": "pdf",
        "filename": "service.pdf",
        "media_type": "application/pdf",
        "converter_name": "docling",
        "converter_task_id": "task-99",
        "converter_submitted_at": datetime.now(UTC),
    }
    base.update(overrides)
    return JobRecord(**base)


async def persist_pending_upload(
    repository: InMemoryRepository, job: JobRecord
) -> None:
    await repository.create_upload_job(
        job,
        StagedSourceObjectRecord(
            job_id=job.job_id,
            document_id=document_uuid(job.collection_id, job.external_id),
            collection_id=job.collection_id,
            external_id=job.external_id,
            filename=PDF["filename"],
            media_type=PDF["media_type"],
            byte_size=len(PDF["content"]),
            checksum=hashlib.sha256(PDF["content"]).hexdigest(),
            storage_backend="memory",
            storage_key=storage_key(job.job_id, ORIGINAL_VARIANT),
        ),
    )


async def restart(
    settings: Settings, runtime_factory, job_ids: list[UUID]
) -> dict[UUID, dict]:
    """Start a fresh application over existing durable state and settle its work."""
    app = create_app(settings, runtime_factory=runtime_factory)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://seshat"
        ) as http_client:
            await app.state.runtime.ingestion.wait_for_pending()
            bodies = {}
            for job_id in job_ids:
                response = await http_client.get(
                    f"/v1/jobs/{job_id}", headers=auth_headers()
                )
                response.raise_for_status()
                bodies[job_id] = response.json()
    return bodies


async def test_the_converter_task_is_recorded_before_any_polling(
    client: AsyncClient,
    async_converter: RecordingAsyncConverter,
    repository: InMemoryRepository,
) -> None:
    async with held(async_converter):
        accepted = await upload(client, **PDF)
        job_id = UUID(accepted.json()["job_id"])
        await reach(lambda: bool(async_converter.awaited))

        stored = repository.jobs[job_id]
        assert stored.status == "processing"
        assert stored.converter_name == "docling"
        assert stored.converter_task_id == "task-1"
        assert stored.converter_submitted_at is not None
        assert stored.resumable is True

    assert (await settle(client, str(job_id)))["status"] == "completed"


async def test_cross_section_passage_keeps_its_page_range_on_the_async_path(
    client: AsyncClient,
    async_converter: RecordingAsyncConverter,
    vector_store: RecordingVectorStore,
) -> None:
    async_converter.segments = LOCATED

    job = await settle(client, (await upload(client, **PDF)).json()["job_id"])
    assert job["status"] == "completed"

    assert len(vector_store.nodes) == 1
    indexed = next(iter(vector_store.nodes.values()))
    assert indexed.metadata["page"] == 4
    assert indexed.metadata["page_end"] == 17
    assert "section" not in indexed.metadata

    found = await client.post(
        "/v1/search",
        json={"query": "controller module", "collection_ids": [COLLECTION]},
        headers=auth_headers(),
    )
    top = found.json()["items"][0]
    assert top["page"] == 4
    assert top["page_end"] == 17
    assert "section_path" not in top
    assert (await sources(client))[0]["page_count"] == 17


async def test_re_uploading_a_pending_file_reuses_the_job_and_the_docling_task(
    client: AsyncClient, async_converter: RecordingAsyncConverter
) -> None:
    async with held(async_converter):
        first = await upload(client, **PDF)
        await reach(lambda: bool(async_converter.awaited))
        second = await upload(client, **PDF)

        # The same job is handed back, still running, rather than a second one.
        assert second.status_code == 202
        assert second.json()["job_id"] == first.json()["job_id"]
        assert second.json()["status"] == "processing"

    job = await settle(client, first.json()["job_id"])

    assert job["status"] == "completed"
    # The point of the whole exercise: one file, one conversion.
    assert async_converter.submission_count == 1
    assert async_converter.awaited == ["task-1"]
    assert len(await sources(client)) == 1


async def test_an_indexed_upload_stays_idempotent(
    client: AsyncClient, async_converter: RecordingAsyncConverter
) -> None:
    """A completed upload still re-checks its checksum rather than reusing the job."""
    first = await settle(client, (await upload(client, **PDF)).json()["job_id"])
    second = await settle(client, (await upload(client, **PDF)).json()["job_id"])

    assert first["unchanged"] is False
    assert second["unchanged"] is True
    assert second["job_id"] != first["job_id"]
    assert len(await sources(client)) == 1


async def test_a_restart_resumes_the_persisted_task_without_resubmitting(
    settings: Settings,
    runtime_factory,
    repository: InMemoryRepository,
    async_converter: RecordingAsyncConverter,
    vector_store: RecordingVectorStore,
) -> None:
    async_converter.segments = LOCATED
    job = pending_job()
    await persist_pending_upload(repository, job)
    orphaned = TextNode(text="staged before the service stopped")
    orphaned.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
        node_id=str(job.job_id)
    )
    vector_store.add([orphaned])

    bodies = await restart(settings, runtime_factory, [job.job_id])

    assert bodies[job.job_id]["status"] == "completed"
    assert bodies[job.job_id]["chunk_count"] >= 1
    # Nothing was handed to the converter again; the existing task was followed.
    assert async_converter.submissions == []
    assert async_converter.awaited == ["task-99"]
    assert str(job.job_id) in vector_store.deleted_refs
    assert all(node.text != orphaned.text for node in vector_store.nodes.values())


async def test_a_restart_fails_only_the_jobs_it_cannot_resume(
    settings: Settings,
    runtime_factory,
    repository: InMemoryRepository,
    async_converter: RecordingAsyncConverter,
) -> None:
    resumable = pending_job()
    interrupted = JobRecord(
        job_id=uuid4(),
        collection_id=COLLECTION,
        external_id="example-manual",
        status="processing",
    )
    await persist_pending_upload(repository, resumable)
    await repository.create_job(interrupted)

    bodies = await restart(
        settings, runtime_factory, [resumable.job_id, interrupted.job_id]
    )

    assert bodies[resumable.job_id]["status"] == "completed"
    assert bodies[interrupted.job_id]["status"] == "failed"
    assert "restart" in bodies[interrupted.job_id]["detail"]


async def test_a_task_from_another_converter_fails_with_a_retryable_reason(
    settings: Settings,
    runtime_factory,
    repository: InMemoryRepository,
    async_converter: RecordingAsyncConverter,
) -> None:
    job = pending_job(converter_name="azure-document-intelligence")
    await repository.create_job(job)

    bodies = await restart(settings, runtime_factory, [job.job_id])

    assert bodies[job.job_id]["status"] == "failed"
    assert bodies[job.job_id]["detail"] == CONVERTER_CHANGED_DETAIL
    assert async_converter.awaited == []


@pytest.mark.parametrize("conversion_enabled", [False], indirect=True)
async def test_a_task_is_not_left_waiting_when_conversion_is_switched_off(
    settings: Settings,
    runtime_factory,
    repository: InMemoryRepository,
) -> None:
    job = pending_job()
    await repository.create_job(job)

    bodies = await restart(settings, runtime_factory, [job.job_id])

    assert bodies[job.job_id]["status"] == "failed"
    assert bodies[job.job_id]["detail"] == CONVERTER_CHANGED_DETAIL


async def test_an_expired_task_fails_retryably_and_the_next_upload_starts_fresh(
    settings: Settings,
    runtime_factory,
    repository: InMemoryRepository,
    async_converter: RecordingAsyncConverter,
) -> None:
    async_converter.result_error = ConversionTaskLostError(
        "The document converter no longer holds this conversion. Upload the file again."
    )
    job = pending_job()
    await repository.create_job(job)

    bodies = await restart(settings, runtime_factory, [job.job_id])
    assert bodies[job.job_id]["status"] == "failed"
    assert "Upload the file again" in bodies[job.job_id]["detail"]
    assert async_converter.submissions == []

    # Retrying is a new upload of the same bytes: a fresh job and a fresh task.
    async_converter.result_error = None
    app = create_app(settings, runtime_factory=runtime_factory)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://seshat"
        ) as http_client:
            http_client.app = app  # type: ignore[attr-defined]
            accepted = await upload(http_client, **PDF)
            retried = await settle(http_client, accepted.json()["job_id"])

    assert accepted.json()["job_id"] != str(job.job_id)
    assert retried["status"] == "completed"
    assert async_converter.submission_count == 1
    assert async_converter.awaited == ["task-99", "task-1"]


async def test_a_conversion_past_its_deadline_is_reported_as_such(
    settings: Settings,
    runtime_factory,
    repository: InMemoryRepository,
    async_converter: RecordingAsyncConverter,
) -> None:
    async_converter.result_error = ConversionDeadlineExceededError(
        "Document conversion did not finish within the conversion deadline."
    )
    job = pending_job(converter_submitted_at=datetime.now(UTC) - timedelta(hours=2))
    await repository.create_job(job)

    bodies = await restart(settings, runtime_factory, [job.job_id])

    assert bodies[job.job_id]["status"] == "failed"
    assert "conversion deadline" in bodies[job.job_id]["detail"]


async def test_a_refused_submission_fails_the_job_without_recording_a_task(
    client: AsyncClient,
    async_converter: RecordingAsyncConverter,
    repository: InMemoryRepository,
) -> None:
    async_converter.submit_error = ConversionSubmissionError(
        "The document converter refused this file."
    )

    accepted = await upload(client, **PDF)
    job = await settle(client, accepted.json()["job_id"])

    assert job["status"] == "failed"
    assert job["detail"] == "The document converter refused this file."
    stored = repository.jobs[UUID(accepted.json()["job_id"])]
    assert stored.converter_task_id is None
    assert async_converter.awaited == []


async def test_a_failure_to_record_the_task_fails_the_job_without_a_second_submission(
    client: AsyncClient,
    async_converter: RecordingAsyncConverter,
    repository: InMemoryRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The submit-then-record window is narrow but real, so its outcome is defined.

    Seshat cannot make the two atomic -- Docling mints the task id -- so if the
    record fails, the job fails rather than polling something it cannot resume.
    The orphaned Docling task is wasted work, but no second one is started here.
    """

    async def refuse(*_args, **_kwargs) -> None:
        raise RuntimeError("database write failed")

    monkeypatch.setattr(repository, "set_job_conversion_task", refuse)

    accepted = await upload(client, **PDF)
    job = await settle(client, accepted.json()["job_id"])

    assert job["status"] == "failed"
    assert "failed unexpectedly" in job["detail"]
    assert "database write failed" not in job["detail"]
    assert async_converter.submission_count == 1
    assert async_converter.awaited == []
    assert repository.jobs[UUID(accepted.json()["job_id"])].converter_task_id is None


async def test_text_uploads_never_reach_the_converter(
    client: AsyncClient, async_converter: RecordingAsyncConverter
) -> None:
    job = await settle(client, (await upload(client)).json()["job_id"])

    assert job["status"] == "completed"
    assert async_converter.submissions == []
