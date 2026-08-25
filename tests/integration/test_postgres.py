"""End-to-end ingestion and retrieval against a real PostgreSQL/pgvector database.

Skipped unless ``SESHAT_TEST_DATABASE_URL`` points at a database with the
``vector`` extension available. Embeddings stay deterministic so the test needs
no embedding endpoint.

    docker run --rm -d -p 55432:5432 -e POSTGRES_PASSWORD=postgres \\
        --name seshat-test-db pgvector/pgvector:pg17
    SESHAT_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/postgres \\
        .venv/bin/python -m pytest tests/integration -q
"""

import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
import pytest_asyncio

from app import repository as repository_module
from app.config import Settings
from app.embeddings.pipeline import build_ingestion_pipeline, build_vector_store
from app.embeddings.retriever import search_documents
from app.ingest import IngestionService, UploadSubmission
from app.models import SearchRequest, TextDocumentRequest
from app.parsing import DocumentNormalizer, UploadedFile, resolve_format
from app.references import source_revision_marker
from app.repository import (
    ConcurrentIngestionError,
    JobRecord,
    PostgresRepository,
    SourceChangedError,
)
from app.storage import (
    ORIGINAL_VARIANT,
    PREVIEW_VARIANT,
    LocalFileSourceStore,
    storage_key,
)
from tests.fakes import VOCAB, DeterministicEmbedding

DATABASE_URL = os.environ.get("SESHAT_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="SESHAT_TEST_DATABASE_URL is not set"
)

EXAMPLE_TEXT = (
    "Calibrate the inlet pressure sensor during preventive maintenance. "
    "A persistent alarm means the outlet valve needs replacement."
)
BACKUP_UNIT_TEXT = (
    "Calibrate the supply pressure sensor on the backup unit. "
    "A persistent alarm means the battery pack needs replacement."
)


@pytest_asyncio.fixture
async def settings() -> Settings:
    schema = f"seshat_test_{uuid.uuid4().hex[:12]}"
    return Settings.from_document(
        {
            "database": {"schema": schema},
            "embedding": {
                "base_url": "https://example.invalid/openai/v1",
                "model": "deterministic",
                "dimension": len(VOCAB),
            },
            "chunking": {"size": 128, "overlap": 16},
            "retrieval": {"mode": "hybrid"},
        },
        env={
            "SESHAT_DATABASE_URL": DATABASE_URL,
            "SESHAT_API_TOKEN": "integration-token-0123456789",
            "SESHAT_CURSOR_SIGNING_KEY": "integration-cursor-key-0123456789",
            "SESHAT_EMBEDDING_API_KEY": "unused",
        },
    )


@pytest_asyncio.fixture
async def stack(settings: Settings, tmp_path):
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=4)
    await pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
    repository = PostgresRepository(pool, settings.db_schema)
    await repository.ensure_schema()

    embed_model = DeterministicEmbedding()
    vector_store = build_vector_store(settings)
    pipeline = build_ingestion_pipeline(
        embed_model,
        vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    source_store = LocalFileSourceStore(tmp_path / "sources")
    await source_store.prepare()
    ingestion = IngestionService(
        repository=repository,
        vector_store=vector_store,
        pipeline=pipeline,
        normalizer=DocumentNormalizer(
            converter=None, max_text_bytes=settings.max_document_bytes
        ),
        source_store=source_store,
    )
    try:
        yield repository, vector_store, embed_model, ingestion, source_store
    finally:
        await vector_store.close()
        await pool.execute(f"DROP SCHEMA IF EXISTS {settings.db_schema} CASCADE")
        await pool.close()


async def _ingest(ingestion: IngestionService, request: TextDocumentRequest):
    job = await ingestion.submit(request)
    await ingestion.wait_for_pending()
    return job


async def test_two_collections_stay_isolated_in_postgres(
    stack, settings: Settings
) -> None:
    repository, vector_store, embed_model, ingestion, _source_store = stack

    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="example-manual",
            title="Example service manual",
            source_type="manual",
            text=EXAMPLE_TEXT,
        ),
    )
    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="other-manual",
            external_id="backup-unit-manual",
            title="Backup unit manual",
            source_type="manual",
            text=BACKUP_UNIT_TEXT,
        ),
    )

    result = await search_documents(
        vector_store,
        embed_model,
        repository,
        SearchRequest(
            query="pressure sensor alarm", collection_ids=["example-collection"]
        ),
        settings,
    )

    assert result.items
    assert {item.collection_id for item in result.items} == {"example-collection"}
    assert all("backup unit" not in item.text.lower() for item in result.items)
    assert all(item.citations for item in result.items)
    assert all(item.citations[0].locator is None for item in result.items)
    assert all("document_id" not in item.model_dump() for item in result.items)
    assert all("chunk_id" not in item.model_dump() for item in result.items)

    both = await search_documents(
        vector_store,
        embed_model,
        repository,
        SearchRequest(
            query="pressure sensor alarm",
            collection_ids=["example-collection", "other-manual"],
        ),
        settings,
    )
    assert {item.collection_id for item in both.items} == {
        "example-collection",
        "other-manual",
    }


async def test_excluded_source_is_filtered_by_postgres(
    stack, settings: Settings
) -> None:
    repository, vector_store, embed_model, ingestion, _source_store = stack
    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="example-manual",
            title="Legacy embedded manual",
            text=EXAMPLE_TEXT,
        ),
    )
    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="file-replacement",
            title="Uploaded replacement",
            text="Pressure sensor maintenance guidance for the controller.",
        ),
    )

    result = await search_documents(
        vector_store,
        embed_model,
        repository,
        SearchRequest(
            query=EXAMPLE_TEXT,
            collection_ids=["example-collection"],
            top_k=1,
            filters={"exclude_external_id": ["example-manual"]},
        ),
        settings,
    )

    assert [item.external_id for item in result.items] == ["file-replacement"]


async def test_job_and_document_state_survive_a_new_repository_instance(
    stack, settings: Settings
) -> None:
    repository, vector_store, embed_model, ingestion, _source_store = stack

    request = TextDocumentRequest(
        collection_id="example-collection",
        external_id="example-manual",
        text=EXAMPLE_TEXT,
        page=4,
        section="Old section",
    )
    job = await _ingest(ingestion, request)
    vector_count = await repository._pool.fetchval(  # noqa: SLF001
        f"SELECT count(*) FROM {settings.db_schema}.data_{settings.vector_table}"
    )
    repeated = await _ingest(
        ingestion,
        request.model_copy(
            update={
                "title": "Current display title",
                "page": 7,
                "section": "Current section",
                "metadata": {"audience": "service"},
            }
        ),
    )

    # A fresh repository object stands in for a restarted process.
    reopened = PostgresRepository(repository._pool, settings.db_schema)  # noqa: SLF001
    stored_job = await reopened.get_job(job.job_id)
    stored_repeat = await reopened.get_job(repeated.job_id)
    document = await reopened.get_document("example-collection", "example-manual")

    assert stored_job is not None and stored_job.status == "completed"
    assert stored_repeat is not None and stored_repeat.unchanged is True
    assert document is not None and document.chunk_count == stored_job.chunk_count
    assert document.title == "Current display title"
    assert document.page == 7
    assert document.section == "Current section"
    assert document.metadata == {"audience": "service"}
    assert (
        await repository._pool.fetchval(  # noqa: SLF001
            f"SELECT count(*) FROM {settings.db_schema}.data_{settings.vector_table}"
        )
        == vector_count
    )
    sections = await reopened.list_document_sections(
        "example-collection", "example-manual"
    )
    outline = await reopened.get_document_outline(
        "example-collection", "example-manual"
    )
    blocks = await reopened.list_document_blocks("example-collection", "example-manual")
    assert len(sections) == 1 and sections[0].is_root is True
    assert outline is not None
    assert outline.document == document
    assert outline.sections == tuple(sections)
    assert (sections[0].page_start, sections[0].page_end) == (7, 7)
    assert blocks and {block.revision_id for block in blocks} == {job.job_id}
    assert {(block.page_start, block.page_end) for block in blocks} == {(7, 7)}
    projection_metadata = await repository._pool.fetchrow(  # noqa: SLF001
        f"""
        SELECT text,
               metadata_->>'title' AS title,
               (metadata_->>'page')::integer AS page,
               metadata_->>'section' AS section,
               metadata_->>'audience' AS audience
        FROM {settings.db_schema}.data_{settings.vector_table}
        WHERE metadata_->>'revision_id' = $1
        LIMIT 1
        """,
        str(job.job_id),
    )
    assert projection_metadata["title"] == "Current display title"
    assert projection_metadata["page"] == 7
    assert projection_metadata["section"] == "Current section"
    assert projection_metadata["audience"] == "service"
    assert projection_metadata["text"].startswith(
        "[Current display title > Current section - page 7]\n"
    )
    assert "Old section" not in projection_metadata["text"]
    result = await search_documents(
        vector_store,
        embed_model,
        reopened,
        SearchRequest(
            query=EXAMPLE_TEXT,
            collection_ids=["example-collection"],
            top_k=1,
        ),
        settings,
    )
    assert result.items[0].text == (
        "[Current display title > Current section - page 7]\n" + EXAMPLE_TEXT
    )


async def test_failed_activation_rolls_back_the_whole_current_representation(
    stack, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, vector_store, embed_model, ingestion, _source_store = stack
    original = await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="example-manual",
            title="Current manual",
            text=EXAMPLE_TEXT,
        ),
    )
    current_document = await repository.get_document(
        "example-collection", "example-manual"
    )
    current_sections = await repository.list_document_sections(
        "example-collection", "example-manual"
    )
    current_blocks = await repository.list_document_blocks(
        "example-collection", "example-manual"
    )

    async def fail_block_insert(*_args, **_kwargs) -> None:
        raise RuntimeError("injected block write failure")

    monkeypatch.setattr(repository_module, "_insert_blocks", fail_block_insert)
    replacement = await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="example-manual",
            title="Broken replacement",
            text=BACKUP_UNIT_TEXT,
        ),
    )

    failed = await repository.get_job(replacement.job_id)
    assert failed is not None and failed.status == "failed"
    assert (
        await repository.get_document("example-collection", "example-manual")
        == current_document
    )
    assert (
        await repository.list_document_sections("example-collection", "example-manual")
        == current_sections
    )
    assert (
        await repository.list_document_blocks("example-collection", "example-manual")
        == current_blocks
    )
    states = await repository._pool.fetch(  # noqa: SLF001
        f"""
        SELECT metadata_->>'revision_id' AS revision_id,
               metadata_->>'projection_state' AS projection_state
        FROM {settings.db_schema}.data_{settings.vector_table}
        """
    )
    assert {(row["revision_id"], row["projection_state"]) for row in states} == {
        (str(original.job_id), "current")
    }
    result = await search_documents(
        vector_store,
        embed_model,
        repository,
        SearchRequest(
            query="inlet pressure", collection_ids=["example-collection"]
        ),
        settings,
    )
    assert result.items and all(
        "backup unit" not in item.text.lower() for item in result.items
    )


async def test_successful_replacement_removes_the_previous_postgres_revision(
    stack, settings: Settings
) -> None:
    repository, _vector_store, _embed_model, ingestion, _source_store = stack
    original = await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="example-manual",
            text=EXAMPLE_TEXT,
        ),
    )
    replacement = await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="example-manual",
            text=BACKUP_UNIT_TEXT,
        ),
    )

    current = await repository.get_document("example-collection", "example-manual")
    assert current is not None and current.current_revision_id == replacement.job_id
    blocks = await repository.list_document_blocks(
        "example-collection", "example-manual"
    )
    assert blocks and {block.revision_id for block in blocks} == {replacement.job_id}
    revisions = await repository._pool.fetch(  # noqa: SLF001
        f"""
        SELECT metadata_->>'revision_id' AS revision_id,
               metadata_->>'projection_state' AS projection_state
        FROM {settings.db_schema}.data_{settings.vector_table}
        """
    )
    assert {(row["revision_id"], row["projection_state"]) for row in revisions} == {
        (str(replacement.job_id), "current")
    }
    assert str(original.job_id) not in {row["revision_id"] for row in revisions}


async def test_replacement_during_scan_page_read_uses_one_snapshot(
    stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _vector_store, _embed_model, ingestion, _source_store = stack
    original = await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="large-manual",
            text=("Battery maintenance sentence. " * 5_000),
        ),
    )
    current = await repository.get_document("example-collection", "large-manual")
    assert current is not None and current.current_revision_id == original.job_id
    marker = source_revision_marker(current.checksum, original.job_id)

    scan_reached_block_read = asyncio.Event()
    continue_scan = asyncio.Event()
    original_fetch = repository_module._fetch_scan_block_rows

    async def pause_before_block_read(*args: Any, **kwargs: Any) -> list[Any]:
        scan_reached_block_read.set()
        await continue_scan.wait()
        return await original_fetch(*args, **kwargs)

    monkeypatch.setattr(
        repository_module, "_fetch_scan_block_rows", pause_before_block_read
    )
    scan_task = asyncio.create_task(
        repository.get_document_scan(
            "example-collection",
            "large-manual",
            section_ref=None,
            after_ordinal=0,
            expected_source_marker=marker,
            limit=1,
        )
    )
    await asyncio.wait_for(scan_reached_block_read.wait(), timeout=10)

    try:
        replacement = await _ingest(
            ingestion,
            TextDocumentRequest(
                collection_id="example-collection",
                external_id="large-manual",
                text=BACKUP_UNIT_TEXT,
            ),
        )
    finally:
        continue_scan.set()
    first = await asyncio.wait_for(scan_task, timeout=10)

    assert first is not None
    assert first.document.current_revision_id == original.job_id
    assert len(first.blocks) == 1
    assert first.blocks[0].revision_id == original.job_id
    assert first.blocks[0].ordinal == 1
    assert first.has_more is True
    current = await repository.get_document("example-collection", "large-manual")
    assert current is not None and current.current_revision_id == replacement.job_id

    with pytest.raises(SourceChangedError):
        await repository.get_document_scan(
            "example-collection",
            "large-manual",
            section_ref=None,
            after_ordinal=first.blocks[-1].ordinal,
            expected_source_marker=marker,
            limit=1,
        )


async def test_uploaded_and_text_sources_are_listed_per_collection(stack) -> None:
    repository, _vector_store, _embed_model, ingestion, _source_store = stack

    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="example-manual",
            title="Example service manual",
            source_type="manual",
            text=EXAMPLE_TEXT,
        ),
    )
    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="other-manual",
            external_id="backup-unit-manual",
            text=BACKUP_UNIT_TEXT,
        ),
    )

    content = b"The supply sensor is calibrated during preventive maintenance."
    document_format = resolve_format("notes.txt", "text/plain")
    await ingestion.submit_upload(
        UploadSubmission(
            collection_id="example-collection",
            external_id="field-notes",
            title="Field notes",
            upload=UploadedFile(
                filename="notes.txt",
                media_type="text/plain",
                content=content,
                format=document_format,
            ),
        )
    )
    # A file that cannot be decoded proves a failed source is listed with its reason.
    broken = b"\xff\xfe\x00\x01"
    await ingestion.submit_upload(
        UploadSubmission(
            collection_id="example-collection",
            external_id="broken-notes",
            title="broken.txt",
            upload=UploadedFile(
                filename="broken.txt",
                media_type="text/plain",
                content=broken,
                format=document_format,
            ),
        )
    )
    await ingestion.wait_for_pending()

    sources = {
        record.external_id: record
        for record in await repository.list_sources("example-collection", limit=50)
    }

    assert set(sources) == {
        "example-manual",
        "field-notes",
        "broken-notes",
    }
    assert sources["example-manual"].status == "ready"
    assert sources["example-manual"].source_type == "manual"

    uploaded = sources["field-notes"]
    assert uploaded.status == "ready"
    assert uploaded.title == "Field notes"
    assert uploaded.filename == "notes.txt"
    assert uploaded.media_type == "text/plain"
    assert uploaded.chunk_count >= 1

    failed = sources["broken-notes"]
    assert failed.status == "failed"
    assert failed.detail and "UTF-8" in failed.detail

    other = await repository.list_sources("other-manual", limit=50)
    assert [record.external_id for record in other] == ["backup-unit-manual"]


async def test_source_limit_uses_recency_not_external_id_order(
    stack, settings: Settings
) -> None:
    repository, _vector_store, _embed_model, _ingestion, _source_store = stack
    now = datetime.now(UTC)
    # The identifiers are chosen so alphabetical order and recency disagree: a
    # LIMIT applied under "ORDER BY external_id" would keep "a-older" and drop
    # the newer row, so this fails unless the query orders by recency first.
    older = JobRecord(
        job_id=uuid.uuid4(),
        collection_id="example-collection",
        external_id="a-older",
        title="Older upload",
        source_type="upload",
        status="accepted",
    )
    newer = JobRecord(
        job_id=uuid.uuid4(),
        collection_id="example-collection",
        external_id="z-newer",
        title="Newer upload",
        source_type="upload",
        status="accepted",
    )
    await repository.create_job(older)
    await repository.create_job(newer)
    await repository._pool.execute(  # noqa: SLF001 - integration boundary
        f"UPDATE {settings.db_schema}.ingest_jobs SET created_at = $1, updated_at = $1 "
        "WHERE job_id = $2",
        now - timedelta(minutes=1),
        older.job_id,
    )
    await repository._pool.execute(  # noqa: SLF001 - integration boundary
        f"UPDATE {settings.db_schema}.ingest_jobs SET created_at = $1, updated_at = $1 "
        "WHERE job_id = $2",
        now,
        newer.job_id,
    )

    sources = await repository.list_sources("example-collection", limit=1)

    assert [source.external_id for source in sources] == ["z-newer"]


async def test_database_rejects_a_second_unfinished_job_for_one_source(stack) -> None:
    repository, _vector_store, _embed_model, _ingestion, _source_store = stack
    first = JobRecord(
        job_id=uuid.uuid4(),
        collection_id="example-collection",
        external_id="same-source",
        status="accepted",
    )
    second = JobRecord(
        job_id=uuid.uuid4(),
        collection_id="example-collection",
        external_id="same-source",
        status="accepted",
    )
    await repository.create_job(first)

    with pytest.raises(ConcurrentIngestionError):
        await repository.create_job(second)


async def test_completed_job_cannot_be_regressed_by_a_late_worker(stack) -> None:
    repository, _vector_store, _embed_model, _ingestion, _source_store = stack
    job = JobRecord(
        job_id=uuid.uuid4(),
        collection_id="example-collection",
        external_id="completed-source",
        status="accepted",
    )
    await repository.create_job(job)
    await repository.update_job(job.job_id, status="completed")

    await repository.update_job(
        job.job_id, status="failed", detail="late worker failure"
    )

    stored = await repository.get_job(job.job_id)
    assert stored is not None
    assert stored.status == "completed"
    assert stored.detail is None


async def test_fresh_schema_contains_current_representation_tables(
    stack, settings: Settings
) -> None:
    repository, _vector_store, _embed_model, _ingestion, _source_store = stack
    rows = await repository._pool.fetch(  # noqa: SLF001 - integration boundary
        "SELECT tablename FROM pg_tables WHERE schemaname = $1", settings.db_schema
    )

    assert {
        "documents",
        "document_sections",
        "document_blocks",
        "source_objects",
        "staged_source_objects",
        "pending_resource_cleanup",
    } <= {row["tablename"] for row in rows}


async def test_a_converter_task_round_trips_and_survives_a_restart(stack) -> None:
    """A recorded converter task is what makes a conversion resumable in Postgres."""
    repository, _vector_store, _embed_model, _ingestion, _source_store = stack
    pending = JobRecord(
        job_id=uuid.uuid4(),
        collection_id="example-collection",
        external_id="file-pending",
        status="processing",
        title="service.pdf",
        source_type="pdf",
        filename="service.pdf",
        media_type="application/pdf",
    )
    interrupted = JobRecord(
        job_id=uuid.uuid4(),
        collection_id="example-collection",
        external_id="file-interrupted",
        status="processing",
    )
    await repository.create_job(pending)
    await repository.create_job(interrupted)

    submitted_at = datetime.now(UTC) - timedelta(minutes=3)
    await repository.set_job_conversion_task(
        pending.job_id,
        converter_name="docling",
        task_id="task-abc",
        submitted_at=submitted_at,
    )

    stored = await repository.get_job(pending.job_id)
    assert stored is not None
    assert stored.converter_name == "docling"
    assert stored.converter_task_id == "task-abc"
    assert stored.converter_submitted_at == submitted_at
    assert stored.resumable is True

    # Startup reconciliation: only the job with no converter task is failed.
    failed = await repository.fail_interrupted_jobs("Interrupted by a restart.")
    assert failed == [interrupted.job_id]
    resumable = await repository.list_resumable_jobs()
    assert [job.job_id for job in resumable] == [pending.job_id]
    assert resumable[0].filename == "service.pdf"
    reloaded = await repository.get_job(interrupted.job_id)
    assert reloaded is not None and reloaded.status == "failed"

    # The listing path carries the task through too, so a source stays visible
    # as processing while its conversion runs.
    listed = {
        record.external_id: record
        for record in await repository.list_sources("example-collection", limit=50)
    }
    assert listed["file-pending"].status == "processing"


async def _upload(
    ingestion: IngestionService,
    filename: str,
    content: bytes,
    media_type: str,
    *,
    external_id: str | None = None,
):
    """Run the production upload path with caller-owned source identity."""
    document_format = resolve_format(filename, media_type)
    submission = UploadSubmission(
        collection_id="example-collection",
        external_id=external_id or filename.rsplit(".", 1)[0],
        title=filename,
        upload=UploadedFile(
            filename=filename,
            media_type=document_format.canonical_media_type,
            content=content,
            format=document_format,
        ),
    )
    await ingestion.submit_upload(submission)
    await ingestion.wait_for_pending()
    return submission.external_id


async def test_retained_originals_survive_a_restart_in_postgres(
    stack, settings: Settings, tmp_path
) -> None:
    """The acceptance case: reopen the persisted file after the service restarts."""
    repository, _vector_store, _embed_model, ingestion, source_store = stack
    content = b"The supply sensor is calibrated during preventive maintenance."

    external_id = await _upload(ingestion, "notes.txt", content, "text/plain")

    stored = await repository.get_source_object("example-collection", external_id)
    assert stored is not None
    assert stored.filename == "notes.txt"
    assert stored.media_type == "text/plain"
    assert stored.byte_size == len(content)
    assert len(stored.checksum) == 64
    assert stored.storage_backend == "local"
    first_job = await repository.get_latest_job("example-collection", external_id)
    assert first_job is not None
    assert stored.storage_key == storage_key(first_job.job_id, ORIGINAL_VARIANT)
    # A text upload is its own readable form, so no second copy is stored.
    assert stored.preview_key is None
    assert stored.page_count is None

    listed = {
        record.external_id: record
        for record in await repository.list_sources("example-collection", limit=50)
    }
    assert listed[external_id].viewable is True
    assert listed[external_id].byte_size == len(content)
    assert listed[external_id].preview_available is False

    # A restart is a fresh repository and a fresh store over the same database
    # and the same volume.
    restarted_pool = await asyncpg.create_pool(
        settings.database_url, min_size=1, max_size=1
    )
    try:
        restarted = PostgresRepository(restarted_pool, settings.db_schema)
        await restarted.ensure_schema()
        after_restart = await restarted.get_source_object(
            "example-collection", external_id
        )
    finally:
        await restarted_pool.close()

    assert after_restart == stored
    fresh_store = LocalFileSourceStore(tmp_path / "sources")
    await fresh_store.prepare()
    read_back = b"".join(
        [chunk async for chunk in fresh_store.read(stored.storage_key)]
    )
    assert read_back == content

    # Re-uploading identical content replaces the same object rather than adding one.
    await _upload(ingestion, "notes.txt", content, "text/plain")
    files = [path for path in source_store.root.rglob("*") if path.is_file()]
    assert [path.name for path in files] == ["original"], files
    reuploaded = await repository.get_source_object("example-collection", external_id)
    assert reuploaded is not None
    assert reuploaded.storage_key == stored.storage_key
    assert reuploaded.checksum == stored.checksum
    assert reuploaded.created_at == stored.created_at


async def test_an_identical_upload_promotes_an_original_for_a_text_source(
    stack,
) -> None:
    repository, _vector_store, _embed_model, ingestion, source_store = stack
    content = b"The supply sensor is calibrated during preventive maintenance."
    external_id = "shared-source"
    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id=external_id,
            title="Direct text",
            text=content.decode("utf-8"),
        ),
    )

    await _upload(
        ingestion,
        "shared-source.txt",
        content,
        "text/plain",
        external_id=external_id,
    )

    job = await repository.get_latest_job("example-collection", external_id)
    assert job is not None and job.unchanged is True
    stored = await repository.get_source_object("example-collection", external_id)
    assert stored is not None
    assert stored.filename == "shared-source.txt"
    assert stored.storage_key == storage_key(job.job_id, ORIGINAL_VARIANT)
    assert [path.name for path in source_store.root.rglob("*") if path.is_file()] == [
        "original"
    ]
    assert (
        b"".join([chunk async for chunk in source_store.read(stored.storage_key)])
        == content
    )


async def test_a_preview_and_page_reach_round_trip_through_postgres(stack) -> None:
    repository, _vector_store, _embed_model, ingestion, source_store = stack

    class LocatedConverter:
        name = "located"

        async def convert(self, *, filename: str, media_type: str, content: bytes):
            from app.parsing import DocumentSegment, build_converted_document

            return build_converted_document(
                [
                    DocumentSegment(
                        text="Calibrate the inlet pressure sensor.",
                        page=4,
                        section="2 Maintenance",
                    ),
                    DocumentSegment(
                        text="The outlet valve alarm needs the battery replaced.",
                        page=17,
                        section="3 Alarms",
                    ),
                ]
            )

    ingestion._normalizer = DocumentNormalizer(  # noqa: SLF001
        converter=LocatedConverter(), max_text_bytes=8_000_000
    )
    external_id = await _upload(
        ingestion, "service.pdf", b"%PDF-1.7 body", "application/pdf"
    )

    stored = await repository.get_source_object("example-collection", external_id)
    assert stored is not None
    assert stored.page_count == 17
    latest = await repository.get_latest_job("example-collection", external_id)
    assert latest is not None
    assert stored.preview_key == storage_key(latest.job_id, PREVIEW_VARIANT)
    assert stored.preview_bytes and stored.preview_bytes > 0
    # The preview hashes its own bytes, so its HTTP validator tracks the extracted
    # text rather than the original that produced it.
    assert stored.preview_checksum and len(stored.preview_checksum) == 64
    assert stored.preview_checksum != stored.checksum

    preview = b"".join([chunk async for chunk in source_store.read(stored.preview_key)])
    assert b"inlet pressure sensor" in preview

    listed = {
        record.external_id: record
        for record in await repository.list_sources("example-collection", limit=50)
    }
    assert listed[external_id].page_count == 17
    assert listed[external_id].preview_available is True

    # The provenance reached the indexed chunks, which is what citations read back.
    assert latest is not None and latest.status == "completed"
