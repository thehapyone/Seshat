"""How document rows and ingest jobs collapse into one selectable source."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.repository import DocumentRecord, JobRecord, build_source_record, merge_sources

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


def document(external_id: str = "doc-1", *, chunk_count: int = 4, **overrides) -> DocumentRecord:
    return DocumentRecord(
        document_id=uuid4(),
        collection_id="example-collection",
        external_id=external_id,
        title=overrides.pop("title", "Service notes"),
        source_type=overrides.pop("source_type", "text"),
        source_uri="upload:doc-1",
        checksum="abc",
        version="",
        page=None,
        section="",
        metadata=overrides.pop("metadata", {"filename": "notes.txt", "media_type": "text/plain"}),
        chunk_count=chunk_count,
        created_at=overrides.pop("created_at", NOW),
        updated_at=overrides.pop("updated_at", NOW),
    )


def job(external_id: str = "doc-1", *, status: str = "completed", **overrides) -> JobRecord:
    return JobRecord(
        job_id=overrides.pop("job_id", uuid4()),
        collection_id="example-collection",
        external_id=external_id,
        status=status,  # type: ignore[arg-type]
        chunk_count=overrides.pop("chunk_count", 0),
        detail=overrides.pop("detail", None),
        created_at=overrides.pop("created_at", NOW),
        updated_at=overrides.pop("updated_at", NOW),
    )


def test_an_indexed_document_is_ready() -> None:
    record = build_source_record(document(), job())

    assert record.status == "ready"
    assert record.detail is None
    assert record.chunk_count == 4
    assert record.filename == "notes.txt"
    assert record.media_type == "text/plain"


def test_an_upload_with_no_document_row_yet_is_processing() -> None:
    record = build_source_record(None, job(status="accepted"))

    assert record.status == "processing"
    # The listing still needs a label before a title exists.
    assert record.title == "doc-1"
    assert record.source_type == "text"
    assert record.chunk_count == 0


def test_a_re_index_reads_as_processing_rather_than_as_an_empty_source() -> None:
    # _index resets chunk_count to zero before it rewrites the vectors.
    record = build_source_record(document(chunk_count=0), job(status="processing"))

    assert record.status == "processing"


def test_a_re_index_of_a_still_searchable_document_stays_ready() -> None:
    record = build_source_record(document(chunk_count=4), job(status="processing"))

    assert record.status == "ready"


def test_a_failed_job_surfaces_its_detail() -> None:
    record = build_source_record(None, job(status="failed", detail="The file is not valid UTF-8."))

    assert record.status == "failed"
    assert record.detail == "The file is not valid UTF-8."


def test_a_completed_job_that_indexed_nothing_is_failed() -> None:
    record = build_source_record(document(chunk_count=0), job(status="completed"))

    assert record.status == "failed"
    assert record.detail == "Indexing produced no searchable content."


def test_a_document_keeps_its_ready_status_when_no_job_survives() -> None:
    record = build_source_record(document(), None)

    assert record.status == "ready"


def test_merge_pairs_each_document_with_its_latest_job_newest_first() -> None:
    older = document("older", updated_at=NOW - timedelta(hours=2))
    newer = document("newer", updated_at=NOW)
    stale_job = job(
        "older",
        status="completed",
        created_at=NOW - timedelta(hours=3),
        updated_at=NOW - timedelta(hours=3),
    )
    latest_job = job(
        "older",
        status="failed",
        detail="boom",
        created_at=NOW - timedelta(hours=1),
        updated_at=NOW - timedelta(hours=1),
    )
    orphan = job("uploading", status="processing", updated_at=NOW + timedelta(minutes=1))

    records = merge_sources([older, newer], [stale_job, latest_job, orphan], limit=10)

    assert [record.external_id for record in records] == ["uploading", "newer", "older"]
    # "older" still has chunks, so the newer failed job does not hide it.
    assert records[2].status == "ready"
    assert records[0].status == "processing"


def test_merge_bounds_the_result() -> None:
    documents = [
        document(f"doc-{index}", updated_at=NOW + timedelta(seconds=index)) for index in range(10)
    ]

    records = merge_sources(documents, [], limit=3)

    assert [record.external_id for record in records] == ["doc-9", "doc-8", "doc-7"]
