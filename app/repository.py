"""Durable state: collection identity, document identity, and ingestion jobs."""

import hmac
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import asyncpg

from app.config import VECTOR_TABLE_NAME
from app.models import SEARCH_CONTEXT_METADATA_KEY, JobStatus, SourceStatus
from app.references import section_reference, source_revision_marker
from app.representation import DocumentBlock, DocumentSection
from app.search_text import context_header, contextual_search_text, search_body

ProvenanceMode = Literal["document", "block"]
CleanupResourceType = Literal["source_object", "vector_revision"]


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: UUID
    collection_id: str
    external_id: str
    title: str
    source_type: str
    source_uri: str
    checksum: str
    version: str
    page: int | None
    section: str
    provenance_mode: ProvenanceMode = "document"
    current_revision_id: UUID | None = None
    normalized_checksum: str = ""
    page_count: int | None = None
    recognized_section_count: int | None = None
    recognized_table_count: int | None = None
    recognized_figure_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class DocumentOutlineRecord:
    """One current document and its sections from the same database snapshot."""

    document: DocumentRecord
    sections: tuple[DocumentSection, ...]


@dataclass(frozen=True, slots=True)
class DocumentScanRecord:
    """One bounded page from a current canonical representation."""

    document: DocumentRecord
    sections: tuple[DocumentSection, ...]
    blocks: tuple[DocumentBlock, ...]
    has_more: bool


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: UUID
    collection_id: str
    external_id: str
    status: JobStatus
    title: str = ""
    source_type: str = "text"
    filename: str | None = None
    media_type: str | None = None
    document_id: UUID | None = None
    chunk_count: int = 0
    unchanged: bool = False
    detail: str | None = None
    # Persisted before polling so a restart resumes the same converter task.
    converter_name: str | None = None
    converter_task_id: str | None = None
    converter_submitted_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def resumable(self) -> bool:
        """Whether this job is waiting on remote work that outlived Seshat."""
        return bool(
            self.status in ("accepted", "processing")
            and self.converter_task_id
            and self.converter_submitted_at is not None
        )


@dataclass(frozen=True, slots=True)
class SourceObjectRecord:
    """The original and optional preview for one active source revision."""

    document_id: UUID
    collection_id: str
    external_id: str
    filename: str
    media_type: str
    byte_size: int
    checksum: str
    storage_backend: str
    storage_key: str
    preview_key: str | None = None
    preview_bytes: int | None = None
    # The preview can change when its original does not, for example after
    # converter reprocessing.
    preview_checksum: str | None = None
    page_count: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class StagedSourceObjectRecord:
    """Private job-scoped source bytes that are not yet readable by callers."""

    job_id: UUID
    document_id: UUID
    collection_id: str
    external_id: str
    filename: str
    media_type: str
    byte_size: int
    checksum: str
    storage_backend: str
    storage_key: str
    preview_key: str | None = None
    preview_bytes: int | None = None
    preview_checksum: str | None = None
    page_count: int | None = None


@dataclass(frozen=True, slots=True)
class ActivationResult:
    """Previous private resources that may be removed after activation commits."""

    previous_revision_id: UUID | None = None
    previous_source: SourceObjectRecord | None = None


@dataclass(frozen=True, slots=True)
class PendingCleanupRecord:
    """An external resource whose deletion must survive process restarts."""

    resource_type: CleanupResourceType
    resource_key: str
    storage_backend: str | None = None


class ConcurrentIngestionError(RuntimeError):
    """Raised when another process already accepted work for one source."""


class SourceChangedError(RuntimeError):
    """Raised when a cursor names a representation that is no longer current."""


class SectionNotFoundError(LookupError):
    """Raised when a public section reference is not current for its source."""


class InvalidScanPositionError(ValueError):
    """Raised when a cursor position falls outside its bound scan scope."""


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One selectable document source, as a UI needs to render it."""

    external_id: str
    title: str
    source_type: str
    status: SourceStatus
    chunk_count: int
    detail: str | None = None
    filename: str | None = None
    media_type: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    # Viewer state. ``viewable`` is false for normalized text submitted without
    # an uploaded original.
    viewable: bool = False
    byte_size: int | None = None
    page_count: int | None = None
    preview_available: bool = False


@dataclass(frozen=True, slots=True)
class EmbeddingState:
    model_name: str
    model_dim: int


def derive_source_status(
    document: DocumentRecord | None,
    job: JobRecord | None,
) -> tuple[SourceStatus, str | None]:
    """Collapse document and latest-job state into one status plus a detail.

    A document keeps its ``ready`` status while a replacement runs because the
    existing revision stays current until activation. A first ingest without a
    current document follows its latest job state.
    """
    if document is not None and document.chunk_count > 0:
        return "ready", None
    if job is not None and job.status in ("accepted", "processing"):
        return "processing", None
    if job is not None and job.status == "failed":
        return "failed", job.detail
    if job is None and document is None:  # pragma: no cover - defensive
        return "failed", None
    return "failed", "Indexing produced no searchable content."


def build_source_record(
    document: DocumentRecord | None,
    job: JobRecord | None,
    source_object: SourceObjectRecord | None = None,
) -> SourceRecord:
    """Merge a document row, its latest job, and any stored original into one record."""
    if document is None and job is None:  # pragma: no cover - defensive
        raise ValueError("A source needs either a document or a job.")
    status, detail = derive_source_status(document, job)
    external_id = document.external_id if document is not None else job.external_id  # type: ignore[union-attr]
    created_at = _earliest(
        document.created_at if document else None, job.created_at if job else None
    )
    updated_at = _latest(
        document.updated_at if document else None, job.updated_at if job else None
    )
    return SourceRecord(
        external_id=external_id,
        title=(document.title if document else job.title) or external_id,  # type: ignore[union-attr]
        source_type=document.source_type if document else job.source_type,  # type: ignore[union-attr]
        status=status,
        chunk_count=document.chunk_count if document else 0,
        detail=detail,
        filename=(
            _optional_text(document.metadata.get("filename"))
            if document is not None
            else job.filename  # type: ignore[union-attr]
        )
        or (source_object.filename if source_object is not None else None),
        media_type=(
            _optional_text(document.metadata.get("media_type"))
            if document is not None
            else job.media_type  # type: ignore[union-attr]
        )
        or (source_object.media_type if source_object is not None else None),
        created_at=created_at,
        updated_at=updated_at,
        viewable=source_object is not None,
        byte_size=source_object.byte_size if source_object is not None else None,
        page_count=source_object.page_count if source_object is not None else None,
        preview_available=bool(source_object is not None and source_object.preview_key),
    )


class Repository(ABC):
    """Storage contract used by the ingestion and search paths."""

    @abstractmethod
    async def ensure_collection(self, collection_id: str) -> None: ...

    @abstractmethod
    async def get_document(
        self, collection_id: str, external_id: str
    ) -> DocumentRecord | None: ...

    @abstractmethod
    async def list_current_documents(
        self, collection_ids: list[str]
    ) -> list[DocumentRecord]: ...

    @abstractmethod
    async def list_document_sections(
        self, collection_id: str, external_id: str
    ) -> list[DocumentSection]: ...

    @abstractmethod
    async def get_document_outline(
        self, collection_id: str, external_id: str
    ) -> DocumentOutlineRecord | None: ...

    @abstractmethod
    async def get_document_scan(
        self,
        collection_id: str,
        external_id: str,
        *,
        section_ref: str | None,
        after_ordinal: int | None,
        expected_source_marker: str | None,
        limit: int,
    ) -> DocumentScanRecord | None: ...

    @abstractmethod
    async def list_document_blocks(
        self, collection_id: str, external_id: str
    ) -> list[DocumentBlock]: ...

    @abstractmethod
    async def list_sources(
        self, collection_id: str, *, limit: int
    ) -> list[SourceRecord]: ...

    @abstractmethod
    async def create_upload_job(
        self, job: JobRecord, source: StagedSourceObjectRecord
    ) -> None: ...

    @abstractmethod
    async def set_staged_source_preview(
        self,
        job_id: UUID,
        *,
        preview_key: str | None,
        preview_bytes: int | None,
        preview_checksum: str | None,
        page_count: int | None,
    ) -> None: ...

    @abstractmethod
    async def get_staged_source_object(
        self, job_id: UUID
    ) -> StagedSourceObjectRecord | None: ...

    @abstractmethod
    async def discard_staged_source(
        self, job_id: UUID
    ) -> StagedSourceObjectRecord | None: ...

    @abstractmethod
    async def discard_failed_staged_sources(self) -> list[StagedSourceObjectRecord]: ...

    @abstractmethod
    async def activate_document(
        self,
        job_id: UUID,
        document: DocumentRecord,
        sections: tuple[DocumentSection, ...],
        blocks: tuple[DocumentBlock, ...],
    ) -> ActivationResult: ...

    @abstractmethod
    async def complete_unchanged(
        self, job_id: UUID, document: DocumentRecord
    ) -> StagedSourceObjectRecord | None: ...

    @abstractmethod
    async def enqueue_cleanup(self, record: PendingCleanupRecord) -> None: ...

    @abstractmethod
    async def list_pending_cleanup(self) -> list[PendingCleanupRecord]: ...

    @abstractmethod
    async def complete_cleanup(self, record: PendingCleanupRecord) -> None: ...

    @abstractmethod
    async def get_source_object(
        self, collection_id: str, external_id: str
    ) -> SourceObjectRecord | None: ...

    @abstractmethod
    async def create_job(self, record: JobRecord) -> None: ...

    @abstractmethod
    async def update_job(
        self,
        job_id: UUID,
        *,
        status: JobStatus,
        document_id: UUID | None = None,
        chunk_count: int | None = None,
        unchanged: bool | None = None,
        detail: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def set_job_conversion_task(
        self,
        job_id: UUID,
        *,
        converter_name: str,
        task_id: str,
        submitted_at: datetime,
    ) -> None: ...

    @abstractmethod
    async def get_job(self, job_id: UUID) -> JobRecord | None: ...

    @abstractmethod
    async def get_latest_job(
        self, collection_id: str, external_id: str
    ) -> JobRecord | None: ...

    @abstractmethod
    async def list_resumable_jobs(self) -> list[JobRecord]: ...

    @abstractmethod
    async def fail_interrupted_jobs(self, detail: str) -> list[UUID]: ...

    @abstractmethod
    async def get_embedding_state(self) -> EmbeddingState | None: ...

    @abstractmethod
    async def set_embedding_state(self, state: EmbeddingState) -> None: ...

    @abstractmethod
    async def probe(self) -> None: ...


def schema_ddl(schema: str) -> tuple[str, ...]:
    """Return idempotent DDL statements for *schema*.

    ``schema`` is validated as a SQL identifier in :mod:`app.config` before it
    reaches this module; it is interpolated because PostgreSQL does not accept
    identifiers as bind parameters.
    """
    return (
        f"CREATE SCHEMA IF NOT EXISTS {schema}",
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.collections (
            collection_id text PRIMARY KEY,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.documents (
            document_id uuid PRIMARY KEY,
            collection_id text NOT NULL REFERENCES {schema}.collections (collection_id),
            external_id text NOT NULL,
            current_revision_id uuid NOT NULL,
            title text NOT NULL DEFAULT '',
            source_type text NOT NULL DEFAULT 'text',
            source_uri text NOT NULL DEFAULT '',
            checksum text NOT NULL,
            normalized_checksum text NOT NULL,
            provenance_mode text NOT NULL
                CHECK (provenance_mode IN ('document', 'block')),
            version text NOT NULL DEFAULT '',
            page integer,
            section text NOT NULL DEFAULT '',
            page_count integer CHECK (page_count IS NULL OR page_count >= 0),
            recognized_section_count integer
                CHECK (recognized_section_count IS NULL OR recognized_section_count >= 0),
            recognized_table_count integer
                CHECK (recognized_table_count IS NULL OR recognized_table_count >= 0),
            recognized_figure_count integer
                CHECK (recognized_figure_count IS NULL OR recognized_figure_count >= 0),
            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            chunk_count integer NOT NULL CHECK (chunk_count > 0),
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (collection_id, external_id),
            UNIQUE (document_id, current_revision_id),
            UNIQUE (document_id, collection_id, external_id)
        )
        """,
        (
            f"CREATE INDEX IF NOT EXISTS documents_collection_idx "
            f"ON {schema}.documents (collection_id)"
        ),
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.ingest_jobs (
            job_id uuid PRIMARY KEY,
            collection_id text NOT NULL,
            external_id text NOT NULL,
            title text NOT NULL DEFAULT '',
            source_type text NOT NULL DEFAULT 'text',
            filename text,
            media_type text,
            document_id uuid,
            status text NOT NULL
                CHECK (status IN ('accepted', 'processing', 'completed', 'failed')),
            chunk_count integer NOT NULL DEFAULT 0,
            unchanged boolean NOT NULL DEFAULT false,
            detail text,
            converter_name text,
            converter_task_id text,
            converter_submitted_at timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (job_id, collection_id, external_id)
        )
        """,
        (
            f"CREATE INDEX IF NOT EXISTS ingest_jobs_status_idx "
            f"ON {schema}.ingest_jobs (status)"
        ),
        # Startup reads exactly this set to decide what to resume, so it is worth
        # a partial index rather than a scan of every job ever recorded.
        (
            f"CREATE INDEX IF NOT EXISTS ingest_jobs_resumable_idx "
            f"ON {schema}.ingest_jobs (status) WHERE converter_task_id IS NOT NULL"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS ingest_jobs_identity_idx "
            f"ON {schema}.ingest_jobs (collection_id, external_id, created_at DESC)"
        ),
        (
            f"CREATE UNIQUE INDEX IF NOT EXISTS ingest_jobs_one_unfinished_source_idx "
            f"ON {schema}.ingest_jobs (collection_id, external_id) "
            f"WHERE status IN ('accepted', 'processing')"
        ),
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.document_sections (
            section_id uuid PRIMARY KEY,
            document_id uuid NOT NULL
                REFERENCES {schema}.documents (document_id) ON DELETE CASCADE,
            revision_id uuid NOT NULL,
            parent_section_id uuid,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            title text NOT NULL,
            path jsonb NOT NULL,
            source_ref text,
            first_block_ordinal integer,
            last_block_ordinal integer,
            page_start integer,
            page_end integer,
            is_root boolean NOT NULL DEFAULT false,
            UNIQUE (document_id, ordinal),
            UNIQUE (document_id, section_id),
            FOREIGN KEY (document_id, revision_id)
                REFERENCES {schema}.documents (document_id, current_revision_id)
                ON DELETE CASCADE,
            FOREIGN KEY (document_id, parent_section_id)
                REFERENCES {schema}.document_sections (document_id, section_id)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.document_blocks (
            document_id uuid NOT NULL
                REFERENCES {schema}.documents (document_id) ON DELETE CASCADE,
            revision_id uuid NOT NULL,
            section_id uuid NOT NULL,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            kind text NOT NULL CHECK (kind IN ('text', 'table_part')),
            text text NOT NULL,
            page_start integer,
            page_end integer,
            table_id text,
            table_caption text NOT NULL DEFAULT '',
            table_header text NOT NULL DEFAULT '',
            part integer,
            parts integer,
            PRIMARY KEY (document_id, ordinal),
            FOREIGN KEY (document_id, section_id)
                REFERENCES {schema}.document_sections (document_id, section_id)
                ON DELETE CASCADE
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.source_objects (
            document_id uuid PRIMARY KEY
                REFERENCES {schema}.documents (document_id) ON DELETE CASCADE,
            collection_id text NOT NULL,
            external_id text NOT NULL,
            filename text NOT NULL,
            media_type text NOT NULL,
            byte_size bigint NOT NULL,
            checksum text NOT NULL,
            storage_backend text NOT NULL,
            storage_key text NOT NULL,
            preview_key text,
            preview_bytes bigint,
            preview_checksum text,
            page_count integer,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (collection_id, external_id),
            FOREIGN KEY (document_id, collection_id, external_id)
                REFERENCES {schema}.documents (document_id, collection_id, external_id)
                ON DELETE CASCADE
        )
        """,
        (
            f"CREATE INDEX IF NOT EXISTS source_objects_collection_idx "
            f"ON {schema}.source_objects (collection_id)"
        ),
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.staged_source_objects (
            job_id uuid PRIMARY KEY
                REFERENCES {schema}.ingest_jobs (job_id) ON DELETE CASCADE,
            document_id uuid NOT NULL,
            collection_id text NOT NULL,
            external_id text NOT NULL,
            filename text NOT NULL,
            media_type text NOT NULL,
            byte_size bigint NOT NULL,
            checksum text NOT NULL,
            storage_backend text NOT NULL,
            storage_key text NOT NULL,
            preview_key text,
            preview_bytes bigint,
            preview_checksum text,
            page_count integer,
            UNIQUE (collection_id, external_id, job_id),
            FOREIGN KEY (job_id, collection_id, external_id)
                REFERENCES {schema}.ingest_jobs (job_id, collection_id, external_id)
                ON DELETE CASCADE
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.pending_resource_cleanup (
            resource_type text NOT NULL
                CHECK (resource_type IN ('source_object', 'vector_revision')),
            resource_key text NOT NULL,
            storage_backend text,
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (resource_type, resource_key)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.embedding_state (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            model_name text NOT NULL,
            model_dim integer NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """,
    )


class PostgresRepository(Repository):
    """asyncpg-backed implementation."""

    def __init__(
        self, pool: asyncpg.Pool, schema: str, vector_table: str = VECTOR_TABLE_NAME
    ) -> None:
        self._pool = pool
        self._schema = schema
        self._vector_table = f"data_{vector_table}"

    async def ensure_schema(self) -> None:
        async with self._pool.acquire() as connection:
            for statement in schema_ddl(self._schema):
                await connection.execute(statement)

    async def probe(self) -> None:
        async with self._pool.acquire() as connection:
            await connection.execute("SELECT 1")

    async def ensure_collection(self, collection_id: str) -> None:
        await self._pool.execute(
            f"""
            INSERT INTO {self._schema}.collections (collection_id)
            VALUES ($1)
            ON CONFLICT (collection_id) DO UPDATE SET updated_at = now()
            """,
            collection_id,
        )

    async def get_document(
        self, collection_id: str, external_id: str
    ) -> DocumentRecord | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT document_id, collection_id, external_id, title, source_type, source_uri,
                   checksum, normalized_checksum, provenance_mode, current_revision_id,
                   version, page, section,
                   page_count, recognized_section_count, recognized_table_count,
                   recognized_figure_count, metadata, chunk_count, created_at, updated_at
            FROM {self._schema}.documents
            WHERE collection_id = $1 AND external_id = $2
            """,
            collection_id,
            external_id,
        )
        return _document_from_row(row) if row else None

    async def list_current_documents(
        self, collection_ids: list[str]
    ) -> list[DocumentRecord]:
        if not collection_ids:
            return []
        rows = await self._pool.fetch(
            f"""
            SELECT document_id, collection_id, external_id, title, source_type, source_uri,
                   checksum, normalized_checksum, provenance_mode, current_revision_id,
                   version, page, section,
                   page_count, recognized_section_count, recognized_table_count,
                   recognized_figure_count, metadata, chunk_count, created_at, updated_at
            FROM {self._schema}.documents
            WHERE collection_id = ANY($1::text[])
            """,
            collection_ids,
        )
        return [_document_from_row(row) for row in rows]

    async def list_document_sections(
        self, collection_id: str, external_id: str
    ) -> list[DocumentSection]:
        rows = await self._pool.fetch(
            f"""
            SELECT section.section_id, section.document_id, section.revision_id,
                   section.parent_section_id, section.ordinal, section.title, section.path,
                   section.source_ref, section.first_block_ordinal,
                   section.last_block_ordinal, section.page_start, section.page_end,
                   section.is_root
            FROM {self._schema}.document_sections AS section
            JOIN {self._schema}.documents AS document
              ON document.document_id = section.document_id
            WHERE document.collection_id = $1 AND document.external_id = $2
            ORDER BY section.ordinal
            """,
            collection_id,
            external_id,
        )
        return [_section_from_row(row) for row in rows]

    async def get_document_outline(
        self, collection_id: str, external_id: str
    ) -> DocumentOutlineRecord | None:
        async with (
            self._pool.acquire() as connection,
            connection.transaction(isolation="repeatable_read", readonly=True),
        ):
            document_row = await connection.fetchrow(
                f"""
                SELECT document_id, collection_id, external_id, title, source_type,
                       source_uri, checksum, normalized_checksum, provenance_mode,
                       current_revision_id, version, page, section, page_count,
                       recognized_section_count, recognized_table_count,
                       recognized_figure_count, metadata, chunk_count, created_at,
                       updated_at
                FROM {self._schema}.documents
                WHERE collection_id = $1 AND external_id = $2
                """,
                collection_id,
                external_id,
            )
            if document_row is None:
                return None
            section_rows = await connection.fetch(
                f"""
                SELECT section.section_id, section.document_id, section.revision_id,
                       section.parent_section_id, section.ordinal, section.title,
                       section.path, section.source_ref, section.first_block_ordinal,
                       section.last_block_ordinal, section.page_start, section.page_end,
                       section.is_root
                FROM {self._schema}.document_sections AS section
                WHERE section.document_id = $1 AND section.revision_id = $2
                ORDER BY section.ordinal
                """,
                document_row["document_id"],
                document_row["current_revision_id"],
            )
        return DocumentOutlineRecord(
            document=_document_from_row(document_row),
            sections=tuple(_section_from_row(row) for row in section_rows),
        )

    async def get_document_scan(
        self,
        collection_id: str,
        external_id: str,
        *,
        section_ref: str | None,
        after_ordinal: int | None,
        expected_source_marker: str | None,
        limit: int,
    ) -> DocumentScanRecord | None:
        if limit < 1:
            raise ValueError("A document scan limit must be positive.")
        async with (
            self._pool.acquire() as connection,
            connection.transaction(isolation="repeatable_read", readonly=True),
        ):
            document_row = await connection.fetchrow(
                f"""
                SELECT document_id, collection_id, external_id, title, source_type,
                       source_uri, checksum, normalized_checksum, provenance_mode,
                       current_revision_id, version, page, section, page_count,
                       recognized_section_count, recognized_table_count,
                       recognized_figure_count, metadata, chunk_count, created_at,
                       updated_at
                FROM {self._schema}.documents
                WHERE collection_id = $1 AND external_id = $2
                """,
                collection_id,
                external_id,
            )
            if document_row is None:
                return None
            section_rows = await connection.fetch(
                f"""
                SELECT section.section_id, section.document_id, section.revision_id,
                       section.parent_section_id, section.ordinal, section.title,
                       section.path, section.source_ref, section.first_block_ordinal,
                       section.last_block_ordinal, section.page_start, section.page_end,
                       section.is_root
                FROM {self._schema}.document_sections AS section
                WHERE section.document_id = $1 AND section.revision_id = $2
                ORDER BY section.ordinal
                """,
                document_row["document_id"],
                document_row["current_revision_id"],
            )
            document = _document_from_row(document_row)
            sections = tuple(_section_from_row(row) for row in section_rows)
            first_ordinal, last_ordinal = _scan_bounds(
                document,
                sections,
                section_ref=section_ref,
                after_ordinal=after_ordinal,
                expected_source_marker=expected_source_marker,
            )
            rows = await _fetch_scan_block_rows(
                connection,
                self._schema,
                document,
                first_ordinal=first_ordinal,
                last_ordinal=last_ordinal,
                after_ordinal=after_ordinal,
                limit=limit,
            )
        blocks = tuple(_block_from_row(row) for row in rows[:limit])
        return DocumentScanRecord(
            document=document,
            sections=sections,
            blocks=blocks,
            has_more=len(rows) > limit,
        )

    async def list_document_blocks(
        self, collection_id: str, external_id: str
    ) -> list[DocumentBlock]:
        rows = await self._pool.fetch(
            f"""
            SELECT block.document_id, block.revision_id, block.section_id, block.ordinal,
                   block.kind, block.text, block.page_start, block.page_end, block.table_id,
                   block.table_caption, block.table_header, block.part, block.parts
            FROM {self._schema}.document_blocks AS block
            JOIN {self._schema}.documents AS document
              ON document.document_id = block.document_id
            WHERE document.collection_id = $1 AND document.external_id = $2
            ORDER BY block.ordinal
            """,
            collection_id,
            external_id,
        )
        return [_block_from_row(row) for row in rows]

    async def list_sources(
        self, collection_id: str, *, limit: int
    ) -> list[SourceRecord]:
        """List every document and in-flight ingest in *collection_id*.

        Jobs are included on their own so an upload is visible as ``processing``
        before its document row exists. Both halves are bounded by *limit*; a
        collection larger than that is truncated to the most recent entries.
        """
        documents = await self._pool.fetch(
            f"""
            SELECT document_id, collection_id, external_id, title, source_type, source_uri,
                   checksum, normalized_checksum, provenance_mode, current_revision_id,
                   version, page, section,
                   page_count, recognized_section_count, recognized_table_count,
                   recognized_figure_count, metadata, chunk_count, created_at, updated_at
            FROM {self._schema}.documents
            WHERE collection_id = $1
            ORDER BY updated_at DESC
            LIMIT $2
            """,
            collection_id,
            limit,
        )
        jobs = await self._pool.fetch(
            f"""
            SELECT job_id, collection_id, external_id, title, source_type, filename,
                   media_type, document_id, status, chunk_count, unchanged, detail,
                   converter_name, converter_task_id, converter_submitted_at,
                   created_at, updated_at
            FROM (
                SELECT DISTINCT ON (external_id)
                       job_id, collection_id, external_id, title, source_type, filename,
                       media_type, document_id, status, chunk_count, unchanged, detail,
                       converter_name, converter_task_id, converter_submitted_at,
                       created_at, updated_at
                FROM {self._schema}.ingest_jobs
                WHERE collection_id = $1
                ORDER BY external_id, created_at DESC, job_id DESC
            ) AS latest_jobs
            ORDER BY updated_at DESC, job_id DESC
            LIMIT $2
            """,
            collection_id,
            limit,
        )
        source_objects = await self._pool.fetch(
            f"""
            SELECT document_id, collection_id, external_id, filename, media_type, byte_size,
                   checksum, storage_backend, storage_key, preview_key, preview_bytes,
                   preview_checksum, page_count, created_at, updated_at
            FROM {self._schema}.source_objects
            WHERE collection_id = $1
            ORDER BY updated_at DESC
            LIMIT $2
            """,
            collection_id,
            limit,
        )
        return merge_sources(
            [_document_from_row(row) for row in documents],
            [_job_from_row(row) for row in jobs],
            [_source_object_from_row(row) for row in source_objects],
            limit=limit,
        )

    async def create_upload_job(
        self, job: JobRecord, source: StagedSourceObjectRecord
    ) -> None:
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                await _insert_job(connection, self._schema, job)
                await connection.execute(
                    f"""
                    INSERT INTO {self._schema}.staged_source_objects (
                        job_id, document_id, collection_id, external_id, filename,
                        media_type, byte_size, checksum, storage_backend, storage_key,
                        preview_key, preview_bytes, preview_checksum, page_count
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    """,
                    source.job_id,
                    source.document_id,
                    source.collection_id,
                    source.external_id,
                    source.filename,
                    source.media_type,
                    source.byte_size,
                    source.checksum,
                    source.storage_backend,
                    source.storage_key,
                    source.preview_key,
                    source.preview_bytes,
                    source.preview_checksum,
                    source.page_count,
                )
        except asyncpg.UniqueViolationError as exc:
            raise ConcurrentIngestionError(
                "Another ingestion job is already active for this source."
            ) from exc

    async def set_staged_source_preview(
        self,
        job_id: UUID,
        *,
        preview_key: str | None,
        preview_bytes: int | None,
        preview_checksum: str | None,
        page_count: int | None,
    ) -> None:
        await self._pool.execute(
            f"""
            UPDATE {self._schema}.staged_source_objects
            SET preview_key = $2,
                preview_bytes = $3,
                preview_checksum = $4,
                page_count = $5
            WHERE job_id = $1
            """,
            job_id,
            preview_key,
            preview_bytes,
            preview_checksum,
            page_count,
        )

    async def get_staged_source_object(
        self, job_id: UUID
    ) -> StagedSourceObjectRecord | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT job_id, document_id, collection_id, external_id, filename, media_type,
                   byte_size, checksum, storage_backend, storage_key, preview_key,
                   preview_bytes, preview_checksum, page_count
            FROM {self._schema}.staged_source_objects
            WHERE job_id = $1
            """,
            job_id,
        )
        return _staged_source_from_row(row) if row else None

    async def discard_staged_source(
        self, job_id: UUID
    ) -> StagedSourceObjectRecord | None:
        async with self._pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                f"""
                DELETE FROM {self._schema}.staged_source_objects
                WHERE job_id = $1
                RETURNING job_id, document_id, collection_id, external_id, filename,
                          media_type, byte_size, checksum, storage_backend, storage_key,
                          preview_key, preview_bytes, preview_checksum, page_count
                """,
                job_id,
            )
            if row is not None:
                await _enqueue_cleanup_records(
                    connection,
                    self._schema,
                    _source_cleanup_records(_staged_source_from_row(row)),
                )
        return _staged_source_from_row(row) if row is not None else None

    async def discard_failed_staged_sources(self) -> list[StagedSourceObjectRecord]:
        async with self._pool.acquire() as connection, connection.transaction():
            rows = await connection.fetch(
                f"""
                DELETE FROM {self._schema}.staged_source_objects AS staged
                USING {self._schema}.ingest_jobs AS job
                WHERE staged.job_id = job.job_id AND job.status = 'failed'
                RETURNING staged.job_id, staged.document_id, staged.collection_id,
                          staged.external_id, staged.filename, staged.media_type,
                          staged.byte_size, staged.checksum, staged.storage_backend,
                          staged.storage_key, staged.preview_key, staged.preview_bytes,
                          staged.preview_checksum, staged.page_count
                """
            )
            records = [_staged_source_from_row(row) for row in rows]
            for record in records:
                await _enqueue_cleanup_records(
                    connection, self._schema, _source_cleanup_records(record)
                )
        return records

    async def activate_document(
        self,
        job_id: UUID,
        document: DocumentRecord,
        sections: tuple[DocumentSection, ...],
        blocks: tuple[DocumentBlock, ...],
    ) -> ActivationResult:
        _validate_activation_payload(job_id, document, sections, blocks)

        async with self._pool.acquire() as connection, connection.transaction():
            job_row = await connection.fetchrow(
                f"""
                SELECT collection_id, external_id, status
                FROM {self._schema}.ingest_jobs
                WHERE job_id = $1
                FOR UPDATE
                """,
                job_id,
            )
            _require_matching_active_job(job_row, document)
            previous_document_row = await connection.fetchrow(
                f"""
                SELECT current_revision_id
                FROM {self._schema}.documents
                WHERE collection_id = $1 AND external_id = $2
                FOR UPDATE
                """,
                document.collection_id,
                document.external_id,
            )
            previous_source_row = await connection.fetchrow(
                f"""
                SELECT document_id, collection_id, external_id, filename, media_type,
                       byte_size, checksum, storage_backend, storage_key, preview_key,
                       preview_bytes, preview_checksum, page_count, created_at, updated_at
                FROM {self._schema}.source_objects
                WHERE collection_id = $1 AND external_id = $2
                FOR UPDATE
                """,
                document.collection_id,
                document.external_id,
            )
            projection_count = await connection.fetchval(
                f"""
                SELECT count(*)
                FROM {self._schema}.{self._vector_table}
                WHERE metadata_->>'revision_id' = $1
                """,
                str(document.current_revision_id),
            )
            if projection_count != document.chunk_count:
                raise RuntimeError(
                    "The staged search projection is incomplete; activation was refused."
                )

            # Search projection rows are written privately with state "staged".
            # These updates become visible at the same commit as the current
            # document, section, block, and source-object state.
            await connection.execute(
                f"""
                UPDATE {self._schema}.{self._vector_table}
                SET metadata_ = jsonb_set(metadata_, '{{projection_state}}', '"retired"'::jsonb, true)
                WHERE metadata_->>'collection_id' = $1
                  AND metadata_->>'external_id' = $2
                  AND metadata_->>'revision_id' <> $3
                  AND metadata_->>'projection_state' = 'current'
                """,
                document.collection_id,
                document.external_id,
                str(document.current_revision_id),
            )
            await connection.execute(
                f"""
                UPDATE {self._schema}.{self._vector_table}
                SET metadata_ = jsonb_set(metadata_, '{{projection_state}}', '"current"'::jsonb, true)
                WHERE metadata_->>'revision_id' = $1
                """,
                str(document.current_revision_id),
            )

            await connection.execute(
                f"DELETE FROM {self._schema}.document_sections WHERE document_id = $1",
                document.document_id,
            )
            await _upsert_document(connection, self._schema, document)
            await _insert_sections(connection, self._schema, sections)
            await _insert_blocks(connection, self._schema, blocks)

            staged_source_row = await connection.fetchrow(
                f"""
                DELETE FROM {self._schema}.staged_source_objects
                WHERE job_id = $1
                RETURNING job_id, document_id, collection_id, external_id, filename,
                          media_type, byte_size, checksum, storage_backend, storage_key,
                          preview_key, preview_bytes, preview_checksum, page_count
                """,
                job_id,
            )
            if staged_source_row is None:
                await connection.execute(
                    f"DELETE FROM {self._schema}.source_objects WHERE document_id = $1",
                    document.document_id,
                )
            else:
                staged = _staged_source_from_row(staged_source_row)
                if (
                    staged.document_id != document.document_id
                    or staged.collection_id != document.collection_id
                    or staged.external_id != document.external_id
                ):
                    raise RuntimeError(
                        "The staged source does not belong to this document."
                    )
                await _upsert_current_source(connection, self._schema, staged)

            if previous_document_row is not None:
                previous_revision_id = previous_document_row["current_revision_id"]
                if previous_revision_id != document.current_revision_id:
                    await _enqueue_cleanup_records(
                        connection,
                        self._schema,
                        (
                            PendingCleanupRecord(
                                resource_type="vector_revision",
                                resource_key=str(previous_revision_id),
                            ),
                        ),
                    )
            if previous_source_row is not None:
                await _enqueue_cleanup_records(
                    connection,
                    self._schema,
                    _source_cleanup_records(
                        _source_object_from_row(previous_source_row)
                    ),
                )
            await _complete_job(
                connection,
                self._schema,
                job_id,
                document.document_id,
                document.chunk_count,
                unchanged=False,
            )

        return ActivationResult(
            previous_revision_id=(
                previous_document_row["current_revision_id"]
                if previous_document_row is not None
                else None
            ),
            previous_source=(
                _source_object_from_row(previous_source_row)
                if previous_source_row is not None
                else None
            ),
        )

    async def complete_unchanged(
        self, job_id: UUID, document: DocumentRecord
    ) -> StagedSourceObjectRecord | None:
        if document.current_revision_id is None:
            raise ValueError("An unchanged document requires its current revision id.")
        discarded_staged: StagedSourceObjectRecord | None = None
        async with self._pool.acquire() as connection, connection.transaction():
            job_row = await connection.fetchrow(
                f"""
                SELECT collection_id, external_id, status
                FROM {self._schema}.ingest_jobs
                WHERE job_id = $1
                FOR UPDATE
                """,
                job_id,
            )
            _require_matching_active_job(job_row, document)
            current_row = await connection.fetchrow(
                f"""
                SELECT current_revision_id, checksum, metadata
                FROM {self._schema}.documents
                WHERE document_id = $1 AND collection_id = $2 AND external_id = $3
                FOR UPDATE
                """,
                document.document_id,
                document.collection_id,
                document.external_id,
            )
            if (
                current_row is None
                or current_row["current_revision_id"] != document.current_revision_id
                or current_row["checksum"] != document.checksum
            ):
                raise RuntimeError(
                    "The current source changed before metadata activation."
                )
            await _upsert_document(connection, self._schema, document)
            previous_metadata = current_row["metadata"]
            if isinstance(previous_metadata, str):
                previous_metadata = json.loads(previous_metadata)
            transaction_time = await connection.fetchval("SELECT now()")
            projection_metadata: dict[str, Any] = {
                **document.metadata,
                "title": document.title,
                "source_type": document.source_type,
                "source_uri": document.source_uri,
                "checksum": document.checksum,
                "version": document.version,
                "updated_at": transaction_time.isoformat(),
                "updated_at_ts": int(transaction_time.timestamp()),
            }
            if document.provenance_mode == "document":
                projection_metadata.update(
                    {
                        "page": document.page,
                        "page_end": document.page,
                        "section": document.section,
                    }
                )
                await connection.execute(
                    f"""
                    UPDATE {self._schema}.document_blocks
                    SET page_start = $2, page_end = $2
                    WHERE document_id = $1
                    """,
                    document.document_id,
                    document.page,
                )
                await connection.execute(
                    f"""
                    UPDATE {self._schema}.document_sections
                    SET page_start = $2, page_end = $2
                    WHERE document_id = $1 AND is_root
                    """,
                    document.document_id,
                    document.page,
                )
            projection_rows = await connection.fetch(
                f"""
                SELECT id, text, metadata_
                FROM {self._schema}.{self._vector_table}
                WHERE metadata_->>'revision_id' = $1
                  AND metadata_->>'projection_state' = 'current'
                FOR UPDATE
                """,
                str(document.current_revision_id),
            )
            await connection.executemany(
                f"""
                UPDATE {self._schema}.{self._vector_table}
                SET text = $2, metadata_ = $3::jsonb
                WHERE id = $1
                """,
                [
                    _refreshed_projection_row(
                        row,
                        previous_metadata=dict(previous_metadata or {}),
                        projection_metadata=projection_metadata,
                    )
                    for row in projection_rows
                ],
            )
            staged_row = await connection.fetchrow(
                f"""
                DELETE FROM {self._schema}.staged_source_objects
                WHERE job_id = $1
                RETURNING job_id, document_id, collection_id, external_id, filename,
                          media_type, byte_size, checksum, storage_backend, storage_key,
                          preview_key, preview_bytes, preview_checksum, page_count
                """,
                job_id,
            )
            if staged_row is not None:
                staged = _staged_source_from_row(staged_row)
                if (
                    staged.document_id != document.document_id
                    or staged.collection_id != document.collection_id
                    or staged.external_id != document.external_id
                    or staged.checksum != document.checksum
                ):
                    raise RuntimeError(
                        "The staged source does not match the current source."
                    )
                updated_source = await connection.fetchval(
                    f"""
                    UPDATE {self._schema}.source_objects
                    SET filename = $2, media_type = $3, updated_at = now()
                    WHERE document_id = $1
                    RETURNING document_id
                    """,
                    document.document_id,
                    staged.filename,
                    staged.media_type,
                )
                if updated_source is None:
                    await _upsert_current_source(connection, self._schema, staged)
                else:
                    discarded_staged = staged
                    await _enqueue_cleanup_records(
                        connection, self._schema, _source_cleanup_records(staged)
                    )
            await _complete_job(
                connection,
                self._schema,
                job_id,
                document.document_id,
                document.chunk_count,
                unchanged=True,
            )
        return discarded_staged

    async def enqueue_cleanup(self, record: PendingCleanupRecord) -> None:
        await _enqueue_cleanup_records(self._pool, self._schema, (record,))

    async def list_pending_cleanup(self) -> list[PendingCleanupRecord]:
        rows = await self._pool.fetch(
            f"""
            SELECT resource_type, resource_key, storage_backend
            FROM {self._schema}.pending_resource_cleanup
            ORDER BY created_at, resource_type, resource_key
            """
        )
        return [
            PendingCleanupRecord(
                resource_type=row["resource_type"],
                resource_key=row["resource_key"],
                storage_backend=row["storage_backend"],
            )
            for row in rows
        ]

    async def complete_cleanup(self, record: PendingCleanupRecord) -> None:
        await self._pool.execute(
            f"""
            DELETE FROM {self._schema}.pending_resource_cleanup
            WHERE resource_type = $1 AND resource_key = $2
            """,
            record.resource_type,
            record.resource_key,
        )

    async def get_source_object(
        self, collection_id: str, external_id: str
    ) -> SourceObjectRecord | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT document_id, collection_id, external_id, filename, media_type, byte_size,
                   checksum, storage_backend, storage_key, preview_key, preview_bytes,
                   preview_checksum, page_count, created_at, updated_at
            FROM {self._schema}.source_objects
            WHERE collection_id = $1 AND external_id = $2
            """,
            collection_id,
            external_id,
        )
        return _source_object_from_row(row) if row else None

    async def create_job(self, record: JobRecord) -> None:
        try:
            await _insert_job(self._pool, self._schema, record)
        except asyncpg.UniqueViolationError as exc:
            raise ConcurrentIngestionError(
                "Another ingestion job is already active for this source."
            ) from exc

    async def update_job(
        self,
        job_id: UUID,
        *,
        status: JobStatus,
        document_id: UUID | None = None,
        chunk_count: int | None = None,
        unchanged: bool | None = None,
        detail: str | None = None,
    ) -> None:
        await self._pool.execute(
            f"""
            UPDATE {self._schema}.ingest_jobs
            SET status = $2,
                document_id = COALESCE($3, document_id),
                chunk_count = COALESCE($4, chunk_count),
                unchanged = COALESCE($5, unchanged),
                detail = $6,
                updated_at = now()
            WHERE job_id = $1
              AND (status <> 'completed' OR $2 = 'completed')
            """,
            job_id,
            status,
            document_id,
            chunk_count,
            unchanged,
            detail,
        )

    async def set_job_conversion_task(
        self,
        job_id: UUID,
        *,
        converter_name: str,
        task_id: str,
        submitted_at: datetime,
    ) -> None:
        """Record the converter task this job is waiting on.

        Written before the first poll, so a crash between submission and the
        first status read still leaves the remote task findable.
        """
        await self._pool.execute(
            f"""
            UPDATE {self._schema}.ingest_jobs
            SET converter_name = $2,
                converter_task_id = $3,
                converter_submitted_at = $4,
                updated_at = now()
            WHERE job_id = $1
            """,
            job_id,
            converter_name,
            task_id,
            submitted_at,
        )

    async def get_job(self, job_id: UUID) -> JobRecord | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT job_id, collection_id, external_id, document_id, status,
                   title, source_type, filename, media_type, chunk_count, unchanged,
                   detail, converter_name, converter_task_id, converter_submitted_at,
                   created_at, updated_at
            FROM {self._schema}.ingest_jobs
            WHERE job_id = $1
            """,
            job_id,
        )
        return _job_from_row(row) if row else None

    async def get_latest_job(
        self, collection_id: str, external_id: str
    ) -> JobRecord | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT job_id, collection_id, external_id, document_id, status,
                   title, source_type, filename, media_type, chunk_count, unchanged,
                   detail, converter_name, converter_task_id, converter_submitted_at,
                   created_at, updated_at
            FROM {self._schema}.ingest_jobs
            WHERE collection_id = $1 AND external_id = $2
            ORDER BY created_at DESC, job_id DESC
            LIMIT 1
            """,
            collection_id,
            external_id,
        )
        return _job_from_row(row) if row else None

    async def list_resumable_jobs(self) -> list[JobRecord]:
        rows = await self._pool.fetch(
            f"""
            SELECT job_id, collection_id, external_id, document_id, status,
                   title, source_type, filename, media_type, chunk_count, unchanged,
                   detail, converter_name, converter_task_id, converter_submitted_at,
                   created_at, updated_at
            FROM {self._schema}.ingest_jobs
            WHERE status IN ('accepted', 'processing')
              AND converter_task_id IS NOT NULL
              AND converter_submitted_at IS NOT NULL
            ORDER BY converter_submitted_at
            """
        )
        return [_job_from_row(row) for row in rows]

    async def fail_interrupted_jobs(self, detail: str) -> list[UUID]:
        """Fail unfinished jobs that hold no resumable converter task.

        A job whose converter task is recorded is deliberately left alone: the
        remote conversion is still running, and failing it here would throw away
        work Seshat has already paid for.
        """
        rows = await self._pool.fetch(
            f"""
            UPDATE {self._schema}.ingest_jobs
            SET status = 'failed', detail = $1, updated_at = now()
            WHERE status IN ('accepted', 'processing')
              AND (converter_task_id IS NULL OR converter_submitted_at IS NULL)
            RETURNING job_id
            """,
            detail,
        )
        return [row["job_id"] for row in rows]

    async def get_embedding_state(self) -> EmbeddingState | None:
        row = await self._pool.fetchrow(
            f"SELECT model_name, model_dim FROM {self._schema}.embedding_state WHERE singleton"
        )
        if row is None:
            return None
        return EmbeddingState(model_name=row["model_name"], model_dim=row["model_dim"])

    async def set_embedding_state(self, state: EmbeddingState) -> None:
        await self._pool.execute(
            f"""
            INSERT INTO {self._schema}.embedding_state (singleton, model_name, model_dim)
            VALUES (true, $1, $2)
            ON CONFLICT (singleton) DO UPDATE SET
                model_name = EXCLUDED.model_name,
                model_dim = EXCLUDED.model_dim,
                updated_at = now()
            """,
            state.model_name,
            state.model_dim,
        )


async def _insert_job(connection: Any, schema: str, record: JobRecord) -> None:
    await connection.execute(
        f"""
        INSERT INTO {schema}.ingest_jobs (
            job_id, collection_id, external_id, title, source_type, filename,
            media_type, document_id, status, chunk_count, unchanged, detail,
            converter_name, converter_task_id, converter_submitted_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        """,
        record.job_id,
        record.collection_id,
        record.external_id,
        record.title,
        record.source_type,
        record.filename,
        record.media_type,
        record.document_id,
        record.status,
        record.chunk_count,
        record.unchanged,
        record.detail,
        record.converter_name,
        record.converter_task_id,
        record.converter_submitted_at,
    )


def _validate_activation_payload(
    job_id: UUID,
    document: DocumentRecord,
    sections: tuple[DocumentSection, ...],
    blocks: tuple[DocumentBlock, ...],
) -> None:
    revision_id = document.current_revision_id
    if revision_id is None or revision_id != job_id:
        raise ValueError(
            "An activated document must use its job id as the revision id."
        )
    if document.chunk_count < 1 or not blocks:
        raise ValueError("An activated document requires blocks and search chunks.")
    if [section.ordinal for section in sections] != list(range(len(sections))):
        raise ValueError("Document section ordinals must be contiguous.")
    roots = [section for section in sections if section.is_root]
    if (
        len(roots) != 1
        or roots[0].ordinal != 0
        or roots[0].parent_section_id is not None
    ):
        raise ValueError("An activated document requires one leading root section.")
    section_ids = {section.section_id for section in sections}
    if len(section_ids) != len(sections):
        raise ValueError("Document section ids must be unique.")
    seen_sections: set[UUID] = set()
    last_block_ordinal = len(blocks) - 1
    for section in sections:
        if (
            section.document_id != document.document_id
            or section.revision_id != revision_id
        ):
            raise ValueError(
                "A section does not belong to the activated document revision."
            )
        if (
            section.parent_section_id is not None
            and section.parent_section_id not in seen_sections
        ):
            raise ValueError("A section parent must precede its children.")
        if (
            section.first_block_ordinal is None
            or section.last_block_ordinal is None
            or section.first_block_ordinal < 0
            or section.first_block_ordinal > section.last_block_ordinal
            or section.last_block_ordinal > last_block_ordinal
        ):
            raise ValueError(
                "Every persisted section must cover a canonical block range."
            )
        seen_sections.add(section.section_id)
    root = roots[0]
    if (root.first_block_ordinal, root.last_block_ordinal) != (
        0,
        last_block_ordinal,
    ):
        raise ValueError("The root section must cover every canonical block.")
    if document.recognized_section_count is not None:
        recognized_count = sum(not section.is_root for section in sections)
        if document.recognized_section_count != recognized_count:
            raise ValueError(
                "The recognized section count does not match persisted sections."
            )
    if [block.ordinal for block in blocks] != list(range(len(blocks))):
        raise ValueError("Document block ordinals must be contiguous.")
    for block in blocks:
        if (
            block.document_id != document.document_id
            or block.revision_id != revision_id
        ):
            raise ValueError(
                "A block does not belong to the activated document revision."
            )
        if block.section_id not in section_ids:
            raise ValueError("A block references an unknown section.")
        if not block.text:
            raise ValueError("Canonical blocks may not be empty.")


def _require_matching_active_job(job_row: Any, document: DocumentRecord) -> None:
    if job_row is None:
        raise RuntimeError("The ingestion job does not exist.")
    if (
        job_row["collection_id"] != document.collection_id
        or job_row["external_id"] != document.external_id
        or job_row["status"] not in ("accepted", "processing")
    ):
        raise RuntimeError("The ingestion job does not match the document activation.")


async def _upsert_document(
    connection: Any, schema: str, record: DocumentRecord
) -> None:
    await connection.execute(
        f"""
        INSERT INTO {schema}.documents (
            document_id, collection_id, external_id, current_revision_id, title,
            source_type, source_uri, checksum, normalized_checksum, provenance_mode,
            version, page, section, page_count, recognized_section_count,
            recognized_table_count, recognized_figure_count, metadata, chunk_count
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
            $15, $16, $17, $18::jsonb, $19
        )
        ON CONFLICT (document_id) DO UPDATE SET
            current_revision_id = EXCLUDED.current_revision_id,
            title = EXCLUDED.title,
            source_type = EXCLUDED.source_type,
            source_uri = EXCLUDED.source_uri,
            checksum = EXCLUDED.checksum,
            normalized_checksum = EXCLUDED.normalized_checksum,
            provenance_mode = EXCLUDED.provenance_mode,
            version = EXCLUDED.version,
            page = EXCLUDED.page,
            section = EXCLUDED.section,
            page_count = EXCLUDED.page_count,
            recognized_section_count = EXCLUDED.recognized_section_count,
            recognized_table_count = EXCLUDED.recognized_table_count,
            recognized_figure_count = EXCLUDED.recognized_figure_count,
            metadata = EXCLUDED.metadata,
            chunk_count = EXCLUDED.chunk_count,
            updated_at = now()
        """,
        record.document_id,
        record.collection_id,
        record.external_id,
        record.current_revision_id,
        record.title,
        record.source_type,
        record.source_uri,
        record.checksum,
        record.normalized_checksum,
        record.provenance_mode,
        record.version,
        record.page,
        record.section,
        record.page_count,
        record.recognized_section_count,
        record.recognized_table_count,
        record.recognized_figure_count,
        json.dumps(record.metadata),
        record.chunk_count,
    )


async def _insert_sections(
    connection: Any, schema: str, sections: tuple[DocumentSection, ...]
) -> None:
    await connection.executemany(
        f"""
        INSERT INTO {schema}.document_sections (
            section_id, document_id, revision_id, parent_section_id, ordinal, title,
            path, source_ref, first_block_ordinal, last_block_ordinal, page_start,
            page_end, is_root
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12, $13)
        """,
        [
            (
                section.section_id,
                section.document_id,
                section.revision_id,
                section.parent_section_id,
                section.ordinal,
                section.title,
                json.dumps(section.path),
                section.source_ref,
                section.first_block_ordinal,
                section.last_block_ordinal,
                section.page_start,
                section.page_end,
                section.is_root,
            )
            for section in sections
        ],
    )


async def _insert_blocks(
    connection: Any, schema: str, blocks: tuple[DocumentBlock, ...]
) -> None:
    await connection.executemany(
        f"""
        INSERT INTO {schema}.document_blocks (
            document_id, revision_id, section_id, ordinal, kind, text, page_start,
            page_end, table_id, table_caption, table_header, part, parts
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """,
        [
            (
                block.document_id,
                block.revision_id,
                block.section_id,
                block.ordinal,
                block.kind,
                block.text,
                block.page_start,
                block.page_end,
                block.table_id,
                block.table_caption,
                block.table_header,
                block.part,
                block.parts,
            )
            for block in blocks
        ],
    )


def _refreshed_projection_row(
    row: Any,
    *,
    previous_metadata: dict[str, Any],
    projection_metadata: dict[str, Any],
) -> tuple[int, str, str]:
    stored_metadata = row["metadata_"]
    if isinstance(stored_metadata, str):
        stored_metadata = json.loads(stored_metadata)
    stored_projection_metadata = dict(stored_metadata or {})
    metadata = dict(stored_projection_metadata)
    for key in previous_metadata:
        metadata.pop(key, None)
    metadata.update(projection_metadata)

    legacy_metadata = stored_projection_metadata
    node_metadata = metadata
    serialized_node = metadata.get("_node_content")
    if isinstance(serialized_node, str):
        node = json.loads(serialized_node)
        raw_node_metadata = node.get("metadata")
        if isinstance(raw_node_metadata, dict):
            legacy_metadata = dict(raw_node_metadata)
            node_metadata = dict(legacy_metadata)
            for key in previous_metadata:
                node_metadata.pop(key, None)
            node_metadata.update(projection_metadata)

    stored_context = legacy_metadata.get(SEARCH_CONTEXT_METADATA_KEY)
    body = search_body(
        str(row["text"]),
        stored_context if isinstance(stored_context, str) else "",
        legacy_metadata=legacy_metadata,
    )
    refreshed_context = context_header(node_metadata, body)
    if refreshed_context and body.startswith(refreshed_context):
        refreshed_context = ""
    node_metadata[SEARCH_CONTEXT_METADATA_KEY] = refreshed_context
    metadata[SEARCH_CONTEXT_METADATA_KEY] = refreshed_context
    if isinstance(serialized_node, str):
        node["metadata"] = node_metadata
        metadata["_node_content"] = json.dumps(node, ensure_ascii=False)
    text = contextual_search_text(node_metadata, body)
    return row["id"], text, json.dumps(metadata)


async def _upsert_current_source(
    connection: Any, schema: str, source: StagedSourceObjectRecord
) -> None:
    await connection.execute(
        f"""
        INSERT INTO {schema}.source_objects (
            document_id, collection_id, external_id, filename, media_type, byte_size,
            checksum, storage_backend, storage_key, preview_key, preview_bytes,
            preview_checksum, page_count
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        ON CONFLICT (document_id) DO UPDATE SET
            filename = EXCLUDED.filename,
            media_type = EXCLUDED.media_type,
            byte_size = EXCLUDED.byte_size,
            checksum = EXCLUDED.checksum,
            storage_backend = EXCLUDED.storage_backend,
            storage_key = EXCLUDED.storage_key,
            preview_key = EXCLUDED.preview_key,
            preview_bytes = EXCLUDED.preview_bytes,
            preview_checksum = EXCLUDED.preview_checksum,
            page_count = EXCLUDED.page_count,
            updated_at = now()
        """,
        source.document_id,
        source.collection_id,
        source.external_id,
        source.filename,
        source.media_type,
        source.byte_size,
        source.checksum,
        source.storage_backend,
        source.storage_key,
        source.preview_key,
        source.preview_bytes,
        source.preview_checksum,
        source.page_count,
    )


async def _complete_job(
    connection: Any,
    schema: str,
    job_id: UUID,
    document_id: UUID,
    chunk_count: int,
    *,
    unchanged: bool,
) -> None:
    await connection.execute(
        f"""
        UPDATE {schema}.ingest_jobs
        SET status = 'completed', document_id = $2, chunk_count = $3,
            unchanged = $4, detail = NULL, updated_at = now()
        WHERE job_id = $1
        """,
        job_id,
        document_id,
        chunk_count,
        unchanged,
    )


def _source_cleanup_records(
    source: SourceObjectRecord | StagedSourceObjectRecord,
) -> tuple[PendingCleanupRecord, ...]:
    keys = tuple(dict.fromkeys((source.storage_key, source.preview_key)))
    return tuple(
        PendingCleanupRecord(
            resource_type="source_object",
            resource_key=key,
            storage_backend=source.storage_backend,
        )
        for key in keys
        if key is not None
    )


def _scan_bounds(
    document: DocumentRecord,
    sections: tuple[DocumentSection, ...],
    *,
    section_ref: str | None,
    after_ordinal: int | None,
    expected_source_marker: str | None,
) -> tuple[int, int]:
    revision_id = document.current_revision_id
    if revision_id is None:
        raise RuntimeError("A current document must have a revision.")
    current_marker = source_revision_marker(document.checksum, revision_id)
    if expected_source_marker is not None and not hmac.compare_digest(
        expected_source_marker, current_marker
    ):
        raise SourceChangedError("The source changed after this scan began.")

    scope: DocumentSection | None = None
    if section_ref is None:
        scope = next((section for section in sections if section.is_root), None)
    else:
        for section in sections:
            if section.is_root:
                continue
            candidate = section_reference(
                document.collection_id,
                document.external_id,
                revision_id,
                section.section_id,
            )
            if hmac.compare_digest(candidate, section_ref):
                scope = section
                break
        if scope is None:
            raise SectionNotFoundError(
                "The section reference is not current for this source."
            )

    if (
        scope is None
        or scope.first_block_ordinal is None
        or scope.last_block_ordinal is None
    ):
        raise RuntimeError("The current source has no valid canonical scan scope.")
    first_ordinal = scope.first_block_ordinal
    last_ordinal = scope.last_block_ordinal
    if after_ordinal is not None and not (
        first_ordinal - 1 <= after_ordinal < last_ordinal
    ):
        raise InvalidScanPositionError(
            "The scan cursor position is outside its section scope."
        )
    return first_ordinal, last_ordinal


async def _fetch_scan_block_rows(
    connection: Any,
    schema: str,
    document: DocumentRecord,
    *,
    first_ordinal: int,
    last_ordinal: int,
    after_ordinal: int | None,
    limit: int,
) -> list[Any]:
    """Read one bounded block page within the caller's open snapshot."""
    return await connection.fetch(
        f"""
        SELECT block.document_id, block.revision_id, block.section_id,
               block.ordinal, block.kind, block.text, block.page_start,
               block.page_end, block.table_id, block.table_caption,
               block.table_header, block.part, block.parts
        FROM {schema}.document_blocks AS block
        WHERE block.document_id = $1
          AND block.revision_id = $2
          AND block.ordinal BETWEEN $3 AND $4
          AND block.ordinal > $5
        ORDER BY block.ordinal
        LIMIT $6
        """,
        document.document_id,
        document.current_revision_id,
        first_ordinal,
        last_ordinal,
        after_ordinal if after_ordinal is not None else first_ordinal - 1,
        limit + 1,
    )


async def _enqueue_cleanup_records(
    connection: Any,
    schema: str,
    records: tuple[PendingCleanupRecord, ...],
) -> None:
    if not records:
        return
    await connection.executemany(
        f"""
        INSERT INTO {schema}.pending_resource_cleanup (
            resource_type, resource_key, storage_backend
        )
        VALUES ($1, $2, $3)
        ON CONFLICT (resource_type, resource_key) DO NOTHING
        """,
        [
            (record.resource_type, record.resource_key, record.storage_backend)
            for record in records
        ],
    )


def merge_sources(
    documents: Iterable[DocumentRecord],
    jobs: Iterable[JobRecord],
    source_objects: Iterable[SourceObjectRecord] = (),
    *,
    limit: int,
) -> list[SourceRecord]:
    """Pair documents with their latest job and stored original, newest first."""
    latest_jobs: dict[str, JobRecord] = {}
    for job in jobs:
        current = latest_jobs.get(job.external_id)
        if current is None or _sorts_after(job.created_at, current.created_at):
            latest_jobs[job.external_id] = job
    originals = {record.external_id: record for record in source_objects}

    records: list[SourceRecord] = []
    seen: set[str] = set()
    for document in documents:
        seen.add(document.external_id)
        records.append(
            build_source_record(
                document,
                latest_jobs.get(document.external_id),
                originals.get(document.external_id),
            )
        )
    for external_id, job in latest_jobs.items():
        if external_id not in seen:
            records.append(build_source_record(None, job, originals.get(external_id)))

    records.sort(key=lambda record: record.updated_at or _EPOCH, reverse=True)
    return records[:limit]


_EPOCH = datetime.fromtimestamp(0, tz=UTC)


def _sorts_after(candidate: datetime | None, current: datetime | None) -> bool:
    return (candidate or _EPOCH) >= (current or _EPOCH)


def _earliest(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None or right is None:
        return left or right
    return min(left, right)


def _latest(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None or right is None:
        return left or right
    return max(left, right)


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _document_from_row(row: Any) -> DocumentRecord:
    metadata = row["metadata"]
    return DocumentRecord(
        document_id=row["document_id"],
        collection_id=row["collection_id"],
        external_id=row["external_id"],
        title=row["title"],
        source_type=row["source_type"],
        source_uri=row["source_uri"],
        checksum=row["checksum"],
        normalized_checksum=row["normalized_checksum"],
        provenance_mode=row["provenance_mode"],
        current_revision_id=row["current_revision_id"],
        version=row["version"],
        page=row["page"],
        section=row["section"],
        page_count=row["page_count"],
        recognized_section_count=row["recognized_section_count"],
        recognized_table_count=row["recognized_table_count"],
        recognized_figure_count=row["recognized_figure_count"],
        metadata=json.loads(metadata)
        if isinstance(metadata, str)
        else dict(metadata or {}),
        chunk_count=row["chunk_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _section_from_row(row: Any) -> DocumentSection:
    path = row["path"]
    return DocumentSection(
        section_id=row["section_id"],
        document_id=row["document_id"],
        revision_id=row["revision_id"],
        parent_section_id=row["parent_section_id"],
        ordinal=row["ordinal"],
        title=row["title"],
        path=tuple(json.loads(path) if isinstance(path, str) else path or ()),
        source_ref=row["source_ref"],
        first_block_ordinal=row["first_block_ordinal"],
        last_block_ordinal=row["last_block_ordinal"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        is_root=row["is_root"],
    )


def _block_from_row(row: Any) -> DocumentBlock:
    return DocumentBlock(
        document_id=row["document_id"],
        revision_id=row["revision_id"],
        section_id=row["section_id"],
        ordinal=row["ordinal"],
        kind=row["kind"],
        text=row["text"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        table_id=row["table_id"],
        table_caption=row["table_caption"],
        table_header=row["table_header"],
        part=row["part"],
        parts=row["parts"],
    )


def _source_object_from_row(row: Any) -> SourceObjectRecord:
    return SourceObjectRecord(
        document_id=row["document_id"],
        collection_id=row["collection_id"],
        external_id=row["external_id"],
        filename=row["filename"],
        media_type=row["media_type"],
        byte_size=row["byte_size"],
        checksum=row["checksum"],
        storage_backend=row["storage_backend"],
        storage_key=row["storage_key"],
        preview_key=row["preview_key"],
        preview_bytes=row["preview_bytes"],
        preview_checksum=row["preview_checksum"],
        page_count=row["page_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _staged_source_from_row(row: Any) -> StagedSourceObjectRecord:
    return StagedSourceObjectRecord(
        job_id=row["job_id"],
        document_id=row["document_id"],
        collection_id=row["collection_id"],
        external_id=row["external_id"],
        filename=row["filename"],
        media_type=row["media_type"],
        byte_size=row["byte_size"],
        checksum=row["checksum"],
        storage_backend=row["storage_backend"],
        storage_key=row["storage_key"],
        preview_key=row["preview_key"],
        preview_bytes=row["preview_bytes"],
        preview_checksum=row["preview_checksum"],
        page_count=row["page_count"],
    )


def _job_from_row(row: Any) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        collection_id=row["collection_id"],
        external_id=row["external_id"],
        status=row["status"],
        title=row["title"],
        source_type=row["source_type"],
        filename=row["filename"],
        media_type=row["media_type"],
        document_id=row["document_id"],
        chunk_count=row["chunk_count"],
        unchanged=row["unchanged"],
        detail=row["detail"],
        converter_name=row["converter_name"],
        converter_task_id=row["converter_task_id"],
        converter_submitted_at=row["converter_submitted_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
