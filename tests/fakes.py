"""Deterministic doubles used by the unit tests.

The vector store deliberately ignores metadata filters so that the collection
isolation tests exercise the service's own guard rather than a cooperative
backend.
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from app.parsing import ConvertedDocument, DocumentSegment, build_converted_document
from app.repository import (
    ActivationResult,
    ConcurrentIngestionError,
    DocumentOutlineRecord,
    DocumentRecord,
    DocumentScanRecord,
    EmbeddingState,
    JobRecord,
    PendingCleanupRecord,
    Repository,
    SourceObjectRecord,
    SourceRecord,
    StagedSourceObjectRecord,
    _scan_bounds,
    _source_cleanup_records,
    merge_sources,
)
from app.representation import DocumentBlock, DocumentSection
from app.storage import SourceObjectStore, StoredObject, StoredObjectMissingError

VOCAB = (
    "alarm",
    "battery",
    "calibration",
    "outlet",
    "filter",
    "inlet",
    "maintenance",
    "supply",
    "pressure",
    "sensor",
    "valve",
    "controller",
)


class DeterministicEmbedding(BaseEmbedding):
    """Keyword-presence embeddings: no network, identical text -> identical vector."""

    _vocab: tuple[str, ...] = PrivateAttr()

    def __init__(self, vocab: tuple[str, ...] = VOCAB) -> None:
        super().__init__(model_name="deterministic", embed_batch_size=8)
        self._vocab = vocab

    @classmethod
    def class_name(cls) -> str:
        return "deterministic"

    @property
    def dim(self) -> int:
        return len(self._vocab)

    def _vectorize(self, text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if token in lowered else 0.0 for token in self._vocab]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vectorize(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vectorize(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_text_embeddings(texts)


class RecordingVectorStore(BasePydanticVectorStore):
    """In-memory vector store that records queries and ignores metadata filters."""

    stores_text: bool = True
    is_embedding_query: bool = True

    _nodes: dict[str, BaseNode] = PrivateAttr(default_factory=dict)
    _queries: list[VectorStoreQuery] = PrivateAttr(default_factory=list)
    _cleared: int = PrivateAttr(default=0)
    _deleted_refs: list[str] = PrivateAttr(default_factory=list)
    _delete_error: Exception | None = PrivateAttr(default=None)

    @property
    def client(self) -> None:
        return None

    @property
    def queries(self) -> list[VectorStoreQuery]:
        return self._queries

    @property
    def deleted_refs(self) -> list[str]:
        return self._deleted_refs

    @property
    def clear_count(self) -> int:
        return self._cleared

    @property
    def delete_error(self) -> Exception | None:
        return self._delete_error

    @delete_error.setter
    def delete_error(self, value: Exception | None) -> None:
        self._delete_error = value

    @property
    def nodes(self) -> dict[str, BaseNode]:
        return self._nodes

    def add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        for node in nodes:
            self._nodes[str(node.node_id)] = node
        return [str(node.node_id) for node in nodes]

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        if self._delete_error is not None:
            raise self._delete_error
        self._deleted_refs.append(ref_doc_id)
        for node_id in [
            node_id
            for node_id, node in self._nodes.items()
            if node.ref_doc_id == ref_doc_id
            or node.metadata.get("document_id") == ref_doc_id
        ]:
            self._nodes.pop(node_id, None)

    def clear(self) -> None:
        self._cleared += 1
        self._nodes.clear()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        self._queries.append(query)
        if not query.query_embedding:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        scored: list[tuple[float, BaseNode]] = []
        for node in self._nodes.values():
            if not node.embedding:
                continue
            score = sum(
                a * b
                for a, b in zip(node.embedding, query.query_embedding, strict=False)
            )
            scored.append((score, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: (query.similarity_top_k or len(scored))]
        return VectorStoreQueryResult(
            nodes=[node for _, node in top],
            similarities=[score for score, _ in top],
            ids=[str(node.node_id) for _, node in top],
        )


class RecordingConverter:
    """Document converter double: records calls and can be made to fail.

    ``segments`` is what a provenance-carrying converter returns; leaving it
    ``None`` yields one unlocated segment, which reports no page or section.
    """

    name = "recording"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.markdown = "# Converted\n\nThe outlet valve needs calibration."
        self.segments: list[DocumentSegment] | None = None
        self.converted_document: ConvertedDocument | None = None
        self.error: Exception | None = None

    async def convert(
        self, *, filename: str, media_type: str, content: bytes
    ) -> ConvertedDocument:
        self.calls.append(
            {"filename": filename, "media_type": media_type, "content": content}
        )
        if self.error is not None:
            raise self.error
        if self.converted_document is not None:
            return self.converted_document
        if self.segments is not None:
            return build_converted_document(list(self.segments))
        return build_converted_document([DocumentSegment(text=self.markdown)])


class RecordingAsyncConverter(RecordingConverter):
    """A converter whose work is addressable by task id, like Docling's async API.

    Submissions and result waits are counted separately, which is what the
    restart-resume and duplicate-upload tests assert on: the point of those tests
    is that a file is submitted exactly once.
    """

    name = "docling"

    def __init__(self) -> None:
        super().__init__()
        self.submissions: list[dict[str, Any]] = []
        self.awaited: list[str] = []
        self.submit_error: Exception | None = None
        self.result_error: Exception | None = None
        # When set, a result wait blocks until the test releases it, which models a
        # conversion that is still running.
        self.released: asyncio.Event | None = None

    @property
    def submission_count(self) -> int:
        return len(self.submissions)

    async def submit(self, *, filename: str, media_type: str, content: bytes) -> str:
        self.submissions.append(
            {"filename": filename, "media_type": media_type, "content": content}
        )
        if self.submit_error is not None:
            raise self.submit_error
        return f"task-{len(self.submissions)}"

    async def await_result(
        self, task_id: str, *, submitted_at: datetime
    ) -> ConvertedDocument:
        self.awaited.append(task_id)
        if self.released is not None:
            await self.released.wait()
        if self.result_error is not None:
            raise self.result_error
        if self.converted_document is not None:
            return self.converted_document
        if self.segments is not None:
            return build_converted_document(list(self.segments))
        return build_converted_document([DocumentSegment(text=self.markdown)])

    async def convert(
        self, *, filename: str, media_type: str, content: bytes
    ) -> ConvertedDocument:
        self.calls.append(
            {"filename": filename, "media_type": media_type, "content": content}
        )
        task_id = await self.submit(
            filename=filename, media_type=media_type, content=content
        )
        return await self.await_result(task_id, submitted_at=datetime.now(UTC))


class InMemorySourceStore(SourceObjectStore):
    """Source storage double that keeps objects in a dict."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.put_error: Exception | None = None
        self.delete_error: Exception | None = None

    @property
    def backend(self) -> str:
        return "memory"

    async def put(self, key: str, content: bytes) -> StoredObject:
        if self.put_error is not None:
            raise self.put_error
        self.objects[key] = content
        return StoredObject(
            key=key,
            byte_size=len(content),
            checksum=hashlib.sha256(content).hexdigest(),
        )

    async def stat(self, key: str) -> StoredObject | None:
        content = self.objects.get(key)
        if content is None:
            return None
        return StoredObject(key=key, byte_size=len(content), checksum="")

    async def read(
        self, key: str, *, offset: int = 0, length: int | None = None
    ) -> AsyncIterator[bytes]:
        content = self.objects.get(key)
        if content is None:
            raise StoredObjectMissingError(
                "The stored source file is no longer available."
            )
        window = (
            content[offset:] if length is None else content[offset : offset + length]
        )
        # Deliberately chunked so range assembly is exercised, not just sliced.
        for start in range(0, len(window), 8):
            yield window[start : start + 8]

    async def delete(self, key: str) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.objects.pop(key, None)


class InMemoryRepository(Repository):
    """Repository double whose state survives being handed to a new application."""

    def __init__(self) -> None:
        self.collections: dict[str, datetime] = {}
        self.documents: dict[tuple[str, str], DocumentRecord] = {}
        self.sections: dict[tuple[UUID, int], DocumentSection] = {}
        self.blocks: dict[tuple[UUID, int], DocumentBlock] = {}
        self.jobs: dict[UUID, JobRecord] = {}
        self.source_objects: dict[tuple[str, str], SourceObjectRecord] = {}
        self.staged_source_objects: dict[UUID, StagedSourceObjectRecord] = {}
        self.pending_cleanup: dict[tuple[str, str], PendingCleanupRecord] = {}
        self.embedding_state: EmbeddingState | None = None
        self.probe_error: Exception | None = None

    async def ensure_collection(self, collection_id: str) -> None:
        self.collections[collection_id] = datetime.now(UTC)

    async def get_document(
        self, collection_id: str, external_id: str
    ) -> DocumentRecord | None:
        return self.documents.get((collection_id, external_id))

    async def list_current_documents(
        self, collection_ids: list[str]
    ) -> list[DocumentRecord]:
        allowed = set(collection_ids)
        return [
            document
            for (collection_id, _), document in self.documents.items()
            if collection_id in allowed
        ]

    async def list_document_sections(
        self, collection_id: str, external_id: str
    ) -> list[DocumentSection]:
        document = self.documents.get((collection_id, external_id))
        if document is None:
            return []
        return sorted(
            (
                section
                for (document_id, _), section in self.sections.items()
                if document_id == document.document_id
            ),
            key=lambda section: section.ordinal,
        )

    async def get_document_outline(
        self, collection_id: str, external_id: str
    ) -> DocumentOutlineRecord | None:
        document = self.documents.get((collection_id, external_id))
        if document is None:
            return None
        sections = await self.list_document_sections(collection_id, external_id)
        return DocumentOutlineRecord(document=document, sections=tuple(sections))

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
        document = self.documents.get((collection_id, external_id))
        if document is None:
            return None
        sections = tuple(
            await self.list_document_sections(collection_id, external_id)
        )
        first_ordinal, last_ordinal = _scan_bounds(
            document,
            sections,
            section_ref=section_ref,
            after_ordinal=after_ordinal,
            expected_source_marker=expected_source_marker,
        )
        after = after_ordinal if after_ordinal is not None else first_ordinal - 1
        matching = [
            block
            for block in await self.list_document_blocks(collection_id, external_id)
            if first_ordinal <= block.ordinal <= last_ordinal
            and block.ordinal > after
        ]
        return DocumentScanRecord(
            document=document,
            sections=sections,
            blocks=tuple(matching[:limit]),
            has_more=len(matching) > limit,
        )

    async def list_document_blocks(
        self, collection_id: str, external_id: str
    ) -> list[DocumentBlock]:
        document = self.documents.get((collection_id, external_id))
        if document is None:
            return []
        return sorted(
            (
                block
                for (document_id, _), block in self.blocks.items()
                if document_id == document.document_id
            ),
            key=lambda block: block.ordinal,
        )

    async def list_sources(
        self, collection_id: str, *, limit: int
    ) -> list[SourceRecord]:
        return merge_sources(
            [
                document
                for (collection, _), document in self.documents.items()
                if collection == collection_id
            ],
            [job for job in self.jobs.values() if job.collection_id == collection_id],
            [
                record
                for (collection, _), record in self.source_objects.items()
                if collection == collection_id
            ],
            limit=limit,
        )

    async def create_upload_job(
        self, job: JobRecord, source: StagedSourceObjectRecord
    ) -> None:
        await self.create_job(job)
        self.staged_source_objects[job.job_id] = source

    async def set_staged_source_preview(
        self,
        job_id: UUID,
        *,
        preview_key: str | None,
        preview_bytes: int | None,
        preview_checksum: str | None,
        page_count: int | None,
    ) -> None:
        record = self.staged_source_objects[job_id]
        self.staged_source_objects[job_id] = replace(
            record,
            preview_key=preview_key,
            preview_bytes=preview_bytes,
            preview_checksum=preview_checksum,
            page_count=page_count,
        )

    async def get_staged_source_object(
        self, job_id: UUID
    ) -> StagedSourceObjectRecord | None:
        return self.staged_source_objects.get(job_id)

    async def discard_staged_source(
        self, job_id: UUID
    ) -> StagedSourceObjectRecord | None:
        source = self.staged_source_objects.pop(job_id, None)
        if source is not None:
            for record in _source_cleanup_records(source):
                await self.enqueue_cleanup(record)
        return source

    async def discard_failed_staged_sources(self) -> list[StagedSourceObjectRecord]:
        failed_ids = {
            job_id for job_id, job in self.jobs.items() if job.status == "failed"
        }
        discarded = [
            source
            for job_id, source in self.staged_source_objects.items()
            if job_id in failed_ids
        ]
        for source in discarded:
            self.staged_source_objects.pop(source.job_id, None)
            for record in _source_cleanup_records(source):
                await self.enqueue_cleanup(record)
        return discarded

    async def activate_document(
        self,
        job_id: UUID,
        document: DocumentRecord,
        sections: tuple[DocumentSection, ...],
        blocks: tuple[DocumentBlock, ...],
    ) -> ActivationResult:
        now = datetime.now(UTC)
        key = (document.collection_id, document.external_id)
        previous = self.documents.get(key)
        previous_source = self.source_objects.get(key)
        self.documents[key] = replace(
            document,
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        self.sections = {
            key: value
            for key, value in self.sections.items()
            if key[0] != document.document_id
        }
        self.sections.update(
            {(section.document_id, section.ordinal): section for section in sections}
        )
        self.blocks = {
            key: value
            for key, value in self.blocks.items()
            if key[0] != document.document_id
        }
        self.blocks.update(
            {(block.document_id, block.ordinal): block for block in blocks}
        )

        staged = self.staged_source_objects.pop(job_id, None)
        if staged is None:
            self.source_objects.pop(key, None)
        else:
            self.source_objects[key] = SourceObjectRecord(
                document_id=staged.document_id,
                collection_id=staged.collection_id,
                external_id=staged.external_id,
                filename=staged.filename,
                media_type=staged.media_type,
                byte_size=staged.byte_size,
                checksum=staged.checksum,
                storage_backend=staged.storage_backend,
                storage_key=staged.storage_key,
                preview_key=staged.preview_key,
                preview_bytes=staged.preview_bytes,
                preview_checksum=staged.preview_checksum,
                page_count=staged.page_count,
                created_at=previous_source.created_at if previous_source else now,
                updated_at=now,
            )
        if (
            previous is not None
            and previous.current_revision_id is not None
            and previous.current_revision_id != job_id
        ):
            await self.enqueue_cleanup(
                PendingCleanupRecord(
                    resource_type="vector_revision",
                    resource_key=str(previous.current_revision_id),
                )
            )
        if previous_source is not None:
            for record in _source_cleanup_records(previous_source):
                await self.enqueue_cleanup(record)
        await self.update_job(
            job_id,
            status="completed",
            document_id=document.document_id,
            chunk_count=document.chunk_count,
            unchanged=False,
            detail=None,
        )
        return ActivationResult(
            previous_revision_id=(previous.current_revision_id if previous else None),
            previous_source=previous_source,
        )

    async def complete_unchanged(
        self, job_id: UUID, document: DocumentRecord
    ) -> StagedSourceObjectRecord | None:
        now = datetime.now(UTC)
        key = (document.collection_id, document.external_id)
        existing = self.documents[key]
        self.documents[key] = replace(
            document, created_at=existing.created_at, updated_at=now
        )
        if document.provenance_mode == "document":
            self.blocks = {
                key: (
                    replace(block, page_start=document.page, page_end=document.page)
                    if key[0] == document.document_id
                    else block
                )
                for key, block in self.blocks.items()
            }
            self.sections = {
                key: (
                    replace(section, page_start=document.page, page_end=document.page)
                    if key[0] == document.document_id and section.is_root
                    else section
                )
                for key, section in self.sections.items()
            }
        staged = self.staged_source_objects.pop(job_id, None)
        if staged is not None:
            current_source = self.source_objects.get(key)
            if current_source is None:
                self.source_objects[key] = SourceObjectRecord(
                    document_id=staged.document_id,
                    collection_id=staged.collection_id,
                    external_id=staged.external_id,
                    filename=staged.filename,
                    media_type=staged.media_type,
                    byte_size=staged.byte_size,
                    checksum=staged.checksum,
                    storage_backend=staged.storage_backend,
                    storage_key=staged.storage_key,
                    preview_key=staged.preview_key,
                    preview_bytes=staged.preview_bytes,
                    preview_checksum=staged.preview_checksum,
                    page_count=staged.page_count,
                    created_at=now,
                    updated_at=now,
                )
                staged = None
            else:
                self.source_objects[key] = replace(
                    current_source,
                    filename=staged.filename,
                    media_type=staged.media_type,
                    updated_at=now,
                )
                for record in _source_cleanup_records(staged):
                    await self.enqueue_cleanup(record)
        await self.update_job(
            job_id,
            status="completed",
            document_id=document.document_id,
            chunk_count=document.chunk_count,
            unchanged=True,
            detail=None,
        )
        return staged

    async def enqueue_cleanup(self, record: PendingCleanupRecord) -> None:
        self.pending_cleanup[(record.resource_type, record.resource_key)] = record

    async def list_pending_cleanup(self) -> list[PendingCleanupRecord]:
        return list(self.pending_cleanup.values())

    async def complete_cleanup(self, record: PendingCleanupRecord) -> None:
        self.pending_cleanup.pop((record.resource_type, record.resource_key), None)

    async def get_source_object(
        self, collection_id: str, external_id: str
    ) -> SourceObjectRecord | None:
        return self.source_objects.get((collection_id, external_id))

    async def create_job(self, record: JobRecord) -> None:
        if any(
            job.collection_id == record.collection_id
            and job.external_id == record.external_id
            and job.status in ("accepted", "processing")
            for job in self.jobs.values()
        ):
            raise ConcurrentIngestionError(
                "Another ingestion job is already active for this source."
            )
        now = datetime.now(UTC)
        self.jobs[record.job_id] = replace(record, created_at=now, updated_at=now)

    async def update_job(
        self,
        job_id: UUID,
        *,
        status: str,
        document_id: UUID | None = None,
        chunk_count: int | None = None,
        unchanged: bool | None = None,
        detail: str | None = None,
    ) -> None:
        job = self.jobs[job_id]
        if job.status == "completed" and status != "completed":
            return
        self.jobs[job_id] = replace(
            job,
            status=status,
            document_id=document_id if document_id is not None else job.document_id,
            chunk_count=chunk_count if chunk_count is not None else job.chunk_count,
            unchanged=unchanged if unchanged is not None else job.unchanged,
            detail=detail,
            updated_at=datetime.now(UTC),
        )

    async def set_job_conversion_task(
        self,
        job_id: UUID,
        *,
        converter_name: str,
        task_id: str,
        submitted_at: datetime,
    ) -> None:
        job = self.jobs[job_id]
        self.jobs[job_id] = replace(
            job,
            converter_name=converter_name,
            converter_task_id=task_id,
            converter_submitted_at=submitted_at,
            updated_at=datetime.now(UTC),
        )

    async def get_job(self, job_id: UUID) -> JobRecord | None:
        return self.jobs.get(job_id)

    async def get_latest_job(
        self, collection_id: str, external_id: str
    ) -> JobRecord | None:
        matching = [
            job
            for job in self.jobs.values()
            if job.collection_id == collection_id and job.external_id == external_id
        ]
        if not matching:
            return None
        return max(
            matching,
            key=lambda job: job.created_at or datetime.fromtimestamp(0, tz=UTC),
        )

    async def list_resumable_jobs(self) -> list[JobRecord]:
        return sorted(
            (job for job in self.jobs.values() if job.resumable),
            key=lambda job: (
                job.converter_submitted_at or datetime.fromtimestamp(0, tz=UTC)
            ),
        )

    async def fail_interrupted_jobs(self, detail: str) -> list[UUID]:
        failed: list[UUID] = []
        for job_id, job in list(self.jobs.items()):
            if job.status in ("accepted", "processing") and not job.resumable:
                self.jobs[job_id] = replace(
                    job, status="failed", detail=detail, updated_at=datetime.now(UTC)
                )
                failed.append(job_id)
        return failed

    async def get_embedding_state(self) -> EmbeddingState | None:
        return self.embedding_state

    async def set_embedding_state(self, state: EmbeddingState) -> None:
        self.embedding_state = state

    async def probe(self) -> None:
        if self.probe_error is not None:
            raise self.probe_error
