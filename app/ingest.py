"""Idempotent text-document ingestion with durable job state."""

import asyncio
import hashlib
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import (
    BaseNode,
    NodeRelationship,
    ObjectType,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from app.embeddings.chunking import CanonicalBlock, build_canonical_blocks
from app.embeddings.pipeline import delete_document_nodes, run_ingestion
from app.log import logger
from app.models import RESERVED_METADATA_KEYS, TextDocumentRequest
from app.parsing import (
    ConvertedDocument,
    DocumentError,
    DocumentNormalizer,
    DocumentSegment,
    UploadedFile,
    resolve_format,
)
from app.repository import (
    ActivationResult,
    ConcurrentIngestionError,
    DocumentRecord,
    JobRecord,
    PendingCleanupRecord,
    Repository,
    StagedSourceObjectRecord,
)
from app.representation import DocumentBlock, DocumentSection, build_document_structure
from app.storage import (
    ORIGINAL_VARIANT,
    PREVIEW_VARIANT,
    SourceObjectStore,
    StorageError,
    storage_key,
)

MAXIMUM_JOB_DETAIL_CHARACTERS = 500
INTERRUPTED_JOB_DETAIL = "Ingestion was interrupted by a service restart."
DEFAULT_MAX_CANONICAL_BLOCK_BYTES = 64_000
UNEXPECTED_JOB_DETAIL = "Document processing failed unexpectedly. Try again or ask the service owner to check the logs."
# Left on a job whose converter task cannot be picked up again because this
# deployment now talks to a different converter than the one holding it.
CONVERTER_CHANGED_DETAIL = (
    "The document converter that was processing this file is no longer configured on this "
    "service. Upload the file again."
)
UNRESUMABLE_UPLOAD_DETAIL = "The upload this conversion belongs to can no longer be identified. Upload the file again."
_UNFINISHED_JOB_STATUSES = ("accepted", "processing")


class MetadataConflictError(ValueError):
    """Raised when caller metadata would overwrite a service-owned key."""


def document_uuid(collection_id: str, external_id: str) -> UUID:
    """Stable document identity for a collection-scoped external ID."""
    # Keep the namespace stable so moving the service does not orphan indexed documents.
    return uuid5(NAMESPACE_URL, f"seshat:{collection_id}:{external_id}")


def text_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class UploadSubmission:
    """One accepted file upload awaiting normalization and indexing."""

    collection_id: str
    external_id: str
    title: str
    upload: UploadedFile


@dataclass(frozen=True, slots=True)
class StagedRepresentation:
    """Validated current-state candidates built under one private revision."""

    sections: tuple[DocumentSection, ...]
    blocks: tuple[DocumentBlock, ...]
    nodes: list[TextNode]


@dataclass(slots=True)
class _DocumentLock:
    """A per-document lock plus the number of holders waiting on it."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class IngestionService:
    """Owns the ingest lifecycle: job creation, change detection, and indexing."""

    def __init__(
        self,
        *,
        repository: Repository,
        vector_store: BasePydanticVectorStore,
        pipeline: IngestionPipeline,
        normalizer: DocumentNormalizer | None = None,
        source_store: SourceObjectStore | None = None,
        max_canonical_block_bytes: int = DEFAULT_MAX_CANONICAL_BLOCK_BYTES,
    ) -> None:
        self._repository = repository
        self._vector_store = vector_store
        self._pipeline = pipeline
        self._normalizer = normalizer
        self._source_store = source_store
        self._max_canonical_block_bytes = max_canonical_block_bytes
        self._locks: dict[tuple[str, str], _DocumentLock] = {}
        # Limit job acceptance, not the potentially long conversion.
        self._accept_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def _document_lock(self, key: tuple[str, str]) -> AsyncIterator[None]:
        """Serialize ingestion per document without leaking a lock per key."""
        entry = self._locks.get(key)
        if entry is None:
            entry = self._locks[key] = _DocumentLock()
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                self._locks.pop(key, None)

    async def submit(self, request: TextDocumentRequest) -> JobRecord:
        """Record an accepted job and index the document in the background."""
        _reject_reserved_metadata(request.metadata)
        async with self._accept_lock:
            pending = await self._unfinished_job(
                request.collection_id, request.external_id
            )
            if pending is not None:
                return pending
            return await self._accept(
                request.collection_id,
                request.external_id,
                lambda job_id: self.run(job_id, request),
                title=request.title,
                source_type=request.source_type,
            )

    async def submit_upload(self, submission: UploadSubmission) -> JobRecord:
        """Record an accepted job, then convert and index the file in the background.

        Conversion happens inside the job rather than inside the HTTP request so a
        slow PDF conversion is observable as ``processing`` instead of holding the
        upload connection open.

        Re-uploading a source whose job is still running returns that job instead
        of starting a second replacement. Original bytes are written under the
        private job id before the accepted job becomes visible.
        """
        async with self._accept_lock:
            pending = await self._unfinished_job(
                submission.collection_id, submission.external_id
            )
            if pending is not None:
                return pending
            if (
                self._source_store is None
            ):  # pragma: no cover - production always has one
                raise StorageError("Source storage is not configured.")

            job = JobRecord(
                job_id=uuid4(),
                collection_id=submission.collection_id,
                external_id=submission.external_id,
                status="accepted",
                title=submission.title,
                source_type=submission.upload.format.source_type,
                filename=submission.upload.filename,
                media_type=submission.upload.media_type,
            )
            key = storage_key(job.job_id, ORIGINAL_VARIANT)
            stored = await self._source_store.put(key, submission.upload.content)
            staged = StagedSourceObjectRecord(
                job_id=job.job_id,
                document_id=document_uuid(
                    submission.collection_id, submission.external_id
                ),
                collection_id=submission.collection_id,
                external_id=submission.external_id,
                filename=submission.upload.filename,
                media_type=submission.upload.media_type,
                byte_size=stored.byte_size,
                checksum=stored.checksum,
                storage_backend=self._source_store.backend,
                storage_key=stored.key,
                preview_key=(
                    storage_key(job.job_id, PREVIEW_VARIANT)
                    if submission.upload.format.needs_conversion
                    else None
                ),
            )
            try:
                await self._repository.create_upload_job(job, staged)
            except ConcurrentIngestionError:
                await self._discard_source_key(
                    stored.key, storage_backend=self._source_store.backend
                )
                pending = await self._unfinished_job(
                    submission.collection_id, submission.external_id
                )
                if pending is not None:
                    return pending
                raise
            except Exception:
                await self._discard_source_key(
                    stored.key, storage_backend=self._source_store.backend
                )
                raise
            self._spawn(job.job_id, lambda job_id: self.run_upload(job_id, submission))
            return job

    async def _accept(
        self,
        collection_id: str,
        external_id: str,
        start: Callable[[UUID], Coroutine[Any, Any, None]],
        *,
        title: str = "",
        source_type: str = "text",
        filename: str | None = None,
        media_type: str | None = None,
    ) -> JobRecord:
        job = JobRecord(
            job_id=uuid4(),
            collection_id=collection_id,
            external_id=external_id,
            status="accepted",
            title=title,
            source_type=source_type,
            filename=filename,
            media_type=media_type,
        )
        try:
            await self._repository.create_job(job)
        except ConcurrentIngestionError:
            pending = await self._unfinished_job(collection_id, external_id)
            if pending is not None:
                return pending
            raise
        self._spawn(job.job_id, start)
        return job

    async def _unfinished_job(
        self, collection_id: str, external_id: str
    ) -> JobRecord | None:
        job = await self._repository.get_latest_job(collection_id, external_id)
        if job is None or job.status not in _UNFINISHED_JOB_STATUSES:
            return None
        logger.info(
            "Source %s/%s is already being processed by job %s",
            collection_id,
            external_id,
            job.job_id,
        )
        return job

    async def resume_conversions(self) -> int:
        """Pick up converter tasks that outlived the last Seshat process.

        Returns the number of jobs resumed. A job whose task belongs to a
        converter this deployment no longer uses is failed here with a retryable
        reason rather than left waiting on something nobody will poll.
        """
        jobs = await self._repository.list_resumable_jobs()
        converter = (
            self._normalizer.resumable_converter
            if self._normalizer is not None
            else None
        )
        resumed = 0
        for job in jobs:
            if converter is None or converter.name != job.converter_name:
                logger.warning(
                    "Cannot resume conversion %s for job %s: converter %r is not configured",
                    job.converter_task_id,
                    job.job_id,
                    job.converter_name,
                )
                await self._record_failure(
                    job.job_id,
                    job.collection_id,
                    job.external_id,
                    DocumentError(CONVERTER_CHANGED_DETAIL),
                )
                continue
            logger.info(
                "Resuming converter task %s for job %s",
                job.converter_task_id,
                job.job_id,
            )

            async def resume_job(_job_id: UUID, persisted_job: JobRecord = job) -> None:
                await self.resume_upload(persisted_job)

            self._spawn(job.job_id, resume_job)
            resumed += 1
        return resumed

    async def discard_failed_staging(
        self, interrupted_revision_ids: tuple[UUID, ...] = ()
    ) -> int:
        """Remove private source/vector state left by jobs failed at startup."""
        staged_sources = await self._repository.discard_failed_staged_sources()
        revision_ids = set(interrupted_revision_ids)
        for source in staged_sources:
            await self._delete_staged_source(source)
            revision_ids.add(source.job_id)
        for revision_id in revision_ids:
            await self._discard_projection(revision_id)
        return len(revision_ids)

    async def retry_pending_cleanup(self) -> int:
        """Retry durable cleanup work left by a prior process or transient failure."""
        completed = 0
        for record in await self._repository.list_pending_cleanup():
            if await self._delete_pending_resource(record):
                completed += 1
        return completed

    def _spawn(
        self,
        job_id: UUID,
        start: Callable[[UUID], Coroutine[Any, Any, None]],
    ) -> None:
        task = asyncio.create_task(self._run_guarded(job_id, start))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_for_pending(self) -> None:
        """Await in-flight ingestion so shutdown does not orphan job state."""
        if not self._tasks:
            return
        await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def _run_guarded(
        self,
        job_id: UUID,
        start: Callable[[UUID], Coroutine[Any, Any, None]],
    ) -> None:
        try:
            await start(job_id)
        except Exception:  # pragma: no cover - run()/run_upload() record failures
            logger.exception("Ingestion job %s crashed", job_id)

    async def run(self, job_id: UUID, request: TextDocumentRequest) -> None:
        """Execute one ingestion job, recording every state transition."""
        async with self._document_lock((request.collection_id, request.external_id)):
            await self._repository.update_job(job_id, status="processing")
            try:
                await self._index(job_id, request)
            except Exception as exc:
                await self._record_failure(
                    job_id, request.collection_id, request.external_id, exc
                )

    async def run_upload(self, job_id: UUID, submission: UploadSubmission) -> None:
        """Normalize an uploaded file and index the text it produced."""
        async with self._document_lock(
            (submission.collection_id, submission.external_id)
        ):
            await self._repository.update_job(job_id, status="processing")
            try:
                if await self._complete_unchanged_upload(job_id, submission):
                    return
                converted = await self._convert(job_id, submission)
                await self._finish_upload(job_id, submission, converted)
            except Exception as exc:
                await self._record_failure(
                    job_id, submission.collection_id, submission.external_id, exc
                )

    async def resume_upload(self, job: JobRecord) -> None:
        """Continue a conversion that was submitted before the last restart.

        Nothing is resubmitted: the converter still holds the file under the task
        id recorded on the job, and its own deadline still runs from when it was
        submitted, so a task that has since expired fails here rather than being
        waited on forever.
        """
        async with self._document_lock((job.collection_id, job.external_id)):
            await self._repository.update_job(job.job_id, status="processing")
            try:
                # A crash can happen after the vector store accepted rows but
                # before activation committed. Resuming the same revision must
                # replace that partial projection rather than append to it.
                await delete_document_nodes(self._vector_store, str(job.job_id))
                submission = self._resumed_submission(job)
                converted = await self._await_conversion(job)
                await self._finish_upload(job.job_id, submission, converted)
            except Exception as exc:
                await self._record_failure(
                    job.job_id, job.collection_id, job.external_id, exc
                )

    def _resumed_submission(self, job: JobRecord) -> UploadSubmission:
        """Rebuild the upload identity a resumed job needs, from its own row.

        The file's bytes are deliberately not reloaded: the converter already has
        them, and everything left to do -- metadata, the preview, the index --
        needs only the identity and the format.
        """
        if self._normalizer is None or not job.filename:  # pragma: no cover - defensive
            raise DocumentError(UNRESUMABLE_UPLOAD_DETAIL)
        upload_format = resolve_format(job.filename, job.media_type)
        return UploadSubmission(
            collection_id=job.collection_id,
            external_id=job.external_id,
            title=job.title,
            upload=UploadedFile(
                filename=job.filename,
                media_type=job.media_type or upload_format.canonical_media_type,
                content=b"",
                format=upload_format,
            ),
        )

    async def _convert(
        self, job_id: UUID, submission: UploadSubmission
    ) -> ConvertedDocument:
        """Produce normalized text for *submission*, recording resumable work first."""
        if self._normalizer is None:  # pragma: no cover - defensive
            raise RuntimeError("File uploads are not enabled on this service.")
        converter = self._normalizer.resumable_converter
        if not submission.upload.format.needs_conversion or converter is None:
            return await self._normalizer.to_document(submission.upload)

        task_id = await self._normalizer.submit_conversion(submission.upload)
        submitted_at = datetime.now(UTC)
        # Persist before polling so a restart resumes rather than resubmits.
        await self._repository.set_job_conversion_task(
            job_id,
            converter_name=converter.name,
            task_id=task_id,
            submitted_at=submitted_at,
        )
        logger.info(
            "Submitted %s to %s as task %s",
            submission.upload.filename,
            converter.name,
            task_id,
        )
        return await self._normalizer.await_conversion(
            task_id, submitted_at=submitted_at
        )

    async def _await_conversion(self, job: JobRecord) -> ConvertedDocument:
        if self._normalizer is None:  # pragma: no cover - defensive
            raise RuntimeError("File uploads are not enabled on this service.")
        if (
            not job.converter_task_id or job.converter_submitted_at is None
        ):  # pragma: no cover
            raise DocumentError(UNRESUMABLE_UPLOAD_DETAIL)
        return await self._normalizer.await_conversion(
            job.converter_task_id, submitted_at=job.converter_submitted_at
        )

    async def _finish_upload(
        self,
        job_id: UUID,
        submission: UploadSubmission,
        converted: ConvertedDocument,
    ) -> None:
        staged = await self._repository.get_staged_source_object(job_id)
        if staged is None:
            raise DocumentError(UNRESUMABLE_UPLOAD_DETAIL)
        await self._record_normalized_form(job_id, submission, converted)
        request = _upload_request(submission, converted)
        await self._index(job_id, request, converted, source_checksum=staged.checksum)

    async def _complete_unchanged_upload(
        self, job_id: UUID, submission: UploadSubmission
    ) -> bool:
        staged = await self._repository.get_staged_source_object(job_id)
        if staged is None:
            raise DocumentError(UNRESUMABLE_UPLOAD_DETAIL)
        existing = await self._repository.get_document(
            submission.collection_id, submission.external_id
        )
        current_source = await self._repository.get_source_object(
            submission.collection_id, submission.external_id
        )
        if (
            existing is None
            or current_source is None
            or existing.checksum != staged.checksum
            or existing.source_type != submission.upload.format.source_type
        ):
            return False

        refreshed = replace(
            existing,
            title=submission.title,
            source_uri=f"upload:{submission.external_id}",
            metadata={
                **existing.metadata,
                "filename": submission.upload.filename,
                "media_type": submission.upload.media_type,
            },
        )
        discarded = await self._repository.complete_unchanged(job_id, refreshed)
        await self._delete_staged_source(discarded)
        logger.info(
            "Source %s/%s is unchanged; updated metadata without conversion or embedding",
            submission.collection_id,
            submission.external_id,
        )
        return True

    async def _record_failure(
        self,
        job_id: UUID,
        collection_id: str,
        external_id: str,
        exc: Exception,
    ) -> None:
        # A DocumentError is an expected outcome for a bad or unconvertible file,
        # so it is logged as a warning rather than as a service fault.
        if isinstance(exc, DocumentError):
            logger.warning(
                "Ingestion rejected %s/%s: %s", collection_id, external_id, exc
            )
        else:
            logger.exception("Ingestion failed for %s/%s", collection_id, external_id)
        current = await self._repository.get_document(collection_id, external_id)
        if current is not None and current.current_revision_id == job_id:
            logger.warning(
                "Ignoring a late failure for already-current revision %s", job_id
            )
            return
        staged = await self._repository.discard_staged_source(job_id)
        await self._delete_staged_source(staged)
        await self._discard_projection(job_id)
        await self._repository.update_job(
            job_id, status="failed", detail=_job_detail(exc)
        )

    async def _record_normalized_form(
        self,
        job_id: UUID,
        submission: UploadSubmission,
        converted: ConvertedDocument,
    ) -> None:
        """Store the normalized text and page reach for the viewer.

        A format that needs conversion gets a stored preview, because its original
        bytes cannot be shown safely inline. Text and Markdown do not: their
        original *is* the readable form, so a second copy would only drift.
        """
        if self._source_store is None:
            return
        preview_key: str | None = None
        preview_bytes: int | None = None
        preview_checksum: str | None = None
        if submission.upload.format.needs_conversion:
            key = storage_key(job_id, PREVIEW_VARIANT)
            stored = await self._source_store.put(key, converted.text.encode("utf-8"))
            preview_key = stored.key
            preview_bytes = stored.byte_size
            # Recorded so the preview's HTTP validator tracks the extracted text
            # rather than the original that produced it.
            preview_checksum = stored.checksum
        await self._repository.set_staged_source_preview(
            job_id,
            preview_key=preview_key,
            preview_bytes=preview_bytes,
            preview_checksum=preview_checksum,
            page_count=converted.page_count,
        )

    async def _index(
        self,
        job_id: UUID,
        request: TextDocumentRequest,
        converted: ConvertedDocument | None = None,
        *,
        source_checksum: str | None = None,
    ) -> None:
        checksum = source_checksum or text_checksum(request.text)
        normalized_checksum = text_checksum(request.text)
        document_id = document_uuid(request.collection_id, request.external_id)

        await self._repository.ensure_collection(request.collection_id)
        existing = await self._repository.get_document(
            request.collection_id, request.external_id
        )

        if (
            existing is not None
            and existing.checksum == checksum
            and existing.normalized_checksum == normalized_checksum
            and existing.source_type == request.source_type
            and existing.chunk_count > 0
            and not request.force_reindex
        ):
            logger.info(
                "Document %s/%s is unchanged; skipping re-embedding",
                request.collection_id,
                request.external_id,
            )
            refreshed = replace(
                existing,
                title=request.title,
                source_type=request.source_type,
                source_uri=request.source_uri,
                version=request.version,
                page=request.page,
                section=request.section,
                metadata=dict(request.metadata),
            )
            discarded = await self._repository.complete_unchanged(job_id, refreshed)
            await self._delete_staged_source(discarded)
            return

        revision_id = job_id
        updated_at = datetime.now(UTC)
        representation = _build_staged_representation(
            request,
            document_id,
            revision_id,
            checksum,
            updated_at,
            converted.segments if converted is not None else (),
            max_canonical_block_bytes=self._max_canonical_block_bytes,
        )
        produced = await run_ingestion(self._pipeline, representation.nodes)
        _validate_search_projection(
            produced,
            representation.sections,
            representation.blocks,
            revision_id,
        )

        document = DocumentRecord(
            document_id=document_id,
            collection_id=request.collection_id,
            external_id=request.external_id,
            title=request.title,
            source_type=request.source_type,
            source_uri=request.source_uri,
            checksum=checksum,
            normalized_checksum=normalized_checksum,
            provenance_mode="block" if converted is not None else "document",
            version=request.version,
            page=request.page,
            section=request.section,
            current_revision_id=revision_id,
            page_count=converted.page_count if converted is not None else None,
            recognized_section_count=(
                converted.recognized_section_count if converted is not None else None
            ),
            recognized_table_count=(
                converted.recognized_table_count if converted is not None else None
            ),
            recognized_figure_count=(
                converted.recognized_figure_count if converted is not None else None
            ),
            metadata=dict(request.metadata),
            chunk_count=len(produced),
        )
        activation = await self._repository.activate_document(
            job_id,
            document,
            representation.sections,
            representation.blocks,
        )
        await self._delete_replaced_resources(activation)
        logger.info(
            "Indexed %d chunk(s) for %s/%s",
            len(produced),
            request.collection_id,
            request.external_id,
        )

    async def _discard_projection(self, revision_id: UUID) -> None:
        record = PendingCleanupRecord(
            resource_type="vector_revision", resource_key=str(revision_id)
        )
        await self._repository.enqueue_cleanup(record)
        await self._delete_pending_resource(record)

    async def _delete_staged_source(
        self, source: StagedSourceObjectRecord | None
    ) -> None:
        if source is None or self._source_store is None:
            return
        keys = {
            source.storage_key,
            storage_key(source.job_id, PREVIEW_VARIANT),
        }
        if source.preview_key is not None:
            keys.add(source.preview_key)
        for key in keys:
            await self._discard_source_key(key, storage_backend=source.storage_backend)

    async def _delete_replaced_resources(self, activation: ActivationResult) -> None:
        if activation.previous_revision_id is not None:
            await self._discard_projection(activation.previous_revision_id)
        source = activation.previous_source
        if source is None or self._source_store is None:
            return
        for key in (source.storage_key, source.preview_key):
            if key is None:
                continue
            await self._discard_source_key(key, storage_backend=source.storage_backend)

    async def _discard_source_key(self, key: str, *, storage_backend: str) -> None:
        record = PendingCleanupRecord(
            resource_type="source_object",
            resource_key=key,
            storage_backend=storage_backend,
        )
        await self._repository.enqueue_cleanup(record)
        await self._delete_pending_resource(record)

    async def _delete_pending_resource(self, record: PendingCleanupRecord) -> bool:
        try:
            if record.resource_type == "vector_revision":
                await delete_document_nodes(self._vector_store, record.resource_key)
            elif (
                self._source_store is not None
                and record.storage_backend == self._source_store.backend
            ):
                await self._source_store.delete(record.resource_key)
            else:
                logger.warning(
                    "Cannot clean source object %s for unavailable backend %r",
                    record.resource_key,
                    record.storage_backend,
                )
                return False
            await self._repository.complete_cleanup(record)
        except Exception:
            logger.warning(
                "Could not remove pending %s %s",
                record.resource_type,
                record.resource_key,
                exc_info=True,
            )
            return False
        return True


def _upload_request(
    submission: UploadSubmission, converted: ConvertedDocument
) -> TextDocumentRequest:
    """The normalized-text request an uploaded file reduces to."""
    upload = submission.upload
    return TextDocumentRequest(
        collection_id=submission.collection_id,
        external_id=submission.external_id,
        text=converted.text,
        title=submission.title,
        source_type=upload.format.source_type,
        source_uri=f"upload:{submission.external_id}",
        metadata={"filename": upload.filename, "media_type": upload.media_type},
    )


def _build_staged_representation(
    request: TextDocumentRequest,
    document_id: UUID,
    revision_id: UUID,
    checksum: str,
    updated_at: datetime,
    segments: tuple[DocumentSegment, ...] = (),
    *,
    max_canonical_block_bytes: int = DEFAULT_MAX_CANONICAL_BLOCK_BYTES,
) -> StagedRepresentation:
    """Build persistable structure and one search input per canonical block.

    Canonical blocks are bounded and non-overlapping before the search pipeline
    introduces retrieval overlap. Every resulting search chunk carries the
    ordinals of its contributing canonical blocks.
    """
    located = segments or (
        DocumentSegment(text=request.text, page=request.page, section=request.section),
    )
    blocks = build_canonical_blocks(
        located, max_rendered_bytes=max_canonical_block_bytes
    )
    structure = build_document_structure(document_id, revision_id, blocks)
    return StagedRepresentation(
        sections=structure.sections,
        blocks=structure.blocks,
        nodes=[
            _build_node(
                request,
                document_id,
                revision_id,
                checksum,
                updated_at,
                block=block,
                section_id=structure.block_section_ids[block.ordinal],
            )
            for block in blocks
        ],
    )


def _validate_search_projection(
    nodes: list[BaseNode],
    sections: tuple[DocumentSection, ...],
    blocks: tuple[DocumentBlock, ...],
    revision_id: UUID,
) -> None:
    """Refuse activation unless every search chunk maps to current staged blocks."""
    if not nodes:
        raise RuntimeError("Indexing produced no searchable content.")
    expected_ordinals = list(range(len(blocks)))
    if [block.ordinal for block in blocks] != expected_ordinals:
        raise RuntimeError("Canonical block ordinals are not contiguous.")
    block_by_ordinal = {block.ordinal: block for block in blocks}
    section_by_id = {section.section_id: section for section in sections}
    available = set(expected_ordinals)
    covered: set[int] = set()
    for node in nodes:
        metadata = node.metadata
        if metadata.get("revision_id") != str(revision_id):
            raise RuntimeError("A search chunk does not belong to the staged revision.")
        if metadata.get("projection_state") != "staged":
            raise RuntimeError(
                "A staged search chunk was marked current before activation."
            )
        ordinals = metadata.get("block_ordinals")
        if not (
            isinstance(ordinals, list)
            and ordinals
            and all(
                isinstance(ordinal, int)
                and not isinstance(ordinal, bool)
                and ordinal in available
                for ordinal in ordinals
            )
        ):
            raise RuntimeError("A search chunk has invalid canonical block provenance.")
        section_ids = {block_by_ordinal[ordinal].section_id for ordinal in ordinals}
        section_id = next(iter(section_ids)) if len(section_ids) == 1 else None
        section = section_by_id.get(section_id) if section_id is not None else None
        if (
            section is None
            or metadata.get("section_id") != str(section.section_id)
            or metadata.get("section_path") != list(section.path)
        ):
            raise RuntimeError(
                "A search chunk does not match its canonical block section."
            )
        covered.update(ordinals)
    if covered != available:
        raise RuntimeError(
            "The search projection does not cover every canonical block."
        )


def _build_nodes(
    request: TextDocumentRequest,
    document_id: UUID,
    checksum: str,
    updated_at: datetime,
    segments: tuple[DocumentSegment, ...] = (),
    *,
    max_canonical_block_bytes: int = DEFAULT_MAX_CANONICAL_BLOCK_BYTES,
) -> list[TextNode]:
    """Compatibility helper for focused tests of ingest-shaped nodes."""
    return _build_staged_representation(
        request,
        document_id,
        document_id,
        checksum,
        updated_at,
        segments,
        max_canonical_block_bytes=max_canonical_block_bytes,
    ).nodes


def _build_node(
    request: TextDocumentRequest,
    document_id: UUID,
    revision_id: UUID,
    checksum: str,
    updated_at: datetime,
    *,
    block: CanonicalBlock,
    section_id: UUID,
) -> TextNode:
    section = block.section_path[-1] if block.section_path else request.section
    metadata: dict[str, object] = {
        "collection_id": request.collection_id,
        "document_id": str(document_id),
        "revision_id": str(revision_id),
        # The vector row is written before activation. PostgreSQL flips this
        # top-level metadata state in the same transaction as the current
        # document/section/block swap; search also verifies the revision pointer.
        "projection_state": "staged",
        "section_id": str(section_id),
        "external_id": request.external_id,
        "title": request.title,
        "source_type": request.source_type,
        "source_uri": request.source_uri,
        "checksum": checksum,
        "version": request.version,
        "section": section,
        "section_path": list(block.section_path),
        "block_ordinals": [block.ordinal],
        "updated_at": updated_at.isoformat(),
        "updated_at_ts": int(updated_at.timestamp()),
    }
    if block.page_start is not None:
        metadata["page"] = block.page_start
    if block.page_end is not None:
        metadata["page_end"] = block.page_end
    if block.source_section_ref is not None:
        metadata["source_section_ref"] = block.source_section_ref
    # Structure the chunker reads and embeddings must not see.
    if block.kind == "table_part":
        metadata["is_table"] = True
        metadata["table_part"] = block.part
        metadata["table_parts"] = block.parts
    if block.table_ref is not None:
        metadata["table_ref"] = block.table_ref
    if block.table_caption:
        metadata["caption"] = block.table_caption
    if block.table_header:
        metadata["table_header"] = block.table_header
    metadata.update(request.metadata)

    node_text = block.text if block.kind == "table_part" else block.rendered_text
    node = TextNode(text=node_text, metadata=metadata)
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
        node_id=str(revision_id), node_type=ObjectType.DOCUMENT
    )

    node.excluded_embed_metadata_keys = list(metadata)
    node.excluded_llm_metadata_keys = list(metadata)
    return node


def _reject_reserved_metadata(metadata: dict[str, object]) -> None:
    conflicts = sorted(set(metadata) & RESERVED_METADATA_KEYS)
    if conflicts:
        raise MetadataConflictError(
            "metadata may not contain service-owned keys: " + ", ".join(conflicts)
        )


def _job_detail(exc: Exception) -> str:
    # DocumentError messages are deliberately user-safe. Unexpected exceptions
    # stay in service logs; provider URLs and database diagnostics must not be
    # persisted into API-visible job state.
    detail = str(exc) if isinstance(exc, DocumentError) else UNEXPECTED_JOB_DETAIL
    if len(detail) > MAXIMUM_JOB_DETAIL_CHARACTERS:
        return f"{detail[:MAXIMUM_JOB_DETAIL_CHARACTERS]}…"
    return detail
