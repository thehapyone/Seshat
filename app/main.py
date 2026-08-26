"""FastAPI application for Seshat."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from uuid import UUID

import httpx
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from pydantic import TypeAdapter, ValidationError

from app.auth import require_api_token
from app.config import MAXIMUM_LISTED_SOURCES, Settings
from app.db import create_pool
from app.embeddings.config import build_embed_model
from app.embeddings.pipeline import build_ingestion_pipeline, build_vector_store
from app.embeddings.retriever import search_documents
from app.ingest import (
    INTERRUPTED_JOB_DETAIL,
    IngestionService,
    MetadataConflictError,
    UploadSubmission,
)
from app.cursors import InvalidScanCursorError
from app.knowledge import ScanLimitError, get_source_outline, scan_source
from app.log import logger, setup_logging
from app.models import (
    CollectionId,
    ExternalId,
    HealthResponse,
    IngestAcceptedResponse,
    JobResponse,
    ScanRequest,
    ScanResponse,
    SearchRequest,
    SearchResponse,
    SourceContentResponse,
    SourceItem,
    SourceListResponse,
    SourceOutlineResponse,
    TextDocumentRequest,
)
from app.parsing import (
    AzureDocumentIntelligenceClient,
    DoclingClient,
    DocumentConverter,
    DocumentDecodeError,
    DocumentError,
    DocumentNormalizer,
    DocumentTooLargeError,
    UploadedFile,
    resolve_format,
    sanitize_filename,
)
from app.repository import (
    EmbeddingState,
    InvalidScanPositionError,
    PostgresRepository,
    Repository,
    SectionNotFoundError,
    SourceChangedError,
    SourceObjectRecord,
    build_source_record,
)
from app.serving import (
    UnsatisfiableRangeError,
    content_disposition,
    parse_range,
    resolve_media_type,
)
from app.storage import (
    ORIGINAL_VARIANT,
    PREVIEW_VARIANT,
    VARIANTS,
    LocalFileSourceStore,
    SourceObjectStore,
    StorageError,
)

API_PREFIX = "/v1"
# Allow room for JSON envelope and multipart framing above the raw payload limit.
REQUEST_OVERHEAD_BYTES = 128 * 1024
# Azure analysis responses include geometry and duplicate text, so their limit
# is independent of extracted-text size. The extracted text is bounded after
# parsing; 500 MiB matches Azure's documented response ceiling.
_AZURE_OCR_MAX_RESPONSE_BYTES = 500 * 1024 * 1024
MAXIMUM_TITLE_CHARACTERS = 512
_COLLECTION_ID = TypeAdapter(CollectionId)
_EXTERNAL_ID = TypeAdapter(ExternalId)
_STORAGE_UNAVAILABLE_DETAIL = (
    "The uploaded file could not be stored, so it was not accepted. Try again or ask "
    "the service owner to check the source storage volume."
)
_UNAVAILABLE_SOURCE_DETAIL = (
    "No original file is stored for this source. Sources indexed before originals "
    "were retained have to be uploaded again to be viewable."
)
_UNAVAILABLE_PREVIEW_DETAIL = "No extracted text preview is stored for this source."


@dataclass(slots=True)
class Runtime:
    """Everything the request handlers need, built once at startup."""

    repository: Repository
    vector_store: BasePydanticVectorStore
    embed_model: BaseEmbedding
    ingestion: IngestionService
    source_store: SourceObjectStore | None = None
    close: Callable[[], Awaitable[None]] | None = None


RuntimeFactory = Callable[[Settings], Awaitable[Runtime]]


class EmbeddingModelMismatchError(RuntimeError):
    """Raised when the configured embedding model differs from the indexed one."""


def _build_converter(
    settings: Settings,
) -> tuple[httpx.AsyncClient | None, DocumentConverter | None]:
    """Build the configured document converter, if any."""
    if not settings.conversion_configured:
        logger.info(
            "%s is unset; only text and Markdown uploads can be normalized",
            "converter.azure.endpoint/AZURE_OCR_API_KEY"
            if settings.document_converter == "azure"
            else "converter.docling.base_url",
        )
        return None, None

    if settings.document_converter == "azure":
        # The converter bounds each request by its remaining conversion deadline.
        http_client = httpx.AsyncClient()
        converter: DocumentConverter = AzureDocumentIntelligenceClient(
            settings.azure_ocr_endpoint,
            settings.azure_ocr_api_key,
            http_client,
            model_id=settings.azure_ocr_model_id,
            timeout_seconds=settings.azure_ocr_timeout_seconds,
            max_response_bytes=_AZURE_OCR_MAX_RESPONSE_BYTES,
        )
        logger.info(
            "Document conversion enabled via Azure Document Intelligence (%s)",
            settings.azure_ocr_endpoint,
        )
        return http_client, converter

    # The converter bounds each request by its remaining conversion deadline.
    http_client = httpx.AsyncClient()
    converter = DoclingClient(
        settings.docling_base_url,
        http_client,
        max_response_bytes=settings.max_document_bytes + REQUEST_OVERHEAD_BYTES,
        request_timeout_seconds=settings.docling_timeout_seconds,
        deadline_seconds=settings.docling_conversion_deadline_seconds,
        poll_interval_seconds=settings.docling_poll_interval_seconds,
    )
    logger.info(
        "Document conversion enabled via %s, deadline %ds",
        settings.docling_base_url,
        settings.docling_conversion_deadline_seconds,
    )
    return http_client, converter


async def build_runtime(settings: Settings) -> Runtime:
    """Create the production runtime: PostgreSQL, pgvector, and the embedder."""
    pool = await create_pool(settings)
    repository = PostgresRepository(
        pool, settings.db_schema, vector_table=settings.vector_table
    )
    await repository.ensure_schema()
    await repository.probe()

    embed_model = build_embed_model(settings)
    vector_store = build_vector_store(settings)
    # Fail startup on missing vector storage rather than the first ingest.
    await vector_store.adelete("startup-probe")
    pipeline = build_ingestion_pipeline(
        embed_model,
        vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    converter_http, converter = _build_converter(settings)
    source_store = LocalFileSourceStore(settings.source_storage_dir)
    # Fail startup when retained uploads could not be read back.
    await source_store.prepare()
    ingestion = IngestionService(
        repository=repository,
        vector_store=vector_store,
        pipeline=pipeline,
        normalizer=DocumentNormalizer(
            converter=converter, max_text_bytes=settings.max_document_bytes
        ),
        source_store=source_store,
        max_canonical_block_bytes=settings.max_canonical_block_bytes,
    )

    async def close() -> None:
        if converter_http is not None:
            await converter_http.aclose()
        await vector_store.close()
        await pool.close()

    return Runtime(
        repository=repository,
        vector_store=vector_store,
        embed_model=embed_model,
        ingestion=ingestion,
        source_store=source_store,
        close=close,
    )


async def prepare_runtime(runtime: Runtime, settings: Settings) -> None:
    """Reconcile durable state before the service accepts traffic."""
    await verify_embedding_state(runtime.repository, settings)
    interrupted = await runtime.repository.fail_interrupted_jobs(INTERRUPTED_JOB_DETAIL)
    if interrupted:
        logger.warning(
            "Marked %d interrupted ingestion job(s) as failed", len(interrupted)
        )
    discarded = await runtime.ingestion.discard_failed_staging(tuple(interrupted))
    if discarded:
        logger.warning("Discarded staging for %d failed ingestion job(s)", discarded)
    cleaned = await runtime.ingestion.retry_pending_cleanup()
    if cleaned:
        logger.info("Completed %d deferred resource cleanup operation(s)", cleaned)
    resumed = await runtime.ingestion.resume_conversions()
    if resumed:
        logger.info("Resumed %d in-flight document conversion(s)", resumed)


async def verify_embedding_state(repository: Repository, settings: Settings) -> None:
    """Refuse to start when the index was built with a different embedding model.

    Re-embedding on drift is deliberately not automatic: it would rewrite every
    collection, and one collection's configuration change must never invalidate
    the others.
    """
    desired = EmbeddingState(
        model_name=settings.embedding_model, model_dim=settings.embedding_dimension
    )
    stored = await repository.get_embedding_state()
    if stored is None:
        await repository.set_embedding_state(desired)
        return
    if stored.model_name != desired.model_name or stored.model_dim != desired.model_dim:
        raise EmbeddingModelMismatchError(
            "The indexed embedding model "
            f"({stored.model_name}/{stored.model_dim}) does not match the configured model "
            f"({desired.model_name}/{desired.model_dim}). Point database.schema at a fresh "
            "schema or drop the existing schema and re-ingest."
        )


def create_app(
    settings: Settings, runtime_factory: RuntimeFactory | None = None
) -> FastAPI:
    """Build the FastAPI application for *settings*."""
    setup_logging(settings.log_level)
    factory = runtime_factory or build_runtime

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logger.info("Starting Seshat with %s", settings.redacted())
        runtime = await factory(settings)
        try:
            await prepare_runtime(runtime, settings)
        except BaseException:
            # A failed startup must not leak the pool or vector-store connections.
            if runtime.close is not None:
                await runtime.close()
            raise
        application.state.runtime = runtime
        try:
            yield
        finally:
            application.state.runtime = None
            await runtime.ingestion.wait_for_pending()
            if runtime.close is not None:
                await runtime.close()

    application = FastAPI(title="Seshat", version="1.0.0", lifespan=lifespan)
    application.state.settings = settings
    application.state.runtime = None

    maximum_request_bytes = (
        max(settings.max_document_bytes, settings.max_upload_bytes)
        + REQUEST_OVERHEAD_BYTES
    )

    @application.middleware("http")
    async def limit_request_size(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit():
            if int(declared) > maximum_request_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body is too large."},
                )
        return await call_next(request)

    application.include_router(_health_router())
    application.include_router(_api_router())
    return application


def create_app_from_config() -> FastAPI:
    """Uvicorn factory entrypoint (``uvicorn app.main:create_app_from_config --factory``)."""
    return create_app(Settings.load())


def _runtime(request: Request) -> Runtime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The service is not ready.",
        )
    return runtime


def _validated_collection_id(value: str) -> str:
    try:
        return _COLLECTION_ID.validate_python(value)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "collection_id must be 1-63 lowercase characters using letters, digits, "
                "'.', '_' or '-'."
            ),
        ) from exc


def _validated_identity(collection_id: str, external_id: str) -> tuple[str, str]:
    """Validate both halves of a source's identity before any lookup."""
    collection = _validated_collection_id(collection_id)
    try:
        external = _EXTERNAL_ID.validate_python(external_id)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail="external_id must be 1-256 characters."
        ) from exc
    return collection, external


def _knowledge_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def _variant_key(record: SourceObjectRecord, variant: str) -> str | None:
    if variant == PREVIEW_VARIANT:
        return record.preview_key
    return record.storage_key


def _variant_filename(record: SourceObjectRecord, variant: str) -> str:
    if variant == PREVIEW_VARIANT:
        return f"{record.filename}.extracted.txt"
    return record.filename


def _entity_tag(record: SourceObjectRecord, variant: str) -> str:
    """Strong validator for one representation of one source."""
    if variant == PREVIEW_VARIANT:
        # Rows written before the preview hash was recorded fall back to the byte
        # count, which is the only thing about the old preview that was stored.
        # Reprocessing writes a real hash, so a preview that changes is covered.
        identity = (
            record.preview_checksum
            or f"legacy-{record.checksum}-{record.preview_bytes or 0}"
        )
        return f'"{identity}-{PREVIEW_VARIANT}"'
    return f'"{record.checksum}-{ORIGINAL_VARIANT}"'


async def _read_upload(file: UploadFile, limit: int) -> bytes:
    """Read at most *limit* bytes, bounding resident memory per upload.

    The declared request size is refused earlier by the size middleware; this
    also stops a body that understated its ``Content-Length`` from being loaded
    in full. Starlette spools the part to a temporary file first, so the
    container's ``/tmp`` must be larger than ``limits.max_upload_bytes``.
    """
    chunks: list[bytes] = []
    received = 0
    while chunk := await file.read(64 * 1024):
        received += len(chunk)
        if received > limit:
            raise DocumentTooLargeError(
                f"The file is larger than the {limit // (1024 * 1024)} MiB upload limit."
            )
        chunks.append(chunk)
    if not received:
        raise DocumentDecodeError("The uploaded file is empty.")
    return b"".join(chunks)


def _document_error(exc: DocumentError) -> HTTPException:
    status_code = 413 if isinstance(exc, DocumentTooLargeError) else 415
    if isinstance(exc, DocumentDecodeError):
        status_code = 422
    return HTTPException(status_code=status_code, detail=str(exc))


def _health_router() -> APIRouter:
    router = APIRouter(tags=["health"])

    @router.get("/health/live", response_model=HealthResponse)
    async def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @router.get("/health/ready", response_model=HealthResponse)
    async def ready(request: Request, response: Response) -> HealthResponse:
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(
                status="starting", detail="Runtime is not initialised."
            )
        try:
            await runtime.repository.probe()
        except Exception as exc:
            logger.warning("Readiness probe failed: %s", exc)
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(
                status="unavailable",
                database="error",
                detail="Database is unreachable.",
            )
        return HealthResponse(status="ready", database="ok")

    return router


def _api_router() -> APIRouter:
    router = APIRouter(prefix=API_PREFIX, dependencies=[Depends(require_api_token)])

    @router.post(
        "/documents/text",
        response_model=IngestAcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["documents"],
    )
    async def ingest_text(
        payload: TextDocumentRequest, request: Request
    ) -> IngestAcceptedResponse:
        settings: Settings = request.app.state.settings
        if len(payload.text.encode("utf-8")) > settings.max_document_bytes:
            raise HTTPException(
                status_code=413,
                detail="Document text exceeds the configured maximum size.",
            )
        runtime = _runtime(request)
        try:
            job = await runtime.ingestion.submit(payload)
        except MetadataConflictError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IngestAcceptedResponse(
            job_id=job.job_id,
            status=job.status,
            collection_id=job.collection_id,
            external_id=job.external_id,
        )

    @router.post(
        "/documents/file",
        response_model=IngestAcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["documents"],
    )
    async def ingest_file(
        request: Request,
        file: UploadFile = File(..., description="One document to index."),
        collection_id: str = Form(...),
        external_id: str = Form(...),
        title: str = Form(""),
    ) -> IngestAcceptedResponse:
        """Accept one bounded file and index it in the background."""
        settings: Settings = request.app.state.settings
        collection, external = _validated_identity(collection_id, external_id)
        try:
            filename = sanitize_filename(
                file.filename or "", maximum_characters=settings.max_filename_characters
            )
            document_format = resolve_format(filename, file.content_type)
            content = await _read_upload(file, settings.max_upload_bytes)
        except DocumentError as exc:
            raise _document_error(exc) from exc

        runtime = _runtime(request)
        submission = UploadSubmission(
            collection_id=collection,
            external_id=external,
            title=(title.strip() or filename)[:MAXIMUM_TITLE_CHARACTERS],
            upload=UploadedFile(
                filename=filename,
                media_type=document_format.canonical_media_type,
                content=content,
                format=document_format,
            ),
        )
        try:
            job = await runtime.ingestion.submit_upload(submission)
        except StorageError as exc:
            logger.error("Could not retain the original upload: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=_STORAGE_UNAVAILABLE_DETAIL,
            ) from exc
        return IngestAcceptedResponse(
            job_id=job.job_id,
            status=job.status,
            collection_id=job.collection_id,
            external_id=job.external_id,
        )

    @router.get("/documents", response_model=SourceListResponse, tags=["documents"])
    async def list_documents(
        request: Request,
        collection_id: str = Query(...),
    ) -> SourceListResponse:
        """List the sources in one collection, including in-flight uploads."""
        collection = _validated_collection_id(collection_id)
        runtime = _runtime(request)
        records = await runtime.repository.list_sources(
            collection, limit=MAXIMUM_LISTED_SOURCES
        )
        return SourceListResponse(
            collection_id=collection,
            items=[SourceItem(**asdict(record)) for record in records],
            truncated=len(records) >= MAXIMUM_LISTED_SOURCES,
        )

    @router.get(
        "/documents/source", response_model=SourceContentResponse, tags=["documents"]
    )
    async def get_source(
        request: Request,
        collection_id: str = Query(...),
        external_id: str = Query(...),
    ) -> SourceContentResponse:
        """Describe the stored original for one source in one collection."""
        collection, external = _validated_identity(collection_id, external_id)
        runtime = _runtime(request)
        record = await runtime.repository.get_source_object(collection, external)
        if record is None:
            raise HTTPException(status_code=404, detail=_UNAVAILABLE_SOURCE_DETAIL)
        document = await runtime.repository.get_document(collection, external)
        job = await runtime.repository.get_latest_job(collection, external)
        source = (
            build_source_record(document, job, record)
            if document is not None or job is not None
            else None
        )
        return SourceContentResponse(
            collection_id=collection,
            external_id=external,
            title=(source.title if source else record.filename),
            source_type=(source.source_type if source else "upload"),
            status=(source.status if source else "processing"),
            filename=record.filename,
            media_type=record.media_type,
            byte_size=record.byte_size,
            checksum=record.checksum,
            page_count=record.page_count,
            preview_available=bool(record.preview_key),
            preview_bytes=record.preview_bytes,
            chunk_count=(source.chunk_count if source else 0),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @router.get(
        "/documents/source/outline",
        response_model=SourceOutlineResponse,
        response_model_exclude_unset=True,
        tags=["knowledge"],
    )
    async def get_source_document_outline(
        request: Request,
        collection_id: str = Query(...),
        external_id: str = Query(...),
    ) -> SourceOutlineResponse:
        """Return normalized structural facts for one current source."""
        collection, external = _validated_identity(collection_id, external_id)
        outline = await get_source_outline(
            _runtime(request).repository, collection, external
        )
        if outline is None:
            raise _knowledge_error(404, "source_not_found", "Current source not found.")
        return outline

    @router.get("/documents/source/content", tags=["documents"])
    async def get_source_content(
        request: Request,
        collection_id: str = Query(...),
        external_id: str = Query(...),
        variant: str = Query(ORIGINAL_VARIANT),
    ) -> Response:
        """Stream the stored bytes for one source, with Range support."""
        collection, external = _validated_identity(collection_id, external_id)
        if variant not in VARIANTS:
            raise HTTPException(
                status_code=422,
                detail=f"variant must be one of: {', '.join(VARIANTS)}.",
            )
        runtime = _runtime(request)
        if runtime.source_store is None:  # pragma: no cover - storage is always built
            raise HTTPException(status_code=404, detail=_UNAVAILABLE_SOURCE_DETAIL)
        record = await runtime.repository.get_source_object(collection, external)
        if record is None:
            raise HTTPException(status_code=404, detail=_UNAVAILABLE_SOURCE_DETAIL)

        key = _variant_key(record, variant)
        if key is None:
            raise HTTPException(status_code=404, detail=_UNAVAILABLE_PREVIEW_DETAIL)
        # Storage is authoritative for the response length and range calculations.
        stored = await runtime.source_store.stat(key)
        if stored is None:
            raise HTTPException(status_code=404, detail=_UNAVAILABLE_SOURCE_DETAIL)
        byte_size = stored.byte_size

        media_type, inline = resolve_media_type(record.media_type, variant=variant)
        # Per-representation validators make viewer reloads conditional and private.
        etag = _entity_tag(record, variant)
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, no-cache",
            "ETag": etag,
            "Content-Disposition": content_disposition(
                _variant_filename(record, variant), inline=inline
            ),
            "X-Content-Type-Options": "nosniff",
        }
        if (
            request.headers.get("if-none-match") == etag
            and "range" not in request.headers
        ):
            return Response(status_code=304, headers=headers)
        try:
            requested = parse_range(request.headers.get("range"), byte_size)
        except UnsatisfiableRangeError:
            return Response(
                status_code=416,
                headers={**headers, "Content-Range": f"bytes */{byte_size}"},
            )

        offset = requested.start if requested else 0
        length = requested.length if requested else byte_size
        if requested is not None:
            headers["Content-Range"] = (
                f"bytes {requested.start}-{requested.end}/{byte_size}"
            )
        headers["Content-Length"] = str(length)

        store = runtime.source_store

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in store.read(key, offset=offset, length=length):
                    yield chunk
            except StorageError:
                # The response has already started, so the only honest signal left
                # is a truncated body; the log carries the reason.
                logger.error(
                    "Source content for %s/%s became unreadable mid-response",
                    collection,
                    external,
                    exc_info=True,
                )
                raise

        return StreamingResponse(
            stream(),
            status_code=206 if requested is not None else 200,
            media_type=media_type,
            headers=headers,
        )

    @router.get("/jobs/{job_id}", response_model=JobResponse, tags=["documents"])
    async def get_job(job_id: UUID, request: Request) -> JobResponse:
        runtime = _runtime(request)
        job = await runtime.repository.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found."
            )
        return JobResponse(**asdict(job))

    @router.post(
        "/search",
        response_model=SearchResponse,
        response_model_exclude_none=True,
        tags=["search"],
    )
    async def search(payload: SearchRequest, request: Request) -> SearchResponse:
        runtime = _runtime(request)
        settings: Settings = request.app.state.settings
        return await search_documents(
            runtime.vector_store,
            runtime.embed_model,
            runtime.repository,
            payload,
            settings,
        )

    @router.post(
        "/scan",
        response_model=ScanResponse,
        response_model_exclude_unset=True,
        tags=["knowledge"],
    )
    async def scan(payload: ScanRequest, request: Request) -> ScanResponse:
        runtime = _runtime(request)
        settings: Settings = request.app.state.settings
        try:
            response = await scan_source(runtime.repository, payload, settings)
        except ScanLimitError as exc:
            raise _knowledge_error(422, "invalid_limit", str(exc)) from exc
        except InvalidScanCursorError as exc:
            raise _knowledge_error(400, "invalid_cursor", str(exc)) from exc
        except SourceChangedError as exc:
            raise _knowledge_error(409, "source_changed", str(exc)) from exc
        except SectionNotFoundError as exc:
            raise _knowledge_error(404, "section_not_found", str(exc)) from exc
        except InvalidScanPositionError as exc:
            raise _knowledge_error(400, "invalid_cursor", str(exc)) from exc
        if response is None:
            raise _knowledge_error(404, "source_not_found", "Current source not found.")
        return response

    return router
