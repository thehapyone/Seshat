"""Hybrid retrieval with mandatory collection isolation and provenance."""

import hashlib
import inspect
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQueryMode,
)

from app.config import Settings
from app.log import logger
from app.models import (
    SEARCH_CONTEXT_METADATA_KEY,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from app.references import section_reference
from app.repository import DocumentRecord, Repository
from app.search_text import search_body


async def search_documents(
    vector_store: BasePydanticVectorStore,
    embed_model: BaseEmbedding,
    repository: Repository,
    request: SearchRequest,
    settings: Settings,
) -> SearchResponse:
    """Retrieve ranked chunks from the requested collections.

    Collection scoping is applied twice: once as a backend metadata filter and
    once as a post-retrieval guard. The guard is what makes isolation a service
    guarantee rather than a property of whichever vector backend is configured.
    """
    allowed = list(dict.fromkeys(request.collection_ids))
    top_k = min(request.top_k or settings.default_top_k, settings.max_top_k)
    documents = _filter_current_documents(
        await repository.list_current_documents(allowed), request
    )
    current_by_revision = {
        str(document.current_revision_id): document
        for document in documents
        if document.current_revision_id is not None
    }
    if not current_by_revision:
        return SearchResponse(items=[])

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    retriever = _build_retriever(
        index,
        similarity_top_k=top_k,
        filters=_build_metadata_filters(allowed, list(current_by_revision)),
        retrieval_mode=settings.retrieval_mode,
    )

    nodes = await retriever.aretrieve(request.query)
    retrieved = len(nodes)

    nodes, leaked = _enforce_current_projection(
        nodes, set(allowed), set(current_by_revision)
    )
    if leaked:
        logger.error(
            "Vector backend returned %d chunk(s) outside the requested current revisions; "
            "they were dropped before the response was built",
            leaked,
        )
    nodes = _dedupe(nodes)
    nodes.sort(key=lambda nws: nws.score or 0.0, reverse=True)
    nodes = nodes[:top_k]

    items = [
        _build_item(nws, current_by_revision[str(nws.node.metadata["revision_id"])])
        for nws in nodes
    ]
    logger.info(
        "Search over %s returned %d of %d retrieved chunk(s)",
        ",".join(allowed),
        len(items),
        retrieved,
    )
    return SearchResponse(items=items)


def _build_metadata_filters(
    collection_ids: list[str],
    revision_ids: list[str],
) -> MetadataFilters:
    filters = [
        MetadataFilter(
            key="collection_id",
            value=collection_ids,
            operator=FilterOperator.IN,
        ),
        MetadataFilter(
            key="revision_id",
            value=revision_ids,
            operator=FilterOperator.IN,
        ),
        MetadataFilter(
            key="projection_state",
            value="current",
            operator=FilterOperator.EQ,
        ),
    ]
    return MetadataFilters(filters=filters, condition=FilterCondition.AND)


def _build_retriever(
    index: VectorStoreIndex,
    *,
    similarity_top_k: int,
    filters: MetadataFilters,
    retrieval_mode: str,
) -> Any:
    kwargs: dict[str, Any] = {"similarity_top_k": similarity_top_k, "filters": filters}
    if retrieval_mode == "hybrid":
        kwargs["vector_store_query_mode"] = VectorStoreQueryMode.HYBRID
    return index.as_retriever(**_supported_kwargs(index.as_retriever, kwargs))


def _supported_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs the retriever factory does not accept."""
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _enforce_current_projection(
    nodes: Iterable[Any],
    allowed: set[str],
    current_revisions: set[str],
) -> tuple[list[Any], int]:
    kept: list[Any] = []
    dropped = 0
    for nws in nodes:
        metadata = nws.node.metadata
        if (
            metadata.get("collection_id") in allowed
            and metadata.get("revision_id") in current_revisions
        ):
            kept.append(nws)
        else:
            dropped += 1
    return kept, dropped


def _filter_current_documents(
    documents: Iterable[DocumentRecord], request: SearchRequest
) -> list[DocumentRecord]:
    """Resolve source filters against current document state before retrieval."""
    extra = request.filters
    if extra is None:
        return list(documents)

    source_types = set(extra.source_type or ())
    external_ids = set(extra.external_id or ())
    excluded_external_ids = set(extra.exclude_external_id or ())
    cutoff = _ensure_tz(extra.updated_after) if extra.updated_after else None

    kept: list[DocumentRecord] = []
    for document in documents:
        if source_types and document.source_type not in source_types:
            continue
        if external_ids and document.external_id not in external_ids:
            continue
        if excluded_external_ids and document.external_id in excluded_external_ids:
            continue
        if cutoff is not None:
            updated_at = document.updated_at
            if updated_at is None or updated_at < cutoff:
                continue
        kept.append(document)
    return kept


def _dedupe(nodes: Iterable[Any]) -> list[Any]:
    """Keep the highest-scoring node per (document, chunk text)."""
    best: dict[tuple[str, str], Any] = {}
    for nws in nodes:
        node = nws.node
        document_key = (
            node.metadata.get("document_id")
            or getattr(node, "ref_doc_id", None)
            or node.node_id
        )
        text_hash = hashlib.sha256(node.get_content().encode("utf-8")).hexdigest()
        key = (str(document_key), text_hash)
        previous = best.get(key)
        if previous is None or (nws.score or 0.0) > (previous.score or 0.0):
            best[key] = nws
    return list(best.values())


def _build_item(nws: Any, document: DocumentRecord) -> SearchResultItem:
    metadata = dict(nws.node.metadata)
    if document.provenance_mode == "document":
        metadata["page"] = document.page
        metadata["section"] = document.section
    score = float(nws.score or 0.0)
    page = _page_number(metadata.get("page"))
    page_end = _page_number(metadata.get("page_end"))
    public_section_ref, section_path = _public_section_provenance(metadata, document)
    if section_path is None and document.provenance_mode == "document":
        section_path = [document.section] if document.section else None
    stored_text = nws.node.get_content()
    stored_context = nws.node.metadata.get(SEARCH_CONTEXT_METADATA_KEY)
    body = search_body(
        stored_text, stored_context if isinstance(stored_context, str) else ""
    )
    return SearchResultItem(
        text=body,
        score=score,
        collection_id=document.collection_id,
        external_id=document.external_id,
        title=document.title or None,
        page=page,
        page_end=page_end if page_end != page else None,
        section_ref=public_section_ref,
        section_path=section_path,
    )


def _public_section_provenance(
    metadata: dict[str, Any], document: DocumentRecord
) -> tuple[str | None, list[str] | None]:
    raw_path = metadata.get("section_path")
    if not (
        isinstance(raw_path, (list, tuple))
        and raw_path
        and all(isinstance(part, str) and part for part in raw_path)
    ):
        return None, None
    path = [part for part in raw_path if isinstance(part, str)]
    revision_id = document.current_revision_id
    raw_section_id = metadata.get("section_id")
    if revision_id is None or raw_section_id is None:
        return None, path
    try:
        section_id = UUID(str(raw_section_id))
    except ValueError:
        return None, path
    return (
        section_reference(
            document.collection_id,
            document.external_id,
            revision_id,
            section_id,
        ),
        path,
    )


def _page_number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _ensure_tz(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)
