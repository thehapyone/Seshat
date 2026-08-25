"""LlamaIndex ingestion pipeline over PostgreSQL/pgvector."""

from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.postgres import PGVectorStore

from app.config import Settings
from app.embeddings.chunking import StructuralChunker
from app.log import logger

# pgvector's HNSW index cannot be built above 2000 dimensions. Larger embeddings
# still work, they just fall back to an exact scan instead of an ANN index.
MAXIMUM_HNSW_DIMENSION = 2000

_HNSW_KWARGS: dict[str, Any] = {
    "hnsw_m": 16,
    "hnsw_ef_construction": 64,
    "hnsw_ef_search": 40,
    "hnsw_dist_method": "vector_cosine_ops",
}


def build_vector_store(settings: Settings) -> PGVectorStore:
    """Create the pgvector store for chunk embeddings."""
    hnsw_kwargs = (
        _HNSW_KWARGS if settings.embedding_dimension <= MAXIMUM_HNSW_DIMENSION else None
    )
    if hnsw_kwargs is None:
        logger.warning(
            "Embedding dimension %d exceeds the pgvector HNSW limit of %d; "
            "creating the vector table without an ANN index",
            settings.embedding_dimension,
            MAXIMUM_HNSW_DIMENSION,
        )
    return PGVectorStore.from_params(
        connection_string=settings.sync_database_url,
        async_connection_string=settings.async_database_url,
        schema_name=settings.db_schema,
        table_name=settings.vector_table,
        embed_dim=settings.embedding_dimension,
        hybrid_search=settings.retrieval_mode == "hybrid",
        text_search_config="english",
        perform_setup=True,
        use_jsonb=True,
        hnsw_kwargs=hnsw_kwargs,
    )


def build_ingestion_pipeline(
    embed_model: BaseEmbedding,
    vector_store: BasePydanticVectorStore,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestionPipeline:
    """Build a chunk + embed pipeline that writes staged rows to *vector_store*.

    No docstore is attached: change detection is handled by Seshat's document
    rows. New chunks carry a private revision id and remain non-current until the
    repository activates them atomically with the canonical representation.
    """
    return IngestionPipeline(
        transformations=[
            StructuralChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            embed_model,
        ],
        vector_store=vector_store,
    )


async def delete_document_nodes(
    vector_store: BasePydanticVectorStore,
    document_id: str,
) -> None:
    """Delete every chunk belonging to one document.

    Deletion is always scoped by ``ref_doc_id``; the store is never cleared.
    """
    try:
        await vector_store.adelete(document_id)
    except Exception:
        logger.warning(
            "Failed to delete existing chunks for document %s",
            document_id,
            exc_info=True,
        )
        raise


async def run_ingestion(
    pipeline: IngestionPipeline,
    nodes: list[BaseNode],
) -> list[BaseNode]:
    """Build and privately store one revision's search projection.

    Segments are handed in as nodes rather than documents because every segment of
    one source must share a single ``ref_doc_id`` — that is what keeps deletion
    scoped to this document — while keeping its own page and section. Documents
    that share an id have their metadata collapsed by the splitter, so the
    document form cannot express that.
    """
    produced = await pipeline.arun(nodes=nodes, show_progress=False)
    return list(produced)
