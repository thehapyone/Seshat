"""Shared fixtures: a fully wired application with deterministic infrastructure."""

from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.embeddings.pipeline import build_ingestion_pipeline
from app.ingest import IngestionService
from app.main import Runtime, create_app
from app.parsing import DocumentNormalizer
from tests.fakes import (
    DeterministicEmbedding,
    InMemoryRepository,
    InMemorySourceStore,
    RecordingAsyncConverter,
    RecordingConverter,
    RecordingVectorStore,
)

API_TOKEN = "test-service-token-0123456789"

BASE_SECRETS = {
    "SESHAT_DATABASE_URL": "postgresql://user:pass@db:5432/seshat",
    "SESHAT_API_TOKEN": API_TOKEN,
    "SESHAT_CURSOR_SIGNING_KEY": "test-cursor-signing-key-0123456789",
    "SESHAT_EMBEDDING_API_KEY": "embedding-key",
}

BASE_CONFIG: dict[str, Any] = {
    "embedding": {
        "base_url": "https://example-resource.cognitiveservices.azure.com/openai/v1",
        "model": "text-embedding-3-large",
        "dimension": 12,
    },
    "chunking": {"size": 128, "overlap": 16},
    "retrieval": {"mode": "vector", "default_top_k": 5, "max_top_k": 20},
    "limits": {"max_document_bytes": 4096},
}


@pytest.fixture
def base_secrets() -> dict[str, str]:
    return dict(BASE_SECRETS)


@pytest.fixture
def base_config() -> dict[str, Any]:
    return deepcopy(BASE_CONFIG)


@pytest.fixture
def settings(base_config: dict[str, Any], base_secrets: dict[str, str]) -> Settings:
    return Settings.from_document(base_config, env=base_secrets)


@pytest.fixture
def repository() -> InMemoryRepository:
    return InMemoryRepository()


@pytest.fixture
def vector_store() -> RecordingVectorStore:
    return RecordingVectorStore()


@pytest.fixture
def embed_model() -> DeterministicEmbedding:
    return DeterministicEmbedding()


@pytest.fixture
def converter() -> RecordingConverter:
    return RecordingConverter()


@pytest.fixture
def async_converter() -> RecordingAsyncConverter:
    """A converter whose conversions are resumable by task id, as Docling's are."""
    return RecordingAsyncConverter()


@pytest.fixture
def converter_mode(request: pytest.FixtureRequest) -> str:
    """Which converter double the runtime is wired with.

    Override with
    ``@pytest.mark.parametrize("converter_mode", ["async"], indirect=True)`` to
    exercise the submit-then-poll path the Docling client uses.
    """
    return getattr(request, "param", "sync")


@pytest.fixture
def source_store() -> InMemorySourceStore:
    return InMemorySourceStore()


@pytest.fixture
def conversion_enabled(request: pytest.FixtureRequest) -> bool:
    """Whether the runtime is built with a converter.

    Override per test with
    ``@pytest.mark.parametrize("conversion_enabled", [False], indirect=True)``;
    the runtime is built by the ``client`` fixture, so a test body cannot flip
    this after the fact.
    """
    return getattr(request, "param", True)


@pytest.fixture
def runtime_factory(
    repository: InMemoryRepository,
    vector_store: RecordingVectorStore,
    embed_model: DeterministicEmbedding,
    converter: RecordingConverter,
    async_converter: RecordingAsyncConverter,
    converter_mode: str,
    conversion_enabled: bool,
    source_store: InMemorySourceStore,
):
    """Build the same Runtime shape as production, with deterministic parts."""
    selected = async_converter if converter_mode == "async" else converter

    async def factory(settings: Settings) -> Runtime:
        pipeline = build_ingestion_pipeline(
            embed_model,
            vector_store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        return Runtime(
            repository=repository,
            vector_store=vector_store,
            embed_model=embed_model,
            ingestion=IngestionService(
                repository=repository,
                vector_store=vector_store,
                pipeline=pipeline,
                normalizer=DocumentNormalizer(
                    converter=selected if conversion_enabled else None,
                    max_text_bytes=settings.max_document_bytes,
                ),
                source_store=source_store,
                max_canonical_block_bytes=settings.max_canonical_block_bytes,
            ),
            source_store=source_store,
        )

    return factory


@pytest_asyncio.fixture
async def client(settings: Settings, runtime_factory) -> AsyncIterator[AsyncClient]:
    app = create_app(settings, runtime_factory=runtime_factory)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://seshat") as http_client:
            http_client.app = app  # type: ignore[attr-defined]
            yield http_client


def auth_headers(token: str = API_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def ingest(
    http_client: AsyncClient,
    payload: dict,
    *,
    token: str = API_TOKEN,
) -> dict:
    """POST a document, wait for the background job, and return the job body."""
    response = await http_client.post(
        "/v1/documents/text", json=payload, headers=auth_headers(token)
    )
    response.raise_for_status()
    job_id = response.json()["job_id"]
    await http_client.app.state.runtime.ingestion.wait_for_pending()  # type: ignore[attr-defined]
    job = await http_client.get(f"/v1/jobs/{job_id}", headers=auth_headers(token))
    job.raise_for_status()
    return job.json()
