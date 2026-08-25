"""The real docling-serve contract, exercised end to end through the upload API.

The unit tests drive the client against a mock transport, which proves Seshat's
half of the contract but not Docling's. This proves the other half against the
pinned image: that the asynchronous chunk route exists, that a task id stays
addressable across a Seshat restart, and that one file is converted once.

Skipped unless both a database and a Docling instance are pointed at:

    docker run --rm -d -p 55432:5432 -e POSTGRES_PASSWORD=postgres \\
        --name seshat-test-db pgvector/pgvector:pg17
    docker run --rm -d -p 55001:5001 -e DOCLING_SERVE_ENABLE_UI=false \\
        --name seshat-test-docling \\
        quay.io/docling-project/docling-serve-cpu:v1.30.0
    SESHAT_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/postgres \\
    SESHAT_TEST_DOCLING_URL=http://127.0.0.1:55001 \\
        uv run pytest tests/integration/test_docling_live.py -q
"""

import asyncio
import os
import uuid

import asyncpg
import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.embeddings.pipeline import build_ingestion_pipeline, build_vector_store
from app.ingest import IngestionService
from app.main import Runtime, create_app
from app.parsing import DocumentNormalizer
from app.parsing.docling import DoclingClient
from app.repository import PostgresRepository
from app.storage import LocalFileSourceStore
from tests.fakes import VOCAB, DeterministicEmbedding

DATABASE_URL = os.environ.get("SESHAT_TEST_DATABASE_URL", "")
DOCLING_URL = os.environ.get("SESHAT_TEST_DOCLING_URL", "")

pytestmark = pytest.mark.skipif(
    not (DATABASE_URL and DOCLING_URL),
    reason="SESHAT_TEST_DATABASE_URL and SESHAT_TEST_DOCLING_URL are not both set",
)

API_TOKEN = "integration-token-0123456789"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
COLLECTION = "docling-live"
# A conversion on CPU takes well over a minute even for two pages, so the test
# has to be allowed to wait for it.
CONVERSION_TIMEOUT_SECONDS = 480


def two_page_pdf() -> bytes:
    """A minimal PDF with real extractable text on two pages.

    Written by hand rather than fixtured so the repository carries no binary and
    the expected page provenance is visible next to the assertion that checks it.
    """
    pages = [
        ["1 Preventive maintenance", "Inspect the controller module every three years."],
        ["2 Alarms", "The outlet valve alarm persists after calibration."],
    ]
    page_ids = [3 + index for index in range(len(pages))]
    content_ids = [3 + len(pages) + index for index in range(len(pages))]
    font_id = 3 + 2 * len(pages)

    objects: list[tuple[int, bytes]] = [(1, b"<< /Type /Catalog /Pages 2 0 R >>")]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids).encode()
    objects.append(
        (
            2,
            b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(pages)).encode() + b" >>",
        )
    )
    for page_id, content_id in zip(page_ids, content_ids, strict=True):
        objects.append(
            (
                page_id,
                (
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources "
                    f"<< /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
                ).encode(),
            )
        )
    for content_id, lines in zip(content_ids, pages, strict=True):
        body = ["BT", "/F1 18 Tf", "72 720 Td", "18 TL"]
        for line in lines:
            body += [f"({line}) Tj", "T*"]
        stream = "\n".join(body + ["ET"]).encode("latin-1")
        objects.append(
            (
                content_id,
                b"<< /Length "
                + str(len(stream)).encode()
                + b" >>\nstream\n"
                + stream
                + b"\nendstream",
            )
        )
    objects.append((font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number, payload in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode() + payload + b"\nendobj\n"
    start_xref = len(out)
    count = max(offsets) + 1
    out += f"xref\n0 {count}\n".encode() + b"0000000000 65535 f \n"
    for number in range(1, count):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{start_xref}\n%%EOF\n".encode()
    return bytes(out)


@pytest_asyncio.fixture
async def settings(tmp_path) -> Settings:
    schema = f"seshat_live_{uuid.uuid4().hex[:12]}"
    built = Settings.from_document(
        {
            "database": {"schema": schema},
            "embedding": {
                "base_url": "https://example.invalid/openai/v1",
                "model": "deterministic",
                "dimension": len(VOCAB),
            },
            "chunking": {"size": 128, "overlap": 16},
            "retrieval": {"mode": "vector"},
            "storage": {"source_dir": str(tmp_path / "sources")},
            "converter": {"docling": {"base_url": DOCLING_URL, "poll_interval_seconds": 2}},
        },
        env={
            "SESHAT_DATABASE_URL": DATABASE_URL,
            "SESHAT_API_TOKEN": API_TOKEN,
            "SESHAT_CURSOR_SIGNING_KEY": "docling-cursor-key-0123456789012",
            "SESHAT_EMBEDDING_API_KEY": "unused",
        },
    )
    try:
        yield built
    finally:
        pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=1)
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


class CountingDoclingClient(DoclingClient):
    """The production client, counting how often a file is handed to Docling."""

    submissions: list[str] = []

    async def submit(self, **kwargs: object) -> str:
        task_id = await super().submit(**kwargs)  # type: ignore[arg-type]
        CountingDoclingClient.submissions.append(task_id)
        return task_id


@pytest_asyncio.fixture
async def runtime_factory():
    CountingDoclingClient.submissions = []
    async with httpx.AsyncClient() as http:

        async def factory(settings: Settings) -> Runtime:
            pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=4)
            await pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
            repository = PostgresRepository(pool, settings.db_schema)
            await repository.ensure_schema()
            vector_store = build_vector_store(settings)
            embed_model = DeterministicEmbedding()
            source_store = LocalFileSourceStore(settings.source_storage_dir)
            await source_store.prepare()

            async def close() -> None:
                await vector_store.close()
                await pool.close()

            return Runtime(
                repository=repository,
                vector_store=vector_store,
                embed_model=embed_model,
                ingestion=IngestionService(
                    repository=repository,
                    vector_store=vector_store,
                    pipeline=build_ingestion_pipeline(
                        embed_model,
                        vector_store,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                    ),
                    normalizer=DocumentNormalizer(
                        converter=CountingDoclingClient(
                            settings.docling_base_url,
                            http,
                            max_response_bytes=settings.max_document_bytes,
                            request_timeout_seconds=settings.docling_timeout_seconds,
                            deadline_seconds=settings.docling_conversion_deadline_seconds,
                            poll_interval_seconds=settings.docling_poll_interval_seconds,
                        ),
                        max_text_bytes=settings.max_document_bytes,
                    ),
                    source_store=source_store,
                ),
                source_store=source_store,
                close=close,
            )

        yield factory


async def _await_recorded_task(settings: Settings, job_id: uuid.UUID) -> str:
    """Wait until the converter task id is durable, which must precede polling."""
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=1)
    try:
        for _ in range(300):
            row = await pool.fetchrow(
                f"SELECT converter_name, converter_task_id FROM {settings.db_schema}"
                ".ingest_jobs WHERE job_id = $1",
                job_id,
            )
            if row is not None and row["converter_task_id"]:
                assert row["converter_name"] == "docling"
                return row["converter_task_id"]
            await asyncio.sleep(0.2)
    finally:
        await pool.close()
    raise AssertionError("Docling's task id was never recorded on the job.")


async def test_a_real_docling_task_survives_an_seshat_restart(
    settings: Settings, runtime_factory
) -> None:
    content = two_page_pdf()

    app = create_app(settings, runtime_factory=runtime_factory)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://seshat") as api:
            accepted = await api.post(
                "/v1/documents/file",
                files={"file": ("service.pdf", content, "application/pdf")},
                data={"collection_id": COLLECTION, "external_id": "service-manual"},
                headers=HEADERS,
            )
            assert accepted.status_code == 202
            job_id = uuid.UUID(accepted.json()["job_id"])
            task_id = await _await_recorded_task(settings, job_id)

            # Re-uploading while the conversion runs must not start a second one.
            duplicate = await api.post(
                "/v1/documents/file",
                files={"file": ("service.pdf", content, "application/pdf")},
                data={"collection_id": COLLECTION, "external_id": "service-manual"},
                headers=HEADERS,
            )
            assert duplicate.json()["job_id"] == str(job_id)

        # A crash rather than a graceful stop: the polling task is abandoned
        # mid-conversion, exactly as a killed container would abandon it.
        ingestion = app.state.runtime.ingestion
        for task in tuple(ingestion._tasks):  # noqa: SLF001
            task.cancel()
        ingestion._tasks.clear()  # noqa: SLF001

    assert CountingDoclingClient.submissions == [task_id]

    restarted = create_app(settings, runtime_factory=runtime_factory)
    async with restarted.router.lifespan_context(restarted):
        transport = ASGITransport(app=restarted)
        async with AsyncClient(transport=transport, base_url="http://seshat") as api:
            await asyncio.wait_for(
                restarted.state.runtime.ingestion.wait_for_pending(),
                timeout=CONVERSION_TIMEOUT_SECONDS,
            )
            job = (await api.get(f"/v1/jobs/{job_id}", headers=HEADERS)).json()
            found = (
                await api.post(
                    "/v1/search",
                    json={"query": "controller module", "collection_ids": [COLLECTION]},
                    headers=HEADERS,
                )
            ).json()
            listed = (
                await api.get(
                    "/v1/documents", params={"collection_id": COLLECTION}, headers=HEADERS
                )
            ).json()["items"]

    assert job["status"] == "completed", job
    assert job["chunk_count"] >= 1
    # The same Docling task carried the conversion across the restart.
    assert CountingDoclingClient.submissions == [task_id]
    # Page provenance came through the asynchronous chunk route.
    assert found["items"][0]["page"] == 1
    # The chunk route locates passages but does not expose an exact source page
    # count, so Seshat must not infer one from the highest page containing text.
    assert listed[0]["page_count"] is None
