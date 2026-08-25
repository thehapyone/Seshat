"""The production startup path against a real database.

Proves the service starts from the shipped example configuration plus the
documented secrets and PostgreSQL, with no embedding call involved. Skipped
unless ``SESHAT_TEST_DATABASE_URL`` is set.
"""

import os
import tomllib
import uuid
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]

DATABASE_URL = os.environ.get("SESHAT_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="SESHAT_TEST_DATABASE_URL is not set"
)

API_TOKEN = "integration-token-0123456789"


@pytest_asyncio.fixture
async def documented_settings(tmp_path):
    """Settings built the way a deployment builds them: the example file plus secrets."""
    schema = f"seshat_start_{uuid.uuid4().hex[:12]}"
    document = tomllib.loads((REPO_ROOT / "seshat.example.toml").read_text(encoding="utf-8"))
    document.setdefault("database", {})["schema"] = schema
    # The deployment defaults live inside the container; a test run needs a
    # directory it can actually write to and no Docling instance to reach.
    document.setdefault("storage", {})["source_dir"] = str(tmp_path / "sources")
    document["converter"]["docling"]["base_url"] = ""
    try:
        yield Settings.from_document(
            document,
            env={
                "SESHAT_DATABASE_URL": DATABASE_URL,
                "SESHAT_API_TOKEN": API_TOKEN,
                "SESHAT_CURSOR_SIGNING_KEY": "startup-cursor-key-0123456789012",
                "SESHAT_EMBEDDING_API_KEY": "unused-during-startup",
            },
            origin="seshat.example.toml",
        )
    finally:
        pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=1)
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


async def test_service_starts_and_serves_health_and_auth(documented_settings: Settings) -> None:
    app = create_app(documented_settings)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://seshat") as client:
            live = await client.get("/health/live")
            ready = await client.get("/health/ready")
            unauthorized = await client.post(
                "/v1/search", json={"query": "x", "collection_ids": ["example-collection"]}
            )
            missing_job = await client.get(
                f"/v1/jobs/{uuid.uuid4()}", headers={"Authorization": f"Bearer {API_TOKEN}"}
            )

    assert live.status_code == 200
    assert ready.status_code == 200 and ready.json() == {
        "status": "ready",
        "database": "ok",
        "detail": None,
    }
    assert unauthorized.status_code == 401
    assert missing_job.status_code == 404


async def test_startup_creates_the_schema_and_vector_table(documented_settings: Settings) -> None:
    settings = documented_settings
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        pass

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=1)
    try:
        tables = {
            row["tablename"]
            for row in await pool.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = $1", settings.db_schema
            )
        }
    finally:
        await pool.close()

    assert {
        "collections",
        "documents",
        "ingest_jobs",
        "source_objects",
        "embedding_state",
    } <= tables
    assert any(name.endswith(settings.vector_table) for name in tables), tables
