"""PostgreSQL connection pool."""

import asyncpg

from app.config import Settings


async def create_pool(settings: Settings) -> asyncpg.Pool:
    """Create the shared asyncpg pool for durable ingestion state."""
    return await asyncpg.create_pool(settings.database_url, min_size=1, max_size=10)
