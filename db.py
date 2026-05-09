from __future__ import annotations

import logging

import asyncpg

from config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS connections (
    connection_id TEXT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    can_reply BOOLEAN NOT NULL DEFAULT TRUE,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    connection_id TEXT NOT NULL REFERENCES connections(connection_id),
    chat_id BIGINT NOT NULL,
    tg_message_id INTEGER,
    from_user_id BIGINT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    has_embedding BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(connection_id, chat_id, created_at);
"""


async def init_db() -> None:
    global _pool
    _pool = await asyncpg.create_pool(settings.database_url)
    async with _pool.acquire() as conn:
        await conn.execute(_SCHEMA)
    logger.info("Database initialized")


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialized. Call init_db() first.")
    return _pool


async def upsert_connection(
    connection_id: str,
    user_id: int,
    can_reply: bool,
    is_enabled: bool,
) -> None:
    await _get_pool().execute(
        """
        INSERT INTO connections (connection_id, user_id, can_reply, is_enabled, updated_at)
        VALUES ($1, $2, $3, $4, NOW())
        ON CONFLICT (connection_id) DO UPDATE
            SET user_id = $2, can_reply = $3, is_enabled = $4, updated_at = NOW()
        """,
        connection_id,
        user_id,
        can_reply,
        is_enabled,
    )


async def get_connection(connection_id: str) -> dict | None:
    row = await _get_pool().fetchrow(
        "SELECT connection_id, user_id, can_reply, is_enabled FROM connections WHERE connection_id = $1",
        connection_id,
    )
    return dict(row) if row else None


async def save_message(
    connection_id: str,
    chat_id: int,
    tg_message_id: int | None,
    from_user_id: int,
    role: str,
    content: str,
) -> int:
    return await _get_pool().fetchval(
        """
        INSERT INTO messages (connection_id, chat_id, tg_message_id, from_user_id, role, content)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """,
        connection_id,
        chat_id,
        tg_message_id,
        from_user_id,
        role,
        content,
    )


async def mark_embedding(message_id: int) -> None:
    await _get_pool().execute(
        "UPDATE messages SET has_embedding = TRUE WHERE id = $1",
        message_id,
    )


async def get_recent_messages(
    connection_id: str, chat_id: int, limit: int
) -> list[dict]:
    rows = await _get_pool().fetch(
        """
        SELECT id, role, content, created_at
        FROM messages
        WHERE connection_id = $1 AND chat_id = $2
        ORDER BY created_at DESC
        LIMIT $3
        """,
        connection_id,
        chat_id,
        limit,
    )
    return [dict(r) for r in reversed(rows)]


async def get_messages_by_ids(ids: list[int]) -> list[dict]:
    if not ids:
        return []
    rows = await _get_pool().fetch(
        """
        SELECT id, role, content, created_at
        FROM messages
        WHERE id = ANY($1::int[])
        ORDER BY created_at
        """,
        ids,
    )
    return [dict(r) for r in rows]
