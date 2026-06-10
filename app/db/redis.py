from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_pool: aioredis.ConnectionPool | None = None


def _get_pool() -> aioredis.ConnectionPool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = aioredis.ConnectionPool.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=20,
        )
    return _pool


def get_redis() -> aioredis.Redis:
    return aioredis.Redis(connection_pool=_get_pool())


# ── Chat history helpers ──────────────────────────────────────────────────────

def _history_key(session_id: str) -> str:
    return f"chat:history:{session_id}"


async def append_message(session_id: str, role: str, content: str) -> None:
    """Append a single message to the session history list."""
    settings = get_settings()
    redis = get_redis()
    key = _history_key(session_id)
    message = json.dumps({"role": role, "content": content})
    await redis.rpush(key, message)
    await redis.expire(key, settings.redis_chat_ttl_seconds)
    logger.debug("redis.message_appended", session_id=session_id, role=role)


async def get_history(session_id: str) -> list[dict[str, str]]:
    """Return full message history for a session."""
    redis = get_redis()
    raw: list[str] = await redis.lrange(_history_key(session_id), 0, -1)
    return [json.loads(m) for m in raw]


async def clear_history(session_id: str) -> int:
    """Delete all messages for a session. Returns number of keys deleted."""
    redis = get_redis()
    deleted: int = await redis.delete(_history_key(session_id))
    logger.info("redis.history_cleared", session_id=session_id)
    return deleted


async def close_redis() -> None:
    global _pool
    if _pool:
        await _pool.disconnect()
        _pool = None