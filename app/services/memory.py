from __future__ import annotations

from app.db.redis import append_message, clear_history, get_history


async def add_user_message(session_id: str, content: str) -> None:
    await append_message(session_id, role="user", content=content)


async def add_assistant_message(session_id: str, content: str) -> None:
    await append_message(session_id, role="assistant", content=content)


async def get_session_history(session_id: str) -> list[dict[str, str]]:
    """Return all messages for the session as a list of {role, content} dicts."""
    return await get_history(session_id)


async def clear_session(session_id: str) -> int:
    return await clear_history(session_id)