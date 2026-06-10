from __future__ import annotations

from openai.types.chat import ChatCompletionMessageParam
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.schemas.conversation import MessageResponse, SourceChunk
from app.services.booking import try_extract_and_save_booking
from app.services.llm import RAG_SYSTEM_PROMPT, chat_completion
from app.services.memory import (
    add_assistant_message,
    add_user_message,
    get_session_history,
)
from app.services.retrieval import retrieve, to_source_chunks

logger = get_logger(__name__)

_CONTEXT_WINDOW = 10   # number of previous messages to include in LLM prompt


def _build_rag_prompt(
    user_message: str,
    context_chunks: list[str],
    history: list[dict[str, str]],
) -> list[ChatCompletionMessageParam]:
    """Construct the messages list for the LLM: history + context + user query."""
    context_block = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."

    # Include last N message pairs from history
    recent_history = history[-_CONTEXT_WINDOW:]

    messages: list[ChatCompletionMessageParam] = [
        *[{"role": m["role"], "content": m["content"]} for m in recent_history],
        {
            "role": "user",
            "content": (
                f"Context from documents:\n\n{context_block}\n\n"
                f"User question: {user_message}"
            ),
        },
    ]
    return messages


async def process_message(
    *,
    session_id: str,
    user_message: str,
    db: AsyncSession,
) -> MessageResponse:
    # ── 1. Load chat history ──────────────────────────────────────────────────
    history = await get_session_history(session_id)

    # ── 2. Check for booking intent in parallel with retrieval ────────────────
    booking_record = await try_extract_and_save_booking(
        session_id=session_id,
        user_message=user_message,
        history=history,
        db=db,
    )

    # ── 3. Retrieve relevant chunks ───────────────────────────────────────────
    retrieved = await retrieve(user_message)
    context_texts = [c.text for c in retrieved]
    source_chunks: list[SourceChunk] = to_source_chunks(retrieved)

    # ── 4. Build prompt & call LLM ────────────────────────────────────────────
    messages = _build_rag_prompt(
        user_message=user_message,
        context_chunks=context_texts,
        history=history,
    )
    answer = await chat_completion(messages=messages, system_prompt=RAG_SYSTEM_PROMPT)

    # ── 5. If booking was just created, append confirmation ───────────────────
    if booking_record:
        answer = (
            f"{answer}\n\n✅ Your interview has been booked!\n"
            f"**Name:** {booking_record.name}\n"
            f"**Email:** {booking_record.email}\n"
            f"**Date:** {booking_record.date} at {booking_record.time}"
        )

    # ── 6. Persist to Redis memory ────────────────────────────────────────────
    await add_user_message(session_id, user_message)
    await add_assistant_message(session_id, answer)

    logger.info(
        "conversation.processed",
        session_id=session_id,
        sources=len(source_chunks),
        booking=bool(booking_record),
    )

    return MessageResponse(
        session_id=session_id,
        answer=answer,
        sources=source_chunks,
        booking_created=bool(booking_record),
        booking_id=booking_record.id if booking_record else None,
    )