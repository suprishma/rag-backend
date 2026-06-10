from __future__ import annotations
import json
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

RAG_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based strictly on the provided context.
If the answer cannot be found in the context, say so clearly.
Be concise and accurate.\
"""

BOOKING_DETECTION_PROMPT = """\
You are an assistant that detects whether a user wants to book an interview.
Given the latest user message and conversation history, extract booking details if present.

Respond ONLY with valid JSON (no markdown) in one of these shapes:
1. If booking intended and all fields present:
   {"intent": "book", "name": "...", "email": "...", "date": "YYYY-MM-DD", "time": "HH:MM", "notes": "..."}
2. If booking intended but fields missing:
   {"intent": "incomplete", "missing": ["field1"]}
3. If no booking intent:
   {"intent": "none"}\
"""

def _get_client() -> AsyncOpenAI:
    # Ollama exposes an OpenAI-compatible API at localhost:11434
    return AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required but ignored by Ollama
    )

async def chat_completion(
    messages,
    system_prompt: str = RAG_SYSTEM_PROMPT,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    client = _get_client()
    all_messages = [
        {"role": "system", "content": system_prompt},
        *messages,
    ]
    response = await client.chat.completions.create(
        model="llama3.2",
        messages=all_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""

async def detect_booking_intent(
    user_message: str,
    history_messages: list[dict[str, str]],
) -> dict:
    client = _get_client()
    messages = [
        *[{"role": m["role"], "content": m["content"]} for m in history_messages[-6:]],
        {"role": "user", "content": user_message},
    ]
    all_messages = [
        {"role": "system", "content": BOOKING_DETECTION_PROMPT},
        *messages,
    ]
    response = await client.chat.completions.create(
        model="llama3.2",
        messages=all_messages,
        temperature=0.0,
        max_tokens=256,
    )
    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("llm.booking_parse_failed", raw=raw)
        return {"intent": "none"}