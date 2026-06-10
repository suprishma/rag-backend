from __future__ import annotations
from openai import AsyncOpenAI
from app.core.logging import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 100

def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

async def _embed_batch(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    response = await client.embeddings.create(
        model="nomic-embed-text",
        input=texts,
    )
    return [item.embedding for item in response.data]

async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    all_embeddings = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i: i + _BATCH_SIZE]
        embeddings = await _embed_batch(batch)
        all_embeddings.extend(embeddings)
    logger.info("embedding.completed", count=len(all_embeddings))
    return all_embeddings

async def embed_query(text: str) -> list[float]:
    results = await embed_texts([text])
    return results[0]