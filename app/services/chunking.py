from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.utils.text import split_into_sentences

logger = get_logger(__name__)

_TOKENIZER = None


def _count_tokens(text: str) -> int:
    """Approximate token count — 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def _split_into_tokens(text: str) -> list[str]:
    """Split text into word-level tokens."""
    return text.split()


def _join_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)


@dataclass
class Chunk:
    index: int
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


# ── Fixed-size chunking ───────────────────────────────────────────────────────

def fixed_chunking(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size - 1  # Or raise a ValueError
        
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    
    tokens = _split_into_tokens(text)
    chunks: list[Chunk] = []

    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _join_tokens(chunk_tokens)
        chunks.append(
            Chunk(index=idx, text=chunk_text, token_count=len(chunk_tokens))
        )
        idx += 1
        if end == len(tokens):
            break
        start += chunk_size - chunk_overlap

    logger.info("chunking.fixed", total_chunks=len(chunks))
    return chunks


# ── Semantic chunking ─────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


async def semantic_chunking(
    text: str,
    embed_fn,
    similarity_threshold: float | None = None,
) -> list[Chunk]:
    settings = get_settings()
    threshold = similarity_threshold or settings.semantic_similarity_threshold

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    embeddings: list[list[float]] = await embed_fn(sentences)

    chunks: list[Chunk] = []
    current_sentences: list[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim >= threshold:
            current_sentences.append(sentences[i])
        else:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=chunk_text,
                    token_count=_count_tokens(chunk_text),
                )
            )
            current_sentences = [sentences[i]]

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(
            Chunk(
                index=len(chunks),
                text=chunk_text,
                token_count=_count_tokens(chunk_text),
            )
        )

    logger.info("chunking.semantic", total_chunks=len(chunks))
    return chunks