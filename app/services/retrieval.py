from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.qdrant import search_vectors
from app.schemas.conversation import SourceChunk
from app.services.embedding import embed_query

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


async def retrieve(
    query: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
    filter_doc_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Embed the query and return the top-k matching chunks from Qdrant."""
    settings = get_settings()
    top_k = top_k or settings.rag_top_k
    score_threshold = score_threshold or settings.rag_score_threshold

    query_vector = await embed_query(query)
    scored_points = await search_vectors(
        query_vector=query_vector,
        top_k=top_k,
        score_threshold=score_threshold,
        filter_doc_ids=filter_doc_ids,
    )

    results: list[RetrievedChunk] = []
    for point in scored_points:
        payload = point.payload or {}
        results.append(
            RetrievedChunk(
                document_id=payload.get("document_id", ""),
                filename=payload.get("filename", ""),
                chunk_index=payload.get("chunk_index", 0),
                score=point.score,
                text=payload.get("text", ""),
            )
        )

    logger.info("retrieval.results", query_len=len(query), hits=len(results))
    return results


def to_source_chunks(chunks: list[RetrievedChunk]) -> list[SourceChunk]:
    return [
        SourceChunk(
            document_id=c.document_id,
            filename=c.filename,
            chunk_index=c.chunk_index,
            score=round(c.score, 4),
            text_preview=c.text[:200],
        )
        for c in chunks
    ]