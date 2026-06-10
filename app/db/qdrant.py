from __future__ import annotations

from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_client: AsyncQdrantClient | None = None


async def get_qdrant_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        settings = get_settings()
        kwargs: dict[str, Any] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _client = AsyncQdrantClient(**kwargs)
        await _ensure_collection(_client)
    return _client


async def _ensure_collection(client: AsyncQdrantClient) -> None:
    settings = get_settings()
    name = settings.qdrant_collection_name
    try:
        await client.get_collection(name)
        logger.info("qdrant.collection_exists", collection=name)
    except (UnexpectedResponse, Exception):
        await client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=settings.openai_embedding_dimensions,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info("qdrant.collection_created", collection=name)


async def upsert_vectors(
    points: list[dict[str, Any]],
) -> None:
    """Upsert a batch of vector points into Qdrant.

    Each dict must contain: id (str UUID), vector (list[float]), payload (dict).
    """
    settings = get_settings()
    client = await get_qdrant_client()
    qdrant_points = [
        qmodels.PointStruct(
            id=p["id"],
            vector=p["vector"],
            payload=p["payload"],
        )
        for p in points
    ]
    await client.upsert(
        collection_name=settings.qdrant_collection_name,
        points=qdrant_points,
        wait=True,
    )
    logger.info("qdrant.upserted", count=len(qdrant_points))


async def search_vectors(
    query_vector: list[float],
    top_k: int,
    score_threshold: float,
    filter_doc_ids: list[str] | None = None,
) -> list[qmodels.ScoredPoint]:
    settings = get_settings()
    client = await get_qdrant_client()

    query_filter: qmodels.Filter | None = None
    if filter_doc_ids:
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchAny(any=filter_doc_ids),
                )
            ]
        )

    results = await client.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=query_filter,
        with_payload=True,
    )
    logger.info("qdrant.search", hits=len(results))
    return results


async def delete_by_document_id(document_id: str) -> None:
    settings = get_settings()
    client = await get_qdrant_client()
    await client.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )
        ),
    )
    logger.info("qdrant.deleted_by_doc", document_id=document_id)