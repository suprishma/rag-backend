from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.qdrant import upsert_vectors
from app.models.booking import DocumentRecord
from app.schemas.ingestion import ChunkingStrategy, IngestResponse
from app.services.chunking import Chunk, fixed_chunking, semantic_chunking
from app.services.embedding import embed_texts
from app.utils.pdf import extract_text_from_pdf, extract_text_from_txt
from app.utils.text import clean_text

logger = get_logger(__name__)


async def ingest_document(
    *,
    filename: str,
    content: bytes,
    chunking_strategy: ChunkingStrategy,
    chunk_size: int | None,
    chunk_overlap: int | None,
    db: AsyncSession,
) -> IngestResponse:
    # ── 1. Extract text ───────────────────────────────────────────────────────
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        raw_text = extract_text_from_pdf(content)
        file_type = "pdf"
    else:
        raw_text = extract_text_from_txt(content)
        file_type = "txt"

    text = clean_text(raw_text)
    logger.info("ingestion.text_extracted", filename=filename, chars=len(text))

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    chunks: list[Chunk]
    if chunking_strategy == ChunkingStrategy.semantic:
        chunks = await semantic_chunking(text, embed_fn=embed_texts)
    else:
        chunks = fixed_chunking(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info("ingestion.chunked", strategy=chunking_strategy, count=len(chunks))

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    document_id = str(uuid.uuid4())
    texts = [c.text for c in chunks]
    embeddings = await embed_texts(texts)

    # ── 4. Build Qdrant points ────────────────────────────────────────────────
    points: list[dict[str, Any]] = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                },
            }
        )

    await upsert_vectors(points)

    # ── 5. Persist metadata ───────────────────────────────────────────────────
    record = DocumentRecord(
        id=uuid.UUID(document_id),
        filename=filename,
        file_type=file_type,
        chunking_strategy=chunking_strategy.value,
        chunk_count=len(chunks),
        char_count=len(text),
    )
    db.add(record)
    await db.flush()

    return IngestResponse(
        document_id=uuid.UUID(document_id),
        filename=filename,
        file_type=file_type,
        chunking_strategy=chunking_strategy,
        chunk_count=len(chunks),
        char_count=len(text),
    )