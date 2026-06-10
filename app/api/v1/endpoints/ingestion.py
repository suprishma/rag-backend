from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.qdrant import delete_by_document_id
from app.db.sql import get_db
from app.models.booking import DocumentRecord
from app.schemas.ingestion import (
    ChunkingStrategy,
    DeleteResponse,
    DocumentListItem,
    DocumentListResponse,
    IngestResponse,
)
from app.services.ingestion import ingest_document

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

_ALLOWED_TYPES = {"application/pdf", "text/plain"}
_ALLOWED_EXTENSIONS = {"pdf", "txt"}
_MAX_FILE_SIZE = 50 * 1024 * 1024   # 50 MB


@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF or TXT file to ingest")],
    chunking_strategy: Annotated[
        ChunkingStrategy,
        Form(description="Chunking strategy: 'fixed' or 'semantic'"),
    ] = ChunkingStrategy.fixed,
    chunk_size: Annotated[
        int | None,
        Form(description="Tokens per chunk (fixed strategy only)", ge=64, le=4096),
    ] = None,
    chunk_overlap: Annotated[
        int | None,
        Form(description="Overlap tokens between chunks", ge=0, le=512),
    ] = None,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    # ── Validation ────────────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '.{ext}'. Allowed: pdf, txt.",
        )

    content = await file.read()
    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {_MAX_FILE_SIZE // (1024*1024)} MB.",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info(
        "ingestion.request",
        filename=file.filename,
        strategy=chunking_strategy,
        size_bytes=len(content),
    )

    response = await ingest_document(
        filename=file.filename,
        content=content,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        db=db,
    )
    return response


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
async def list_documents(
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    stmt = select(DocumentRecord).order_by(DocumentRecord.created_at.desc())
    result = await db.execute(stmt)
    records = list(result.scalars().all())
    items = [
        DocumentListItem(
            document_id=r.id,
            filename=r.filename,
            file_type=r.file_type,
            chunking_strategy=r.chunking_strategy,
            chunk_count=r.chunk_count,
            char_count=r.char_count,
            created_at=r.created_at,
        )
        for r in records
    ]
    return DocumentListResponse(total=len(items), documents=items)


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    summary="Delete a document and its vectors",
)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> DeleteResponse:
    record = await db.get(DocumentRecord, document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found.")

    await delete_by_document_id(str(document_id))
    await db.delete(record)

    logger.info("ingestion.deleted", document_id=str(document_id))
    return DeleteResponse(
        document_id=document_id,
        message=f"Document '{record.filename}' and its vectors have been deleted.",
    )