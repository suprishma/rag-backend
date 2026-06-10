from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    fixed = "fixed"
    semantic = "semantic"


class IngestRequest(BaseModel):
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.fixed
    chunk_size: int | None = Field(default=None, ge=64, le=4096)
    chunk_overlap: int | None = Field(default=None, ge=0, le=512)


class ChunkInfo(BaseModel):
    chunk_index: int
    text: str
    token_count: int


class IngestResponse(BaseModel):
    document_id: UUID
    filename: str
    file_type: str
    chunking_strategy: ChunkingStrategy
    chunk_count: int
    char_count: int
    message: str = "Document ingested successfully."


class DocumentListItem(BaseModel):
    document_id: UUID
    filename: str
    file_type: str
    chunking_strategy: str
    chunk_count: int
    char_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentListItem]


class DeleteResponse(BaseModel):
    document_id: UUID
    message: str