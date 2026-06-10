from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# ── Chat ──────────────────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=256)
    message: str = Field(..., min_length=1, max_length=8192)


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text_preview: str   # first 200 chars


class MessageResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk] = []
    booking_created: bool = False
    booking_id: UUID | None = None


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class ClearHistoryResponse(BaseModel):
    session_id: str
    message: str


# ── Booking ───────────────────────────────────────────────────────────────────

class BookingOut(BaseModel):
    booking_id: UUID
    session_id: str
    name: str
    email: str
    date: str
    time: str
    notes: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class BookingListResponse(BaseModel):
    total: int
    bookings: list[BookingOut]