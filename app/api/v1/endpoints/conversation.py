from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.sql import get_db
from app.schemas.conversation import (
    BookingListResponse,
    BookingOut,
    ClearHistoryResponse,
    HistoryMessage,
    HistoryResponse,
    MessageRequest,
    MessageResponse,
)
from app.services.booking import get_booking, list_bookings
from app.services.conversation import process_message
from app.services.memory import clear_session, get_session_history

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Conversation"])


@router.post(
    "/message",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message and receive a RAG-powered response",
)
async def send_message(
    body: MessageRequest,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    logger.info(
        "conversation.message_received",
        session_id=body.session_id,
        message_len=len(body.message),
    )
    return await process_message(
        session_id=body.session_id,
        user_message=body.message,
        db=db,
    )


@router.get(
    "/history/{session_id}",
    response_model=HistoryResponse,
    summary="Retrieve chat history for a session",
)
async def get_chat_history(session_id: str) -> HistoryResponse:
    messages = await get_session_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[HistoryMessage(**m) for m in messages],
    )


@router.delete(
    "/history/{session_id}",
    response_model=ClearHistoryResponse,
    summary="Clear chat history for a session",
)
async def clear_chat_history(session_id: str) -> ClearHistoryResponse:
    await clear_session(session_id)
    return ClearHistoryResponse(
        session_id=session_id,
        message="Chat history cleared.",
    )


@router.get(
    "/bookings",
    response_model=BookingListResponse,
    summary="List all interview bookings",
)
async def get_bookings(db: AsyncSession = Depends(get_db)) -> BookingListResponse:
    records = await list_bookings(db)
    return BookingListResponse(
        total=len(records),
        bookings=[
            BookingOut(
                booking_id=r.id,
                session_id=r.session_id,
                name=r.name,
                email=r.email,
                date=r.date,
                time=r.time,
                notes=r.notes,
                created_at=r.created_at,
            )
            for r in records
        ],
    )


@router.get(
    "/bookings/{booking_id}",
    response_model=BookingOut,
    summary="Get a single booking by ID",
)
async def get_single_booking(
    booking_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> BookingOut:
    record = await get_booking(booking_id, db)
    if not record:
        raise HTTPException(status_code=404, detail="Booking not found.")
    return BookingOut(
        booking_id=record.id,
        session_id=record.session_id,
        name=record.name,
        email=record.email,
        date=record.date,
        time=record.time,
        notes=record.notes,
        created_at=record.created_at,
    )