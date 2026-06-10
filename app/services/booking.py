from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.booking import BookingRecord
from app.schemas.conversation import BookingOut
from app.services.llm import detect_booking_intent

logger = get_logger(__name__)


async def try_extract_and_save_booking(
    *,
    session_id: str,
    user_message: str,
    history: list[dict[str, str]],
    db: AsyncSession,
) -> BookingRecord | None:
    """
    Use the LLM to detect booking intent. If all fields are present,
    persist the booking and return the record; otherwise return None.
    """
    result: dict[str, Any] = await detect_booking_intent(
        user_message=user_message,
        history_messages=history,
    )

    intent = result.get("intent", "none")
    logger.info("booking.intent", session_id=session_id, intent=intent)

    if intent != "book":
        return None

    # Validate required fields
    required = ("name", "email", "date", "time")
    if not all(result.get(f) for f in required):
        logger.warning("booking.missing_fields", result=result)
        return None

    record = BookingRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        name=result["name"],
        email=result["email"],
        date=result["date"],
        time=result["time"],
        notes=result.get("notes"),
    )
    db.add(record)
    await db.flush()
    logger.info("booking.created", booking_id=str(record.id))
    return record


async def list_bookings(db: AsyncSession) -> list[BookingRecord]:
    stmt = select(BookingRecord).order_by(BookingRecord.created_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_booking(booking_id: uuid.UUID, db: AsyncSession) -> BookingRecord | None:
    return await db.get(BookingRecord, booking_id)