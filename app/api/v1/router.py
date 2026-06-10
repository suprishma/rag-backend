from fastapi import APIRouter

from app.api.v1.endpoints.conversation import router as conversation_router
from app.api.v1.endpoints.ingestion import router as ingestion_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(ingestion_router)
api_router.include_router(conversation_router)