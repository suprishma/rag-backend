from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.qdrant import get_qdrant_client
from app.db.redis import close_redis, get_redis
from app.db.sql import Base, engine

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logger.info("app.starting", env=settings.app_env)

    # ── Create SQL tables ─────────────────────────────────────────────────────
    # In production use Alembic migrations instead.

    #async with engine.begin() as conn:
        #await conn.run_sync(Base.metadata.create_all)

    # ── Warm up clients ───────────────────────────────────────────────────────
    await get_qdrant_client()
    _ = get_redis()   # creates connection pool
    logger.info("app.ready")

    yield

    # ── Teardown ──────────────────────────────────────────────────────────────
    await close_redis()
    await engine.dispose()
    logger.info("app.stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG Backend",
        description=(
            "Document Ingestion and Conversational RAG APIs with Qdrant, Redis, and PostgreSQL."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/health", tags=["Health"])
    async def health() -> dict:
        return {"status": "ok", "env": settings.app_env}

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("app.unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred."},
        )

    return app


app = create_app()