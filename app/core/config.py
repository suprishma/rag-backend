from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o"
    openai_embedding_dimensions: int = 1536

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "documents"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_chat_ttl_seconds: int = 86400

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag_db"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_similarity_threshold: float = 0.85

    # RAG
    rag_top_k: int = 5
    rag_score_threshold: float = 0.30


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()