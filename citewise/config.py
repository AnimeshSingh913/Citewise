from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

INSECURE_DEV_SECRET = "citewise-dev-secret-do-not-use-in-production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CITEWISE_", extra="ignore")

    env: Literal["development", "testing", "production"] = "development"

    # Persistence
    database_url: str = "sqlite+aiosqlite:///./data/citewise.db"
    qdrant_url: str | None = None  # None -> embedded on-disk Qdrant at qdrant_path
    qdrant_path: str = "./data/qdrant"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "citewise_chunks"

    # File storage
    storage_backend: Literal["local", "s3"] = "local"
    upload_dir: Path = Path("./data/uploads")
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str | None = None

    # Auth
    jwt_secret: str | None = None
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 15
    refresh_token_days: int = 14

    # Limits
    max_upload_mb: int = 25
    cors_origins: list[str] = ["http://localhost:8000"]

    # Embeddings (local, no API key needed)
    embedding_backend: Literal["fastembed", "hashing"] = "fastembed"
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    embedding_batch_size: int = 64

    # Chunking
    parent_chunk_chars: int = 1200
    parent_chunk_overlap: int = 150
    child_chunk_chars: int = 300

    # Retrieval
    retrieval_top_k: int = 24
    context_parent_limit: int = 8
    history_messages: int = 12

    # Generation. "claude" uses ANTHROPIC_API_KEY (or an `ant auth login` profile);
    # "gemini" uses GEMINI_API_KEY; "extractive" needs no LLM and returns cited passages verbatim.
    generator: Literal["claude", "gemini", "extractive"] = "claude"

    gemini_api_key: str | None = Field(default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    # Alias that tracks Google's current Flash model; pin a specific one if you prefer.
    gemini_model: str = "gemini-flash-latest"
    # Used when the main model is overloaded or rate limited. Empty to disable.
    gemini_fallback_model: str = "gemini-flash-lite-latest"
    gemini_max_output_tokens: int = 8192

    # Read from ANTHROPIC_API_KEY (also via .env). If unset, the SDK falls back to its own
    # credential chain (e.g. an `ant auth login` profile).
    anthropic_api_key: str | None = Field(default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY"))
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    anthropic_max_tokens: int = 16000
    anthropic_fallbacks: bool = True

    # Worker
    worker_concurrency: int = 2
    job_lease_seconds: int = 120
    job_max_attempts: int = 3
    worker_poll_seconds: float = 5.0
    run_worker_in_api: bool = True

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    @model_validator(mode="after")
    def _require_secret_in_production(self) -> "Settings":
        if self.env == "production" and not self.jwt_secret:
            raise ValueError("CITEWISE_JWT_SECRET must be set when CITEWISE_ENV=production")
        if self.storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("CITEWISE_S3_BUCKET is required when CITEWISE_STORAGE_BACKEND=s3")
        if not self.jwt_secret:
            self.jwt_secret = INSECURE_DEV_SECRET
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
