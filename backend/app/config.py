from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"

    # Ollama (dev)
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen2.5:7b"

    # Claude (prod)
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"

    # Qdrant — backend container uses Docker service name
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "hr_documents"

    # Qdrant — ingest script uses localhost (runs on host)
    qdrant_external_host: str = "localhost"

    # Embedding — identical in dev and prod
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"


@lru_cache
def get_settings() -> Settings:
    return Settings()
