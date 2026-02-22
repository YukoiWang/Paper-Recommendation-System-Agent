"""
Configuration management using Pydantic Settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # App
    APP_NAME: str = "IntelliScholar"
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]
    
    # Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "yukiwang"
    POSTGRES_PASSWORD: str = "yukiwang"
    POSTGRES_DB: str = "intellischolar"
    DATABASE_URL: str = ""
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Vector Database
    VECTOR_DB_TYPE: str = "chroma"  # chroma or milvus
    CHROMA_PERSIST_DIR: str = "/tmp/chroma_db"
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    
    # LLM
    LLM_PROVIDER: str = "openai"  # openai, qwen, deepseek
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    
    # Embedding (BGE English for English papers; override via .env)
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DEVICE: str = "cuda"  # cuda or cpu
    
    # Ranking Model
    RANKING_MODEL_PATH: str = "./models/ranking/lightgbm.pkl"
    
    # API Keys
    ARXIV_API_KEY: str = ""
    SEMANTIC_SCHOLAR_API_KEY: str = ""
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Cache
    CACHE_TTL: int = 3600  # seconds


settings = Settings()

# Auto-generate DATABASE_URL if not provided
if not settings.DATABASE_URL:
    settings.DATABASE_URL = (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
