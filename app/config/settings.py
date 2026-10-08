from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = Field("development", alias="APP_ENV")
    app_name: str = Field("conversational-memory-agent", alias="APP_NAME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # OpenAI
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_deployment: str | None = Field(None, alias="OPENAI_DEPLOYMENT")
    openai_api_version: str | None = Field(None, alias="OPENAI_API_VERSION")
    openai_endpoint: str | None = Field(None, alias="OPENAI_ENDPOINT")

    # Embeddings
    # embedding_model: str = Field(
    #     "BAAI/bge-m3",
    #     alias="EMBEDDING_MODEL",
    # )
    embedding_provider: str = "sentence_transformer"
    embedding_model: str = "BAAI/bge-m3"

    # Vector store
    vector_store_type: str = Field("chroma", alias="VECTOR_STORE_TYPE")
    vector_store_path: str = Field("./vectorstore", alias="VECTOR_STORE_PATH")
    stm_collection: str = Field("stm_collection", alias="STM_COLLECTION")
    ltm_collection: str = Field("ltm_collection", alias="LTM_COLLECTION")
    pdf_collection: str = Field("pdf_collection", alias="PDF_COLLECTION")
    pdf_upload_path: str = Field("./data/uploads/pdfs", alias="PDF_UPLOAD_PATH")
    max_pdf_upload_bytes: int = Field(
        25 * 1024 * 1024,
        alias="MAX_PDF_UPLOAD_BYTES",
        ge=1,
    )

    # Conversation store
    conversation_store_type: str = Field("jsonl", alias="CONVERSATION_STORE_TYPE")
    conversation_data_path: str = Field(
        "./data/conversations",
        alias="CONVERSATION_DATA_PATH",
    )
    ltm_memory_path: str = Field(
        "./data/memories/ltm.json",
        alias="LTM_MEMORY_PATH",
    )
    trace_data_path: str = Field(
        "./data/traces/development",
        alias="TRACE_DATA_PATH",
    )

    # Retrieval
    stm_top_k: int = Field(5, alias="STM_TOP_K", ge=1)
    ltm_top_k: int = Field(5, alias="LTM_TOP_K", ge=1)
    pdf_top_k: int = Field(5, alias="PDF_TOP_K", ge=1)

    # Agent
    max_retries: int = Field(2, alias="MAX_RETRIES", ge=0)
    max_context_tokens: int = Field(6000, alias="MAX_CONTEXT_TOKENS", ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()