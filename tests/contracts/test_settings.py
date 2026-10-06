from app.config.settings import Settings


def test_default_settings() -> None:
    settings = Settings()

    assert settings.app_env == "development"
    assert settings.app_name == "conversational-memory-agent"

    assert settings.vector_store_type == "chroma"

    assert settings.stm_collection == "stm_collection"
    assert settings.ltm_collection == "ltm_collection"
    assert settings.pdf_collection == "pdf_collection"

    assert settings.stm_top_k == 5
    assert settings.ltm_top_k == 5
    assert settings.pdf_top_k == 5

    assert settings.max_retries == 2
    assert settings.max_context_tokens == 6000