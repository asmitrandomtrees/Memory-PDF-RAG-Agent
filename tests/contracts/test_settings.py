from app.config.settings import Settings


def test_default_settings(monkeypatch) -> None:
    monkeypatch.delenv("CONVERSATION_DATA_PATH", raising=False)
    monkeypatch.delenv("TRACE_DATA_PATH", raising=False)
    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.app_name == "conversational-memory-agent"
    assert settings.conversation_data_path == "./data/conversations"
    assert settings.ltm_memory_path == "./data/memories/ltm.json"
    assert settings.trace_data_path == "./data/traces/development"

    assert settings.vector_store_type == "chroma"

    assert settings.stm_collection == "stm_collection"
    assert settings.ltm_collection == "ltm_collection"
    assert settings.pdf_collection == "pdf_collection"

    assert settings.stm_top_k == 5
    assert settings.ltm_top_k == 5
    assert settings.pdf_top_k == 5

    assert settings.max_retries == 2
    assert settings.max_context_tokens == 6000