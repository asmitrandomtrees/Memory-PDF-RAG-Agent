from pathlib import Path

import pytest

from app.config.instance_settings import settings_for_instance
from app.config.settings import Settings


def test_api_instances_get_separate_storage_paths(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        VECTOR_STORE_PATH=str(tmp_path / "cli" / "vectorstore"),
        CONVERSATION_DATA_PATH=str(tmp_path / "data" / "conversations"),
        LTM_MEMORY_PATH=str(tmp_path / "data" / "memories" / "ltm.json"),
        PDF_UPLOAD_PATH=str(tmp_path / "data" / "uploads" / "pdfs"),
        TRACE_DATA_PATH=str(tmp_path / "data" / "traces"),
    )

    api_settings = settings_for_instance(settings, "api")
    multiuser_settings = settings_for_instance(settings, "api_multiuser")

    assert settings.vector_store_path == str(tmp_path / "cli" / "vectorstore")
    assert api_settings.vector_store_path != multiuser_settings.vector_store_path
    assert (
        api_settings.conversation_data_path
        != multiuser_settings.conversation_data_path
    )
    assert (
        api_settings.ltm_memory_path
        != multiuser_settings.ltm_memory_path
    )
    assert (
        api_settings.pdf_upload_path
        != multiuser_settings.pdf_upload_path
    )
    assert (
        api_settings.trace_data_path
        != multiuser_settings.trace_data_path
    )
    assert Path(api_settings.vector_store_path).is_relative_to(
        tmp_path / "data" / "api"
    )
    assert Path(multiuser_settings.vector_store_path).is_relative_to(
        tmp_path / "data" / "api_multiuser"
    )


def test_settings_for_instance_rejects_unknown_instance() -> None:
    with pytest.raises(ValueError, match="Unsupported runtime instance"):
        settings_for_instance(Settings(_env_file=None), "cli")


def test_api_service_builders_use_isolated_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api as single_user_api
    import app.api_multiuser as multiuser_api

    settings = Settings(
        _env_file=None,
        VECTOR_STORE_PATH=str(tmp_path / "cli" / "vectorstore"),
        CONVERSATION_DATA_PATH=str(tmp_path / "data" / "conversations"),
        LTM_MEMORY_PATH=str(tmp_path / "data" / "memories" / "ltm.json"),
        PDF_UPLOAD_PATH=str(tmp_path / "data" / "uploads" / "pdfs"),
    )
    monkeypatch.setattr(single_user_api, "get_settings", lambda: settings)
    monkeypatch.setattr(multiuser_api, "get_settings", lambda: settings)
    monkeypatch.setattr(
        single_user_api,
        "create_embedding_provider",
        lambda configured: configured.embedding_model,
    )
    monkeypatch.setattr(
        multiuser_api,
        "create_embedding_provider",
        lambda configured: configured.embedding_model,
    )
    monkeypatch.setattr(
        single_user_api,
        "create_vector_store",
        lambda configured: configured.vector_store_path,
    )
    monkeypatch.setattr(
        multiuser_api,
        "create_vector_store",
        lambda configured: configured.vector_store_path,
    )
    monkeypatch.setattr(
        single_user_api,
        "create_runtime",
        lambda **kwargs: kwargs["settings"],
    )
    monkeypatch.setattr(
        multiuser_api,
        "create_runtime",
        lambda **kwargs: kwargs["settings"],
    )

    api_services = single_user_api._build_services()
    multiuser_services = multiuser_api._build_services()

    assert api_services["vector_store"] == str(
        tmp_path / "data" / "api" / "vectorstore"
    )
    assert multiuser_services["vector_store"] == str(
        tmp_path / "data" / "api_multiuser" / "vectorstore"
    )
    assert api_services["conversation_store"].root_path == (
        tmp_path / "data" / "api" / "conversations"
    )
    assert multiuser_services["conversation_store"].root_path == (
        tmp_path / "data" / "api_multiuser" / "conversations"
    )
    assert multiuser_services["user_store"].database_path == (
        tmp_path / "data" / "api_multiuser" / "db" / "user_profiles.sqlite3"
    )
    assert settings.vector_store_path == str(
        tmp_path / "cli" / "vectorstore"
    )
