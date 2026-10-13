from pathlib import Path

from app.config.settings import Settings


def settings_for_instance(settings: Settings, instance: str) -> Settings:
    if instance not in {"api", "api_multiuser"}:
        raise ValueError(f"Unsupported runtime instance: {instance}")

    data_root = Path(settings.conversation_data_path).resolve().parent / instance
    return settings.model_copy(
        update={
            "vector_store_path": str(data_root / "vectorstore"),
            "conversation_data_path": str(data_root / "conversations"),
            "ltm_memory_path": str(data_root / "memories" / "ltm.json"),
            "pdf_upload_path": str(data_root / "uploads" / "pdfs"),
            "trace_data_path": str(data_root / "traces" / settings.app_env),
        }
    )
