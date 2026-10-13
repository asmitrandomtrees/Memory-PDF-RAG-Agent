"""Upload datasets and manage Azure OpenAI fine-tuning jobs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config.settings import Settings, get_settings
from app.contracts.errors import ConfigurationError


def create_client(settings: Settings | None = None) -> Any:
    settings = settings or get_settings()
    required_settings = {
        "OPENAI_API_KEY": settings.openai_api_key,
        "OPENAI_ENDPOINT": settings.openai_endpoint,
        "FINETUNE_API_VERSION": settings.finetune_api_version,
    }
    missing_settings = [
        name for name, value in required_settings.items() if not value
    ]
    if missing_settings:
        raise ConfigurationError(
            "Missing Azure OpenAI fine-tuning settings: "
            + ", ".join(missing_settings)
        )

    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=settings.openai_endpoint,
        api_key=settings.openai_api_key,
        api_version=settings.finetune_api_version,
    )


def upload_training_file(client: Any, path: str | Path) -> str:
    with Path(path).open("rb") as file:
        uploaded = client.files.create(file=file, purpose="fine-tune")
    return uploaded.id


def create_job(
    client: Any,
    *,
    training_file_id: str,
    base_model: str,
    validation_file_id: str | None = None,
    suffix: str | None = None,
    n_epochs: int | None = None,
    seed: int | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": base_model,
        "training_file": training_file_id,
    }
    if validation_file_id:
        kwargs["validation_file"] = validation_file_id
    if suffix:
        kwargs["suffix"] = suffix
    if seed is not None:
        kwargs["seed"] = seed
    if n_epochs is not None:
        kwargs["hyperparameters"] = {"n_epochs": n_epochs}
    return client.fine_tuning.jobs.create(**kwargs)


def get_job(client: Any, job_id: str) -> Any:
    return client.fine_tuning.jobs.retrieve(job_id)


def list_job_events(client: Any, job_id: str, limit: int = 20) -> list[Any]:
    events = client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=job_id,
        limit=limit,
    )
    return list(events.data)


def cancel_job(client: Any, job_id: str) -> Any:
    return client.fine_tuning.jobs.cancel(job_id)
