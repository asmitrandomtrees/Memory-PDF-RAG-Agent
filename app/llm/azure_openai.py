from __future__ import annotations

from collections.abc import Sequence
import time
from typing import Any, TypeVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.config.settings import get_settings
from app.contracts.errors import ConfigurationError, ContractError, LLMError
from app.llm.provider import ChatMessage, LLMResponse
from app.observability.tracker import record_current_timing

T = TypeVar("T", bound=BaseModel)


class AzureOpenAIProvider:
    def __init__(
        self,
        *,
        chat_model: Any | None = None,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.callbacks = list(callbacks or [])
        # Some deployments (e.g. reasoning models) reject `temperature`.
        # Remember that after the first rejection so every later call does
        # not pay for a failed round trip before retrying.
        self._temperature_supported = True
        self._structured_models: dict[type[BaseModel], Any] = {}
        if chat_model is not None:
            self.chat_model = chat_model
            return

        settings = get_settings()
        required_settings = {
            "OPENAI_API_KEY": settings.openai_api_key,
            "OPENAI_DEPLOYMENT": settings.openai_deployment,
            "OPENAI_API_VERSION": settings.openai_api_version,
            "OPENAI_ENDPOINT": settings.openai_endpoint,
        }
        missing_settings = [
            name
            for name, value in required_settings.items()
            if not value
        ]
        if missing_settings:
            raise ConfigurationError(
                "Missing Azure OpenAI settings: "
                + ", ".join(missing_settings)
            )

        try:
            from langchain_openai import AzureChatOpenAI

            self.chat_model = AzureChatOpenAI(
                azure_endpoint=settings.openai_endpoint,
                api_key=settings.openai_api_key,
                azure_deployment=settings.openai_deployment,
                api_version=settings.openai_api_version,
                timeout=timeout_seconds
                or getattr(settings, "llm_timeout_seconds", 60.0),
                max_retries=getattr(settings, "llm_max_retries", 2),
            )
        except Exception as exc:
            raise ConfigurationError(
                "Failed to initialize Azure OpenAI chat model"
            ) from exc

    def invoke(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = self._call_with_temperature(
                self.chat_model,
                messages,
                temperature,
            )

            if not isinstance(response.content, str):
                raise LLMError(
                    "Azure OpenAI returned non-text content"
                )

            response_metadata = response.response_metadata
            usage = response.usage_metadata or {}

            return LLMResponse(
                content=response.content,
                model=(
                    response_metadata.get("model_name")
                    or response_metadata.get("model")
                ),
                usage=usage,
            )
        except (ContractError, LLMError):
            raise
        except Exception as exc:
            raise LLMError("Azure OpenAI invocation failed") from exc
        finally:
            record_current_timing(
                "llm.invoke",
                (time.perf_counter() - started) * 1000,
            )

    def structured_output(
        self,
        messages: list[ChatMessage],
        output_schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        started = time.perf_counter()
        try:
            result = self._call_with_temperature(
                self._structured_model(output_schema),
                messages,
                temperature,
            )

            if isinstance(result, output_schema):
                return result
            return output_schema.model_validate(result)
        except ContractError:
            raise
        except Exception as exc:
            raise LLMError(
                "Azure OpenAI structured output failed"
            ) from exc
        finally:
            record_current_timing(
                "llm.structured_output",
                (time.perf_counter() - started) * 1000,
            )

    def _structured_model(self, output_schema: type[T]) -> Any:
        structured_model = self._structured_models.get(output_schema)
        if structured_model is not None:
            return structured_model

        try:
            structured_model = self.chat_model.with_structured_output(
                output_schema,
                method="function_calling",
            )
        except (TypeError, ValueError):
            structured_model = self.chat_model.with_structured_output(
                output_schema
            )
        self._structured_models[output_schema] = structured_model
        return structured_model

    def _call_with_temperature(
        self,
        model: Any,
        messages: list[ChatMessage],
        temperature: float | None,
    ) -> Any:
        converted = self._to_langchain_messages(messages)
        temperature_kwargs = self._temperature_kwargs(temperature)
        try:
            return model.invoke(
                converted,
                **temperature_kwargs,
                **self._callback_kwargs(),
            )
        except Exception as exc:
            # Decide on what this call sent, not the shared flag: another
            # thread may have flipped the flag while this call was in flight.
            if (
                not temperature_kwargs
                or "temperature" not in str(exc).lower()
            ):
                raise
            self._temperature_supported = False
            return model.invoke(converted, **self._callback_kwargs())

    def _temperature_kwargs(
        self,
        temperature: float | None,
    ) -> dict[str, float]:
        if temperature is None or not self._temperature_supported:
            return {}
        return {"temperature": temperature}

    def _callback_kwargs(self) -> dict[str, dict[str, list[BaseCallbackHandler]]]:
        if not self.callbacks:
            return {}
        return {"config": {"callbacks": self.callbacks}}

    @staticmethod
    def _to_langchain_messages(
        messages: list[ChatMessage],
    ) -> list[BaseMessage]:
        message_types = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage,
        }

        converted: list[BaseMessage] = []
        for message in messages:
            message_type = message_types.get(message.role)
            if message_type is None:
                raise ContractError(
                    f"Unsupported LLM message role: {message.role}"
                )
            converted.append(message_type(content=message.content))

        return converted
