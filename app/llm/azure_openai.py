from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.config.settings import get_settings
from app.contracts.errors import ConfigurationError, ContractError, LLMError
from app.llm.provider import ChatMessage, LLMResponse

T = TypeVar("T", bound=BaseModel)


class AzureOpenAIProvider:
    def __init__(
        self,
        *,
        chat_model: Any | None = None,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> None:
        self.callbacks = list(callbacks or [])
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
        try:
            try:
                response = self.chat_model.invoke(
                    self._to_langchain_messages(messages),
                    **self._temperature_kwargs(temperature),
                    **self._callback_kwargs(),
                )
            except Exception as inner_exc:
                if "temperature" in str(inner_exc).lower() and temperature is not None:
                    response = self.chat_model.invoke(
                        self._to_langchain_messages(messages),
                        **self._callback_kwargs(),
                    )
                else:
                    raise

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

    def structured_output(
        self,
        messages: list[ChatMessage],
        output_schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        try:
            try:
                structured_model = self.chat_model.with_structured_output(
                    output_schema,
                    method="function_calling",
                )
            except (TypeError, ValueError):
                structured_model = self.chat_model.with_structured_output(
                    output_schema
                )
            try:
                result = structured_model.invoke(
                    self._to_langchain_messages(messages),
                    **self._temperature_kwargs(temperature),
                    **self._callback_kwargs(),
                )
            except Exception as inner_exc:
                if "temperature" in str(inner_exc).lower() and temperature is not None:
                    result = structured_model.invoke(
                        self._to_langchain_messages(messages),
                        **self._callback_kwargs(),
                    )
                else:
                    raise

            if isinstance(result, output_schema):
                return result
            return output_schema.model_validate(result)
        except ContractError:
            raise
        except Exception as exc:
            raise LLMError(
                "Azure OpenAI structured output failed"
            ) from exc

    @staticmethod
    def _temperature_kwargs(
        temperature: float | None,
    ) -> dict[str, float]:
        if temperature is None:
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
