from typing import Any
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.contracts.errors import ConfigurationError, ContractError, LLMError
from app.llm.azure_openai import AzureOpenAIProvider
from app.llm.provider import ChatMessage, LLMResponse


class FakeStructuredModel:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.messages = None
        self.kwargs = None

    def invoke(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return self.result


class FakeChatModel:
    def __init__(self) -> None:
        self.messages = None
        self.kwargs = None
        self.structured_model = None
        self.config = None

    def invoke(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        self.config = kwargs.get("config")
        return AIMessage(
            content="Generated answer",
            response_metadata={"model_name": "fake-deployment"},
            usage_metadata={
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
            },
        )

    def with_structured_output(self, output_schema, **kwargs):
        self.output_schema = output_schema
        self.kwargs = kwargs
        self.structured_model = FakeStructuredModel(
            {"answer": "Structured answer"}
        )
        return self.structured_model


class Answer(BaseModel):
    answer: str


def test_invoke_converts_messages_and_returns_application_response() -> None:
    chat_model = FakeChatModel()
    provider = AzureOpenAIProvider(chat_model=chat_model)

    response = provider.invoke(
        [
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
        ],
        temperature=0.2,
    )

    assert isinstance(response, LLMResponse)
    assert response.content == "Generated answer"
    assert response.model == "fake-deployment"
    assert response.usage == {
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    assert isinstance(chat_model.messages[0], SystemMessage)
    assert isinstance(chat_model.messages[1], HumanMessage)
    assert isinstance(chat_model.messages[2], AIMessage)
    assert chat_model.kwargs == {"temperature": 0.2}


def test_structured_output_returns_requested_pydantic_model() -> None:
    chat_model = FakeChatModel()
    provider = AzureOpenAIProvider(chat_model=chat_model)

    result = provider.structured_output(
        [ChatMessage(role="user", content="Return an answer")],
        Answer,
        temperature=0.0,
    )

    assert result == Answer(answer="Structured answer")
    assert chat_model.output_schema is Answer
    assert isinstance(chat_model.structured_model.messages[0], HumanMessage)
    assert chat_model.structured_model.kwargs == {"temperature": 0.0}


def test_invoke_rejects_unsupported_message_role() -> None:
    provider = AzureOpenAIProvider(chat_model=FakeChatModel())

    with pytest.raises(ContractError, match="Unsupported LLM message role"):
        provider.invoke([ChatMessage(role="tool", content="Tool result")])


def test_invoke_wraps_chat_model_errors() -> None:
    class FailingChatModel:
        def invoke(self, messages, **kwargs):
            raise RuntimeError("provider failure")

    provider = AzureOpenAIProvider(chat_model=FailingChatModel())

    with pytest.raises(LLMError, match="invocation failed"):
        provider.invoke([ChatMessage(role="user", content="Hello")])


def test_invoke_passes_optional_callbacks_to_langchain() -> None:
    callback = BaseCallbackHandler()
    chat_model = FakeChatModel()
    provider = AzureOpenAIProvider(
        chat_model=chat_model,
        callbacks=[callback],
    )

    provider.invoke([ChatMessage(role="user", content="Hello")])

    assert chat_model.config == {"callbacks": [callback]}


def test_provider_initializes_azure_model_from_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.llm.azure_openai.get_settings",
        lambda: SimpleNamespace(
            openai_api_key="test-key",
            openai_deployment="test-deployment",
            openai_api_version="2024-10-21",
            openai_endpoint="https://example.openai.azure.com",
        ),
    )

    provider = AzureOpenAIProvider()

    assert provider.chat_model.azure_endpoint == "https://example.openai.azure.com"
    assert provider.chat_model.deployment_name == "test-deployment"
    assert provider.chat_model.temperature is None or provider.chat_model.temperature == 0.0


def test_structured_output_falls_back_when_temperature_rejected() -> None:
    class ModelRejectingTemperature(FakeChatModel):
        def invoke(self, messages, **kwargs):
            if "temperature" in kwargs:
                raise RuntimeError("Unsupported value: 'temperature' does not support 0.0 with this model.")
            return super().invoke(messages, **kwargs)

        def with_structured_output(self, output_schema):
            class StructuredWrapper:
                def invoke(inner_self, messages, **kwargs):
                    if "temperature" in kwargs:
                        raise RuntimeError("Unsupported value: 'temperature' does not support 0.0 with this model.")
                    return Answer(answer="Fallback structured answer")
            return StructuredWrapper()

    provider = AzureOpenAIProvider(chat_model=ModelRejectingTemperature())
    result = provider.structured_output(
        [ChatMessage(role="user", content="Hello")],
        Answer,
        temperature=0.0,
    )
    assert result == Answer(answer="Fallback structured answer")


def test_provider_rejects_missing_azure_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.llm.azure_openai.get_settings",
        lambda: SimpleNamespace(
            openai_api_key=None,
            openai_deployment=None,
            openai_api_version=None,
            openai_endpoint=None,
        ),
    )

    with pytest.raises(ConfigurationError, match="Missing Azure OpenAI settings"):
        AzureOpenAIProvider()