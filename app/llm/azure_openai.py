from typing import TypeVar

from openai import AzureOpenAI
from pydantic import BaseModel

from app.config.settings import Settings
from app.contracts.errors import ConfigurationError
from app.llm.provider import ChatMessage, LLMResponse

T = TypeVar("T", bound=BaseModel)


class AzureOpenAIProvider:
    def __init__(self, settings: Settings) -> None:
        if not (settings.openai_api_key and settings.openai_endpoint
                and settings.openai_deployment):
            raise ConfigurationError("Azure OpenAI settings missing in .env")
        self.deployment = settings.openai_deployment
        self.client = AzureOpenAI(
            api_key=settings.openai_api_key,
            azure_endpoint=settings.openai_endpoint,
            api_version=settings.openai_api_version,
        )

    def _kwargs(self, messages, temperature):
        kw = {"model": self.deployment,
              "messages": [m.model_dump() for m in messages]}
        if temperature is not None:  # gpt-5-mini only accepts the default
            kw["temperature"] = temperature
        return kw

    def invoke(self, messages: list[ChatMessage], *, temperature=None) -> LLMResponse:
        r = self.client.chat.completions.create(**self._kwargs(messages, temperature))
        return LLMResponse(
            content=r.choices[0].message.content or "",
            model=r.model,
            usage=r.usage.model_dump() if r.usage else {},
        )

    def structured_output(self, messages, output_schema: type[T], *, temperature=None) -> T:
        r = self.client.beta.chat.completions.parse(
            response_format=output_schema, **self._kwargs(messages, temperature)
        )
        return r.choices[0].message.parsed
