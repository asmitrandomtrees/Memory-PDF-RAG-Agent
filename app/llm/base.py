from typing import Any, Protocol


class LLMProvider(Protocol):
    def invoke(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        ...