from app.contracts.retrieval import RetrievedItem
from app.llm.provider import ChatMessage, LLMProvider


class STMContextSummarizer:
    def __init__(
        self,
        *,
        llm: LLMProvider,
    ) -> None:
        self.llm = llm

    def summarize(
        self,
        items: list[RetrievedItem],
        *,
        max_tokens: int | None = None,
    ) -> RetrievedItem | None:
        if not items:
            return None

        ordered_items = sorted(
            items,
            key=lambda item: (
                item.metadata.get("timestamp", ""),
                item.rank or 0,
            ),
        )

        content = self._build_content(ordered_items)

        if not content.strip():
            return None

        system_prompt = (
            "Summarize the conversation context concisely. "
            "Preserve important facts, decisions, requirements, "
            "questions, and unresolved points. Do not invent "
            "information. Return only the summary."
        )
        if max_tokens is not None:
            system_prompt += (
                f" Keep the summary to at most approximately "
                f"{max_tokens} tokens."
            )

        try:
            response = self.llm.invoke(
                [
                    ChatMessage(
                        role="system",
                        content=system_prompt,
                    ),
                    ChatMessage(role="user", content=content),
                ],
                temperature=0.0,
            )
            summary = response.content.strip()
        except Exception as exc:
            raise RuntimeError(
                "Failed to summarize STM context"
            ) from exc

        if not summary:
            return None

        source_ids = [
            item.item_id
            for item in ordered_items
        ]

        return RetrievedItem(
            item_id="stm_summary",
            source="stm",
            content=summary,
            score=None,
            rank=None,
            metadata={
                "context_type": "summary",
                "source_message_ids": source_ids,
                "source_item_ids": source_ids,
            },
        )

    @staticmethod
    def _build_content(
        items: list[RetrievedItem],
    ) -> str:
        parts: list[str] = []

        for item in items:
            role = item.metadata.get("role", "unknown")
            timestamp = item.metadata.get(
                "timestamp",
                "",
            )

            parts.append(
                f"[{timestamp}] {role}: {item.content}"
            )

        return "\n".join(parts)