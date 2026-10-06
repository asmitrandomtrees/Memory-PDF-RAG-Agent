"""Short-term memory: the last few messages of the current thread."""
from __future__ import annotations

from app.contracts.retrieval import RetrievalResult, RetrievedItem


class RecentMessagesSTM:
    def __init__(self, conversation_store, keep: int = 6) -> None:
        self.store = conversation_store
        self.keep = keep

    def retrieve(self, user_id: str, thread_id: str, current_query: str) -> RetrievalResult:
        messages = list(self.store.load(user_id, thread_id).messages)
        # The runtime already saved the current question; don't echo it back.
        if messages and messages[-1].role == "user" and messages[-1].content == current_query:
            messages = messages[:-1]
        recent = messages[-self.keep :]
        items = [
            RetrievedItem(
                item_id=m.message_id,
                source="stm",
                content=f"{m.role}: {m.content}",
                rank=i + 1,
                metadata={"timestamp": m.timestamp.isoformat()},
            )
            for i, m in enumerate(recent)
        ]
        return RetrievalResult(source="stm", query=current_query, items=items)
