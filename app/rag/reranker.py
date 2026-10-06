from typing import Protocol

from app.contracts.retrieval import RetrievedItem


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        items: list[RetrievedItem],
    ) -> list[RetrievedItem]:
        ...