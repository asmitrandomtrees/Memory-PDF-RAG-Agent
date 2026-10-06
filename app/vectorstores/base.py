from typing import Any, Protocol

from app.contracts.retrieval import RetrievedItem


class VectorStore(Protocol):
    def add(
        self,
        *,
        collection: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        ...

    def search(
        self,
        *,
        collection: str,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedItem]:
        ...

    def delete(
        self,
        *,
        collection: str,
        ids: list[str],
    ) -> None:
        ...

    def get_by_metadata(
        self,
        *,
        collection: str,
        filters: dict[str, Any],
    ) -> list[RetrievedItem]:
        ...