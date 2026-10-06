from typing import Any

import chromadb

from app.contracts.errors import VectorStoreError
from app.contracts.retrieval import RetrievedItem


class ChromaVectorStore:
    def __init__(self, path: str) -> None:
        try:
            self.client = chromadb.PersistentClient(path=path)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialize Chroma: {path}"
            ) from exc

    def _collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def _build_where(
        self,
        filters: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not filters:
            return None

        if len(filters) == 1:
            return filters

        return {
            "$and": [
                {key: value}
                for key, value in filters.items()
            ]
        }

    def add(
        self,
        *,
        collection: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return

        if not (
            len(ids)
            == len(documents)
            == len(embeddings)
            == len(metadatas)
        ):
            raise VectorStoreError(
                "Vector store inputs must have the same length"
            )

        try:
            self._collection(collection).upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to add items to {collection}"
            ) from exc

    def search(
        self,
        *,
        collection: str,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedItem]:
        if not query_embedding:
            return []

        if top_k <= 0:
            return []

        try:
            result = self._collection(collection).query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=self._build_where(filters),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to search {collection}"
            ) from exc

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        items: list[RetrievedItem] = []

        for index, item_id in enumerate(ids):
            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            content = (
                documents[index]
                if index < len(documents)
                else ""
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            score = (
                1.0 - distance
                if distance is not None
                else None
            )

            items.append(
                RetrievedItem(
                    item_id=item_id,
                    source=metadata.get("source", collection),
                    content=content,
                    score=score,
                    rank=index + 1,
                    metadata=metadata,
                )
            )

        return items

    def delete(
        self,
        *,
        collection: str,
        ids: list[str],
    ) -> None:
        if not ids:
            return

        try:
            self._collection(collection).delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete items from {collection}"
            ) from exc