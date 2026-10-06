from __future__ import annotations

import time

from app.contracts.retrieval import RetrievalResult


class PDFRetriever:
    def __init__(self, embedder, store, collection: str, top_k: int = 5) -> None:
        self.embedder = embedder
        self.store = store
        self.collection = collection
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        start = time.perf_counter()
        items = self.store.search(
            collection=self.collection,
            query_embedding=self.embedder.embed_query(query),
            top_k=top_k or self.top_k,
        )
        return RetrievalResult(
            source="pdf",
            query=query,
            items=items,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
