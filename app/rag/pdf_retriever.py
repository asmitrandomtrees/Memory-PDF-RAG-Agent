from __future__ import annotations

import time

from app.contracts.errors import RetrievalError
from app.contracts.retrieval import PDFQuery, RetrievalResult
from app.embeddings.provider import EmbeddingProvider
from app.vectorstores.base import VectorStore


class PDFRetriever:
    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        collection: str,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.collection = collection

    def retrieve(self, query: PDFQuery) -> RetrievalResult:
        if not query.query.strip():
            raise RetrievalError("Cannot retrieve PDFs with an empty query")

        start = time.perf_counter()
        try:
            filters = (
                {"document_id": {"$in": query.document_ids}}
                if query.document_ids
                else None
            )
            items = self.vector_store.search(
                collection=self.collection,
                query_embedding=self.embeddings.embed_query(query.query),
                top_k=query.top_k,
                filters=filters,
            )
            return RetrievalResult(
                source="pdf",
                query=query.query,
                items=items,
                latency_ms=(time.perf_counter() - start) * 1000,
                metadata={
                    "top_k": query.top_k,
                    "document_ids": query.document_ids,
                },
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError("Failed to retrieve PDF documents") from exc