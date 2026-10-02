from typing import Any

from app.contracts.retrieval import (
    LTMQuery,
    PDFQuery,
    RetrievalResult,
    STMQuery,
)
from app.graph.state import GraphState
from app.observability.tracker import ObservabilityTracker


class STMNode:
    def __init__(
        self,
        *,
        stm_retriever: Any | None = None,
        episodic_retriever: Any | None = None,
        top_k: int = 5,
        recency_weight: float = 0.5,
        tracker: ObservabilityTracker | None = None,
    ) -> None:
        self.stm_retriever = stm_retriever
        self.episodic_retriever = episodic_retriever
        self.top_k = top_k
        self.recency_weight = recency_weight
        self.tracker = tracker or ObservabilityTracker()

    def __call__(self, state: GraphState) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        try:
            start_at = state.get("date_start_at")
            end_at = state.get("date_end_at")
            if start_at is not None and end_at is not None:
                if self.episodic_retriever is None:
                    raise RuntimeError(
                        "STM episodic retriever is not configured"
                    )
                result = self.episodic_retriever.retrieve(
                    user_id=state["user_id"],
                    query=query,
                    start_at=start_at,
                    end_at=end_at,
                )
            else:
                if self.stm_retriever is None:
                    raise RuntimeError("STM retriever is not configured")
                result = self.stm_retriever.retrieve(
                    STMQuery(
                        user_id=state["user_id"],
                        thread_id=state["thread_id"],
                        query=query,
                        top_k=self.top_k,
                        recency_weight=self.recency_weight,
                    )
                )
            self.tracker.track_retrieval(
                source="stm",
                retrieved_count=len(result.items),
                latency_ms=result.latency_ms or 0.0,
            )
            return _validate_result(result, "stm")
        except Exception as exc:
            self.tracker.track_retrieval(source="stm", retrieved_count=0, error=True)
            return {"stm_error": str(exc)}


class LTMNode:
    def __init__(
        self,
        *,
        retriever: Any | None,
        top_k: int = 5,
        tracker: ObservabilityTracker | None = None,
    ) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.tracker = tracker or ObservabilityTracker()

    def __call__(self, state: GraphState) -> dict[str, Any]:
        try:
            if self.retriever is None:
                raise RuntimeError("LTM retriever is not configured")
            result = self.retriever.retrieve(
                LTMQuery(
                    user_id=state["user_id"],
                    thread_id=state["thread_id"],
                    query=state.get("rewritten_query") or state["query"],
                    top_k=self.top_k,
                )
            )
            self.tracker.track_retrieval(
                source="ltm",
                retrieved_count=len(result.items),
                latency_ms=result.latency_ms or 0.0,
            )
            return _validate_result(result, "ltm")
        except Exception as exc:
            self.tracker.track_retrieval(source="ltm", retrieved_count=0, error=True)
            return {"ltm_error": str(exc)}


class PDFNode:
    def __init__(
        self,
        *,
        retriever: Any | None,
        top_k: int = 5,
        tracker: ObservabilityTracker | None = None,
    ) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.tracker = tracker or ObservabilityTracker()

    def __call__(self, state: GraphState) -> dict[str, Any]:
        try:
            if self.retriever is None:
                raise RuntimeError("PDF retriever is not configured")
            result = self.retriever.retrieve(
                PDFQuery(
                    user_id=state["user_id"],
                    thread_id=state["thread_id"],
                    query=state.get("rewritten_query") or state["query"],
                    top_k=self.top_k,
                    document_ids=state.get("document_ids", []),
                )
            )
            self.tracker.track_retrieval(
                source="pdf",
                retrieved_count=len(result.items),
                latency_ms=result.latency_ms or 0.0,
            )
            return _validate_result(result, "pdf")
        except Exception as exc:
            self.tracker.track_retrieval(source="pdf", retrieved_count=0, error=True)
            return {"pdf_error": str(exc)}


def _validate_result(
    result: RetrievalResult,
    source: str,
) -> dict[str, Any]:
    if result.source != source:
        return {
            f"{source}_error": (
                f"{source} retriever returned source {result.source}"
            )
        }
    return {f"{source}_result": result, f"{source}_error": None}