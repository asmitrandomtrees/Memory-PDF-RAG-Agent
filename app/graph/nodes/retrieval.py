from typing import Any

from app.contracts.retrieval import (
    LTMQuery,
    PDFQuery,
    RetrievalResult,
    STMQuery,
)
from app.contracts.runtime import ConversationStore
from app.graph.state import GraphState
from app.memory.stm.context import STMContextExpander
from app.memory.stm.context_budget import STMContextBudget
from app.memory.stm.summarizer import STMContextSummarizer
from app.observability.tracker import ObservabilityTracker


class STMNode:
    def __init__(
        self,
        *,
        stm_retriever: Any | None = None,
        episodic_retriever: Any | None = None,
        conversation_store: ConversationStore | None = None,
        context_expander: STMContextExpander | None = None,
        context_budget: STMContextBudget | None = None,
        summarizer: STMContextSummarizer | None = None,
        top_k: int = 5,
        recency_weight: float = 0.5,
        tracker: ObservabilityTracker | None = None,
    ) -> None:
        self.stm_retriever = stm_retriever
        self.episodic_retriever = episodic_retriever
        self.conversation_store = conversation_store
        self.context_expander = context_expander or STMContextExpander(
            window_size=1,
            recent_message_count=6,
        )
        self.context_budget = context_budget
        self.summarizer = summarizer
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
                result = self._expand_thread_context(result)
            self.tracker.track_retrieval(
                source="stm",
                retrieved_count=len(result.items),
                latency_ms=result.latency_ms or 0.0,
            )
            return _validate_result(result, "stm")
        except Exception as exc:
            self.tracker.track_retrieval(source="stm", retrieved_count=0, error=True)
            return {"stm_error": str(exc)}

    def _expand_thread_context(
        self,
        result: RetrievalResult,
    ) -> RetrievalResult:
        if self.conversation_store is None:
            return result

        conversation = self.conversation_store.load(
            str(result.metadata.get("user_id", "")),
            str(result.metadata.get("thread_id", "")),
        )
        if not conversation.messages:
            return result

        items = self.context_expander.expand(
            items=result.items,
            conversation=conversation,
            include_recent_messages=True,
        )
        item_ids = {item.item_id for item in items}
        items = [
            *[
                item
                for item in result.items
                if item.item_id not in item_ids
            ],
            *items,
        ]
        context_processing: dict[str, Any] = {
            "max_tokens": (
                self.context_budget.max_tokens
                if self.context_budget is not None
                else None
            ),
            "input_item_count": len(items),
            "input_estimated_tokens": (
                sum(
                    self.context_budget.estimate_tokens(item.content)
                    for item in items
                )
                if self.context_budget is not None
                else None
            ),
            "summary_attempted": False,
            "summary_added": False,
            "summary_source_count": 0,
            "summary_estimated_tokens": 0,
        }
        if self.context_budget is not None:
            selected_items = self.context_budget.select(items)
            full_budget_selection = selected_items
            selected_ids = {
                item.item_id
                for item in selected_items
            }
            omitted_items = [
                item
                for item in items
                if item.item_id not in selected_ids
            ]
            context_processing["overflow"] = bool(omitted_items)
            context_processing["selected_verbatim_count"] = len(selected_items)
            context_processing["omitted_item_count"] = len(omitted_items)
            if omitted_items and self.summarizer is not None:
                total_tokens = sum(
                    self.context_budget.estimate_tokens(item.content)
                    for item in items
                )
                max_tokens = self.context_budget.max_tokens
                summary_reserve = max(1, max_tokens // 4)
                if total_tokens > max_tokens and summary_reserve < max_tokens:
                    raw_budget = STMContextBudget(
                        max_tokens=max_tokens - summary_reserve,
                        chars_per_token=self.context_budget.chars_per_token,
                    )
                    selected_items = raw_budget.select(items)
                    selected_ids = {
                        item.item_id
                        for item in selected_items
                    }
                    omitted_items = [
                        item
                        for item in items
                        if item.item_id not in selected_ids
                    ]
                    context_processing["summary_attempted"] = True
                    context_processing["selected_verbatim_count"] = len(
                        selected_items
                    )
                    context_processing["omitted_item_count"] = len(
                        omitted_items
                    )
                    try:
                        summary = self.summarizer.summarize(
                            omitted_items,
                            max_tokens=summary_reserve,
                        )
                    except Exception as exc:
                        context_processing["summary_error"] = str(exc)
                        summary = None
                    if summary is not None:
                        selected_tokens = sum(
                            self.context_budget.estimate_tokens(item.content)
                            for item in selected_items
                        )
                        summary_tokens = self.context_budget.estimate_tokens(
                            summary.content
                        )
                        context_processing["summary_source_count"] = len(
                            summary.metadata.get("source_item_ids", [])
                        )
                        context_processing["summary_estimated_tokens"] = (
                            summary_tokens
                        )
                        if selected_tokens + summary_tokens <= max_tokens:
                            selected_items.append(summary)
                            context_processing["summary_added"] = True
                        else:
                            selected_items = full_budget_selection
                    else:
                        selected_items = full_budget_selection
                else:
                    selected_items = full_budget_selection
            items = selected_items
            context_processing["output_estimated_tokens"] = sum(
                self.context_budget.estimate_tokens(item.content)
                for item in items
            )

        ranked_items = [
            item.model_copy(update={"rank": rank})
            for rank, item in enumerate(items, start=1)
        ]

        return result.model_copy(
            update={
                "items": ranked_items,
                "metadata": {
                    **result.metadata,
                    "expanded_with_recent_messages": True,
                    "raw_retrieved_count": len(result.items),
                    "context_processing": context_processing,
                },
            }
        )


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
