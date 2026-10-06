from datetime import datetime, timezone
import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.contracts.context import AgentContext
from app.contracts.retrieval import MergedRetrievalResult
from app.contracts.retrieval import RetrievedItem
from app.contracts.runtime import AgentRequest, AgentResponse, ConversationStore
from app.contracts.tracing import GraphTrace, RetrievalTrace
from app.graph.nodes.answer import (
    AnswerEvidenceValidator,
    AnswerGenerator,
    QueryRewriter,
)
from app.graph.nodes.context import ContextBuilder, LLMContextCompressor
from app.graph.nodes.query import QueryAnalyzer, RetrievalPlanner
from app.graph.nodes.retrieval import LTMNode, PDFNode, STMNode
from app.graph.schemas import AnswerDraft, resolve_calendar_day
from app.graph.state import GraphState
from app.llm.provider import LLMProvider
<<<<<<< Updated upstream
from app.rag.reranker import ReciprocalRankFusionReranker, Reranker
=======
from app.graph.reranker import ReciprocalRankFusionReranker, Reranker
from app.memory.stm.context_budget import STMContextBudget
>>>>>>> Stashed changes
from app.observability.tracker import ObservabilityTracker
from app.tracing.tracer import TraceSink


class Phase0Graph:
    def run(
        self,
        request: AgentRequest,
        trace_id: str,
    ) -> AgentResponse:
        return AgentResponse(
            user_id=request.user_id,
            thread_id=request.thread_id,
            answer=(
                "Phase 0 skeleton is working. "
                "The actual agent graph has not been implemented yet."
            ),
            trace_id=trace_id,
        )


class RetrievalPipeline:
    """Integrated retrieval, evidence-grounded answering, and validation graph."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        stm_retriever: Any | None = None,
        episodic_retriever: Any | None = None,
        conversation_store: ConversationStore | None = None,
        ltm_retriever: Any | None = None,
        pdf_retriever: Any | None = None,
        reranker: Reranker | None = None,
        context_builder: ContextBuilder | None = None,
        stm_top_k: int = 5,
        ltm_top_k: int = 5,
        pdf_top_k: int = 5,
        recency_weight: float = 0.5,
        max_context_tokens: int = 6000,
        max_retries: int = 2,
        trace_sink: TraceSink | None = None,
        tracker: ObservabilityTracker | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")

        self.tracker = tracker or ObservabilityTracker()
        self.analyzer = QueryAnalyzer(llm)
        self.planner = RetrievalPlanner(llm)
        self.answer_generator = AnswerGenerator(llm)
        self.answer_validator = AnswerEvidenceValidator(llm)
        self.query_rewriter = QueryRewriter(llm)
        self.max_retries = max_retries
        self.conversation_store = conversation_store
        self.stm_node = STMNode(
            stm_retriever=stm_retriever,
            episodic_retriever=episodic_retriever,
            conversation_store=conversation_store,
            context_budget=STMContextBudget(max_tokens=max_context_tokens),
            top_k=stm_top_k,
            recency_weight=recency_weight,
            tracker=self.tracker,
        )
        self.ltm_node = LTMNode(
            retriever=ltm_retriever,
            top_k=ltm_top_k,
            tracker=self.tracker,
        )
        self.pdf_node = PDFNode(
            retriever=pdf_retriever,
            top_k=pdf_top_k,
            tracker=self.tracker,
        )
        self.reranker = reranker or ReciprocalRankFusionReranker()
        self.trace_sink = trace_sink
        self.context_builder = context_builder or ContextBuilder(
            max_tokens=max_context_tokens,
            compressor=LLMContextCompressor(llm),
        )

        builder = StateGraph(GraphState)
        builder.add_node("analyze", self._analyze)
        builder.add_node("plan", self._plan)
        builder.add_node("stm", self.stm_node)
        builder.add_node("ltm", self.ltm_node)
        builder.add_node("pdf", self.pdf_node)
        builder.add_node("merge", self._merge)
        builder.add_node("rerank", self._rerank)
        builder.add_node("build_context", self._build_context)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("validate_answer", self._validate_answer)
        builder.add_node("rewrite_query", self._rewrite_query)
        builder.add_node("exhausted", self._exhausted)
        builder.add_edge(START, "analyze")
        builder.add_edge("analyze", "plan")
        builder.add_conditional_edges("plan", self._route_sources)
        builder.add_edge("stm", "merge")
        builder.add_edge("ltm", "merge")
        builder.add_edge("pdf", "merge")
        builder.add_edge("merge", "rerank")
        builder.add_edge("rerank", "build_context")
        builder.add_edge("build_context", "generate_answer")
        builder.add_edge("generate_answer", "validate_answer")
        builder.add_conditional_edges(
            "validate_answer",
            self._route_validation,
            {
                "done": END,
                "rewrite": "rewrite_query",
                "exhausted": "exhausted",
            },
        )
        builder.add_edge("rewrite_query", "analyze")
        builder.add_edge("exhausted", END)
        self.compiled = builder.compile()

    @staticmethod
    def _route_sources(state: GraphState) -> list[Send] | str:
        plan = state["retrieval_plan"]
        sends = [
            Send(node_name, dict(state))
            for node_name, selected in (
                ("stm", plan.use_stm),
                ("ltm", plan.use_ltm),
                ("pdf", plan.use_pdf),
            )
            if selected
        ]
        return sends or "merge"

    def invoke(
        self,
        *,
        user_id: str,
        thread_id: str,
        query: str,
        document_ids: list[str] | None = None,
        trace_id: str | None = None,
    ) -> GraphState:
        return self.compiled.invoke(
            {
                "user_id": user_id,
                "thread_id": thread_id,
                "query": query,
                "document_ids": document_ids or [],
                "recent_items": self._recent_thread_items(
                    user_id=user_id,
                    thread_id=thread_id,
                ),
                "trace_id": trace_id or "",
                "retry_count": 0,
            }
        )

    def run(
        self,
        request: AgentRequest,
        trace_id: str,
    ) -> AgentResponse:
        document_ids = request.metadata.get("document_ids", [])
        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        if not isinstance(document_ids, list) or not all(
            isinstance(document_id, str) for document_id in document_ids
        ):
            document_ids = []

        state = self.invoke(
            user_id=request.user_id,
            thread_id=request.thread_id,
            query=request.message,
            document_ids=document_ids,
            trace_id=trace_id,
        )
        completed_at = datetime.now(timezone.utc)
        latency_ms = (time.perf_counter() - started_clock) * 1000
        context = state.get("context") or AgentContext()
        answer = state.get("answer") or self._insufficient_evidence_answer()
        trace_error = None
        if self.trace_sink is not None:
            try:
                self.trace_sink.emit(
                    self._build_trace(
                        request=request,
                        trace_id=trace_id,
                        state=state,
                        started_at=started_at,
                        completed_at=completed_at,
                        latency_ms=latency_ms,
                    )
                )
            except Exception as exc:
                trace_error = str(exc)

        response_metadata: dict[str, Any] = {
            "citations": self._citation_metadata(
                state.get("cited_item_ids", []),
                context,
            ),
            "insufficient_evidence": state.get(
                "insufficient_evidence", True
            ),
            "retrieval_errors": state.get("retrieval_errors", {}),
            "retry_count": state.get("retry_count", 0),
            "validation": state.get("validation", {}),
        }
        if trace_error is not None:
            response_metadata["trace_error"] = trace_error

        return AgentResponse(
            user_id=request.user_id,
            thread_id=request.thread_id,
            answer=answer,
            trace_id=trace_id,
            metadata=response_metadata,
        )

    @staticmethod
    def _build_trace(
        *,
        request: AgentRequest,
        trace_id: str,
        state: GraphState,
        started_at: datetime,
        completed_at: datetime,
        latency_ms: float,
    ) -> GraphTrace:
        def build_retrieval_trace(source: str) -> RetrievalTrace | None:
            result = state.get(f"{source}_result")
            if result is None:
                return None
            return RetrievalTrace(
                source=source,
                query=result.query,
                retrieved_ids=[item.item_id for item in result.items],
                scores=[item.score for item in result.items],
                ranks=[item.rank for item in result.items],
                latency_ms=result.latency_ms,
                metadata=result.metadata,
            )

        merged = state.get("merged_results")
        reranked = state.get("reranked_results")
        context = state.get("context")
        plan = state.get("retrieval_plan")

        return GraphTrace(
            trace_id=trace_id,
            user_id=request.user_id,
            thread_id=request.thread_id,
            query=request.message,
            started_at=started_at,
            completed_at=completed_at,
            planner=plan.model_dump(mode="json") if plan else {},
            stm=build_retrieval_trace("stm"),
            ltm=build_retrieval_trace("ltm"),
            pdf_rag=build_retrieval_trace("pdf"),
            reranker={
                "method": "reciprocal_rank_fusion",
                "input_count": len(merged.items) if merged else 0,
                "output_count": len(reranked.items) if reranked else 0,
            },
            context={
                "item_count": len(context.items) if context else 0,
                "estimated_tokens": (
                    context.estimated_tokens if context else 0
                ),
                "max_tokens": context.max_tokens if context else 0,
            },
            answer={
                "insufficient_evidence": state.get(
                    "insufficient_evidence", True
                ),
                "cited_item_ids": state.get("cited_item_ids", []),
            },
            validation=state.get("validation", {}),
            metadata={
                "latency_ms": latency_ms,
                "retry_count": state.get("retry_count", 0),
                "retrieval_errors": state.get("retrieval_errors", {}),
            },
        )

    def _analyze(self, state: GraphState) -> dict[str, Any]:
        with self.tracker.track_latency("query_analysis"):
            query = state.get("rewritten_query") or state["query"]
            analysis = self.analyzer.analyze(query)
            start_at, end_at = resolve_calendar_day(analysis)
            return {
                "query_analysis": analysis,
                "rewritten_query": analysis.normalized_query,
                "date_start_at": start_at,
                "date_end_at": end_at,
            }

    def _plan(self, state: GraphState) -> dict[str, Any]:
        with self.tracker.track_latency("retrieval_planning"):
            query = state.get("rewritten_query") or state["query"]
            return {
                "retrieval_plan": self.planner.plan(
                    query=query,
                    analysis=state["query_analysis"],
                ),
                "merged_results": None,
                "reranked_results": None,
                "context": None,
                "answer": None,
                "cited_item_ids": [],
                "insufficient_evidence": False,
                "validation": {},
            }

    @staticmethod
    def _merge(state: GraphState) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        results = [
            result
            for key in ("stm_result", "ltm_result", "pdf_result")
            if (result := state.get(key)) is not None
        ]
        retrieval_errors = {
            source: state[f"{source}_error"]
            for source in ("stm", "ltm", "pdf")
            if state.get(f"{source}_error") is not None
        }
        recent_items = state.get("recent_items", [])
        seen_item_ids = {item.item_id for item in recent_items}
        items = [
            *recent_items,
            *[
                item
                for result in results
                for item in result.items
                if item.item_id not in seen_item_ids
            ],
        ]
        return {
            "retrieval_errors": retrieval_errors,
            "merged_results": MergedRetrievalResult(
                query=query,
                items=items,
                metadata={
                    "source_counts": {
                        result.source: len(result.items)
                        for result in results
                    },
                    "retrieval_errors": retrieval_errors,
                },
            ),
        }

    def _rerank(self, state: GraphState) -> dict[str, MergedRetrievalResult]:
        with self.tracker.track_latency("reranking"):
            merged = state["merged_results"]
            reranked_items = self.reranker.rerank(merged.query, merged.items)
            self.tracker.track_reranker(
                input_count=len(merged.items),
                output_count=len(reranked_items),
            )
            return {
                "reranked_results": MergedRetrievalResult(
                    query=merged.query,
                    items=reranked_items,
                    metadata=merged.metadata,
                )
            }

    def _build_context(self, state: GraphState) -> dict[str, Any]:
        with self.tracker.track_latency("context_building"):
            return {
                "context": self.context_builder.build(
                    state["reranked_results"].items
                )
            }

    def _generate_answer(self, state: GraphState) -> dict[str, Any]:
        with self.tracker.track_latency("answer_generation"):
            context = state.get("context") or AgentContext()
            draft = self.answer_generator.generate(
                query=state.get("rewritten_query") or state["query"],
                context=context,
            )
            return {
                "answer": draft.answer,
                "cited_item_ids": draft.cited_item_ids,
                "insufficient_evidence": draft.insufficient_evidence,
            }

    def _validate_answer(self, state: GraphState) -> dict[str, Any]:
        with self.tracker.track_latency("answer_validation"):
            draft = AnswerDraft(
                answer=state["answer"],
                cited_item_ids=state.get("cited_item_ids", []),
                insufficient_evidence=state.get("insufficient_evidence", False),
            )
            validation = self.answer_validator.validate(
                query=state.get("rewritten_query") or state["query"],
                draft=draft,
                context=state.get("context") or AgentContext(),
            )
            return {"validation": validation.model_dump()}

    def _route_validation(self, state: GraphState) -> str:
        validation = state.get("validation", {})
        if validation.get("is_valid"):
            return "done"
        if state.get("retry_count", 0) >= self.max_retries:
            self.tracker.track_exhaustion()
            return "exhausted"
        self.tracker.track_retry(reason=validation.get("reason"))
        return "rewrite"

    def _rewrite_query(self, state: GraphState) -> dict[str, Any]:
        with self.tracker.track_latency("query_rewriting"):
            validation = state.get("validation", {})
            rewritten_query = self.query_rewriter.rewrite(
                query=state["query"],
                reason=validation.get("reason"),
            )
            return {
                "rewritten_query": rewritten_query,
                "retry_count": state.get("retry_count", 0) + 1,
            }

    @staticmethod
    def _exhausted(state: GraphState) -> dict[str, Any]:
        return {
            "answer": RetrievalPipeline._insufficient_evidence_answer(),
            "cited_item_ids": [],
            "insufficient_evidence": True,
            "validation": {
                **state.get("validation", {}),
                "retry_exhausted": True,
            },
        }

    def _recent_thread_items(
        self,
        *,
        user_id: str,
        thread_id: str,
        limit: int = 8,
    ) -> list[RetrievedItem]:
        if self.conversation_store is None:
            return []

        conversation = self.conversation_store.load(user_id, thread_id)
        recent_messages = conversation.messages[-limit:]
        return [
            RetrievedItem(
                item_id=message.message_id,
                source="stm",
                content=message.content,
                score=None,
                rank=index,
                metadata={
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "role": message.role,
                    "timestamp": message.timestamp.isoformat(),
                    "context_type": "recent",
                },
            )
            for index, message in enumerate(recent_messages, start=1)
        ]

    @staticmethod
    def _insufficient_evidence_answer() -> str:
        return "I couldn't verify a sufficiently supported answer from the retrieved information."

    @staticmethod
    def _citation_metadata(
        cited_item_ids: list[str],
        context: AgentContext,
    ) -> list[dict[str, object]]:
        citations: list[dict[str, object]] = []
        for item in context.items:
            source_references = item.metadata.get("source_references", [])
            if item.item_id in cited_item_ids:
                citations.append(
                    {
                        "source": item.source,
                        "item_id": item.item_id,
                        "metadata": item.metadata,
                    }
                )
            if isinstance(source_references, list):
                citations.extend(
                    {
                        "source": item.source,
                        **reference,
                    }
                    for reference in source_references
                    if isinstance(reference, dict)
                    and reference.get("item_id") in cited_item_ids
                )
        return citations