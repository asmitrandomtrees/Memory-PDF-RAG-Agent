from datetime import datetime, timezone
from threading import Barrier

import pytest

from app.contracts.conversation import Conversation, ConversationMessage
from app.contracts.context import AgentContext
from app.contracts.retrieval import RetrievalResult, RetrievedItem
from app.contracts.routing import RetrievalPlan
from app.graph.graph import RetrievalPipeline
from app.graph.nodes.context import ContextBuilder
from app.graph.nodes.context import ContextSummary, LLMContextCompressor
from app.graph.schemas import (
    AnswerDraft,
    AnswerValidation,
    INDIA_TIMEZONE,
    QueryAnalysis,
    RewrittenQuery,
    resolve_calendar_day,
)
from app.contracts.runtime import AgentRequest
from app.memory.ltm.retriever import LTMRetriever
from app.memory.stm.retriever import STMRetriever
from app.rag.reranker import ReciprocalRankFusionReranker


class FakeLLM:
    def __init__(
        self,
        *,
        analysis: QueryAnalysis,
        plan: RetrievalPlan,
        answer_drafts: list[AnswerDraft] | None = None,
        validations: list[AnswerValidation] | None = None,
        rewrites: list[RewrittenQuery] | None = None,
    ) -> None:
        self.analysis = analysis
        self.plan = plan
        self.answer_drafts = list(answer_drafts or [])
        self.validations = list(validations or [])
        self.rewrites = list(rewrites or [])
        self.rewrite_prompts = []

    def structured_output(self, messages, output_schema, *, temperature=None):
        if output_schema is QueryAnalysis:
            return self.analysis
        if output_schema is RetrievalPlan:
            return self.plan
        if output_schema is ContextSummary:
            return ContextSummary(summary="compressed")
        if output_schema is AnswerDraft:
            if self.answer_drafts:
                return self.answer_drafts.pop(0)
            return AnswerDraft(
                answer="No sufficient evidence was supplied.",
                insufficient_evidence=True,
            )
        if output_schema is AnswerValidation:
            if self.validations:
                return self.validations.pop(0)
            return AnswerValidation(is_valid=True)
        if output_schema is RewrittenQuery:
            self.rewrite_prompts.append(messages)
            if self.rewrites:
                return self.rewrites.pop(0)
            return RewrittenQuery(query="clarified query")
        raise AssertionError(f"Unexpected output schema: {output_schema}")


class FakeRetriever:
    def __init__(
        self,
        result: RetrievalResult | None = None,
        *,
        barrier: Barrier | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.barrier = barrier
        self.error = error
        self.query = None
        self.calls = 0

    def retrieve(self, query=None, **kwargs):
        self.calls += 1
        self.query = kwargs if "start_at" in kwargs else query
        if self.barrier is not None:
            self.barrier.wait(timeout=3)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class FakeConversationStore:
    def __init__(self, conversation: Conversation) -> None:
        self.conversation = conversation

    def load(self, user_id: str, thread_id: str) -> Conversation:
        assert user_id == self.conversation.user_id
        assert thread_id == self.conversation.thread_id
        return self.conversation

    def append_message(self, message: ConversationMessage) -> None:
        self.conversation.messages.append(message)


def _result(source: str, item_id: str, content: str, rank: int = 1):
    return RetrievalResult(
        source=source,
        query="query",
        items=[
            RetrievedItem(
                item_id=item_id,
                source=source,
                content=content,
                score=0.8,
                rank=rank,
                metadata={"origin": source},
            )
        ],
    )


def _pipeline(
    *,
    analysis: QueryAnalysis,
    plan: RetrievalPlan,
    stm=None,
    episodic=None,
    ltm=None,
    pdf=None,
    conversation_store=None,
    max_context_tokens: int = 100,
) -> RetrievalPipeline:
    return RetrievalPipeline(
        llm=FakeLLM(analysis=analysis, plan=plan),
        stm_retriever=stm,
        episodic_retriever=episodic,
        conversation_store=conversation_store,
        ltm_retriever=ltm,
        pdf_retriever=pdf,
        context_builder=ContextBuilder(
            max_tokens=max_context_tokens,
            chars_per_token=4,
        ),
    )


def test_pipeline_runs_selected_sources_in_parallel_and_builds_context() -> None:
    barrier = Barrier(3)
    analysis = QueryAnalysis(
        intent="mixed",
        normalized_query="Which database did I prefer and why?",
    )
    plan = RetrievalPlan(use_stm=True, use_ltm=True, use_pdf=True)
    stm = FakeRetriever(_result("stm", "s1", "STM evidence"), barrier=barrier)
    ltm = FakeRetriever(_result("ltm", "l1", "LTM evidence"), barrier=barrier)
    pdf = FakeRetriever(_result("pdf", "p1", "PDF evidence"), barrier=barrier)
    pipeline = _pipeline(
        analysis=analysis,
        plan=plan,
        stm=stm,
        ltm=ltm,
        pdf=pdf,
    )

    state = pipeline.invoke(
        user_id="user_001",
        thread_id="thread_001",
        query="Which database did I prefer and why?",
        document_ids=["doc_001"],
    )

    assert [stm.calls, ltm.calls, pdf.calls] == [1, 1, 1]
    assert pdf.query.document_ids == ["doc_001"]
    assert state["retrieval_errors"] == {}
    assert {item.source for item in state["merged_results"].items} == {
        "stm",
        "ltm",
        "pdf",
    }
    assert [item.rank for item in state["reranked_results"].items] == [1, 2, 3]
    assert isinstance(state["context"], AgentContext)
    assert {item.source for item in state["context"].items} == {
        "stm",
        "ltm",
        "pdf",
    }


def test_pipeline_preserves_successful_sources_when_one_retriever_fails() -> None:
    analysis = QueryAnalysis(
        intent="mixed",
        normalized_query="What did I discuss and remember?",
    )
    plan = RetrievalPlan(use_stm=True, use_ltm=True)
    stm = FakeRetriever(_result("stm", "s1", "Conversation evidence"))
    ltm = FakeRetriever(error=RuntimeError("LTM unavailable"))
    pipeline = _pipeline(
        analysis=analysis,
        plan=plan,
        stm=stm,
        ltm=ltm,
    )

    state = pipeline.invoke(
        user_id="user_001",
        thread_id="thread_001",
        query="What did I discuss and remember?",
    )

    assert [item.item_id for item in state["merged_results"].items] == ["s1"]
    assert state["retrieval_errors"] == {"ltm": "LTM unavailable"}
    assert state["context"].items[0].source == "stm"


def test_episodic_analysis_routes_to_ist_day_range_even_if_plan_omits_stm() -> None:
    analysis = QueryAnalysis(
        intent="episodic_recall",
        normalized_query="What did I do on September 18, 2026?",
        date_month=9,
        date_day=18,
        date_year=2026,
    )
    plan = RetrievalPlan()
    episodic = FakeRetriever(_result("stm", "s1", "Event evidence"))
    pipeline = _pipeline(
        analysis=analysis,
        plan=plan,
        episodic=episodic,
    )

    state = pipeline.invoke(
        user_id="user_001",
        thread_id="thread_001",
        query="What did I do on September 18, 2026?",
    )

    assert episodic.calls == 1
    assert episodic.query["start_at"] == datetime(
        2026, 9, 18, tzinfo=INDIA_TIMEZONE
    )
    assert episodic.query["end_at"] == datetime(
        2026, 9, 19, tzinfo=INDIA_TIMEZONE
    )
    assert state["retrieval_plan"].use_stm is True


def test_date_less_episodic_recall_uses_thread_scoped_stm() -> None:
    analysis = QueryAnalysis(
        intent="episodic_recall",
        normalized_query="What were we discussing?",
    )
    stm = FakeRetriever(_result("stm", "s1", "Conversation evidence"))
    episodic = FakeRetriever(_result("stm", "e1", "unused"))
    pipeline = _pipeline(
        analysis=analysis,
        plan=RetrievalPlan(use_stm=True),
        stm=stm,
        episodic=episodic,
    )

    pipeline.invoke(
        user_id="user_001",
        thread_id="thread_001",
        query="What were we discussing?",
    )

    assert stm.calls == 1
    assert stm.query.thread_id == "thread_001"
    assert episodic.calls == 0


def test_yearless_calendar_date_resolves_to_most_recent_past_occurrence() -> None:
    analysis = QueryAnalysis(
        intent="episodic_recall",
        normalized_query="What did I do on September 18?",
        date_month=9,
        date_day=18,
    )

    start_at, end_at = resolve_calendar_day(
        analysis,
        now=datetime(2026, 10, 2, tzinfo=INDIA_TIMEZONE),
    )

    assert start_at == datetime(2026, 9, 18, tzinfo=INDIA_TIMEZONE)
    assert end_at == datetime(2026, 9, 19, tzinfo=INDIA_TIMEZONE)


def test_query_analysis_discards_partial_calendar_date() -> None:
    analysis = QueryAnalysis.model_validate(
        {
            "intent": "general",
            "normalized_query": "any day of this weekend",
            "date_month": 10,
            "date_day": None,
            "date_year": None,
        }
    )

    assert analysis.date_month is None
    assert analysis.date_day is None
    assert analysis.date_year is None
    assert resolve_calendar_day(analysis) == (None, None)


def test_context_builder_respects_combined_token_budget() -> None:
    builder = ContextBuilder(max_tokens=2, chars_per_token=4)
    items = [
        _result("stm", "s1", "12345678").items[0],
        _result("ltm", "l1", "abcdefgh").items[0],
    ]

    context = builder.build(items)

    assert len(context.items) == 1
    assert context.items[0].source == "stm"
    assert context.estimated_tokens == 2


def test_rank_fusion_uses_source_local_ranks() -> None:
    items = [
        _result("stm", "stm_rank_1", "STM first", rank=1).items[0],
        _result("ltm", "ltm_rank_2", "LTM second", rank=2).items[0],
        _result("pdf", "pdf_rank_4", "PDF fourth", rank=4).items[0],
    ]

    reranked = ReciprocalRankFusionReranker().rerank("query", items)

    assert [item.item_id for item in reranked] == [
        "stm_rank_1",
        "ltm_rank_2",
        "pdf_rank_4",
    ]
    assert reranked[0].metadata["source_rank"] == 1
    assert reranked[0].metadata["source_score"] == 0.8


def test_empty_plan_returns_empty_results_without_retriever_calls() -> None:
    analysis = QueryAnalysis(
        intent="general",
        normalized_query="Hello",
    )
    stm = FakeRetriever(_result("stm", "s1", "unused"))
    ltm = FakeRetriever(_result("ltm", "l1", "unused"))
    pipeline = _pipeline(
        analysis=analysis,
        plan=RetrievalPlan(),
        stm=stm,
        ltm=ltm,
    )

    state = pipeline.invoke(
        user_id="user_001",
        thread_id="thread_001",
        query="Hello",
    )

    assert stm.calls == 0
    assert ltm.calls == 0
    assert state["merged_results"].items == []
    assert state["context"].items == []


def test_short_follow_up_forces_stm_and_includes_recent_thread_context() -> None:
    conversation = Conversation(
        user_id="user_001",
        thread_id="thread_002",
        messages=[
            ConversationMessage(
                message_id="assistant_question",
                user_id="user_001",
                thread_id="thread_002",
                role="assistant",
                content=(
                    "Do you prefer watching in theaters, at home via "
                    "streaming, or both? Any sensitivity to violence?"
                ),
                timestamp=datetime(2026, 10, 5, 16, 16, tzinfo=timezone.utc),
            ),
            ConversationMessage(
                message_id="user_follow_up",
                user_id="user_001",
                thread_id="thread_002",
                role="user",
                content="both, no such sensitivity",
                timestamp=datetime(2026, 10, 5, 16, 17, tzinfo=timezone.utc),
            ),
        ],
    )
    analysis = QueryAnalysis(
        intent="general",
        normalized_query="both, no such sensitivity",
    )
    stm = FakeRetriever(
        RetrievalResult(
            source="stm",
            query="both, no such sensitivity",
            items=[],
            metadata={
                "user_id": "user_001",
                "thread_id": "thread_002",
                "recency_weight": 0.5,
            },
        )
    )
    pipeline = _pipeline(
        analysis=analysis,
        plan=RetrievalPlan(),
        stm=stm,
        conversation_store=FakeConversationStore(conversation),
    )

    state = pipeline.invoke(
        user_id="user_001",
        thread_id="thread_002",
        query="both, no such sensitivity",
    )

    assert state["retrieval_plan"].use_stm is True
    assert stm.calls == 1
    assert [
        item.item_id
        for item in state["merged_results"].items
    ] == [
        "assistant_question",
        "user_follow_up",
    ]
    assert all(
        item.metadata["context_type"] == "recent"
        for item in state["merged_results"].items
    )


def test_tiny_whom_follow_up_forces_stm() -> None:
    analysis = QueryAnalysis(
        intent="general",
        normalized_query="with whom",
    )
    stm = FakeRetriever(
        RetrievalResult(
            source="stm",
            query="with whom",
            items=[],
            metadata={
                "user_id": "user_001",
                "thread_id": "thread_003",
                "recency_weight": 0.5,
            },
        )
    )
    pipeline = _pipeline(
        analysis=analysis,
        plan=RetrievalPlan(),
        stm=stm,
    )

    state = pipeline.invoke(
        user_id="user_001",
        thread_id="thread_003",
        query="with whom",
    )

    assert state["retrieval_plan"].use_stm is True
    assert stm.calls == 1


def test_context_compression_preserves_pdf_chunk_provenance() -> None:
    class FakeCompressionLLM:
        messages = None

        def structured_output(self, messages, output_schema, *, temperature=None):
            assert output_schema is ContextSummary
            self.messages = messages
            return ContextSummary(summary="PDFsum")

    llm = FakeCompressionLLM()
    builder = ContextBuilder(
        max_tokens=6,
        chars_per_token=4,
        compressor=LLMContextCompressor(llm),
    )
    items = [
        RetrievedItem(
            item_id="stm_item",
            source="stm",
            content="1234567890123456",
            score=0.9,
            rank=1,
            metadata={"thread_id": "thread_001"},
        ),
        RetrievedItem(
            item_id="pdf_chunk_7",
            source="pdf",
            content="abcdefghijklmnop",
            score=0.8,
            rank=2,
            metadata={"document_id": "doc_1", "page_number": 7},
        ),
    ]

    context = builder.build(items)

    assert context.estimated_tokens == 6
    assert [item.source for item in context.items] == ["stm", "pdf"]
    compressed = context.items[1]
    assert compressed.content == "PDFsum"
    assert compressed.metadata["source_item_ids"] == ["pdf_chunk_7"]
    assert compressed.metadata["source_references"] == [
        {
            "item_id": "pdf_chunk_7",
            "metadata": {
                "document_id": "doc_1",
                "page_number": 7,
                "retrieval_rank": 2,
            },
        }
    ]
    assert "doc_1" in llm.messages[1].content


def test_graph_combines_production_stm_and_ltm_retrievers() -> None:
    class FakeEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    class FakeVectorStore:
        def __init__(self) -> None:
            self.calls = []

        def search(
            self,
            *,
            collection: str,
            query_embedding: list[float],
            top_k: int,
            filters: dict | None = None,
        ) -> list[RetrievedItem]:
            self.calls.append(
                {
                    "collection": collection,
                    "filters": filters,
                    "top_k": top_k,
                }
            )
            if collection == "stm_collection":
                assert filters == {
                    "user_id": "user_001",
                    "thread_id": "thread_001",
                }
                return [
                    RetrievedItem(
                        item_id="stm_message_1",
                        source="stm",
                        content="We discussed the vector database choice.",
                        score=0.91,
                        rank=1,
                        metadata={"thread_id": "thread_001"},
                    )
                ][:top_k]

            assert collection == "ltm_collection"
            assert filters == {
                "user_id": "user_001",
                "status": "active",
            }
            return [
                RetrievedItem(
                    item_id="ltm_memory_1",
                    source="ltm",
                    content="User prefers Chroma.",
                    score=0.87,
                    rank=1,
                    metadata={"memory_type": "preference"},
                )
            ][:top_k]

    vector_store = FakeVectorStore()
    embeddings = FakeEmbeddings()
    stm_retriever = STMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )
    ltm_retriever = LTMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )
    pipeline = _pipeline(
        analysis=QueryAnalysis(
            intent="mixed",
            normalized_query="What database did I prefer and discuss?",
        ),
        plan=RetrievalPlan(use_stm=True, use_ltm=True),
        stm=stm_retriever,
        ltm=ltm_retriever,
    )

    state = pipeline.invoke(
        user_id="user_001",
        thread_id="thread_001",
        query="What database did I prefer and discuss?",
    )

    assert {call["collection"] for call in vector_store.calls} == {
        "stm_collection",
        "ltm_collection",
    }
    assert [item.source for item in state["merged_results"].items] == [
        "stm",
        "ltm",
    ]
    assert {item.source for item in state["context"].items} == {
        "stm",
        "ltm",
    }
    assert state["retrieval_errors"] == {}


def test_pipeline_compresses_overflowing_context_by_default() -> None:
    analysis = QueryAnalysis(
        intent="memory_recall",
        normalized_query="What do I prefer?",
    )
    ltm = FakeRetriever(_result("ltm", "memory_1", "A" * 80))
    pipeline = RetrievalPipeline(
        llm=FakeLLM(
            analysis=analysis,
            plan=RetrievalPlan(use_ltm=True),
        ),
        ltm_retriever=ltm,
        max_context_tokens=4,
    )

    state = pipeline.invoke(
        user_id="user_001",
        thread_id="thread_001",
        query="What do I prefer?",
    )

    assert len(state["context"].items) == 1
    summary = state["context"].items[0]
    assert summary.content == "compressed"
    assert summary.metadata["source_item_ids"] == ["memory_1"]


def test_pipeline_generates_validated_answer_and_graph_runner_response() -> None:
    llm = FakeLLM(
        analysis=QueryAnalysis(
            intent="memory_recall",
            normalized_query="What database do I prefer?",
        ),
        plan=RetrievalPlan(use_ltm=True),
        answer_drafts=[
            AnswerDraft(
                answer="You prefer Chroma.",
                cited_item_ids=["memory_1"],
            )
        ],
        validations=[AnswerValidation(is_valid=True)],
    )
    pipeline = RetrievalPipeline(
        llm=llm,
        ltm_retriever=FakeRetriever(
            _result("ltm", "memory_1", "User prefers Chroma.")
        ),
    )

    response = pipeline.run(
        AgentRequest(
            user_id="user_001",
            thread_id="thread_001",
            message="What database do I prefer?",
        ),
        "trace_123",
    )

    assert response.answer == "You prefer Chroma."
    assert response.trace_id == "trace_123"
    assert len(response.metadata["citations"]) == 1
    citation = response.metadata["citations"][0]
    assert citation["source"] == "ltm"
    assert citation["item_id"] == "memory_1"
    assert citation["metadata"]["origin"] == "ltm"
    assert citation["metadata"]["retrieval_rank"] == 1
    assert response.metadata["insufficient_evidence"] is False


def test_pipeline_rewrites_invalid_citation_and_retries_once() -> None:
    llm = FakeLLM(
        analysis=QueryAnalysis(
            intent="memory_recall",
            normalized_query="Which database?",
        ),
        plan=RetrievalPlan(use_ltm=True),
        answer_drafts=[
            AnswerDraft(answer="Unsupported.", cited_item_ids=["missing"]),
            AnswerDraft(
                answer="You prefer Chroma.",
                cited_item_ids=["memory_1"],
            ),
        ],
        validations=[AnswerValidation(is_valid=True)],
        rewrites=[RewrittenQuery(query="Which vector database do I prefer?")],
    )
    retriever = FakeRetriever(_result("ltm", "memory_1", "User prefers Chroma."))
    pipeline = RetrievalPipeline(llm=llm, ltm_retriever=retriever, max_retries=1)

    state = pipeline.invoke(
        user_id="user_001",
        thread_id="thread_001",
        query="Which database?",
    )

    assert state["answer"] == "You prefer Chroma."
    assert state["retry_count"] == 1
    assert retriever.calls == 2
    assert state["validation"]["is_valid"] is True


def test_rewrite_retries_keep_original_user_query() -> None:
    llm = FakeLLM(
        analysis=QueryAnalysis(
            intent="general",
            normalized_query="with whom",
        ),
        plan=RetrievalPlan(use_stm=True),
        answer_drafts=[
            AnswerDraft(answer="Unsupported one.", cited_item_ids=["bad1"]),
            AnswerDraft(answer="Unsupported two.", cited_item_ids=["bad2"]),
            AnswerDraft(answer="You planned to go with friends.", cited_item_ids=["friend"]),
        ],
        rewrites=[
            RewrittenQuery(query="first rewritten query"),
            RewrittenQuery(query="second rewritten query"),
        ],
    )
    retriever = FakeRetriever(
        RetrievalResult(
            source="stm",
            query="with whom",
            items=[
                RetrievedItem(
                    item_id="friend",
                    source="stm",
                    content="I am planning to watch a movie at theatre with my friends",
                    metadata={"thread_id": "thread_003"},
                )
            ],
        )
    )
    pipeline = RetrievalPipeline(
        llm=llm,
        stm_retriever=retriever,
        max_retries=2,
    )

    state = pipeline.invoke(
        user_id="user_001",
        thread_id="thread_003",
        query="with whom",
    )

    assert state["answer"] == "You planned to go with friends."
    assert len(llm.rewrite_prompts) == 2
    assert all(
        "Original query: with whom" in prompt[-1].content
        for prompt in llm.rewrite_prompts
    )
    assert all(
        "Original query: first rewritten query" not in prompt[-1].content
        for prompt in llm.rewrite_prompts
    )


def test_pipeline_stops_after_retry_limit_and_returns_safe_fallback() -> None:
    llm = FakeLLM(
        analysis=QueryAnalysis(
            intent="memory_recall",
            normalized_query="Which database?",
        ),
        plan=RetrievalPlan(use_ltm=True),
        answer_drafts=[
            AnswerDraft(answer="Unsupported one.", cited_item_ids=["bad1"]),
            AnswerDraft(answer="Unsupported two.", cited_item_ids=["bad2"]),
        ],
        rewrites=[RewrittenQuery(query="retry query")],
    )
    retriever = FakeRetriever(_result("ltm", "memory_1", "User prefers Chroma."))
    pipeline = RetrievalPipeline(llm=llm, ltm_retriever=retriever, max_retries=1)

    state = pipeline.invoke(
        user_id="user_001",
        thread_id="thread_001",
        query="Which database?",
    )

    assert state["retry_count"] == 1
    assert retriever.calls == 2
    assert state["insufficient_evidence"] is True
    assert state["cited_item_ids"] == []
    assert state["validation"]["retry_exhausted"] is True
    assert "couldn't verify" in state["answer"]