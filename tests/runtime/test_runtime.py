from datetime import datetime, timezone

from app.contracts.runtime import (
    AgentRequest,
    AgentResponse,
)
from app.contracts.retrieval import RetrievedItem
from app.contracts.routing import RetrievalPlan
from app.config.settings import Settings
from app.graph.graph import Phase0Graph
from app.graph.schemas import AnswerDraft, AnswerValidation, QueryAnalysis
from app.memory.ltm.schemas import LTMExtractionOutput
from app.memory.stm.summarizer import STMContextSummarizer
from app.rag.pdf_retriever import PDFRetriever
from app.runtime.agent import AgentRuntime
from app.runtime.conversation_store import (
    JsonlConversationStore,
)
from app.runtime.factory import create_runtime


def test_runtime_executes_phase0_graph(
    tmp_path,
) -> None:
    store = JsonlConversationStore(tmp_path)

    runtime = AgentRuntime(
        conversation_store=store,
        graph=Phase0Graph(),
    )

    request = AgentRequest(
        user_id="user_001",
        thread_id="thread_001",
        message="What can you do?",
    )

    response = runtime.handle(request)

    assert response.user_id == "user_001"
    assert response.thread_id == "thread_001"
    assert response.trace_id
    assert "Phase 0 skeleton" in response.answer

    conversation = store.load(
        user_id="user_001",
        thread_id="thread_001",
    )

    assert len(conversation.messages) == 2

    assert conversation.messages[0].role == "user"
    assert conversation.messages[0].content == "What can you do?"

    assert conversation.messages[1].role == "assistant"


def test_runtime_skips_stm_embedding_for_greeting_messages(tmp_path) -> None:
    class CountingSTMWriter:
        def __init__(self) -> None:
            self.messages = []

        def write(self, message) -> None:
            self.messages.append(message)

    class GreetingGraph:
        def run(self, request, trace_id) -> AgentResponse:
            return AgentResponse(
                user_id=request.user_id,
                thread_id=request.thread_id,
                answer="Hi! How can I help you today?",
                trace_id=trace_id,
            )

    writer = CountingSTMWriter()
    runtime = AgentRuntime(
        conversation_store=JsonlConversationStore(tmp_path),
        graph=GreetingGraph(),
        stm_writer=writer,
    )

    runtime.handle(
        AgentRequest(
            user_id="user_001",
            thread_id="thread_001",
            message="Hi Stan!",
        )
    )

    assert writer.messages == []


def test_runtime_uses_fast_path_for_greeting_messages(tmp_path) -> None:
    class UnexpectedGraph:
        def run(self, request, trace_id):
            raise AssertionError("Greeting should not execute the graph")

    runtime = AgentRuntime(
        conversation_store=JsonlConversationStore(tmp_path),
        graph=UnexpectedGraph(),
    )

    response = runtime.handle(
        AgentRequest(
            user_id="user_001",
            thread_id="thread_001",
            message="Hello",
        )
    )

    conversation = runtime.conversation_store.load("user_001", "thread_001")
    timings = response.metadata["agent_timings"]

    assert response.answer == "Hi! How can I help you today?"
    assert response.metadata["fast_path"] == "small_talk"
    assert "agent.graph_execution" not in timings
    assert [message.role for message in conversation.messages] == [
        "user",
        "assistant",
    ]


def test_runtime_captures_short_profile_facts_for_ltm(tmp_path) -> None:
    class NoopGraph:
        def run(self, request, trace_id) -> AgentResponse:
            return AgentResponse(
                user_id=request.user_id,
                thread_id=request.thread_id,
                answer="Noted.",
                trace_id=trace_id,
            )

    class RecordingExtractor:
        conversations = []

        def extract(self, conversation):
            self.conversations.append(conversation)
            return []

    extractor = RecordingExtractor()
    runtime = AgentRuntime(
        conversation_store=JsonlConversationStore(tmp_path),
        graph=NoopGraph(),
        ltm_extractor=extractor,
        ltm_manager=object(),
        background_memory_writes=False,
    )

    runtime.handle(
        AgentRequest(
            user_id="user_001",
            thread_id="thread_001",
            message="My age is 23",
        )
    )

    assert extractor.conversations
    assert extractor.conversations[0].messages[0].content == "My age is 23"


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.items = {
            "seed_message": RetrievedItem(
                item_id="seed_message",
                source="stm",
                content="We discussed Chroma as the vector database.",
                score=0.95,
                rank=1,
                metadata={
                    "source": "stm",
                    "user_id": "user_001",
                    "thread_id": "thread_001",
                    "role": "user",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        }

    def add(
        self,
        *,
        collection,
        ids,
        documents,
        embeddings,
        metadatas,
    ) -> None:
        for item_id, document, metadata in zip(ids, documents, metadatas):
            self.items[item_id] = RetrievedItem(
                item_id=item_id,
                source=metadata["source"],
                content=document,
                score=1.0,
                rank=1,
                metadata=metadata,
            )

    def search(
        self,
        *,
        collection,
        query_embedding,
        top_k,
        filters=None,
    ) -> list[RetrievedItem]:
        items = [
            item
            for item in self.items.values()
            if item.metadata.get("source") == collection.removesuffix("_collection")
        ]
        if filters:
            items = [
                item
                for item in items
                if all(item.metadata.get(key) == value for key, value in filters.items())
            ]
        return items[:top_k]

    def delete(self, *, collection, ids) -> None:
        for item_id in ids:
            self.items.pop(item_id, None)

    def get_by_metadata(self, *, collection, filters):
        return []


class FakeRuntimeLLM:
    def __init__(self) -> None:
        self.schemas = []

    def structured_output(self, messages, output_schema, *, temperature=None):
        self.schemas.append(output_schema)
        if output_schema is QueryAnalysis:
            return QueryAnalysis(
                intent="episodic_recall",
                normalized_query="What did we discuss about the database?",
            )
        if output_schema is RetrievalPlan:
            return RetrievalPlan(use_stm=True)
        if output_schema is LTMExtractionOutput:
            return LTMExtractionOutput(memories=[])
        if output_schema is AnswerDraft:
            return AnswerDraft(
                answer="We discussed Chroma.",
                cited_item_ids=["seed_message"],
            )
        if output_schema is AnswerValidation:
            return AnswerValidation(is_valid=True)
        raise AssertionError(f"Unexpected model schema: {output_schema}")


def test_runtime_executes_stm_ltm_graph_and_persists_messages(tmp_path) -> None:
    settings = Settings(
        CONVERSATION_DATA_PATH=str(tmp_path / "conversations"),
        VECTOR_STORE_PATH=str(tmp_path / "vectors"),
        LTM_MEMORY_PATH=str(tmp_path / "memories.json"),
        TRACE_DATA_PATH=str(tmp_path / "traces"),
    )
    conversation_store = JsonlConversationStore(
        tmp_path / "conversations"
    )
    vector_store = FakeVectorStore()
    llm = FakeRuntimeLLM()

    runtime = create_runtime(
        settings=settings,
        llm=llm,
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
        conversation_store=conversation_store,
    )
    assert isinstance(runtime.graph.pdf_node.retriever, PDFRetriever)
    assert isinstance(runtime.graph.stm_node.summarizer, STMContextSummarizer)

    response = runtime.handle(
        AgentRequest(
            user_id="user_001",
            thread_id="thread_001",
            message="What did we discuss about the database?",
        )
    )
    runtime.wait_for_background_tasks()

    conversation = conversation_store.load("user_001", "thread_001")
    assert response.answer == "We discussed Chroma."
    assert response.trace_id
    assert response.metadata["citations"][0]["item_id"] == "seed_message"
    timings = response.metadata["agent_timings"]
    assert timings["agent.total"]["total_ms"] > 0
    assert timings["agent.graph_execution"]["total_ms"] > 0
    assert timings["query_analysis"]["total_ms"] > 0
    assert timings["retrieval_planning"]["total_ms"] > 0
    assert timings["answer_generation"]["total_ms"] > 0
    assert timings["answer_validation"]["total_ms"] > 0
    assert len(conversation.messages) == 2
    assert [message.role for message in conversation.messages] == [
        "user",
        "assistant",
    ]
    assert conversation.messages[1].metadata["trace_id"] == response.trace_id
    assert (
        conversation.messages[1].metadata["citations"]
        == response.metadata["citations"]
    )
    assert all(
        message.message_id in vector_store.items
        for message in conversation.messages
    )
    assert LTMExtractionOutput in llm.schemas

    trace_path = tmp_path / "traces" / "traces.jsonl"
    assert trace_path.exists()
    trace_record = __import__("json").loads(
        trace_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert trace_record["trace_id"] == response.trace_id
    assert "seed_message" in trace_record["stm"]["retrieved_ids"]
    assert trace_record["answer"]["cited_item_ids"] == ["seed_message"]
    trace_timings = trace_record["metadata"]["stage_timings_ms"]
    assert trace_timings["query_analysis"]["total_ms"] > 0
    assert trace_timings["agent.conversation_load"]["total_ms"] > 0
    assert "llm.structured_output" not in trace_timings
    agent_timing_record = __import__("json").loads(
        (tmp_path / "traces" / "agent_timings.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert agent_timing_record["trace_id"] == response.trace_id
    assert agent_timing_record["timings_ms"]["agent.total"]["total_ms"] > 0
    assert (
        agent_timing_record["timings_ms"]["agent.graph_execution"]["total_ms"]
        > 0
    )
    text_trace = (tmp_path / "traces" / "traces.txt").read_text(
        encoding="utf-8"
    )
    assert "Agent stage timings (overlapping spans; do not sum)" in text_trace
