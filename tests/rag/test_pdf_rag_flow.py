"""End-to-end test with a fake embedder and fake LLM (no network, no model download)."""
import hashlib
import re

from reportlab.pdfgen import canvas

from app.contracts.conversation import ConversationMessage
from app.contracts.runtime import AgentRequest
from app.graph.rag_graph import RagGraph, Verdict
from app.llm.provider import LLMResponse
from app.memory.stm.recent import RecentMessagesSTM
from app.rag.pdf_ingest import ingest_pdf, pdf_to_chunks
from app.rag.pdf_retriever import PDFRetriever
from app.runtime.agent import AgentRuntime
from app.runtime.conversation_store import JsonlConversationStore
from app.vectorstores.chroma import ChromaVectorStore


class FakeEmbedder:
    DIM = 256

    def _vec(self, text):
        v = [0.0] * self.DIM
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    def __init__(self, verdicts=None):
        self.calls, self.verdicts = [], list(verdicts or [])

    def invoke(self, messages, *, temperature=None):
        self.calls.append(messages)
        return LLMResponse(content="ANSWER based on: " + messages[0].content[-300:])

    def structured_output(self, messages, schema, *, temperature=None):
        if self.verdicts:
            return self.verdicts.pop(0)
        return Verdict(grounded=True, answers_question=True, rewritten_query="")


def make_pdf(path):
    c = canvas.Canvas(str(path))
    c.drawString(72, 750, "Page one: cooling tower general description.")
    c.showPage()
    c.drawString(72, 750, "Fan motor horsepower is 75 HP, voltage 460V.")
    c.showPage()
    c.save()


def build(tmp_path, llm):
    pdf = tmp_path / "Spec.pdf"
    make_pdf(pdf)
    emb, vs = FakeEmbedder(), ChromaVectorStore(str(tmp_path / "vs"))
    ingest_pdf(pdf, emb, vs, "pdf_collection")
    store = JsonlConversationStore(tmp_path / "conv")
    graph = RagGraph(
        stm=RecentMessagesSTM(store),
        pdf_retriever=PDFRetriever(emb, vs, "pdf_collection", 3),
        llm=llm,
        max_retries=2,
    )
    return AgentRuntime(conversation_store=store, graph=graph), emb, vs, pdf


def test_chunks_have_page_numbers(tmp_path):
    pdf = tmp_path / "Spec.pdf"
    make_pdf(pdf)
    chunks = pdf_to_chunks(pdf)
    assert {c.page_number for c in chunks} == {1, 2}
    assert pdf_to_chunks(pdf)[0].chunk_id == chunks[0].chunk_id  # deterministic


def test_reingest_does_not_duplicate(tmp_path):
    rt, emb, vs, pdf = build(tmp_path, FakeLLM())
    before = vs._collection("pdf_collection").count()
    ingest_pdf(pdf, emb, vs, "pdf_collection")
    assert vs._collection("pdf_collection").count() == before


def test_pdf_question_gets_right_page(tmp_path):
    rt, *_ = build(tmp_path, FakeLLM())
    res = rt.handle(AgentRequest(user_id="u", thread_id="t", message="What is the fan motor horsepower?"))
    assert res.metadata["pdf_sources"][0]["page"] == 2
    assert res.answer.startswith("ANSWER")


def test_followup_sees_stm(tmp_path):
    llm = FakeLLM()
    rt, *_ = build(tmp_path, llm)
    rt.handle(AgentRequest(user_id="u", thread_id="t", message="What is the fan motor horsepower?"))
    rt.handle(AgentRequest(user_id="u", thread_id="t", message="and what about its voltage?"))
    second_system_prompt = llm.calls[-1][0].content
    assert "[STM]" in second_system_prompt and "horsepower" in second_system_prompt


def test_retry_is_capped(tmp_path):
    bad = Verdict(grounded=False, answers_question=False, rewritten_query="cooling tower fan hp")
    llm = FakeLLM(verdicts=[bad] * 10)
    rt, *_ = build(tmp_path, llm)
    res = rt.handle(AgentRequest(user_id="u", thread_id="t", message="vague"))
    assert res.metadata["retries"] <= 3  # max_retries + 1, never infinite
    assert res.answer


def test_validator_failure_does_not_break_answer(tmp_path):
    class Broken(FakeLLM):
        def structured_output(self, *a, **k):
            raise RuntimeError("boom")

    rt, *_ = build(tmp_path, Broken())
    res = rt.handle(AgentRequest(user_id="u", thread_id="t", message="hp?"))
    assert res.answer
