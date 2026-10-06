"""The real LangGraph workflow:

planner -> stm -> ltm -> pdf -> merge -> rerank -> context -> llm -> validate
                                  ^                                   |
                                  +------- rewrite query (retry) <----+
"""
from __future__ import annotations

from typing import Any, Protocol

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.contracts.retrieval import MergedRetrievalResult, RetrievalResult
from app.contracts.routing import RetrievalPlan
from app.contracts.runtime import AgentRequest, AgentResponse
from app.graph.state import GraphState
from app.llm.provider import ChatMessage

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the context below. "
    "Context blocks are labelled [STM] recent chat, [LTM] remembered facts about the "
    "user, [PDF] document excerpts (with file and page). Cite PDF answers like "
    "(file, p.N). If the answer is not in the context, say so clearly."
)


class LTMRetriever(Protocol):
    """Plug your long-term memory in here (not built yet)."""

    def retrieve(self, user_id: str, thread_id: str, query: str) -> RetrievalResult: ...


class Verdict(BaseModel):
    grounded: bool          # answer supported by the context?
    answers_question: bool  # does it actually answer what was asked?
    rewritten_query: str    # better search query if a retry is needed


class RagGraph:
    def __init__(
        self,
        *,
        stm,
        pdf_retriever,
        llm,
        ltm: LTMRetriever | None = None,
        max_retries: int = 2,
        pdf_top_k: int = 5,
        max_context_chars: int = 24000,
    ) -> None:
        self.stm, self.pdf, self.ltm, self.llm = stm, pdf_retriever, ltm, llm
        self.max_retries = max_retries
        self.pdf_top_k = pdf_top_k
        self.max_context_chars = max_context_chars
        self.app = self._build()

    # ---------------- nodes ----------------
    def planner_node(self, state: GraphState) -> dict[str, Any]:
        # Simple rules first. STM is cheap, so always on. PDF is on by default
        # because the knowledge base is the main source. LTM only if wired in.
        plan = RetrievalPlan(
            use_stm=True,
            use_pdf=True,
            use_ltm=self.ltm is not None,
            reasoning="default plan: STM + PDF" + (" + LTM" if self.ltm else ""),
        )
        return {"retrieval_plan": plan, "rewritten_query": state["query"]}

    def stm_node(self, state: GraphState) -> dict[str, Any]:
        if not state["retrieval_plan"].use_stm:
            return {}
        return {"stm_result": self.stm.retrieve(state["user_id"], state["thread_id"], state["query"])}

    def ltm_node(self, state: GraphState) -> dict[str, Any]:
        if not (state["retrieval_plan"].use_ltm and self.ltm):
            return {}
        return {"ltm_result": self.ltm.retrieve(state["user_id"], state["thread_id"], state["rewritten_query"])}

    def pdf_node(self, state: GraphState) -> dict[str, Any]:
        if not state["retrieval_plan"].use_pdf:
            return {}
        return {"pdf_result": self.pdf.retrieve(state["rewritten_query"], self.pdf_top_k)}

    def merge_node(self, state: GraphState) -> dict[str, Any]:
        items, seen = [], set()
        for key in ("ltm_result", "pdf_result", "stm_result"):
            result = state.get(key)
            for item in (result.items if result else []):
                if item.item_id not in seen:
                    seen.add(item.item_id)
                    items.append(item)
        return {"merged_results": MergedRetrievalResult(query=state["query"], items=items)}

    def rerank_node(self, state: GraphState) -> dict[str, Any]:
        merged = state["merged_results"]
        # PDF/LTM by relevance (highest first); STM keeps chronological order, last.
        scored = [i for i in merged.items if i.source != "stm"]
        scored.sort(key=lambda i: i.score if i.score is not None else -1, reverse=True)
        stm = [i for i in merged.items if i.source == "stm"]
        return {"reranked_results": MergedRetrievalResult(query=merged.query, items=scored + stm)}

    def context_node(self, state: GraphState) -> dict[str, Any]:
        blocks, used = [], 0
        for item in state["reranked_results"].items:
            if item.source == "pdf":
                label = f"[PDF {item.metadata.get('filename')}, p.{item.metadata.get('page_number')}]"
            else:
                label = f"[{item.source.upper()}]"
            block = f"{label}\n{item.content}"
            if used + len(block) > self.max_context_chars:
                break
            blocks.append(block)
            used += len(block)
        return {"context": "\n\n".join(blocks) or "(no context found)"}

    def llm_node(self, state: GraphState) -> dict[str, Any]:
        reply = self.llm.invoke(
            [
                ChatMessage(role="system", content=f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{state['context']}"),
                ChatMessage(role="user", content=state["query"]),
            ]
        )
        return {"answer": reply.content}

    def validate_node(self, state: GraphState) -> dict[str, Any]:
        retries = state.get("retry_count", 0)
        try:
            verdict = self.llm.structured_output(
                [
                    ChatMessage(
                        role="system",
                        content="Judge the ANSWER against the CONTEXT and QUESTION. "
                        "grounded = every claim is supported by the context. "
                        "answers_question = it actually answers the question. "
                        "If either is false, give a better rewritten search query.",
                    ),
                    ChatMessage(
                        role="user",
                        content=f"QUESTION: {state['query']}\n\nCONTEXT:\n{state['context']}\n\nANSWER:\n{state['answer']}",
                    ),
                ],
                Verdict,
            )
            ok = verdict.grounded and verdict.answers_question
            new_query = verdict.rewritten_query or state["query"]
        except Exception as exc:  # a failing validator must never break the answer
            return {"validation": {"valid": True, "error": str(exc)}}
        return {
            "validation": {"valid": ok, "grounded": verdict.grounded, "answers": verdict.answers_question},
            "rewritten_query": new_query if not ok else state["rewritten_query"],
            "retry_count": retries if ok else retries + 1,
        }

    # ---------------- routing ----------------
    def route_after_validate(self, state: GraphState) -> str:
        if state["validation"].get("valid") or state.get("retry_count", 0) > self.max_retries:
            return "done"
        return "retry"

    # ---------------- wiring ----------------
    def _build(self):
        g = StateGraph(GraphState)
        for name, fn in [
            ("planner", self.planner_node), ("stm", self.stm_node), ("ltm", self.ltm_node),
            ("pdf", self.pdf_node), ("merge", self.merge_node), ("rerank", self.rerank_node),
            ("context", self.context_node), ("llm", self.llm_node), ("validate", self.validate_node),
        ]:
            g.add_node(name, fn)
        g.set_entry_point("planner")
        g.add_edge("planner", "stm")
        g.add_edge("stm", "ltm")
        g.add_edge("ltm", "pdf")
        g.add_edge("pdf", "merge")
        g.add_edge("merge", "rerank")
        g.add_edge("rerank", "context")
        g.add_edge("context", "llm")
        g.add_edge("llm", "validate")
        g.add_conditional_edges("validate", self.route_after_validate, {"done": END, "retry": "pdf"})
        return g.compile()

    # GraphRunner contract used by AgentRuntime
    def run(self, request: AgentRequest, trace_id: str) -> AgentResponse:
        final = self.app.invoke(
            {
                "user_id": request.user_id,
                "thread_id": request.thread_id,
                "query": request.message,
                "retry_count": 0,
                "trace_id": trace_id,
            }
        )
        sources = [
            {"source": i.source, "file": i.metadata.get("filename"), "page": i.metadata.get("page_number")}
            for i in final["reranked_results"].items
            if i.source == "pdf"
        ]
        return AgentResponse(
            user_id=request.user_id,
            thread_id=request.thread_id,
            answer=final["answer"],
            trace_id=trace_id,
            metadata={"validation": final.get("validation"), "retries": final.get("retry_count", 0), "pdf_sources": sources},
        )
