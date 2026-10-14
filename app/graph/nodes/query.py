from datetime import datetime
import re

from langchain_core.prompts import ChatPromptTemplate

from app.contracts.routing import RetrievalPlan
from app.graph.schemas import INDIA_TIMEZONE, QueryAnalysis
from app.llm.prompts import render_chat_prompt
from app.llm.provider import LLMProvider


_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Analyze the user's current query. Classify intent as exactly "
            "one of episodic_recall, memory_recall, document_question, "
            "mixed, or general. Normalize the query without changing its "
            "meaning. For a question about a calendar day, return month and "
            "day, and include year only when the user specified one. Do not "
            "invent a year. The current date in India is {today_ist}.",
        ),
        ("human", "{query}"),
    ]
)

_PLANNING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Choose retrieval sources for this query. Use STM for recent or "
            "episodic conversation recall, LTM for durable personal facts "
            "and preferences, and PDF for questions about supplied documents. "
            "Select every source that may provide useful evidence, and no "
            "source that is irrelevant. Use STM for short follow-up fragments "
            "that depend on the previous turn, such as yes/no answers, "
            "preferences, refinements, or phrases containing this, that, both, "
            "previous, earlier, or same. An episodic_recall intent must use "
            "STM. Return a RetrievalPlan.",
        ),
        ("human", "Original query: {query}\nAnalysis: {analysis}"),
    ]
)


class QueryAnalyzer:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def analyze(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        prefer_pdf: bool = False,
    ) -> QueryAnalysis:
        document_ids = document_ids or []
        if (document_ids or prefer_pdf) and _looks_like_standalone_general_query(
            query
        ):
            return QueryAnalysis(
                intent="document_question",
                normalized_query=query.strip(),
                document_ids=document_ids,
            )
        if not document_ids and _looks_like_standalone_general_query(query):
            return QueryAnalysis(
                intent="general",
                normalized_query=query.strip(),
            )

        today_ist = datetime.now(INDIA_TIMEZONE).date().isoformat()
        return self.llm.structured_output(
            render_chat_prompt(
                _ANALYSIS_PROMPT,
                query=query,
                today_ist=today_ist,
            ),
            QueryAnalysis,
            temperature=0.0,
        )


class RetrievalPlanner:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def plan(
        self,
        *,
        query: str,
        analysis: QueryAnalysis,
        document_ids: list[str] | None = None,
        prefer_pdf: bool = False,
    ) -> RetrievalPlan:
        document_ids = document_ids or analysis.document_ids
        deterministic_plan = _deterministic_source_plan(
            query=query,
            analysis=analysis,
            document_ids=document_ids,
            prefer_pdf=prefer_pdf,
        )
        if deterministic_plan is not None:
            return deterministic_plan

        plan = self.llm.structured_output(
            render_chat_prompt(
                _PLANNING_PROMPT,
                query=query,
                analysis=analysis.model_dump_json(),
            ),
            RetrievalPlan,
            temperature=0.0,
        )

        if analysis.intent == "episodic_recall" and not plan.use_stm:
            return plan.model_copy(update={"use_stm": True})
        if (document_ids or prefer_pdf) and not plan.use_pdf:
            return plan.model_copy(update={"use_pdf": True})
        if _looks_like_contextual_follow_up(query) and not plan.use_stm:
            return plan.model_copy(update={"use_stm": True})
        return plan


_FOLLOW_UP_TERMS = {
    "also",
    "both",
    "either",
    "earlier",
    "whom",
    "it",
    "same",
    "that",
    "this",
    "those",
    "previous",
    "preferably",
    "yes",
    "no",
}

def _looks_like_contextual_follow_up(query: str) -> bool:
    normalized = query.strip().lower()
    if not normalized:
        return False

    tokens = re.findall(r"[a-z0-9']+", normalized)
    if not tokens:
        return False

    token_set = set(tokens)
    if token_set & _FOLLOW_UP_TERMS:
        return True

    if len(tokens) <= 3 and normalized.endswith("?"):
        return True

    if "," in normalized and len(tokens) <= 5 and not normalized.endswith("?"):
        return True

    return False


def _deterministic_source_plan(
    *,
    query: str,
    analysis: QueryAnalysis,
    document_ids: list[str],
    prefer_pdf: bool,
) -> RetrievalPlan | None:
    contextual_follow_up = _looks_like_contextual_follow_up(query)
    pdf_context = bool(document_ids) or prefer_pdf
    if pdf_context:
        return RetrievalPlan(
            use_stm=contextual_follow_up,
            use_pdf=True,
            metadata={
                "router": "deterministic",
                "reason": (
                    "selected_documents"
                    if document_ids
                    else "thread_pdf_context"
                ),
            },
        )

    if _looks_like_explicit_document_query(query) or analysis.intent == "document_question":
        return RetrievalPlan(
            use_pdf=True,
            metadata={
                "router": "deterministic",
                "reason": "document_query",
            },
        )

    if _looks_like_profile_statement(query):
        return RetrievalPlan(
            metadata={
                "router": "deterministic",
                "reason": "profile_statement",
            },
        )

    if contextual_follow_up:
        return RetrievalPlan(
            use_stm=True,
            metadata={
                "router": "deterministic",
                "reason": "contextual_follow_up",
            },
        )

    if analysis.intent == "episodic_recall":
        return RetrievalPlan(
            use_stm=True,
            metadata={
                "router": "deterministic",
                "reason": "episodic_recall",
            },
        )

    if analysis.intent == "memory_recall":
        return RetrievalPlan(
            use_stm=True,
            use_ltm=True,
            metadata={
                "router": "deterministic",
                "reason": "memory_recall",
            },
        )

    if analysis.intent == "general":
        return RetrievalPlan(
            metadata={
                "router": "deterministic",
                "reason": "standalone_general",
            },
        )

    return None


_RETRIEVAL_CUE_TERMS = {
    "document",
    "documents",
    "file",
    "files",
    "i",
    "me",
    "memory",
    "my",
    "our",
    "pdf",
    "previous",
    "remember",
    "said",
    "source",
    "told",
    "uploaded",
    "we",
}

_DOCUMENT_QUERY_TERMS = {
    "document",
    "documents",
    "file",
    "files",
    "pdf",
    "source",
    "uploaded",
}

_PROFILE_STATEMENT_PATTERNS = (
    re.compile(r"\bmy\s+\w+\s+is\b", re.IGNORECASE),
    re.compile(r"\bi\s+am\b", re.IGNORECASE),
    re.compile(r"\bi'm\b", re.IGNORECASE),
)

_GENERAL_QUERY_STARTS = {
    "can",
    "could",
    "define",
    "explain",
    "how",
    "list",
    "should",
    "tell",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


def _looks_like_standalone_general_query(query: str) -> bool:
    normalized = query.strip().lower()
    if not normalized:
        return False

    tokens = re.findall(r"[a-z0-9']+", normalized)
    if not tokens:
        return False

    if _looks_like_contextual_follow_up(query):
        return False

    if set(tokens) & _RETRIEVAL_CUE_TERMS:
        return False

    return tokens[0] in _GENERAL_QUERY_STARTS


def _looks_like_explicit_document_query(query: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9']+", query.strip().lower()))
    return bool(tokens & _DOCUMENT_QUERY_TERMS)


def _looks_like_profile_statement(query: str) -> bool:
    stripped = query.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return False
    return any(pattern.search(stripped) for pattern in _PROFILE_STATEMENT_PATTERNS)
