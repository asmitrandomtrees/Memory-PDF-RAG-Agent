from datetime import datetime

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
            "source that is irrelevant. An episodic_recall intent must use "
            "STM. Return a RetrievalPlan.",
        ),
        ("human", "Original query: {query}\nAnalysis: {analysis}"),
    ]
)


class QueryAnalyzer:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def analyze(self, query: str) -> QueryAnalysis:
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
    ) -> RetrievalPlan:
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
        return plan