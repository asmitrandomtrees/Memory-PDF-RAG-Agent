from datetime import datetime
from typing import Any, TypedDict

from app.contracts.context import AgentContext
from app.contracts.retrieval import (
    MergedRetrievalResult,
    RetrievalResult,
)
from app.contracts.routing import RetrievalPlan
from app.graph.schemas import QueryAnalysis


class GraphState(TypedDict, total=False):
    user_id: str
    thread_id: str
    query: str
    rewritten_query: str
    document_ids: list[str]
    query_analysis: QueryAnalysis
    date_start_at: datetime | None
    date_end_at: datetime | None

    retrieval_plan: RetrievalPlan

    stm_result: RetrievalResult | None
    stm_error: str | None
    ltm_result: RetrievalResult | None
    ltm_error: str | None
    pdf_result: RetrievalResult | None
    pdf_error: str | None
    retrieval_errors: dict[str, str]

    merged_results: MergedRetrievalResult | None
    reranked_results: MergedRetrievalResult | None

    context: AgentContext | None
    answer: str
    cited_item_ids: list[str]
    insufficient_evidence: bool
    validation: dict[str, Any]

    retry_count: int
    trace_id: str