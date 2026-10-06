from typing import Any, TypedDict

from app.contracts.retrieval import (
    MergedRetrievalResult,
    RetrievalResult,
)
from app.contracts.routing import RetrievalPlan


class GraphState(TypedDict, total=False):
    user_id: str
    thread_id: str
    query: str
    rewritten_query: str

    retrieval_plan: RetrievalPlan

    stm_result: RetrievalResult
    ltm_result: RetrievalResult
    pdf_result: RetrievalResult

    merged_results: MergedRetrievalResult
    reranked_results: MergedRetrievalResult

    context: str
    answer: str
    validation: dict[str, Any]

    retry_count: int
    trace_id: str