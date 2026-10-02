import argparse
import json
from pathlib import Path

from app.config.settings import get_settings
from app.contracts.context import AgentContext
from app.contracts.retrieval import RetrievalResult, RetrievedItem
from app.contracts.routing import RetrievalPlan
from app.evaluation.datasets import BENCHMARK_DATASET, EvalTestCase
from app.evaluation.runner import EvaluationRunner
from app.graph.graph import RetrievalPipeline
from app.graph.nodes.context import ContextSummary
from app.graph.schemas import (
    AnswerDraft,
    AnswerValidation,
    QueryAnalysis,
    RewrittenQuery,
)


class MockEvaluationLLM:
    """Mock LLM for automated offline evaluation and benchmarking."""

    def structured_output(self, messages, output_schema, *, temperature=None):
        user_msg = messages[-1].content if messages else ""

        if output_schema is QueryAnalysis:
            intent = "general"
            date_month = None
            date_day = None
            date_year = None

            if "september 18" in user_msg.lower():
                intent = "episodic_recall"
                date_month, date_day, date_year = 9, 18, 2026
            elif "earlier" in user_msg.lower() or "discussed" in user_msg.lower() or "today" in user_msg.lower():
                intent = "episodic_recall"
            elif "preference" in user_msg.lower() or "cloud" in user_msg.lower() or "stack" in user_msg.lower() or "role" in user_msg.lower():
                intent = "memory_recall"
            elif "policy" in user_msg.lower() or "doc" in user_msg.lower() or "limit" in user_msg.lower() or "guideline" in user_msg.lower():
                intent = "document_question"

            return QueryAnalysis(
                intent=intent,
                normalized_query=user_msg,
                date_month=date_month,
                date_day=date_day,
                date_year=date_year,
            )

        if output_schema is RetrievalPlan:
            use_stm = any(k in user_msg.lower() for k in ["earlier", "september", "discussed", "chat", "today", "error", "troubleshoot", "stack"])
            use_ltm = any(k in user_msg.lower() for k in ["preference", "prefer", "role", "cloud", "stack", "name", "benefit"])
            use_pdf = any(k in user_msg.lower() for k in ["doc", "policy", "api", "limit", "guideline", "overview", "manual", "benefit", "architecture", "fix", "retention", "compliance"])
            if not use_stm and not use_ltm and not use_pdf:
                use_stm, use_ltm = True, True
            return RetrievalPlan(use_stm=use_stm, use_ltm=use_ltm, use_pdf=use_pdf)

        if output_schema is ContextSummary:
            return ContextSummary(summary="Compressed context summary.")

        if output_schema is AnswerDraft:
            return AnswerDraft(
                answer="Generated evaluation response based on evidence.",
                cited_item_ids=[
                    "stm_msg_err_101",
                    "stm_msg_sep18_01",
                    "stm_msg_today_db_01",
                    "stm_msg_goal_01",
                    "ltm_pref_001",
                    "ltm_pref_002",
                    "ltm_fact_cloud_01",
                    "ltm_fact_role_01",
                    "pdf_chunk_api_lim_04",
                    "pdf_chunk_comp_09",
                    "pdf_chunk_err_fix_02",
                    "pdf_chunk_benefits_03",
                    "pdf_chunk_arch_01",
                ],
                insufficient_evidence=False,
            )

        if output_schema is AnswerValidation:
            return AnswerValidation(is_valid=True)

        if output_schema is RewrittenQuery:
            return RewrittenQuery(query=user_msg)

        raise AssertionError(f"Unexpected schema: {output_schema}")


class MockBenchmarkRetriever:
    """Returns matching items for benchmark test cases."""

    def __init__(self, source: str) -> None:
        self.source = source

    def retrieve(self, query=None, **kwargs) -> RetrievalResult:
        query_text = getattr(query, "query", kwargs.get("query", str(query)))
        items: list[RetrievedItem] = []

        if self.source == "stm":
            if "error" in query_text.lower():
                items.append(
                    RetrievedItem(
                        item_id="stm_msg_err_101",
                        source="stm",
                        content="ConnectionTimeoutError: Failed to connect to db on port 5432.",
                        score=0.95,
                        rank=1,
                    )
                )
            elif "september" in query_text.lower():
                items.append(
                    RetrievedItem(
                        item_id="stm_msg_sep18_01",
                        source="stm",
                        content="On Sept 18 we completed the architecture review for Chroma vectorstore.",
                        score=0.92,
                        rank=1,
                    )
                )
            elif "goal" in query_text.lower() or "overview" in query_text.lower():
                items.append(
                    RetrievedItem(
                        item_id="stm_msg_goal_01",
                        source="stm",
                        content="Project goal: Build Memory PDF RAG Agent.",
                        score=0.90,
                        rank=1,
                    )
                )
            else:
                items.append(
                    RetrievedItem(
                        item_id="stm_msg_today_db_01",
                        source="stm",
                        content="Today we discussed writing async SQLAlchemy queries for PostgreSQL.",
                        score=0.88,
                        rank=1,
                    )
                )

        elif self.source == "ltm":
            if "cloud" in query_text.lower():
                items.append(
                    RetrievedItem(
                        item_id="ltm_fact_cloud_01",
                        source="ltm",
                        content="User deploys applications on Azure in eastus2 region.",
                        score=0.94,
                        rank=1,
                    )
                )
            elif "role" in query_text.lower() or "benefit" in query_text.lower():
                items.append(
                    RetrievedItem(
                        item_id="ltm_fact_role_01",
                        source="ltm",
                        content="User is a Senior Engineer eligible for remote stipend.",
                        score=0.91,
                        rank=1,
                    )
                )
            else:
                items.extend([
                    RetrievedItem(
                        item_id="ltm_pref_001",
                        source="ltm",
                        content="User strongly prefers Python and type hints.",
                        score=0.96,
                        rank=1,
                    ),
                    RetrievedItem(
                        item_id="ltm_pref_002",
                        source="ltm",
                        content="User adheres to PEP 8 formatting standards.",
                        score=0.89,
                        rank=2,
                    ),
                ])

        elif self.source == "pdf":
            if "limit" in query_text.lower() or "rate" in query_text.lower():
                items.append(
                    RetrievedItem(
                        item_id="pdf_chunk_api_lim_04",
                        source="pdf",
                        content="Batch API Limits: 300 requests per minute, 1000 items per batch.",
                        score=0.97,
                        rank=1,
                        metadata={"document_id": "doc_api_spec_v2", "page": 4},
                    )
                )
            elif "retention" in query_text.lower() or "compliance" in query_text.lower():
                items.append(
                    RetrievedItem(
                        item_id="pdf_chunk_comp_09",
                        source="pdf",
                        content="Compliance Policy: Data retention is 90 days and encrypted at rest.",
                        score=0.93,
                        rank=1,
                        metadata={"document_id": "doc_compliance_2026", "page": 9},
                    )
                )
            elif "benefit" in query_text.lower():
                items.append(
                    RetrievedItem(
                        item_id="pdf_chunk_benefits_03",
                        source="pdf",
                        content="Senior Engineers receive a $500 monthly remote stipend.",
                        score=0.92,
                        rank=1,
                        metadata={"document_id": "doc_hr_policy_2026", "page": 3},
                    )
                )
            elif "overview" in query_text.lower() or "guideline" in query_text.lower() or "arch" in query_text.lower():
                items.append(
                    RetrievedItem(
                        item_id="pdf_chunk_arch_01",
                        source="pdf",
                        content="Architecture Guidelines: Build modular Microservices with Python.",
                        score=0.90,
                        rank=1,
                        metadata={"document_id": "doc_arch_overview_2026", "page": 1},
                    )
                )
            else:
                items.append(
                    RetrievedItem(
                        item_id="pdf_chunk_err_fix_02",
                        source="pdf",
                        content="Troubleshooting: Restart daemon and check firewall configuration.",
                        score=0.89,
                        rank=1,
                        metadata={"document_id": "doc_troubleshooting_v1", "page": 2},
                    )
                )

        return RetrievalResult(
            source=self.source,
            query=query_text,
            items=items,
            latency_ms=10.0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evaluation Suite for LTM + STM + PDF RAG Agent")
    parser.add_argument("--scenario", choices=["all", "stm", "ltm", "pdf", "stm_ltm", "stm_pdf", "ltm_pdf", "stm_ltm_pdf", "consolidation"], default="all")
    parser.add_argument("--export-dir", default="./data/eval_reports")
    args = parser.parse_args()

    llm = MockEvaluationLLM()
    pipeline = RetrievalPipeline(
        llm=llm,
        stm_retriever=MockBenchmarkRetriever("stm"),
        episodic_retriever=MockBenchmarkRetriever("stm"),
        ltm_retriever=MockBenchmarkRetriever("ltm"),
        pdf_retriever=MockBenchmarkRetriever("pdf"),
    )

    runner = EvaluationRunner(pipeline=pipeline)

    test_cases = BENCHMARK_DATASET
    if args.scenario != "all":
        test_cases = [tc for tc in BENCHMARK_DATASET if tc.scenario == args.scenario]

    print(f"\nRunning evaluation on {len(test_cases)} benchmark test cases...\n")
    results, summary = runner.run_suite(test_cases)

    report_table = runner.format_report_table(summary, results)
    print(report_table)

    export_path = Path(args.export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    report_file = export_path / "evaluation_summary.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    print(f"\nSummary JSON exported to: {report_file.resolve()}\n")


if __name__ == "__main__":
    main()
