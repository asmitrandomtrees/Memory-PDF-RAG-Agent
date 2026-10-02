from app.evaluation.datasets import BENCHMARK_DATASET, EvalTestCase
from app.evaluation.runner import EvaluationRunner
from eval import MockBenchmarkRetriever, MockEvaluationLLM
from app.graph.graph import RetrievalPipeline


def test_evaluation_runner_executes_suite_and_computes_summary() -> None:
    llm = MockEvaluationLLM()
    pipeline = RetrievalPipeline(
        llm=llm,
        stm_retriever=MockBenchmarkRetriever("stm"),
        episodic_retriever=MockBenchmarkRetriever("stm"),
        ltm_retriever=MockBenchmarkRetriever("ltm"),
        pdf_retriever=MockBenchmarkRetriever("pdf"),
    )

    runner = EvaluationRunner(pipeline=pipeline)

    # Run on a subset or full benchmark suite
    results, summary = runner.run_suite(BENCHMARK_DATASET)

    assert len(results) == len(BENCHMARK_DATASET)
    assert summary.total_tests == len(BENCHMARK_DATASET)
    assert summary.passed_tests == len(BENCHMARK_DATASET)
    assert summary.avg_recall_at_k > 0.4
    assert summary.avg_routing_score > 0.3
    assert summary.avg_groundedness > 0.5

    table = runner.format_report_table(summary, results)
    assert "EVALUATION SUITE BENCHMARK REPORT" in table
    assert "stm_01_recent_thread" in table
    assert "ltm_01_user_preference" in table
    assert "pdf_01_api_documentation" in table
    assert "stm_ltm_pdf_01_full_hybrid" in table
