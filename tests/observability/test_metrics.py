from app.observability.metrics import MetricsRegistry, MetricSummary
from app.observability.tracker import ObservabilityTracker, tracked_stage


def test_metric_summary_percentiles() -> None:
    summary = MetricSummary(name="test_stage")
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    for v in values:
        summary.count += 1
        summary.total += v
        summary.values.append(v)

    summary.min_value = 10.0
    summary.max_value = 100.0

    assert summary.avg == 55.0
    assert summary.percentile(50) == 50.0
    assert summary.percentile(95) == 100.0
    assert summary.percentile(99) == 100.0

    data = summary.to_dict()
    assert data["name"] == "test_stage"
    assert data["count"] == 10
    assert data["avg"] == 55.0


def test_metrics_registry_recording() -> None:
    registry = MetricsRegistry()
    registry.record_counter("test_counter", 5)
    registry.record_latency("test_latency", 12.5)
    registry.record_tokens("answer_generation", prompt_tokens=100, completion_tokens=25)
    registry.record_error("stm_retriever")
    registry.record_value("batch_size", 32.0)

    assert registry.get_counter("test_counter") == 5
    assert registry.get_error_count("stm_retriever") == 1
    summary = registry.get_latency_summary("test_latency")
    assert summary is not None
    assert summary.count == 1
    assert summary.avg == 12.5

    data = registry.to_dict()
    assert data["counters"]["test_counter"] == 5
    assert data["token_usage"]["answer_generation"]["total_tokens"] == 125
    assert data["errors"]["stm_retriever"] == 1


def test_observability_tracker_track_latency_and_decorator() -> None:
    registry = MetricsRegistry()
    tracker = ObservabilityTracker(registry)

    with tracker.track_latency("mock_stage"):
        pass

    assert registry.get_counter("mock_stage_invocations") == 1
    assert registry.get_latency_summary("mock_stage") is not None

    @tracked_stage("decorated_stage", tracker=tracker)
    def my_func(a: int, b: int) -> int:
        return a + b

    result = my_func(2, 3)
    assert result == 5
    assert registry.get_counter("decorated_stage_invocations") == 1


def test_tracker_helpers() -> None:
    registry = MetricsRegistry()
    tracker = ObservabilityTracker(registry)

    tracker.track_llm_call("planner", prompt_tokens=50, completion_tokens=10, latency_ms=45.0)
    tracker.track_retrieval("pdf", retrieved_count=4, latency_ms=15.0)
    tracker.track_reranker(input_count=10, output_count=5, latency_ms=5.0)
    tracker.track_retry(reason="unsupported_citation")
    tracker.track_exhaustion()

    data = registry.to_dict()
    assert data["counters"]["llm_planner_calls"] == 1
    assert data["counters"]["retrieval_pdf_calls"] == 1
    assert data["counters"]["reranker_calls"] == 1
    assert data["counters"]["validation_retries"] == 1
    assert data["counters"]["validation_retry_reason_unsupported_citation"] == 1
    assert data["counters"]["validation_exhausted"] == 1
