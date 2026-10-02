from contextlib import contextmanager
from functools import wraps
import time
from typing import Any, Callable, Iterator

from app.observability.metrics import MetricsRegistry, get_metrics_registry


class ObservabilityTracker:
    """Context tracker for recording latency, token usage, retrieval, and error metrics."""

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        self.registry = registry or get_metrics_registry()

    @contextmanager
    def track_latency(self, stage_name: str) -> Iterator[dict[str, Any]]:
        """Context manager to measure and record execution time for a stage."""
        context: dict[str, Any] = {"start_time": time.perf_counter(), "error": None}
        self.registry.record_counter(f"{stage_name}_invocations")
        try:
            yield context
        except Exception as exc:
            context["error"] = str(exc)
            self.registry.record_error(stage_name)
            raise
        finally:
            duration_ms = (time.perf_counter() - context["start_time"]) * 1000.0
            context["duration_ms"] = duration_ms
            self.registry.record_latency(stage_name, duration_ms)

    def track_llm_call(
        self,
        category: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        """Record LLM call metrics."""
        self.registry.record_counter(f"llm_{category}_calls")
        self.registry.record_tokens(
            category,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if latency_ms > 0:
            self.registry.record_latency(f"llm_{category}", latency_ms)

    def track_retrieval(
        self,
        source: str,
        retrieved_count: int,
        latency_ms: float = 0.0,
        error: bool = False,
    ) -> None:
        """Record retriever execution metrics."""
        self.registry.record_counter(f"retrieval_{source}_calls")
        self.registry.record_value(f"retrieval_{source}_items_count", float(retrieved_count))
        if latency_ms > 0:
            self.registry.record_latency(f"retrieval_{source}", latency_ms)
        if error:
            self.registry.record_error(f"retrieval_{source}")

    def track_reranker(
        self,
        input_count: int,
        output_count: int,
        latency_ms: float = 0.0,
    ) -> None:
        """Record reranker execution metrics."""
        self.registry.record_counter("reranker_calls")
        self.registry.record_value("reranker_input_count", float(input_count))
        self.registry.record_value("reranker_output_count", float(output_count))
        if latency_ms > 0:
            self.registry.record_latency("reranker", latency_ms)

    def track_retry(self, reason: str | None = None) -> None:
        """Record validation failure and corrective retry."""
        self.registry.record_counter("validation_retries")
        if reason:
            self.registry.record_counter(f"validation_retry_reason_{reason}")

    def track_exhaustion(self) -> None:
        """Record retry budget exhaustion."""
        self.registry.record_counter("validation_exhausted")


def tracked_stage(stage_name: str, tracker: ObservabilityTracker | None = None) -> Callable:
    """Decorator to measure and record execution time of a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            active_tracker = tracker or ObservabilityTracker()
            with active_tracker.track_latency(stage_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
