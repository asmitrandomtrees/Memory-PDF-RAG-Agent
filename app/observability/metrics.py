from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import time
from typing import Any, Callable


@dataclass
class MetricSummary:
    name: str
    count: int = 0
    total: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    values: list[float] = field(default_factory=list)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        index = int(math.ceil((p / 100.0) * len(sorted_values))) - 1
        index = max(0, min(index, len(sorted_values) - 1))
        return sorted_values[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "total": round(self.total, 4),
            "avg": round(self.avg, 4),
            "min": round(self.min_value, 4) if self.count > 0 else 0.0,
            "max": round(self.max_value, 4) if self.count > 0 else 0.0,
            "p50": round(self.percentile(50), 4),
            "p95": round(self.percentile(95), 4),
            "p99": round(self.percentile(99), 4),
        }


class MetricsRegistry:
    """In-memory metrics collector and aggregator for observability."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, MetricSummary] = {}
        self._token_usage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        self._custom_metrics: dict[str, list[float]] = defaultdict(list)
        self._errors: dict[str, int] = defaultdict(int)

    def record_counter(self, name: str, increment: int = 1) -> None:
        self._counters[name] += increment

    def record_latency(self, name: str, duration_ms: float) -> None:
        if name not in self._latencies:
            self._latencies[name] = MetricSummary(name=name)
        summary = self._latencies[name]
        summary.count += 1
        summary.total += duration_ms
        summary.min_value = min(summary.min_value, duration_ms)
        summary.max_value = max(summary.max_value, duration_ms)
        summary.values.append(duration_ms)

    def record_tokens(
        self,
        category: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> None:
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        self._token_usage[category]["prompt_tokens"] += prompt_tokens
        self._token_usage[category]["completion_tokens"] += completion_tokens
        self._token_usage[category]["total_tokens"] += total_tokens

    def record_error(self, component: str, increment: int = 1) -> None:
        self._errors[component] += increment

    def record_value(self, name: str, value: float) -> None:
        self._custom_metrics[name].append(value)

    def get_counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    def get_latency_summary(self, name: str) -> MetricSummary | None:
        return self._latencies.get(name)

    def get_error_count(self, component: str) -> int:
        return self._errors.get(component, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counters": dict(self._counters),
            "errors": dict(self._errors),
            "latencies": {
                name: summary.to_dict()
                for name, summary in self._latencies.items()
            },
            "token_usage": dict(self._token_usage),
            "custom_metrics": {
                name: {
                    "count": len(vals),
                    "avg": round(sum(vals) / len(vals), 4) if vals else 0.0,
                    "min": round(min(vals), 4) if vals else 0.0,
                    "max": round(max(vals), 4) if vals else 0.0,
                }
                for name, vals in self._custom_metrics.items()
            },
        }

    def reset(self) -> None:
        self._counters.clear()
        self._latencies.clear()
        self._token_usage.clear()
        self._custom_metrics.clear()
        self._errors.clear()


# Global default registry
_default_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    return _default_registry
