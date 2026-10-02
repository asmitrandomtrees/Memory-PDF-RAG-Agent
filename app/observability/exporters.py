import json
from pathlib import Path
from typing import Any

from app.observability.metrics import MetricsRegistry, get_metrics_registry


class MetricsExporter:
    """Exports metrics to JSON/JSONL files or structured console outputs."""

    def __init__(
        self,
        export_dir: str | Path = "./data/metrics",
        registry: MetricsRegistry | None = None,
    ) -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry or get_metrics_registry()

    def export_json(self, filename: str = "metrics_summary.json") -> Path:
        output_file = self.export_dir / filename
        data = self.registry.to_dict()
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_file

    def export_jsonl(self, filename: str = "metrics.jsonl") -> Path:
        output_file = self.export_dir / filename
        data = self.registry.to_dict()
        with output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return output_file

    def format_summary_table(self) -> str:
        """Format metrics into a human-readable text table."""
        data = self.registry.to_dict()
        lines: list[str] = [
            "=" * 70,
            f"OBSERVABILITY METRICS SUMMARY ({data['timestamp']})",
            "=" * 70,
            "",
            "--- STAGE LATENCIES (ms) ---",
            f"{'Stage':<28} | {'Count':<6} | {'Avg':<8} | {'P50':<8} | {'P95':<8} | {'Max':<8}",
            "-" * 70,
        ]
        for name, lat in sorted(data.get("latencies", {}).items()):
            lines.append(
                f"{name:<28} | {lat['count']:<6} | {lat['avg']:<8.1f} | {lat['p50']:<8.1f} | {lat['p95']:<8.1f} | {lat['max']:<8.1f}"
            )

        lines.extend([
            "",
            "--- TOKEN USAGE ---",
            f"{'Category':<28} | {'Prompt':<10} | {'Completion':<10} | {'Total':<10}",
            "-" * 70,
        ])
        for cat, tok in sorted(data.get("token_usage", {}).items()):
            lines.append(
                f"{cat:<28} | {tok['prompt_tokens']:<10} | {tok['completion_tokens']:<10} | {tok['total_tokens']:<10}"
            )

        lines.extend([
            "",
            "--- COUNTERS & ERRORS ---",
        ])
        for k, v in sorted(data.get("counters", {}).items()):
            lines.append(f"  • {k}: {v}")
        for k, v in sorted(data.get("errors", {}).items()):
            lines.append(f"  [!] Error in {k}: {v}")

        lines.append("=" * 70)
        return "\n".join(lines)
