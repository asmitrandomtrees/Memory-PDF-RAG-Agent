from pathlib import Path
from app.observability.exporters import MetricsExporter
from app.observability.metrics import MetricsRegistry


def test_metrics_exporter_json_and_table(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    registry.record_counter("test_queries", 3)
    registry.record_latency("test_stage", 25.0)
    registry.record_tokens("llm_gen", prompt_tokens=100, completion_tokens=50)

    exporter = MetricsExporter(export_dir=tmp_path, registry=registry)
    json_path = exporter.export_json("test_summary.json")
    jsonl_path = exporter.export_jsonl("test_traces.jsonl")

    assert json_path.exists()
    assert jsonl_path.exists()

    table_text = exporter.format_summary_table()
    assert "OBSERVABILITY METRICS SUMMARY" in table_text
    assert "test_stage" in table_text
    assert "llm_gen" in table_text
