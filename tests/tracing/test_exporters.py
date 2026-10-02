from datetime import datetime, timezone

from app.contracts.tracing import GraphTrace, RetrievalTrace
from app.tracing.exporters import TextTraceExporter


def test_text_trace_exporter_writes_readable_graph_trace(tmp_path) -> None:
    exporter = TextTraceExporter(tmp_path)
    trace = GraphTrace(
        trace_id="trace_123",
        user_id="user_001",
        thread_id="thread_001",
        query="What did we discuss about authentication?",
        started_at=datetime(2026, 10, 2, tzinfo=timezone.utc),
        completed_at=datetime(2026, 10, 2, 0, 0, 1, tzinfo=timezone.utc),
        planner={"use_stm": True, "use_ltm": False, "use_pdf": False},
        stm=RetrievalTrace(
            source="stm",
            query="What did we discuss about authentication?",
            retrieved_ids=["message_1"],
            latency_ms=12.5,
        ),
        reranker={"input_count": 1, "output_count": 1},
        context={"item_count": 1, "estimated_tokens": 8},
        answer={"cited_item_ids": ["message_1"]},
        validation={"is_valid": True},
        metadata={"latency_ms": 20.0, "retry_count": 0},
    )

    exporter.emit(trace)

    text = (tmp_path / "traces.txt").read_text(encoding="utf-8")
    assert "Trace: trace_123" in text
    assert "Query: What did we discuss about authentication?" in text
    assert "STM: 1 item(s); ids=[\"message_1\"]" in text
    assert "LTM: not used" in text
    assert "use_stm: True" in text
    assert "retry_count=0" in text
