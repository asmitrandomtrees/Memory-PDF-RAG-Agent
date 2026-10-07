import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel


class JsonlTraceExporter:
    """Append Pydantic trace records to a JSONL file."""

    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.path = self.root_path / "traces.jsonl"

    def emit(self, trace: BaseModel) -> None:
        payload: dict[str, Any] = trace.model_dump(mode="json")
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False))
            file.write("\n")

    def export(self, trace: BaseModel) -> None:
        self.emit(trace)


class TextTraceExporter:
    """Append a compact, human-readable summary for each graph trace."""

    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.path = self.root_path / "traces.txt"

    def emit(self, trace: BaseModel) -> None:
        payload: dict[str, Any] = trace.model_dump(mode="json")
        with self.path.open("a", encoding="utf-8") as file:
            file.write(self._render(payload))
            file.write("\n")

    def export(self, trace: BaseModel) -> None:
        self.emit(trace)

    @classmethod
    def _render(cls, payload: dict[str, Any]) -> str:
        lines = [
            "=" * 72,
            f"Trace: {payload.get('trace_id', 'unknown')}",
            f"User: {payload.get('user_id', 'unknown')}",
            f"Thread: {payload.get('thread_id', 'unknown')}",
            f"Started: {payload.get('started_at', 'unknown')}",
            f"Completed: {payload.get('completed_at', 'unknown')}",
            f"Query: {payload.get('query', '')}",
            "",
            "Retrieval plan:",
            cls._format_mapping(payload.get("planner", {})),
            "",
            "Retrieval results:",
            cls._format_retrieval("STM", payload.get("stm")),
            cls._format_retrieval("LTM", payload.get("ltm")),
            cls._format_retrieval("PDF", payload.get("pdf_rag")),
            "",
            "Processing:",
            f"  Reranker: {cls._inline_mapping(payload.get('reranker', {}))}",
            f"  Context: {cls._inline_mapping(payload.get('context', {}))}",
            "",
            "Answer:",
            f"  {cls._inline_mapping(payload.get('answer', {}))}",
            "Validation:",
            f"  {cls._inline_mapping(payload.get('validation', {}))}",
            "Metadata:",
            f"  {cls._inline_mapping(payload.get('metadata', {}))}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_mapping(value: object) -> str:
        if not isinstance(value, dict) or not value:
            return "  none"
        return "\n".join(
            f"  {key}: {TextTraceExporter._format_value(item)}"
            for key, item in value.items()
        )

    @staticmethod
    def _format_retrieval(name: str, value: object) -> str:
        if not isinstance(value, dict):
            return f"  {name}: not used"
        item_ids = value.get("retrieved_ids", [])
        lines = [
            f"  {name}: {len(item_ids)} item(s); "
            f"ids={TextTraceExporter._format_value(item_ids)}; "
            f"latency_ms={value.get('latency_ms')}"
        ]
        metadata = value.get("metadata", {})
        if name == "STM" and isinstance(metadata, dict):
            processing = metadata.get("context_processing")
            if isinstance(processing, dict):
                lines.append(
                    "  STM context: "
                    f"{TextTraceExporter._inline_mapping(processing)}"
                )
        return "\n".join(lines)

    @staticmethod
    def _inline_mapping(value: object) -> str:
        if not isinstance(value, dict) or not value:
            return "none"
        return ", ".join(
            f"{key}={TextTraceExporter._format_value(item)}"
            for key, item in value.items()
        )

    @staticmethod
    def _format_value(value: object) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return str(value)


class CompositeTraceExporter:
    """Emit each trace to every configured exporter."""

    def __init__(self, exporters: Iterable[Any]) -> None:
        self.exporters = list(exporters)

    def emit(self, trace: BaseModel) -> None:
        for exporter in self.exporters:
            exporter.emit(trace)
