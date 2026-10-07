from datetime import datetime, timezone
from unittest.mock import Mock

from app.contracts.conversation import Conversation, ConversationMessage
from app.contracts.retrieval import RetrievalResult, RetrievedItem
from app.graph.nodes.retrieval import STMNode
from app.memory.stm.context import STMContextExpander
from app.memory.stm.context_budget import STMContextBudget


def _build_node(max_tokens: int, summary: str):
    messages = [
        ConversationMessage(
            message_id=f"message_{index}",
            user_id="user_001",
            thread_id="thread_001",
            role="user",
            content=f"message content {index}",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc).replace(
                minute=index
            ),
        )
        for index in range(4)
    ]
    conversation = Conversation(
        user_id="user_001",
        thread_id="thread_001",
        messages=messages,
    )
    store = Mock()
    store.load.return_value = conversation
    retriever = Mock()
    retriever.retrieve.return_value = RetrievalResult(
        source="stm",
        query="recall",
        items=[
            RetrievedItem(
                item_id=message.message_id,
                source="stm",
                content=message.content,
                score=0.9,
                rank=index + 1,
                metadata={
                    "role": message.role,
                    "timestamp": message.timestamp.isoformat(),
                    "context_type": "retrieved",
                },
            )
            for index, message in enumerate(messages)
        ],
        metadata={"user_id": "user_001", "thread_id": "thread_001"},
    )
    summarizer = Mock()
    def make_summary(items, *, max_tokens):
        return RetrievedItem(
            item_id="stm_summary",
            source="stm",
            content=summary,
            metadata={
                "context_type": "summary",
                "source_item_ids": [item.item_id for item in items],
                "max_tokens": max_tokens,
            },
        )

    summarizer.summarize.side_effect = make_summary
    node = STMNode(
        stm_retriever=retriever,
        conversation_store=store,
        context_expander=STMContextExpander(
            window_size=0,
            recent_message_count=0,
        ),
        context_budget=STMContextBudget(
            max_tokens=max_tokens,
            chars_per_token=4,
        ),
        summarizer=summarizer,
    )
    return node, summarizer


def _run_node(node: STMNode):
    return node(
        {
            "user_id": "user_001",
            "thread_id": "thread_001",
            "query": "recall",
        }
    )["stm_result"]


def test_stm_node_summarizes_omitted_context_within_reserved_budget() -> None:
    node, summarizer = _build_node(max_tokens=8, summary="Earlier chat")

    result = _run_node(node)

    summary = next(item for item in result.items if item.item_id == "stm_summary")
    assert summary.metadata["source_item_ids"] == [
        "message_1",
        "message_2",
        "message_3",
    ]
    assert sum(
        node.context_budget.estimate_tokens(item.content)
        for item in result.items
    ) <= 8
    summarizer.summarize.assert_called_once()
    assert summarizer.summarize.call_args.kwargs["max_tokens"] == 2
    assert result.metadata["context_processing"]["summary_attempted"] is True
    assert result.metadata["context_processing"]["summary_added"] is True
    assert result.metadata["context_processing"]["summary_source_count"] == 3


def test_stm_node_skips_summarization_when_all_context_fits() -> None:
    node, summarizer = _build_node(max_tokens=100, summary="unused")

    result = _run_node(node)

    assert len(result.items) == 4
    assert all(item.item_id != "stm_summary" for item in result.items)
    summarizer.summarize.assert_not_called()
    assert result.metadata["context_processing"]["overflow"] is False
    assert result.metadata["context_processing"]["summary_attempted"] is False


def test_stm_node_keeps_full_budget_selection_if_summary_does_not_fit() -> None:
    node, summarizer = _build_node(
        max_tokens=8,
        summary="This summary is much too long to fit the reserved token budget.",
    )

    result = _run_node(node)

    assert len(result.items) == 2
    assert all(item.item_id != "stm_summary" for item in result.items)
    summarizer.summarize.assert_called_once()
    assert result.metadata["context_processing"]["summary_attempted"] is True
    assert result.metadata["context_processing"]["summary_added"] is False


def test_stm_node_falls_back_and_records_summary_error() -> None:
    node, summarizer = _build_node(max_tokens=8, summary="unused")
    summarizer.summarize.side_effect = RuntimeError("summarization unavailable")

    result = _run_node(node)

    assert len(result.items) == 2
    assert result.metadata["context_processing"]["summary_attempted"] is True
    assert result.metadata["context_processing"]["summary_added"] is False
    assert result.metadata["context_processing"]["summary_error"] == (
        "summarization unavailable"
    )
