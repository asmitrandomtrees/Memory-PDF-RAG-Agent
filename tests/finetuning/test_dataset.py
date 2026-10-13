import json
from types import SimpleNamespace

from app.contracts.conversation import Conversation, ConversationMessage
from app.finetuning import azure_job
from app.finetuning.dataset import (
    FineTuneExportConfig,
    build_examples,
    export_dataset,
)
from app.runtime.conversation_store import JsonlConversationStore


def _message(role, content, thread_id="thread_001", **metadata):
    return ConversationMessage(
        user_id="user_001",
        thread_id=thread_id,
        role=role,
        content=content,
        metadata=metadata,
    )


def test_build_examples_uses_history_window() -> None:
    conversation = Conversation(
        user_id="user_001",
        thread_id="thread_001",
        messages=[
            _message("user", "Hi"),
            _message("assistant", "Hello!"),
            _message("user", "What is 2+2?"),
            _message("assistant", "4"),
        ],
    )
    config = FineTuneExportConfig(system_prompt="SYS", max_history_messages=2)

    examples, skipped = build_examples(conversation, config)

    assert skipped == 0
    assert len(examples) == 2
    assert examples[0]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    # Window of 2 drops the opening assistant turn so history starts with user.
    assert examples[1]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]


def test_build_examples_skips_low_quality_replies() -> None:
    conversation = Conversation(
        user_id="user_001",
        thread_id="thread_001",
        messages=[
            _message("user", "Q1"),
            _message("assistant", "I don't know", insufficient_evidence=True),
            _message("user", "Q2"),
            _message("assistant", "Bad", validation={"is_valid": False}),
            _message("user", "Q3"),
            _message("assistant", "Err", retrieval_errors={"pdf": "boom"}),
            _message("user", "Q4"),
            _message("assistant", "Good", validation={"is_valid": True}),
        ],
    )

    examples, skipped = build_examples(conversation, FineTuneExportConfig())

    assert skipped == 3
    assert len(examples) == 1
    assert examples[0]["messages"][-1] == {
        "role": "assistant",
        "content": "Good",
    }


def test_export_dataset_writes_split_and_dedupes(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    store = JsonlConversationStore(conversations)
    for index in range(10):
        thread_id = f"thread_{index:03d}"
        store.append_message(_message("user", f"Question {index}", thread_id))
        store.append_message(_message("assistant", f"Answer {index}", thread_id))
    # Exact duplicate of thread_000 should be dropped.
    store.append_message(_message("user", "Question 0", "thread_dup"))
    store.append_message(_message("assistant", "Answer 0", "thread_dup"))

    result = export_dataset(
        conversations,
        tmp_path / "out",
        FineTuneExportConfig(validation_ratio=0.2),
    )

    assert result.training_count == 8
    assert result.validation_count == 2
    assert result.skipped_count == 1

    lines = result.training_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    for line in lines:
        roles = [m["role"] for m in json.loads(line)["messages"]]
        assert roles == ["system", "user", "assistant"]


def test_create_job_passes_optional_arguments() -> None:
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id="ftjob-1", status="pending")

    client = SimpleNamespace(
        fine_tuning=SimpleNamespace(jobs=SimpleNamespace(create=create))
    )

    job = azure_job.create_job(
        client,
        training_file_id="file-train",
        validation_file_id="file-val",
        base_model="gpt-4o-mini-2024-07-18",
        suffix="agent",
        n_epochs=3,
    )

    assert job.id == "ftjob-1"
    assert captured == {
        "model": "gpt-4o-mini-2024-07-18",
        "training_file": "file-train",
        "validation_file": "file-val",
        "suffix": "agent",
        "hyperparameters": {"n_epochs": 3},
    }
