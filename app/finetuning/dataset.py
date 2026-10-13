"""Build chat-format fine-tuning datasets from stored conversations.

Each assistant reply becomes one training example containing a system
prompt, a window of the preceding user/assistant turns, and the reply as
the final message. Output follows the OpenAI / Azure OpenAI chat
fine-tuning JSONL format: one ``{"messages": [...]}`` object per line.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from app.contracts.conversation import Conversation, ConversationMessage
from app.runtime.conversation_store import JsonlConversationStore


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful conversational assistant. Reply directly and "
    "naturally, as in a normal chat. Keep responses tight and ask at most "
    "one follow-up question, only when the answer would otherwise be "
    "blocked."
)

TRAINABLE_ROLES = {"user", "assistant"}


@dataclass
class FineTuneExportConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Prior user/assistant messages kept before each target reply.
    max_history_messages: int = 10
    min_assistant_chars: int = 1
    # Replies the agent flagged as unanswerable or that failed validation
    # are usually poor targets, so they are skipped unless asked for.
    include_insufficient_evidence: bool = False
    include_failed_validation: bool = False
    include_retrieval_errors: bool = False
    validation_ratio: float = 0.1
    seed: int = 42


@dataclass
class ExportResult:
    training_path: Path
    validation_path: Path | None
    training_count: int
    validation_count: int
    skipped_count: int


def iter_conversations(root_path: str | Path) -> Iterator[Conversation]:
    root = Path(root_path)
    if not root.is_dir():
        return
    store = JsonlConversationStore(root)
    for user_directory in sorted(root.iterdir()):
        if not user_directory.is_dir():
            continue
        for thread_path in sorted(user_directory.glob("*.jsonl")):
            yield store.load(user_directory.name, thread_path.stem)


def is_trainable_reply(
    message: ConversationMessage,
    config: FineTuneExportConfig,
) -> bool:
    if message.role != "assistant":
        return False
    if len(message.content.strip()) < config.min_assistant_chars:
        return False

    metadata = message.metadata
    if (
        not config.include_insufficient_evidence
        and metadata.get("insufficient_evidence") is True
    ):
        return False
    if not config.include_retrieval_errors and metadata.get("retrieval_errors"):
        return False
    validation = metadata.get("validation")
    if (
        not config.include_failed_validation
        and isinstance(validation, dict)
        and validation.get("is_valid") is False
    ):
        return False
    return True


def build_examples(
    conversation: Conversation,
    config: FineTuneExportConfig,
) -> tuple[list[dict[str, object]], int]:
    """Return (examples, skipped_reply_count) for one conversation."""
    turns = [
        message
        for message in conversation.messages
        if message.role in TRAINABLE_ROLES and message.content.strip()
    ]

    examples: list[dict[str, object]] = []
    skipped = 0
    for index, message in enumerate(turns):
        if message.role != "assistant":
            continue
        if index == 0 or turns[index - 1].role != "user":
            skipped += 1
            continue
        if not is_trainable_reply(message, config):
            skipped += 1
            continue

        history = turns[max(0, index - config.max_history_messages):index]
        # Chat fine-tuning expects the history to open with a user turn.
        while history and history[0].role != "user":
            history = history[1:]

        messages = [{"role": "system", "content": config.system_prompt}]
        messages.extend(
            {"role": turn.role, "content": turn.content} for turn in history
        )
        messages.append({"role": "assistant", "content": message.content})
        examples.append({"messages": messages})

    return examples, skipped


def _example_key(example: dict[str, object]) -> str:
    encoded = json.dumps(example, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def collect_examples(
    conversations_path: str | Path,
    config: FineTuneExportConfig,
) -> tuple[list[dict[str, object]], int]:
    examples: list[dict[str, object]] = []
    seen: set[str] = set()
    skipped = 0
    for conversation in iter_conversations(conversations_path):
        conversation_examples, conversation_skipped = build_examples(
            conversation,
            config,
        )
        skipped += conversation_skipped
        for example in conversation_examples:
            key = _example_key(example)
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            examples.append(example)
    return examples, skipped


def write_jsonl(path: Path, examples: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False))
            file.write("\n")


def export_dataset(
    conversations_path: str | Path,
    output_dir: str | Path,
    config: FineTuneExportConfig | None = None,
) -> ExportResult:
    config = config or FineTuneExportConfig()
    if not 0 <= config.validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")

    examples, skipped = collect_examples(conversations_path, config)
    random.Random(config.seed).shuffle(examples)

    validation_count = int(len(examples) * config.validation_ratio)
    validation_examples = examples[:validation_count]
    training_examples = examples[validation_count:]

    output = Path(output_dir)
    training_path = output / "train.jsonl"
    write_jsonl(training_path, training_examples)

    validation_path: Path | None = None
    if validation_examples:
        validation_path = output / "validation.jsonl"
        write_jsonl(validation_path, validation_examples)

    return ExportResult(
        training_path=training_path,
        validation_path=validation_path,
        training_count=len(training_examples),
        validation_count=len(validation_examples),
        skipped_count=skipped,
    )
