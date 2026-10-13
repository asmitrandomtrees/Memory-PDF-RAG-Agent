import json
from datetime import datetime, timezone

import pytest

from app.contracts.conversation import (
    Conversation,
    ConversationMessage,
)
from app.llm.provider import ChatMessage, LLMResponse
from app.memory.ltm.extractor import LTMExtractor


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None
        self.output_schema = None

    def structured_output(
        self,
        messages: list[ChatMessage],
        output_schema,
        *,
        temperature: float | None = None,
    ):
        self.messages = messages
        self.output_schema = output_schema
        return output_schema.model_validate(json.loads(self.response))


class RaisingLLM:
    messages = None
    output_schema = None

    def structured_output(
        self,
        messages: list[ChatMessage],
        output_schema,
        *,
        temperature: float | None = None,
    ):
        raise AssertionError("LLM should not be called")


def _conversation() -> Conversation:
    return Conversation(
        user_id="user_001",
        thread_id="thread_001",
        messages=[
            ConversationMessage(
                message_id="message_001",
                user_id="user_001",
                thread_id="thread_001",
                role="user",
                content="I prefer concise responses.",
                timestamp=datetime(
                    2026,
                    9,
                    20,
                    tzinfo=timezone.utc,
                ),
            ),
            ConversationMessage(
                message_id="message_002",
                user_id="user_001",
                thread_id="thread_001",
                role="assistant",
                content="Understood.",
                timestamp=datetime(
                    2026,
                    9,
                    20,
                    0,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
            ConversationMessage(
                message_id="message_003",
                user_id="user_001",
                thread_id="thread_001",
                role="user",
                content="I'm building a PDF agent.",
                timestamp=datetime(
                    2026,
                    9,
                    20,
                    0,
                    2,
                    tzinfo=timezone.utc,
                ),
            ),
        ],
    )


def test_extractor_returns_candidate_memories() -> None:
    llm = FakeLLM(
        json.dumps(
            {"memories": [
                {
                    "memory_type": "preference",
                    "content": "User prefers concise responses.",
                    "confidence": 0.95,
                    "message_ids": ["message_001"],
                },
                {
                    "memory_type": "goal",
                    "content": "User is building a PDF agent.",
                    "confidence": 0.9,
                    "message_ids": ["message_003"],
                },
            ]}
        )
    )

    extractor = LTMExtractor(llm=llm)

    result = extractor.extract(_conversation())

    assert len(result) == 2
    assert llm.output_schema.__name__ == "LTMExtractionOutput"

    assert result[0].user_id == "user_001"
    assert result[0].memory_type == "preference"
    assert result[0].content == (
        "User prefers concise responses."
    )
    assert result[0].source.thread_id == "thread_001"
    assert result[0].source.message_ids == [
        "message_001"
    ]

    assert result[1].memory_type == "goal"


def test_extractor_sends_only_user_messages() -> None:
    llm = FakeLLM(
        json.dumps({"memories": []})
    )

    extractor = LTMExtractor(llm=llm)

    extractor.extract(_conversation())

    prompt = llm.messages[1].content

    assert "I prefer concise responses." not in prompt
    assert "I'm building a PDF agent." in prompt
    assert "Understood." not in prompt


def test_rule_based_profile_fact_skips_llm() -> None:
    extractor = LTMExtractor(llm=RaisingLLM())

    conversation = Conversation(
        user_id="user_001",
        thread_id="thread_001",
        messages=[
            ConversationMessage(
                message_id="message_001",
                user_id="user_001",
                thread_id="thread_001",
                role="user",
                content="My age is 23.",
                timestamp=datetime(
                    2026,
                    9,
                    20,
                    tzinfo=timezone.utc,
                ),
            ),
        ],
    )

    result = extractor.extract(conversation)

    assert len(result) == 1
    assert result[0].memory_type == "profile"
    assert result[0].content == "User's age is 23."
    assert result[0].confidence == 0.96
    assert result[0].source.message_ids == ["message_001"]
    assert result[0].metadata == {
        "extraction_method": "rule_based",
        "attribute": "age",
    }


def test_rule_based_preference_skips_llm() -> None:
    extractor = LTMExtractor(llm=RaisingLLM())

    conversation = Conversation(
        user_id="user_001",
        thread_id="thread_001",
        messages=[
            ConversationMessage(
                message_id="message_001",
                user_id="user_001",
                thread_id="thread_001",
                role="user",
                content="I prefer concise responses.",
                timestamp=datetime(
                    2026,
                    9,
                    20,
                    tzinfo=timezone.utc,
                ),
            ),
        ],
    )

    result = extractor.extract(conversation)

    assert len(result) == 1
    assert result[0].memory_type == "preference"
    assert result[0].content == "User prefers concise responses."
    assert result[0].metadata == {
        "extraction_method": "rule_based",
    }


def test_extractor_preserves_source_message_ids() -> None:
    llm = FakeLLM(
        json.dumps(
            {"memories": [
                {
                    "memory_type": "preference",
                    "content": "User prefers concise responses.",
                    "confidence": 0.9,
                    "message_ids": [
                        "message_001"
                    ],
                }
            ]}
        )
    )

    extractor = LTMExtractor(llm=llm)

    result = extractor.extract(_conversation())

    assert result[0].source.message_ids == [
        "message_001"
    ]


def test_extractor_returns_empty_for_empty_conversation() -> None:
    llm = FakeLLM(
        json.dumps([])
    )

    extractor = LTMExtractor(llm=llm)

    conversation = Conversation(
        user_id="user_001",
        thread_id="thread_001",
    )

    result = extractor.extract(conversation)

    assert result == []
    assert llm.messages is None


def test_extractor_rejects_invalid_structured_output() -> None:
    llm = FakeLLM(
        json.dumps(
            {"memories": [
                {
                    "memory_type": "preference",
                    "content": "User prefers concise responses.",
                    "confidence": "not-a-number",
                    "message_ids": ["message_001"],
                }
            ]}
        )
    )

    extractor = LTMExtractor(llm=llm)

    with pytest.raises(
        ValueError,
    ):
        extractor.extract(_conversation())


def test_extractor_rejects_unknown_source_message() -> None:
    llm = FakeLLM(
        json.dumps(
            {"memories": [
                {
                    "memory_type": "preference",
                    "content": "User prefers concise responses.",
                    "confidence": 0.9,
                    "message_ids": [
                        "unknown_message"
                    ],
                }
            ]}
        )
    )

    extractor = LTMExtractor(llm=llm)

    with pytest.raises(
        ValueError,
        match="unknown source message",
    ):
        extractor.extract(_conversation())


def test_extractor_rejects_memory_without_source_messages() -> None:
    llm = FakeLLM(
        json.dumps(
            {"memories": [
                {
                    "memory_type": "preference",
                    "content": "User prefers concise responses.",
                    "confidence": 0.9,
                    "message_ids": [],
                }
            ]}
        )
    )

    extractor = LTMExtractor(llm=llm)

    with pytest.raises(
        ValueError,
        match="at least one source message",
    ):
        extractor.extract(_conversation())
