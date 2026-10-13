import re

from langchain_core.prompts import ChatPromptTemplate

from app.contracts.conversation import Conversation, ConversationMessage
from app.contracts.memory import CandidateMemory, MemorySource
from app.llm.prompts import render_chat_prompt
from app.llm.provider import LLMProvider
from app.memory.ltm.schemas import LTMExtractionOutput


_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract durable user-specific memories from the conversation. "
            "Only include information likely to remain useful across "
            "conversations. Exclude temporary requests, questions, and "
            "ordinary conversation. For each memory, provide its type, "
            "content, confidence from 0 to 1, and source user-message IDs. "
            "Only use these types: preference, profile, fact, goal, "
            "relationship, instruction, other.",
        ),
        ("human", "{conversation_text}"),
    ]
)


_MAX_RULE_VALUE_CHARS = 160
_MAX_RULE_VALUE_WORDS = 24

_MY_FIELD_RE = re.compile(
    r"^(?:just\s+)?my\s+(?P<field>[a-z][a-z0-9 _-]{1,40})\s+"
    r"(?:is|=)\s+(?P<value>.+)$",
    re.IGNORECASE,
)
_I_AM_RE = re.compile(
    r"^(?:i am|i'm)\s+(?P<value>.+)$",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"^(?:i\s+)?(?P<verb>prefer|like|love|dislike|hate)\s+"
    r"(?P<value>.+)$",
    re.IGNORECASE,
)
_NEGATIVE_PREFERENCE_RE = re.compile(
    r"^(?:i\s+)?(?:do not|don't)\s+(?:like|prefer)\s+"
    r"(?P<value>.+)$",
    re.IGNORECASE,
)
_REMEMBER_RE = re.compile(
    r"^remember(?:\s+that)?\s+(?P<value>.+)$",
    re.IGNORECASE,
)

_TRANSIENT_I_AM_PREFIXES = (
    "asking",
    "building",
    "checking",
    "creating",
    "debugging",
    "doing",
    "exploring",
    "fixing",
    "implementing",
    "learning",
    "looking",
    "making",
    "reading",
    "testing",
    "trying",
    "using",
    "working",
    "writing",
)


class LTMExtractor:
    def __init__(
        self,
        *,
        llm: LLMProvider,
    ) -> None:
        self.llm = llm

    @staticmethod
    def _is_trivial_user_message(message: ConversationMessage) -> bool:
        content = message.content.strip().lower()
        if not content:
            return True

        tokens = [token for token in content.split() if token]
        if len(tokens) <= 3 and (
            content in {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay"}
            or content.startswith(("hi ", "hello ", "hey "))
        ):
            return True

        return False

    def extract(
        self,
        conversation: Conversation,
    ) -> list[CandidateMemory]:
        if not conversation.messages:
            return []

        messages = [
            message
            for message in conversation.messages
            if message.role == "user"
            and message.content.strip()
        ]

        if not messages:
            return []

        if all(self._is_trivial_user_message(message) for message in messages):
            return []

        rule_candidates, uncovered_messages = self._extract_rule_based(
            conversation=conversation,
            messages=messages,
        )

        if not uncovered_messages:
            return rule_candidates

        conversation_text = self._build_conversation_text(
            uncovered_messages
        )

        output = self.llm.structured_output(
            render_chat_prompt(
                _EXTRACTION_PROMPT,
                conversation_text=conversation_text,
            ),
            LTMExtractionOutput,
            temperature=0.0,
        )

        llm_candidates = self._to_candidates(
            output=output,
            conversation=conversation,
        )

        uncovered_message_ids = {
            message.message_id for message in uncovered_messages
        }
        llm_candidates = [
            candidate
            for candidate in llm_candidates
            if set(candidate.source.message_ids).issubset(
                uncovered_message_ids
            )
        ]

        return rule_candidates + llm_candidates

    @classmethod
    def _extract_rule_based(
        cls,
        *,
        conversation: Conversation,
        messages: list[ConversationMessage],
    ) -> tuple[list[CandidateMemory], list[ConversationMessage]]:
        candidates: list[CandidateMemory] = []
        uncovered_messages: list[ConversationMessage] = []

        for message in messages:
            candidate = cls._extract_rule_based_message(
                conversation=conversation,
                message=message,
            )
            if candidate is None:
                uncovered_messages.append(message)
                continue
            candidates.append(candidate)

        return candidates, uncovered_messages

    @classmethod
    def _extract_rule_based_message(
        cls,
        *,
        conversation: Conversation,
        message: ConversationMessage,
    ) -> CandidateMemory | None:
        content = message.content.strip()
        normalized = " ".join(content.split())

        if (
            not normalized
            or normalized.endswith("?")
            or cls._is_trivial_user_message(message)
        ):
            return None

        candidate = cls._extract_preference(
            conversation=conversation,
            message=message,
            content=normalized,
        )
        if candidate is not None:
            return candidate

        candidate = cls._extract_profile_field(
            conversation=conversation,
            message=message,
            content=normalized,
        )
        if candidate is not None:
            return candidate

        candidate = cls._extract_i_am_profile(
            conversation=conversation,
            message=message,
            content=normalized,
        )
        if candidate is not None:
            return candidate

        return cls._extract_remember_instruction(
            conversation=conversation,
            message=message,
            content=normalized,
        )

    @classmethod
    def _extract_preference(
        cls,
        *,
        conversation: Conversation,
        message: ConversationMessage,
        content: str,
    ) -> CandidateMemory | None:
        negative_match = _NEGATIVE_PREFERENCE_RE.match(content)
        if negative_match:
            value = cls._clean_rule_value(negative_match.group("value"))
            if value is None:
                return None
            return cls._candidate(
                conversation=conversation,
                message=message,
                memory_type="preference",
                content=f"User does not like {value}.",
                confidence=0.95,
                metadata={"extraction_method": "rule_based"},
            )

        match = _PREFERENCE_RE.match(content)
        if not match:
            return None

        value = cls._clean_rule_value(match.group("value"))
        if value is None:
            return None

        verb = match.group("verb").lower()
        if verb in {"dislike", "hate"}:
            text = f"User does not like {value}."
        else:
            text = f"User {verb}s {value}."

        return cls._candidate(
            conversation=conversation,
            message=message,
            memory_type="preference",
            content=text,
            confidence=0.95,
            metadata={"extraction_method": "rule_based"},
        )

    @classmethod
    def _extract_profile_field(
        cls,
        *,
        conversation: Conversation,
        message: ConversationMessage,
        content: str,
    ) -> CandidateMemory | None:
        match = _MY_FIELD_RE.match(content)
        if not match:
            return None

        field = " ".join(match.group("field").lower().split())
        value = cls._clean_rule_value(match.group("value"))
        if value is None:
            return None

        return cls._candidate(
            conversation=conversation,
            message=message,
            memory_type="profile",
            content=f"User's {field} is {value}.",
            confidence=0.96,
            metadata={
                "extraction_method": "rule_based",
                "attribute": field,
            },
        )

    @classmethod
    def _extract_i_am_profile(
        cls,
        *,
        conversation: Conversation,
        message: ConversationMessage,
        content: str,
    ) -> CandidateMemory | None:
        match = _I_AM_RE.match(content)
        if not match:
            return None

        value = cls._clean_rule_value(match.group("value"))
        if value is None:
            return None

        first_word = value.split()[0].lower()
        if first_word in _TRANSIENT_I_AM_PREFIXES:
            return None

        return cls._candidate(
            conversation=conversation,
            message=message,
            memory_type="profile",
            content=f"User is {value}.",
            confidence=0.9,
            metadata={"extraction_method": "rule_based"},
        )

    @classmethod
    def _extract_remember_instruction(
        cls,
        *,
        conversation: Conversation,
        message: ConversationMessage,
        content: str,
    ) -> CandidateMemory | None:
        match = _REMEMBER_RE.match(content)
        if not match:
            return None

        value = cls._clean_rule_value(match.group("value"))
        if value is None:
            return None

        return cls._candidate(
            conversation=conversation,
            message=message,
            memory_type="instruction",
            content=f"User asked to remember that {value}.",
            confidence=0.9,
            metadata={"extraction_method": "rule_based"},
        )

    @staticmethod
    def _clean_rule_value(value: str) -> str | None:
        cleaned = value.strip().strip("\"'")
        cleaned = cleaned.rstrip(".!")
        cleaned = " ".join(cleaned.split())

        if (
            not cleaned
            or len(cleaned) > _MAX_RULE_VALUE_CHARS
            or len(cleaned.split()) > _MAX_RULE_VALUE_WORDS
        ):
            return None

        return cleaned

    @staticmethod
    def _candidate(
        *,
        conversation: Conversation,
        message: ConversationMessage,
        memory_type: str,
        content: str,
        confidence: float,
        metadata: dict[str, object],
    ) -> CandidateMemory:
        return CandidateMemory(
            user_id=conversation.user_id,
            memory_type=memory_type,
            content=content,
            confidence=confidence,
            source=MemorySource(
                thread_id=conversation.thread_id,
                message_ids=[message.message_id],
            ),
            metadata=metadata,
        )

    @staticmethod
    def _build_conversation_text(
        messages: list[ConversationMessage],
    ) -> str:
        lines = []

        for message in messages:
            lines.append(
                f"[message_id={message.message_id}] "
                f"{message.content}"
            )

        return "\n".join(lines)

    @staticmethod
    def _to_candidates(
        *,
        output: LTMExtractionOutput,
        conversation: Conversation,
    ) -> list[CandidateMemory]:
        valid_message_ids = {
            message.message_id
            for message in conversation.messages
        }

        candidates: list[CandidateMemory] = []

        for item in output.memories:
            message_ids = item.message_ids
            if not all(
                message_id in valid_message_ids
                for message_id in message_ids
            ):
                raise ValueError(
                    "Memory contains an unknown source message"
                )

            if not message_ids:
                raise ValueError(
                    "Memory must have at least one source message"
                )

            candidates.append(
                CandidateMemory(
                    user_id=conversation.user_id,
                    memory_type=item.memory_type,
                    content=item.content,
                    confidence=item.confidence,
                    source=MemorySource(
                        thread_id=conversation.thread_id,
                        message_ids=message_ids,
                    ),
                )
            )

        return candidates
