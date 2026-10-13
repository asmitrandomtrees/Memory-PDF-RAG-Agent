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

        conversation_text = self._build_conversation_text(
            messages
        )

        output = self.llm.structured_output(
            render_chat_prompt(
                _EXTRACTION_PROMPT,
                conversation_text=conversation_text,
            ),
            LTMExtractionOutput,
            temperature=0.0,
        )

        return self._to_candidates(
            output=output,
            conversation=conversation,
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