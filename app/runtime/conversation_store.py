import json
from pathlib import Path

from app.contracts.conversation import Conversation, ConversationMessage
from app.contracts.errors import ConversationStoreError


class JsonlConversationStore:
    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _conversation_path(self, user_id: str, thread_id: str) -> Path:
        invalid_characters = ("/", "\\", ":", "\0")
        if (
            not user_id
            or user_id in {".", ".."}
            or any(character in user_id for character in invalid_characters)
            or not thread_id
            or thread_id in {".", ".."}
            or any(character in thread_id for character in invalid_characters)
        ):
            raise ConversationStoreError(
                "user_id and thread_id must be safe path components"
            )

        root = self.root_path.resolve()
        user_directory = (root / user_id).resolve()
        if user_directory.parent != root:
            raise ConversationStoreError("Invalid user_id path")

        path = (user_directory / f"{thread_id}.jsonl").resolve()
        if path.parent != user_directory:
            raise ConversationStoreError("Invalid thread_id path")

        user_directory.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, user_id: str, thread_id: str) -> Conversation:
        path = self._conversation_path(user_id, thread_id)

        if not path.exists():
            return Conversation(
                user_id=user_id,
                thread_id=thread_id,
            )

        messages: list[ConversationMessage] = []

        try:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()

                    if not line:
                        continue

                    payload = json.loads(line)
                    messages.append(
                        ConversationMessage.model_validate(payload)
                    )

        except (OSError, json.JSONDecodeError) as exc:
            raise ConversationStoreError(
                f"Failed to load conversation "
                f"user_id={user_id}, thread_id={thread_id}"
            ) from exc

        return Conversation(
            user_id=user_id,
            thread_id=thread_id,
            messages=messages,
        )

    def list_threads(self, user_id: str) -> list[Conversation]:
        invalid_characters = ("/", "\\", ":", "\0")
        if (
            not user_id
            or user_id in {".", ".."}
            or any(character in user_id for character in invalid_characters)
        ):
            raise ConversationStoreError("user_id must be a safe path component")

        root = self.root_path.resolve()
        user_directory = (root / user_id).resolve()
        if user_directory.parent != root:
            raise ConversationStoreError("Invalid user_id path")
        if not user_directory.is_dir():
            return []

        conversations = [
            self.load(user_id, path.stem)
            for path in user_directory.glob("*.jsonl")
            if path.is_file() and path.resolve().parent == user_directory
        ]
        conversations.sort(
            key=lambda conversation: (
                conversation.messages[-1].timestamp
                if conversation.messages
                else conversation.updated_at
            ),
            reverse=True,
        )
        return conversations

    def append_message(self, message: ConversationMessage) -> None:
        path = self._conversation_path(
            message.user_id,
            message.thread_id,
        )

        try:
            with path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        message.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                )
                file.write("\n")

        except OSError as exc:
            raise ConversationStoreError(
                f"Failed to persist message {message.message_id}"
            ) from exc