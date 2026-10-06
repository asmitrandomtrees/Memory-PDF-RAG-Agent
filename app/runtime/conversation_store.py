import json
from pathlib import Path

from app.contracts.conversation import Conversation, ConversationMessage
from app.contracts.errors import ConversationStoreError


class JsonlConversationStore:
    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _conversation_path(self, user_id: str, thread_id: str) -> Path:
        user_directory = self.root_path / user_id
        user_directory.mkdir(parents=True, exist_ok=True)

        return user_directory / f"{thread_id}.jsonl"

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