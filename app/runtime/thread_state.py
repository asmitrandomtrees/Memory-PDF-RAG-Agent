import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.contracts.errors import ConversationStoreError


class ThreadSourceState(BaseModel):
    source_policy: str | None = None
    document_ids: list[str] = Field(default_factory=list)


class JsonThreadStateStore:
    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _state_path(self, user_id: str, thread_id: str) -> Path:
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

        path = (user_directory / f"{thread_id}.json").resolve()
        if path.parent != user_directory:
            raise ConversationStoreError("Invalid thread_id path")

        user_directory.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, user_id: str, thread_id: str) -> ThreadSourceState:
        path = self._state_path(user_id, thread_id)
        if not path.exists():
            return ThreadSourceState()

        try:
            return ThreadSourceState.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ConversationStoreError(
                f"Failed to load thread state user_id={user_id}, "
                f"thread_id={thread_id}"
            ) from exc

    def save(
        self,
        user_id: str,
        thread_id: str,
        state: ThreadSourceState,
    ) -> None:
        path = self._state_path(user_id, thread_id)
        try:
            path.write_text(
                json.dumps(
                    state.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ConversationStoreError(
                f"Failed to save thread state user_id={user_id}, "
                f"thread_id={thread_id}"
            ) from exc
