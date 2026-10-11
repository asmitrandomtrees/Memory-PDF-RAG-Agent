from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.contracts.conversation import ConversationMessage
from app.contracts.retrieval import RetrievedItem
from app.contracts.runtime import AgentResponse
from app.rag.pdf_ingest import document_id_for
from app.runtime.conversation_store import JsonlConversationStore


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], RetrievedItem] = {}

    def add(
        self,
        *,
        ids,
        documents,
        embeddings,
        metadatas,
        collection,
    ) -> None:
        if len(embeddings) != len(ids):
            raise AssertionError("Expected an embedding for each PDF chunk")
        for item_id, content, metadata in zip(ids, documents, metadatas):
            self.items[(collection, item_id)] = RetrievedItem(
                item_id=item_id,
                source="pdf",
                content=content,
                metadata=metadata,
            )

    def get_by_metadata(self, *, collection, filters):
        return [
            item
            for (item_collection, _), item in self.items.items()
            if item_collection == collection
            and all(item.metadata.get(key) == value for key, value in filters.items())
        ]

    def delete(self, *, collection, ids) -> None:
        for item_id in ids:
            self.items.pop((collection, item_id), None)


class FakeRuntime:
    def __init__(self, conversation_store: JsonlConversationStore) -> None:
        self.conversation_store = conversation_store

    def handle(self, request):
        timestamp = datetime.now(timezone.utc)
        self.conversation_store.append_message(
            ConversationMessage(
                user_id=request.user_id,
                thread_id=request.thread_id,
                role="user",
                content=request.message,
                timestamp=timestamp,
            )
        )
        answer = f"Reply: {request.message}"
        self.conversation_store.append_message(
            ConversationMessage(
                user_id=request.user_id,
                thread_id=request.thread_id,
                role="assistant",
                content=answer,
                timestamp=timestamp,
            )
        )
        return AgentResponse(
            user_id=request.user_id,
            thread_id=request.thread_id,
            answer=answer,
            trace_id=f"trace-{request.user_id}",
        )


def _test_client(tmp_path: Path):
    import app.api_multiuser as multiuser_api

    settings = Settings(
        _env_file=None,
        PDF_UPLOAD_PATH=str(tmp_path / "uploads"),
        CONVERSATION_DATA_PATH=str(tmp_path / "conversations"),
        PDF_COLLECTION="test-pdfs",
    )
    profile_store = multiuser_api.UserProfileStore(
        tmp_path / "profiles" / "users.sqlite3"
    )
    conversation_store = multiuser_api.ProfileConversationStore(
        JsonlConversationStore(settings.conversation_data_path),
        profile_store,
    )
    vector_store = FakeVectorStore()
    services = {
        "settings": settings,
        "embeddings": FakeEmbeddings(),
        "vector_store": vector_store,
        "conversation_store": conversation_store,
        "upload_root": Path(settings.pdf_upload_path),
        "user_store": profile_store,
        "runtimes": {},
    }
    runtime_settings = []

    def runtime_factory(**kwargs):
        runtime_settings.append(kwargs["settings"])
        return FakeRuntime(kwargs["conversation_store"])

    application = multiuser_api.create_app(
        services_factory=lambda: services,
        runtime_factory=runtime_factory,
    )
    return TestClient(application), vector_store, runtime_settings


def _create_profile(client: TestClient, user_id: str) -> dict:
    response = client.post(
        "/api/users",
        json={"display_name": user_id},
    )
    assert response.status_code == 201
    return response.json()


def test_create_and_list_user_profiles(tmp_path) -> None:
    client, _, _ = _test_client(tmp_path)
    with client:
        created = client.post(
            "/api/users",
            json={"display_name": "user_001"},
        )
        assert created.status_code == 201
        profile = created.json()
        assert profile["display_name"] == "user_001"
        assert profile["user_id"] == "user_001"
        assert len(profile["profile_uuid"]) == 32

        profiles = client.get("/api/users")
        assert profiles.status_code == 200
        assert profiles.json() == [profile]

        thread = client.post(
            f"/api/users/{profile['user_id']}/threads",
            json={"thread_id": "thread_001"},
        )
        assert thread.status_code == 201
        thread_data = thread.json()
        assert thread_data["thread_id"] == "thread_001"
        assert len(thread_data["generated_id"]) == 32
        assert thread_data["generated_id"] != thread_data["thread_id"]

        threads = client.get(
            f"/api/users/{profile['user_id']}/threads"
        ).json()
        assert threads[0]["thread_id"] == "thread_001"
        assert threads[0]["generated_id"] == thread_data["generated_id"]
        assert threads[0]["title"] == "New conversation"

        duplicate = client.post(
            f"/api/users/{profile['user_id']}/threads",
            json={"thread_id": "thread_001"},
        )
        assert duplicate.status_code == 409


def test_existing_profile_ids_migrate_to_public_names(tmp_path) -> None:
    import app.api_multiuser as multiuser_api

    database_path = tmp_path / "legacy" / "users.sqlite3"
    database_path.parent.mkdir(parents=True)
    legacy_uuid = "0123456789abcdef0123456789abcdef"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE user_profiles (
                user_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO user_profiles (user_id, display_name, created_at)
            VALUES (?, ?, ?)
            """,
            (legacy_uuid, "user_001", "2026-01-01T00:00:00+00:00"),
        )

    store = multiuser_api.UserProfileStore(database_path)
    profile = store.get("user_001")

    assert profile is not None
    assert profile.user_id == "user_001"
    assert profile.profile_uuid == legacy_uuid
    assert store.get_storage_user_id("user_001") == legacy_uuid
    conversation_root = tmp_path / "legacy" / "conversations"
    old_directory = conversation_root / legacy_uuid
    old_directory.mkdir(parents=True)
    (old_directory / "thread_001.jsonl").write_text(
        '{"user_id":"'
        + legacy_uuid
        + '","thread_id":"thread_001","role":"user",'
        '"content":"legacy history","timestamp":"2026-01-01T00:00:00+00:00",'
        '"metadata":{}}\n',
        encoding="utf-8",
    )
    mapped_store = multiuser_api.ProfileConversationStore(
        JsonlConversationStore(conversation_root),
        store,
    )

    conversation = mapped_store.load(legacy_uuid, "thread_001")

    assert conversation.user_id == legacy_uuid
    assert conversation.messages[0].user_id == legacy_uuid
    assert (conversation_root / "user_001" / "thread_001.jsonl").is_file()
    assert not old_directory.exists()


def test_create_user_rejects_blank_display_name(tmp_path) -> None:
    client, _, _ = _test_client(tmp_path)
    with client:
        response = client.post("/api/users", json={"display_name": "   "})
        assert response.status_code == 422


def test_duplicate_user_id_is_rejected(tmp_path) -> None:
    client, _, _ = _test_client(tmp_path)
    with client:
        _create_profile(client, "user_001")
        duplicate = client.post(
            "/api/users",
            json={"display_name": "user_001"},
        )
        assert duplicate.status_code == 409


def test_threads_and_messages_are_scoped_by_profile(tmp_path) -> None:
    client, _, runtime_settings = _test_client(tmp_path)
    with client:
        for user_id, message in (("alice", "Alice's note"), ("bob", "Bob's note")):
            _create_profile(client, user_id)
            created_thread = client.post(
                f"/api/users/{user_id}/threads",
                json={"thread_id": "thread_001"},
            )
            assert created_thread.status_code == 201
            response = client.post(
                f"/api/users/{user_id}/threads/thread_001/messages",
                json={"message": message},
            )
            assert response.status_code == 200
            assert response.json()["user_id"] == user_id
            assert (
                tmp_path
                / "conversations"
                / user_id
                / "thread_001.jsonl"
            ).is_file()

        alice_messages = client.get(
            "/api/users/alice/threads/thread_001/messages"
        ).json()["messages"]
        bob_messages = client.get(
            "/api/users/bob/threads/thread_001/messages"
        ).json()["messages"]
        assert [message["content"] for message in alice_messages] == [
            "Alice's note",
            "Reply: Alice's note",
        ]
        assert [message["content"] for message in bob_messages] == [
            "Bob's note",
            "Reply: Bob's note",
        ]

        alice_threads = client.get("/api/users/alice/threads").json()
        bob_threads = client.get("/api/users/bob/threads").json()
        assert [thread["thread_id"] for thread in alice_threads] == ["thread_001"]
        assert [thread["thread_id"] for thread in bob_threads] == ["thread_001"]
        assert [thread["title"] for thread in alice_threads] == ["Alice's note"]
        assert [thread["title"] for thread in bob_threads] == ["Bob's note"]
        assert (
            runtime_settings[0].pdf_collection
            != runtime_settings[1].pdf_collection
        )
        assert (
            runtime_settings[0].ltm_memory_path
            != runtime_settings[1].ltm_memory_path
        )


def test_documents_are_scoped_by_profile(tmp_path, monkeypatch) -> None:
    import app.api_multiuser as multiuser_api

    client, vector_store, _ = _test_client(tmp_path)

    def fake_ingest_pdf(pdf_path, embedder, store, collection):
        path = Path(pdf_path)
        store.add(
            collection=collection,
            ids=[document_id_for(path)],
            documents=["test content"],
            embeddings=embedder.embed_documents(["test content"]),
            metadatas=[
                {
                    "document_id": document_id_for(path),
                    "source_path": str(path.resolve()),
                }
            ],
        )
        return 1

    monkeypatch.setattr(multiuser_api, "ingest_pdf", fake_ingest_pdf)
    with client:
        uploaded_ids = {}
        for user_id in ("alice", "bob"):
            _create_profile(client, user_id)
            response = client.post(
                f"/api/users/{user_id}/documents",
                files={"file": ("spec.pdf", b"%PDF-test", "application/pdf")},
            )
            assert response.status_code == 200
            uploaded_ids[user_id] = response.json()["document_id"]

        for user_id in ("alice", "bob"):
            listed = client.get(f"/api/users/{user_id}/documents").json()
            assert [document["document_id"] for document in listed] == [
                uploaded_ids[user_id]
            ]

        wrong_owner_delete = client.delete(
            f"/api/users/alice/documents/{uploaded_ids['bob']}"
        )
        assert wrong_owner_delete.status_code == 404
        assert len(vector_store.items) == 2


def test_invalid_profile_id_is_rejected(tmp_path) -> None:
    client, _, _ = _test_client(tmp_path)
    with client:
        response = client.get("/api/users/../threads")
        assert response.status_code == 404
        response = client.get("/api/users/bad.name/threads")
        assert response.status_code == 422


def test_unregistered_user_and_thread_are_rejected(tmp_path) -> None:
    client, _, _ = _test_client(tmp_path)
    with client:
        missing_user = client.get("/api/users/user_001/threads")
        assert missing_user.status_code == 404

        _create_profile(client, "user_001")
        missing_thread = client.get(
            "/api/users/user_001/threads/thread_001/messages"
        )
        assert missing_thread.status_code == 404
