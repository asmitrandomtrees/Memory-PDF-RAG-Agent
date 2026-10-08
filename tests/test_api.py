from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.contracts.conversation import Conversation, ConversationMessage
from app.contracts.retrieval import RetrievedItem
from app.contracts.runtime import AgentResponse
from app.runtime.conversation_store import JsonlConversationStore


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.items: dict[str, RetrievedItem] = {}

    def add(self, *, ids, documents, embeddings, metadatas, collection) -> None:
        for item_id, document, metadata in zip(ids, documents, metadatas):
            self.items[item_id] = RetrievedItem(
                item_id=item_id,
                source="pdf",
                content=document,
                metadata=metadata,
            )

    def get_by_metadata(self, *, collection, filters):
        return [
            item
            for item in self.items.values()
            if all(item.metadata.get(key) == value for key, value in filters.items())
        ]

    def delete(self, *, collection, ids) -> None:
        for item_id in ids:
            self.items.pop(item_id, None)


class FakeRuntime:
    def __init__(self, conversation_store: JsonlConversationStore) -> None:
        self.conversation_store = conversation_store

    def handle(self, request):
        now = datetime.now(timezone.utc)
        self.conversation_store.append_message(
            ConversationMessage(
                user_id=request.user_id,
                thread_id=request.thread_id,
                role="user",
                content=request.message,
                timestamp=now,
            )
        )
        answer = f"Received: {request.message}"
        self.conversation_store.append_message(
            ConversationMessage(
                user_id=request.user_id,
                thread_id=request.thread_id,
                role="assistant",
                content=answer,
                timestamp=now,
                metadata={"trace_id": "trace_test"},
            )
        )
        return AgentResponse(
            user_id=request.user_id,
            thread_id=request.thread_id,
            answer=answer,
            trace_id="trace_test",
        )


def _make_pdf(path: Path) -> None:
    stream = b"BT /F1 12 Tf 72 750 Td (A small test PDF.) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode()
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010} 00000 n \n".encode())
    pdf.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(pdf)


def _test_client(tmp_path):
    import app.api as api_module

    settings = Settings(
        _env_file=None,
        PDF_UPLOAD_PATH=str(tmp_path / "uploads"),
        CONVERSATION_DATA_PATH=str(tmp_path / "conversations"),
        PDF_COLLECTION="test_pdf_collection",
    )
    conversation_store = JsonlConversationStore(
        settings.conversation_data_path
    )
    vector_store = FakeVectorStore()
    embeddings = FakeEmbeddings()
    services = {
        "settings": settings,
        "embeddings": embeddings,
        "vector_store": vector_store,
        "conversation_store": conversation_store,
        "runtime": FakeRuntime(conversation_store),
        "upload_directory": Path(settings.pdf_upload_path).resolve(),
    }
    application = api_module.create_app(services_factory=lambda: services)
    return TestClient(application), vector_store


def test_chat_threads_and_history(tmp_path) -> None:
    client, _ = _test_client(tmp_path)
    with client:
        created = client.post("/api/threads")
        assert created.status_code == 200
        thread_id = created.json()["thread_id"]

        response = client.post(
            f"/api/threads/{thread_id}/messages",
            json={"message": "Hello", "document_ids": []},
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "Received: Hello"

        messages = client.get(
            f"/api/threads/{thread_id}/messages"
        ).json()["messages"]
        assert [message["role"] for message in messages] == [
            "user",
            "assistant",
        ]

        threads = client.get("/api/threads").json()
        assert threads[0]["thread_id"] == thread_id
        assert threads[0]["title"] == "Hello"


def test_upload_list_and_delete_pdf(tmp_path) -> None:
    client, vector_store = _test_client(tmp_path)
    source_pdf = tmp_path / "source.pdf"
    _make_pdf(source_pdf)
    with client:
        with source_pdf.open("rb") as file:
            response = client.post(
                "/api/documents",
                files={"file": ("Spec.pdf", file, "application/pdf")},
            )
        assert response.status_code == 200
        document = response.json()
        assert document["chunks"] == 1
        assert document["filename"] == "Spec.pdf"
        assert len(vector_store.items) == 1

        listed = client.get("/api/documents").json()
        assert [entry["document_id"] for entry in listed] == [
            document["document_id"]
        ]

        deleted = client.delete(
            f"/api/documents/{document['document_id']}"
        )
        assert deleted.status_code == 200
        assert client.get("/api/documents").json() == []
        assert vector_store.items == {}


def test_upload_rejects_non_pdf_and_blank_chat(tmp_path) -> None:
    client, _ = _test_client(tmp_path)
    with client:
        invalid_file = client.post(
            "/api/documents",
            files={"file": ("notes.txt", b"not a pdf", "text/plain")},
        )
        assert invalid_file.status_code == 415

        blank_message = client.post(
            "/api/threads/thread_001/messages",
            json={"message": "   "},
        )
        assert blank_message.status_code == 422
