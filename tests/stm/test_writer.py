from datetime import datetime, timezone

from chromadb.api.types import validate_metadata

from app.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from app.memory.stm.models import STMMessage
from app.memory.stm.writer import STMWriter
from app.vectorstores.chroma import ChromaVectorStore


def test_stm_writer(tmp_path) -> None:
    embeddings = SentenceTransformerEmbeddings(
        "BAAI/bge-m3"
    )

    vector_store = ChromaVectorStore(
        str(tmp_path / "chroma")
    )

    writer = STMWriter(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    message = STMMessage(
        message_id="message_001",
        user_id="user_001",
        thread_id="thread_001",
        role="user",
        content="I prefer Python.",
        timestamp=datetime.now(timezone.utc),
        metadata={
            "document_ids": [],
            "citations": [{"source": "ltm", "metadata": {"version": 1}}],
        },
    )

    writer.write(message)

    query_embedding = embeddings.embed_query(
        "What programming language do I prefer?"
    )

    results = vector_store.search(
        collection="stm_collection",
        query_embedding=query_embedding,
        top_k=1,
        filters={
            "user_id": "user_001",
            "thread_id": "thread_001",
        },
    )

    assert len(results) == 1
    assert results[0].item_id == "message_001"
    assert results[0].content == "I prefer Python."
    assert results[0].metadata["user_id"] == "user_001"
    assert results[0].metadata["thread_id"] == "thread_001"
    assert results[0].metadata["document_ids"] == "[]"
    assert results[0].metadata["citations"] == (
        '[{"metadata":{"version":1},"source":"ltm"}]'
    )

def test_stm_writer_metadata_cannot_override_identity_fields() -> None:
    class FakeEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2]]

    class FakeVectorStore:
        def add(self, **kwargs) -> None:
            self.metadatas = kwargs["metadatas"]

    vector_store = FakeVectorStore()
    writer = STMWriter(
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
    )
    message = STMMessage(
        message_id="message_001",
        user_id="user_001",
        thread_id="thread_001",
        role="user",
        content="I prefer Python.",
        timestamp=datetime.now(timezone.utc),
        metadata={
            "user_id": "user_002",
            "thread_id": "thread_002",
            "custom_field": "preserved",
            "document_ids": [],
            "citations": [{"source": "ltm", "metadata": {"version": 1}}],
            "labels": ["important", "recent"],
        },
    )

    writer.write(message)

    metadata = vector_store.metadatas[0]
    assert metadata["user_id"] == "user_001"
    assert metadata["thread_id"] == "thread_001"
    assert metadata["custom_field"] == "preserved"
    assert metadata["document_ids"] == "[]"
    assert metadata["citations"] == (
        '[{"metadata":{"version":1},"source":"ltm"}]'
    )
    assert metadata["labels"] == ["important", "recent"]
    validate_metadata(metadata)