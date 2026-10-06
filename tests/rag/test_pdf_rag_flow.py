import hashlib
import re

import pytest

from app.contracts.errors import RetrievalError
from app.contracts.retrieval import PDFQuery
from app.graph.nodes.retrieval import PDFNode
from app.rag.pdf_ingest import ingest_pdf, pdf_to_chunks
from app.rag.pdf_retriever import PDFRetriever
from app.vectorstores.chroma import ChromaVectorStore


class FakeEmbedder:
    dim = 256

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def write_pdf(path, pages: list[str]) -> None:
    font_id = 3 + 2 * len(pages)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{' '.join(f'{3 + 2 * i} 0 R' for i in range(len(pages)))}] "
            f"/Count {len(pages)} >>"
        ).encode(),
    ]
    for index, text in enumerate(pages):
        page_id = 3 + 2 * index
        content_id = page_id + 1
        escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 750 Td ({escaped_text}) Tj ET".encode()
        objects.extend(
            [
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                    f"/Contents {content_id} 0 R >>"
                ).encode(),
                f"<< /Length {len(stream)} >>\nstream\n".encode()
                + stream
                + b"\nendstream",
            ]
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

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


def test_pdf_chunks_preserve_pages_and_stable_ids(tmp_path) -> None:
    pdf = tmp_path / "Spec.pdf"
    write_pdf(
        pdf,
        [
            "Page one: cooling tower general description.",
            "Fan motor horsepower is 75 HP, voltage 460V.",
        ],
    )

    chunks = pdf_to_chunks(pdf)

    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert pdf_to_chunks(pdf)[0].chunk_id == chunks[0].chunk_id
    assert "75 HP" in chunks[1].content


def test_reingestion_is_idempotent_and_removes_obsolete_chunks(tmp_path) -> None:
    pdf = tmp_path / "Spec.pdf"
    write_pdf(pdf, ["Page one.", "Page two."])
    embeddings = FakeEmbedder()
    store = ChromaVectorStore(str(tmp_path / "vectors"))

    assert ingest_pdf(pdf, embeddings, store, "pdf_collection") == 2
    assert ingest_pdf(pdf, embeddings, store, "pdf_collection") == 2
    assert store._collection("pdf_collection").count() == 2

    write_pdf(pdf, ["Updated one-page document."])
    assert ingest_pdf(pdf, embeddings, store, "pdf_collection") == 1
    assert store._collection("pdf_collection").count() == 1


def test_retriever_filters_requested_document_ids(tmp_path) -> None:
    embeddings = FakeEmbedder()
    store = ChromaVectorStore(str(tmp_path / "vectors"))
    store.add(
        collection="pdf_collection",
        ids=["doc_a_chunk", "doc_b_chunk"],
        documents=["document A", "document B"],
        embeddings=embeddings.embed_documents(["document A", "document B"]),
        metadatas=[
            {"source": "pdf", "document_id": "doc_a"},
            {"source": "pdf", "document_id": "doc_b"},
        ],
    )
    retriever = PDFRetriever(
        embeddings=embeddings,
        vector_store=store,
        collection="pdf_collection",
    )

    result = PDFNode(retriever=retriever, top_k=5)(
        {
            "user_id": "u",
            "thread_id": "t",
            "query": "document",
            "rewritten_query": "document",
            "document_ids": ["doc_b"],
        }
    )

    assert [item.item_id for item in result["pdf_result"].items] == [
        "doc_b_chunk"
    ]
    assert result["pdf_error"] is None


def test_retriever_rejects_empty_query(tmp_path) -> None:
    retriever = PDFRetriever(
        embeddings=FakeEmbedder(),
        vector_store=ChromaVectorStore(str(tmp_path / "vectors")),
        collection="pdf_collection",
    )

    with pytest.raises(RetrievalError, match="empty query"):
        retriever.retrieve(
            PDFQuery(user_id="u", thread_id="t", query=" ")
        )
