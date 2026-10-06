"""PDF -> pages -> chunks -> embeddings -> vector store."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pdfplumber

from app.contracts.documents import DocumentChunk


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Deterministic character chunking with overlap."""
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    step = size - overlap
    pieces = (text[i : i + size] for i in range(0, len(text), step))
    return [p.strip() for p in pieces if p.strip()]


def document_id_for(pdf_path: Path) -> str:
    """Same file content -> same id, so re-ingesting is safe."""
    return hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:12]


def pdf_to_chunks(pdf_path: str | Path) -> list[DocumentChunk]:
    pdf_path = Path(pdf_path)
    doc_id = document_id_for(pdf_path)
    chunks: list[DocumentChunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for n, piece in enumerate(chunk_text(text)):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_p{page_no}_c{n}",
                        document_id=doc_id,
                        filename=pdf_path.name,
                        content=piece,
                        page_number=page_no,
                    )
                )
    return chunks


def ingest_pdf(pdf_path, embedder, store, collection: str) -> int:
    """Chunk + embed + store one PDF. Returns number of chunks stored."""
    chunks = pdf_to_chunks(pdf_path)
    if not chunks:
        return 0
    texts = [c.content for c in chunks]
    store.add(
        collection=collection,
        ids=[c.chunk_id for c in chunks],
        documents=texts,
        embeddings=embedder.embed_documents(texts),
        metadatas=[
            {
                "source": "pdf",
                "document_id": c.document_id,
                "filename": c.filename,
                "page_number": c.page_number or 0,
            }
            for c in chunks
        ],
    )
    return len(chunks)


def ingest_folder(folder, embedder, store, collection: str) -> dict[str, int]:
    return {
        p.name: ingest_pdf(p, embedder, store, collection)
        for p in sorted(Path(folder).glob("*.pdf"))
    }
