"""PDF -> pages -> chunks -> embeddings -> vector store."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pdfplumber

from app.contracts.documents import DocumentChunk
from app.embeddings.provider import EmbeddingProvider
from app.vectorstores.base import VectorStore


_SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")


def _split_long(sentence: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    return [sentence[i : i + size] for i in range(0, len(sentence), step)]


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Deterministic sentence-aware chunking with overlap.

    Chunks end on sentence boundaries so a fact is not cut in half, and
    each chunk repeats trailing sentences (up to ``overlap`` chars) from
    the previous one. Sentences longer than ``size`` are split by chars.
    """
    if size <= overlap:
        raise ValueError("size must be larger than overlap")

    sentences: list[str] = []
    for sentence in _SENTENCE_END.split(" ".join(text.split())):
        if len(sentence) > size:
            sentences.extend(_split_long(sentence, size, overlap))
        elif sentence:
            sentences.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        if current and length + 1 + len(sentence) > size:
            chunks.append(" ".join(current))
            carried: list[str] = []
            carried_length = 0
            for previous in reversed(current):
                if carried_length + len(previous) + 1 > overlap:
                    break
                carried.insert(0, previous)
                carried_length += len(previous) + 1
            # Only carry overlap that still leaves room for the new sentence.
            if carried_length + len(sentence) > size:
                carried, carried_length = [], 0
            current, length = carried, carried_length
        current.append(sentence)
        length += len(sentence) + (1 if length else 0)
    if current:
        chunks.append(" ".join(current))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def document_id_for(pdf_path: str | Path) -> str:
    """Return a stable ID for one version of a PDF at a given path."""
    path = Path(pdf_path).resolve()
    digest = hashlib.sha256()
    digest.update(str(path).casefold().encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    return digest.hexdigest()


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


def ingest_pdf(
    pdf_path: str | Path,
    embedder: EmbeddingProvider,
    store: VectorStore,
    collection: str,
) -> int:
    """Chunk + embed + store one PDF. Returns number of chunks stored."""
    pdf_path = Path(pdf_path)
    source_path = str(pdf_path.resolve())
    chunks = pdf_to_chunks(pdf_path)
    existing = store.get_by_metadata(
        collection=collection,
        filters={"source_path": source_path},
    )
    if not chunks:
        if existing:
            store.delete(
                collection=collection,
                ids=[item.item_id for item in existing],
            )
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
                "source_path": source_path,
            }
            for c in chunks
        ],
    )
    current_ids = {c.chunk_id for c in chunks}
    stale_ids = [
        item.item_id for item in existing if item.item_id not in current_ids
    ]
    if stale_ids:
        store.delete(collection=collection, ids=stale_ids)
    return len(chunks)


def ingest_folder(
    folder: str | Path,
    embedder: EmbeddingProvider,
    store: VectorStore,
    collection: str,
) -> dict[str, int]:
    return {
        p.name: ingest_pdf(p, embedder, store, collection)
        for p in sorted(Path(folder).glob("*.pdf"))
    }