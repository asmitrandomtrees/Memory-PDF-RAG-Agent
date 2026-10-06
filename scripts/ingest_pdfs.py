"""Usage: python scripts/ingest_pdfs.py [folder]   (default: data/pdfs)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings
from app.embeddings.factory import create_embedding_provider
from app.rag.pdf_ingest import ingest_folder
from app.vectorstores.factory import create_vector_store

folder = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs"
s = get_settings()
result = ingest_folder(folder, create_embedding_provider(s), create_vector_store(s), s.pdf_collection)
for name, n in result.items():
    print(f"{name}: {n} chunks")
print("Done." if result else f"No PDFs found in {folder}")