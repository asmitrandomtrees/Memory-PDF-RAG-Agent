# Production LangGraph Agent

Conversational AI system with short-term memory (STM), long-term memory (LTM), PDF retrieval-augmented generation, LangGraph orchestration, runtime/session management, persistence, tracing, and evaluation.

The runtime includes STM, LTM, and PDF retrieval in its LangGraph pipeline. PDF ingestion extracts page text, chunks it, creates embeddings, and stores chunks in the configured PDF collection. Image-only/scanned PDFs need OCR before ingestion.

## Ingest PDFs

Place PDFs in `data/pdfs` and run:

```bash
python scripts/ingest_pdfs.py
```

To ingest a different folder:

```bash
python scripts/ingest_pdfs.py path/to/pdfs
```

## Runtime Conversation

```bash
python run.py --user user_001 --thread thread_001
```

Or:

```bash
python run.py --user user_001 --thread thread_001 --query "Hello"
```
