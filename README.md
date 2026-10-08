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

## Web Application

The local web app provides a FastAPI backend and a React chat interface for
conversations and PDF RAG. Configure the existing `.env` settings first,
including the model credentials required by your configured LLM provider.

Start the API from the repository root:

```bash
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the frontend development server:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The Vite development server proxies API requests
to the backend on port 8000.

To serve the built frontend from FastAPI instead, build it and start the API:

```bash
cd frontend
npm install
npm run build
cd ..
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`. The API has no authentication and is intended
for local use only; keep it bound to `127.0.0.1` and do not expose it publicly.
PDFs uploaded through the web app are stored under `PDF_UPLOAD_PATH` and are
limited by `MAX_PDF_UPLOAD_BYTES`.
