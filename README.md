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

## Fine-tuning

Turn stored conversations into an Azure OpenAI chat fine-tuning dataset and
train a custom model:

```bash
# 1. Build data/finetuning/train.jsonl and validation.jsonl
python scripts/finetune.py export

# 2. Upload the files and start a job (base model: FINETUNE_BASE_MODEL)
python scripts/finetune.py submit --suffix agent --epochs 3

# 3. Track progress
python scripts/finetune.py status <job_id>
```

Replies flagged `insufficient_evidence`, failed validation, or retrieval
errors are skipped, and duplicate examples are removed. Azure needs at least
10 training examples. When the job succeeds, deploy the fine-tuned model in
Azure AI Foundry and set `OPENAI_DEPLOYMENT` to the new deployment name.

## Answer evidence

Every answer carries `metadata.evidence`: the exact sentence(s) from the
cited sources, with filename and page number. Quotes the model returns are
checked against the source text; paraphrases are replaced by the closest
real sentence and invented quotes are dropped. `run.py` prints the
evidence under each answer and the web UI shows it below the reply.

## Specification comparator

Compare requirements across cooling tower specifications, vendor proposals
or project documents. One document is the baseline (project requirements);
every value in the others is marked compliant, deviation, or not specified,
with a verified quote and page number.

```bash
python scripts/compare_specs.py --list
python scripts/compare_specs.py --docs Spec.pdf VendorA.pdf VendorB.pdf
python scripts/compare_specs.py --docs Spec.pdf VendorA.pdf --aspects fan_motor pan_heater
python scripts/compare_specs.py --docs Spec.pdf VendorA.pdf --out matrix.csv   # compliance matrix
```

Built-in aspects: `fan_motor`, `control_panel`, `safety_equipment`,
`pan_heater`; any other text (e.g. `"basin materials"`) is compared as a
custom topic. `--docs` accepts indexed filenames, document IDs or PDF paths
(indexed on the fly). Output is Markdown by default, or CSV/JSON by `--out`
extension or `--format`.

API: `POST /api/compare` with `{"document_ids": [...], "baseline_document_id": "...", "aspects": [...]}`
returns the report; `POST /api/compare/export?format=csv|md` turns a report
into a downloadable matrix. In the web UI use **Compare documents** under
Knowledge base. `data/uploads/pdfs/Sample_Vendor*_Proposal.pdf` are
fictional sample proposals for trying it out; delete them when done.

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