from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import re
from typing import Annotated, Any, Callable
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config.instance_settings import settings_for_instance
from app.config.settings import Settings, get_settings
from app.contracts.runtime import AgentRequest
from app.embeddings.factory import create_embedding_provider
from app.runtime.conversation_store import JsonlConversationStore
from app.runtime.factory import create_runtime
from app.runtime.thread_state import JsonThreadStateStore, ThreadSourceState
from app.vectorstores.factory import create_vector_store

LOCAL_USER_ID = "local-user-4"
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
PDF_SIGNATURE = b"%PDF-"
PDF_ONLY_PATTERNS = (
    re.compile(r"\b(uploaded\s+)?pdf\b", re.IGNORECASE),
    re.compile(r"\b(uploaded\s+)?document\b", re.IGNORECASE),
)
PDF_ONLY_QUALIFIERS = (
    "only",
    "based on",
    "regarding",
    "from the pdf",
    "using the pdf",
)
CLEAR_SOURCE_POLICY_PATTERNS = (
    re.compile(r"\bstop\s+(using|checking)\s+(the\s+)?pdf\b", re.IGNORECASE),
    re.compile(r"\banswer\s+generally\b", re.IGNORECASE),
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12_000)
    document_ids: list[str] = Field(default_factory=list, max_length=100)


class NewThreadResponse(BaseModel):
    thread_id: str


class ThreadSummary(BaseModel):
    thread_id: str
    title: str
    updated_at: str | None = None


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    chunks: int
    size_bytes: int


class CompareRequest(BaseModel):
    document_ids: list[str] = Field(min_length=2)
    baseline_document_id: str | None = None
    aspects: list[str] | None = None


def _build_services() -> dict[str, Any]:
    settings = settings_for_instance(get_settings(), "api")
    embeddings = create_embedding_provider(settings)
    vector_store = create_vector_store(settings)
    conversation_store = JsonlConversationStore(settings.conversation_data_path)
    thread_state_store = JsonThreadStateStore(
        Path(settings.conversation_data_path).resolve().parent
        / "thread_state"
    )
    runtime = create_runtime(
        settings=settings,
        embeddings=embeddings,
        vector_store=vector_store,
        conversation_store=conversation_store,
    )
    upload_directory = Path(settings.pdf_upload_path).resolve()
    upload_directory.mkdir(parents=True, exist_ok=True)
    return {
        "settings": settings,
        "embeddings": embeddings,
        "vector_store": vector_store,
        "conversation_store": conversation_store,
        "thread_state_store": thread_state_store,
        "runtime": runtime,
        "upload_directory": upload_directory,
    }


def _message_requests_pdf_only(message: str) -> bool:
    normalized = message.strip().lower()
    if not normalized:
        return False
    return any(pattern.search(message) for pattern in PDF_ONLY_PATTERNS) and any(
        qualifier in normalized for qualifier in PDF_ONLY_QUALIFIERS
    )


def _message_clears_source_policy(message: str) -> bool:
    return any(pattern.search(message) for pattern in CLEAR_SOURCE_POLICY_PATTERNS)


def _available_document_ids(services: dict[str, Any]) -> list[str]:
    directory: Path = services["upload_directory"]
    store = services["vector_store"]
    settings: Settings = services["settings"]
    document_ids: list[str] = []
    for path in sorted(directory.glob("*.pdf")):
        metadata_items = store.get_by_metadata(
            collection=settings.pdf_collection,
            filters={"source_path": str(path.resolve())},
        )
        if not metadata_items:
            continue
        document_id = metadata_items[0].metadata.get("document_id")
        if isinstance(document_id, str) and document_id not in document_ids:
            document_ids.append(document_id)
    return document_ids


def _resolve_thread_source_state(
    *,
    services: dict[str, Any],
    user_id: str,
    thread_id: str,
    message: str,
    requested_document_ids: list[str],
) -> ThreadSourceState:
    store: JsonThreadStateStore = services["thread_state_store"]
    state = store.load(user_id, thread_id)

    if _message_clears_source_policy(message):
        state = ThreadSourceState()
        store.save(user_id, thread_id, state)
        return state

    if requested_document_ids:
        state.document_ids = requested_document_ids

    if _message_requests_pdf_only(message):
        state.source_policy = "pdf_only"
        if not state.document_ids:
            state.document_ids = _available_document_ids(services)

    if state.source_policy == "pdf_only":
        store.save(user_id, thread_id, state)

    return state


def create_app(
    *,
    services_factory: Callable[[], dict[str, Any]] = _build_services,
) -> FastAPI:

    # create and reuse services for entire lifespan
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.services = services_factory()
        yield
    
    application = FastAPI(
        title="Memory PDF RAG Agent API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/api/threads", response_model=list[ThreadSummary])
    def list_threads() -> list[ThreadSummary]:
        store: JsonlConversationStore = application.state.services[
            "conversation_store"
        ]
        return [
            ThreadSummary(
                thread_id=conversation.thread_id,
                title=next(
                    (
                        message.content.strip().splitlines()[0][:72]
                        for message in conversation.messages
                        if message.role == "user" and message.content.strip()
                    ),
                    "New conversation",
                ),
                updated_at=(
                    conversation.messages[-1].timestamp.isoformat()
                    if conversation.messages
                    else None
                ),
            )
            for conversation in store.list_threads(LOCAL_USER_ID)
        ]

    @application.post("/api/threads", response_model=NewThreadResponse)
    def create_thread() -> NewThreadResponse:
        return NewThreadResponse(thread_id=str(uuid4()))

    @application.get("/api/threads/{thread_id}/messages")
    def get_thread_messages(thread_id: str) -> dict[str, Any]:
        store: JsonlConversationStore = application.state.services[
            "conversation_store"
        ]
        conversation = store.load(LOCAL_USER_ID, thread_id)
        return {
            "thread_id": thread_id,
            "messages": [
                message.model_dump(mode="json")
                for message in conversation.messages
                if message.role in {"user", "assistant"}
            ],
        }

    @application.post("/api/threads/{thread_id}/messages")
    def send_message(thread_id: str, body: ChatRequest) -> dict[str, Any]:
        if not body.message.strip():
            raise HTTPException(status_code=422, detail="Message cannot be blank")

        services = application.state.services
        runtime = services["runtime"]
        thread_state = _resolve_thread_source_state(
            services=services,
            user_id=LOCAL_USER_ID,
            thread_id=thread_id,
            message=body.message.strip(),
            requested_document_ids=body.document_ids,
        )
        document_ids = body.document_ids or thread_state.document_ids
        response = runtime.handle(
            AgentRequest(
                user_id=LOCAL_USER_ID,
                thread_id=thread_id,
                message=body.message.strip(),
                metadata={
                    "document_ids": document_ids,
                    "source_policy": thread_state.source_policy,
                },
            )
        )
        return response.model_dump(mode="json")

    @application.get("/api/documents", response_model=list[DocumentSummary])
    def list_documents() -> list[DocumentSummary]:
        services = application.state.services
        directory: Path = services["upload_directory"]
        store = services["vector_store"]
        settings: Settings = services["settings"]
        documents: list[DocumentSummary] = []
        for path in sorted(directory.glob("*.pdf")):
            metadata_items = store.get_by_metadata(
                collection=settings.pdf_collection,
                filters={"source_path": str(path.resolve())},
            )
            if not metadata_items:
                continue
            documents.append(
                DocumentSummary(
                    document_id=str(
                        metadata_items[0].metadata["document_id"]
                    ),
                    filename=path.name,
                    chunks=len(metadata_items),
                    size_bytes=path.stat().st_size,
                )
            )
        return documents

    @application.post("/api/documents", response_model=DocumentSummary)
    async def upload_document(
        file: Annotated[UploadFile, File()],
    ) -> DocumentSummary:
        from app.rag.pdf_ingest import document_id_for, ingest_pdf

        services = application.state.services
        settings: Settings = services["settings"]
        directory: Path = services["upload_directory"]
        filename = file.filename or ""
        if Path(filename.replace("\\", "/")).suffix.lower() != ".pdf":
            raise HTTPException(status_code=415, detail="Upload a .pdf file")

        contents = await file.read(settings.max_pdf_upload_bytes + 1)
        if len(contents) > settings.max_pdf_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"PDF exceeds {settings.max_pdf_upload_bytes} bytes",
            )
        if not contents.startswith(PDF_SIGNATURE):
            raise HTTPException(status_code=415, detail="File is not a valid PDF")

        original_name = Path(filename.replace("\\", "/")).name
        safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", original_name)
        safe_name = safe_name.strip(" .")[:120]
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / safe_name
        if destination.exists():
            destination = directory / f"{destination.stem}-{uuid4().hex[:8]}.pdf"

        try:
            with destination.open("xb") as saved_file:
                saved_file.write(contents)
            chunks = await run_in_threadpool(
                ingest_pdf,          # ingest_pdf is synchronous/blocking so run_in_threadpool() doesn't block FastAPI's async event loop
                destination,
                services["embeddings"],
                services["vector_store"],
                settings.pdf_collection,
            )
            if chunks == 0:
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=422,
                    detail="No text could be extracted. Scanned PDFs require OCR.",
                )
        except HTTPException:
            raise
        except Exception:
            destination.unlink(missing_ok=True)
            raise

        return DocumentSummary(
            document_id=document_id_for(destination),
            filename=destination.name,
            chunks=chunks,
            size_bytes=len(contents),
        )

    @application.delete("/api/documents/{document_id}")
    def delete_document(document_id: str) -> dict[str, str]:
        from app.rag.pdf_ingest import document_id_for

        services = application.state.services
        directory: Path = services["upload_directory"]
        settings: Settings = services["settings"]
        store = services["vector_store"]
        for path in directory.glob("*.pdf"):
            if document_id_for(path) != document_id:
                continue
            items = store.get_by_metadata(
                collection=settings.pdf_collection,
                filters={"source_path": str(path.resolve())},
            )
            if items:
                store.delete(
                    collection=settings.pdf_collection,
                    ids=[item.item_id for item in items],
                )
            path.unlink()
            return {"deleted": document_id}
        raise HTTPException(status_code=404, detail="Document not found")

    @application.get("/api/compare/aspects")
    def list_compare_aspects() -> list[dict[str, Any]]:
        from app.comparison.aspects import DEFAULT_ASPECTS

        return [aspect.model_dump(mode="json") for aspect in DEFAULT_ASPECTS]

    @application.post("/api/compare")
    async def compare_documents(body: CompareRequest) -> dict[str, Any]:
        from app.comparison import (
            COMPARISON_TIMEOUT_SECONDS,
            SpecComparator,
            documents_from_store,
        )
        from app.llm.azure_openai import AzureOpenAIProvider

        services = application.state.services
        settings: Settings = services["settings"]
        store = services["vector_store"]
        embeddings = services["embeddings"]

        try:
            documents = documents_from_store(
                store,
                settings.pdf_collection,
                body.document_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        llm = services.get("compare_llm") or services.get("llm") or AzureOpenAIProvider(
            timeout_seconds=COMPARISON_TIMEOUT_SECONDS
        )
        comparator = SpecComparator(
            llm=llm,
            embeddings=embeddings,
            vector_store=store,
            collection=settings.pdf_collection,
        )
        try:
            report = await run_in_threadpool(
                comparator.compare,
                documents,
                baseline_document_id=body.baseline_document_id,
                aspects=body.aspects,
            )
            return report.model_dump(mode="json")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @application.post("/api/compare/export")
    def export_comparison(
        report: dict[str, Any],
        format: str = "csv",
    ) -> Response:
        from app.comparison import to_csv, to_markdown
        from app.comparison.schemas import ComparisonReport

        try:
            report_model = ComparisonReport.model_validate(report)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid report: {exc}")

        if format.lower() == "csv":
            content = to_csv(report_model)
            media_type = "text/csv; charset=utf-8"
        else:
            content = to_markdown(report_model)
            media_type = "text/markdown; charset=utf-8"

        return Response(content=content, media_type=media_type)

    @application.get("/")
    def frontend_index():
        index_path = FRONTEND_DIST / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)
        return {
            "name": "Memory PDF RAG Agent",
            "api_docs": "/docs",
            "frontend": "Run the Vite development server from frontend/.",
        }

    assets_path = FRONTEND_DIST / "assets"
    if assets_path.is_dir():
        application.mount(
            "/assets",
            StaticFiles(directory=assets_path),
            name="frontend-assets",
        )

    return application


app = create_app()
