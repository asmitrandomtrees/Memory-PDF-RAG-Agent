from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Callable
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api import ChatRequest, DocumentSummary, PDF_SIGNATURE
from app.config.settings import Settings, get_settings
from app.config.instance_settings import settings_for_instance
from app.contracts.runtime import AgentRequest
from app.embeddings.factory import create_embedding_provider
from app.rag.pdf_ingest import document_id_for, ingest_pdf
from app.runtime.agent import AgentRuntime
from app.runtime.conversation_store import JsonlConversationStore
from app.runtime.factory import create_runtime
from app.vectorstores.factory import create_vector_store

USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class CreateUserRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)


class UserProfile(BaseModel):
    user_id: str
    created_at: str


class CreateThreadRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)


class UserThreadSummary(BaseModel):
    thread_id: str
    title: str
    updated_at: str | None = None


class CreateThreadResponse(BaseModel):
    thread_id: str


class UserProfileStore:
    """Small SQLite store for users and their registered threads."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_threads (
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, thread_id),
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                        ON DELETE CASCADE
                )
                """
            )

    def create(self, user_id: str) -> UserProfile:
        profile = UserProfile(
            user_id=user_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT INTO user_profiles (user_id, created_at) VALUES (?, ?)",
                (profile.user_id, profile.created_at),
            )
        return profile

    def list(self) -> list[UserProfile]:
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT user_id, created_at
                FROM user_profiles
                ORDER BY created_at DESC, user_id
                """
            ).fetchall()

        return [UserProfile(user_id=row[0], created_at=row[1]) for row in rows]

    def get(self, user_id: str) -> UserProfile | None:
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT user_id, created_at
                FROM user_profiles
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

        if row is None:
            return None
        return UserProfile(user_id=row[0], created_at=row[1])

    def create_thread(self, user_id: str, thread_id: str) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO user_threads (user_id, thread_id, created_at)
                VALUES (?, ?, ?)
                """,
                (user_id, thread_id, created_at),
            )

    def list_threads(self, user_id: str) -> list[str]:
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT thread_id
                FROM user_threads
                WHERE user_id = ?
                ORDER BY created_at
                """,
                (user_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def get_thread(self, user_id: str, thread_id: str) -> bool:
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM user_threads
                WHERE user_id = ? AND thread_id = ?
                """,
                (user_id, thread_id),
            ).fetchone()
        return row is not None


def _build_services() -> dict[str, Any]:
    settings = settings_for_instance(get_settings(), "api_multiuser")
    embeddings = create_embedding_provider(settings)
    vector_store = create_vector_store(settings)
    user_store = UserProfileStore(
        Path(settings.conversation_data_path).parent
        / "db"
        / "user_profiles.sqlite3"
    )
    conversation_store = JsonlConversationStore(settings.conversation_data_path)
    upload_root = Path(settings.pdf_upload_path).resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    return {
        "settings": settings,
        "embeddings": embeddings,
        "vector_store": vector_store,
        "conversation_store": conversation_store,
        "upload_root": upload_root,
        "user_store": user_store,
        "runtimes": {},
    }


def _validate_user_id(user_id: str) -> str:
    if not USER_ID_PATTERN.fullmatch(user_id):
        raise HTTPException(
            status_code=422,
            detail="user_id must be 1-64 letters, numbers, underscores, or hyphens",
        )
    return user_id.casefold()


def _validate_thread_id(thread_id: str) -> str:
    if not THREAD_ID_PATTERN.fullmatch(thread_id):
        raise HTTPException(
            status_code=422,
            detail="thread_id must be 1-128 letters, numbers, underscores, or hyphens",
        )
    return thread_id


def _user_pdf_collection(base_collection: str, user_id: str) -> str:
    safe_base = re.sub(r"[^a-z0-9._-]", "-", base_collection.lower())
    safe_base = safe_base.strip("._-") or "pdf"
    return f"{safe_base[:44]}-u{_user_hash(user_id)}"


def _user_hash(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def create_app(
    *,
    services_factory: Callable[[], dict[str, Any]] = _build_services,
    runtime_factory: Callable[..., AgentRuntime] = create_runtime,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.services = services_factory()
        yield

    application = FastAPI(
        title="Memory PDF RAG Agent API",
        version="1.0.0",
        description="Backend API for the Memory PDF RAG Agent.",
        lifespan=lifespan,
    )

    def get_runtime(user_id: str) -> AgentRuntime:
        services = application.state.services
        runtimes: dict[str, AgentRuntime] = services["runtimes"]
        if user_id not in runtimes:
            settings: Settings = services["settings"]
            user_hash = _user_hash(user_id)
            memory_path = Path(settings.ltm_memory_path)
            user_settings = settings.model_copy(
                update={
                    "pdf_collection": _user_pdf_collection(
                        settings.pdf_collection,
                        user_id,
                    ),
                    "ltm_memory_path": str(
                        memory_path.with_name(
                            f"{memory_path.stem}-{user_hash}{memory_path.suffix}"
                        )
                    ),
                    "trace_data_path": str(
                        Path(settings.trace_data_path) / f"user-{user_hash}"
                    ),
                }
            )
            runtimes[user_id] = runtime_factory(
                settings=user_settings,
                embeddings=services["embeddings"],
                vector_store=services["vector_store"],
                conversation_store=services["conversation_store"],
            )
        return runtimes[user_id]

    def user_upload_directory(user_id: str) -> Path:
        return application.state.services["upload_root"] / user_id

    def require_user(user_id: str) -> str:
        user_id = _validate_user_id(user_id)
        user_store: UserProfileStore = application.state.services["user_store"]
        if user_store.get(user_id) is None:
            raise HTTPException(status_code=404, detail="User profile not found")
        return user_id

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "unauthenticated"}

    @application.post(
        "/api/users",
        response_model=UserProfile,
        status_code=201,
    )
    def create_user(body: CreateUserRequest) -> UserProfile:
        user_id = _validate_user_id(body.user_id.strip())
        try:
            profile = application.state.services["user_store"].create(user_id)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail="A user with this ID already exists",
            ) from exc
        user_upload_directory(profile.user_id).mkdir(
            parents=True,
            exist_ok=True,
        )
        return profile

    @application.get("/api/users", response_model=list[UserProfile])
    def list_users() -> list[UserProfile]:
        return application.state.services["user_store"].list()

    @application.get(
        "/api/users/{user_id}/threads",
        response_model=list[UserThreadSummary],
    )
    def list_threads(user_id: str) -> list[UserThreadSummary]:
        user_id = require_user(user_id)
        store: JsonlConversationStore = application.state.services[
            "conversation_store"
        ]
        conversations = store.list_threads(user_id)
        user_store: UserProfileStore = application.state.services["user_store"]
        registered_threads = user_store.list_threads(user_id)
        conversation_by_thread = {
            conversation.thread_id: conversation for conversation in conversations
        }
        summaries: list[UserThreadSummary] = []
        for thread_id in registered_threads:
            conversation = conversation_by_thread.get(thread_id)
            summaries.append(
                UserThreadSummary(
                    thread_id=thread_id,
                    title=(
                        next(
                            (
                                message.content.strip().splitlines()[0][:72]
                                for message in conversation.messages
                                if message.role == "user" and message.content.strip()
                            ),
                            "New conversation",
                        )
                        if conversation
                        else "New conversation"
                    ),
                    updated_at=(
                        conversation.messages[-1].timestamp.isoformat()
                        if conversation and conversation.messages
                        else None
                    ),
                )
            )
        summaries.sort(
            key=lambda thread: thread.updated_at or "",
            reverse=True,
        )
        return summaries

    @application.post(
        "/api/users/{user_id}/threads",
        response_model=CreateThreadResponse,
        status_code=201,
    )
    def create_thread(
        user_id: str,
        body: CreateThreadRequest,
    ) -> CreateThreadResponse:
        user_id = require_user(user_id)
        thread_id = _validate_thread_id(body.thread_id)
        user_store: UserProfileStore = application.state.services["user_store"]
        try:
            user_store.create_thread(user_id, thread_id)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail="A thread with this ID already exists for this user",
            ) from exc
        return CreateThreadResponse(thread_id=thread_id)

    @application.get("/api/users/{user_id}/threads/{thread_id}/messages")
    def get_thread_messages(
        user_id: str,
        thread_id: str,
    ) -> dict[str, Any]:
        user_id = require_user(user_id)
        thread_id = _validate_thread_id(thread_id)
        user_store: UserProfileStore = application.state.services["user_store"]
        if not user_store.get_thread(user_id, thread_id):
            raise HTTPException(status_code=404, detail="Thread not found")
        store: JsonlConversationStore = application.state.services[
            "conversation_store"
        ]
        conversation = store.load(user_id, thread_id)
        return {
            "thread_id": thread_id,
            "messages": [
                message.model_dump(mode="json")
                for message in conversation.messages
                if message.role in {"user", "assistant"}
            ],
        }

    @application.post("/api/users/{user_id}/threads/{thread_id}/messages")
    def send_message(
        user_id: str,
        thread_id: str,
        body: ChatRequest,
    ) -> dict[str, Any]:
        user_id = require_user(user_id)
        thread_id = _validate_thread_id(thread_id)
        user_store: UserProfileStore = application.state.services["user_store"]
        if not user_store.get_thread(user_id, thread_id):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not body.message.strip():
            raise HTTPException(status_code=422, detail="Message cannot be blank")
        response = get_runtime(user_id).handle(
            AgentRequest(
                user_id=user_id,
                thread_id=thread_id,
                message=body.message.strip(),
                metadata={"document_ids": body.document_ids},
            )
        )
        return response.model_dump(mode="json")

    @application.get(
        "/api/users/{user_id}/documents",
        response_model=list[DocumentSummary],
    )
    def list_documents(user_id: str) -> list[DocumentSummary]:
        user_id = require_user(user_id)
        services = application.state.services
        directory = user_upload_directory(user_id)
        settings: Settings = services["settings"]
        collection = _user_pdf_collection(settings.pdf_collection, user_id)
        documents: list[DocumentSummary] = []
        if not directory.is_dir():
            return documents
        for path in sorted(directory.glob("*.pdf")):
            metadata_items = services["vector_store"].get_by_metadata(
                collection=collection,
                filters={"source_path": str(path.resolve())},
            )
            if not metadata_items:
                continue
            documents.append(
                DocumentSummary(
                    document_id=str(metadata_items[0].metadata["document_id"]),
                    filename=path.name,
                    chunks=len(metadata_items),
                    size_bytes=path.stat().st_size,
                )
            )
        return documents

    @application.post(
        "/api/users/{user_id}/documents",
        response_model=DocumentSummary,
    )
    async def upload_document(
        user_id: str,
        file: Annotated[UploadFile, File()],
    ) -> DocumentSummary:
        user_id = require_user(user_id)
        services = application.state.services
        settings: Settings = services["settings"]
        directory = user_upload_directory(user_id)
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
        collection = _user_pdf_collection(settings.pdf_collection, user_id)
        try:
            with destination.open("xb") as saved_file:
                saved_file.write(contents)
            chunks = await run_in_threadpool(
                ingest_pdf,
                destination,
                services["embeddings"],
                services["vector_store"],
                collection,
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

    @application.delete("/api/users/{user_id}/documents/{document_id}")
    def delete_document(user_id: str, document_id: str) -> dict[str, str]:
        user_id = require_user(user_id)
        services = application.state.services
        settings: Settings = services["settings"]
        collection = _user_pdf_collection(settings.pdf_collection, user_id)
        directory = user_upload_directory(user_id)
        if directory.is_dir():
            for path in directory.glob("*.pdf"):
                if document_id_for(path) != document_id:
                    continue
                items = services["vector_store"].get_by_metadata(
                    collection=collection,
                    filters={"source_path": str(path.resolve())},
                )
                if items:
                    services["vector_store"].delete(
                        collection=collection,
                        ids=[item.item_id for item in items],
                    )
                path.unlink()
                return {"deleted": document_id}
        raise HTTPException(status_code=404, detail="Document not found")
    return application
app = create_app()
