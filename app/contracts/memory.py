from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


MemoryType = Literal[
    "preference",
    "profile",
    "fact",
    "goal",
    "relationship",
    "instruction",
    "other",
]

MemoryStatus = Literal["active", "superseded", "inactive"]

MemoryAction = Literal[
    "ADD",
    "UPDATE",
    "MERGE",
    "SUPERSEDE",
    "IGNORE",
]


class MemorySource(BaseModel):
    thread_id: str
    message_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid4()))

    user_id: str
    memory_type: MemoryType
    content: str
    status: MemoryStatus = "active"

    source: MemorySource

    version: int = 1
    supersedes_memory_id: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: dict[str, object] = Field(default_factory=dict)


class CandidateMemory(BaseModel):
    user_id: str
    memory_type: MemoryType
    content: str
    source: MemorySource

    confidence: float = Field(ge=0.0, le=1.0)

    metadata: dict[str, object] = Field(default_factory=dict)


class MemoryDecision(BaseModel):
    action: MemoryAction
    candidate: CandidateMemory

    existing_memory_ids: list[str] = Field(default_factory=list)

    reason: str | None = None