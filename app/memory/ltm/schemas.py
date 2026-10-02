from typing import Any
from pydantic import BaseModel, Field, field_validator

from app.contracts.memory import MemoryAction, MemoryType


class ExtractedMemory(BaseModel):
    memory_type: MemoryType
    content: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    message_ids: list[str] = Field(default_factory=list)

    @field_validator("message_ids", mode="before")
    @classmethod
    def coerce_message_ids(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> float:
        if v is None:
            return 1.0
        return float(v)


class LTMExtractionOutput(BaseModel):
    memories: list[ExtractedMemory] = Field(default_factory=list)

    @field_validator("memories", mode="before")
    @classmethod
    def coerce_memories(cls, v: Any) -> list[ExtractedMemory]:
        if v is None:
            return []
        return list(v)


class LTMConsolidationOutput(BaseModel):
    action: MemoryAction
    existing_memory_ids: list[str] = Field(default_factory=list)
    reason: str | None = None

    @field_validator("existing_memory_ids", mode="before")
    @classmethod
    def coerce_existing_memory_ids(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)