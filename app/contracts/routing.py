from typing import Any
from pydantic import BaseModel, Field, field_validator


class RetrievalPlan(BaseModel):
    use_stm: bool = False
    use_ltm: bool = False
    use_pdf: bool = False

    reasoning: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("use_stm", "use_ltm", "use_pdf", mode="before")
    @classmethod
    def coerce_bools(cls, v: Any) -> bool:
        if v is None:
            return False
        return bool(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> float:
        if v is None:
            return 1.0
        return float(v)

    @field_validator("metadata", mode="before")
    @classmethod
    def coerce_metadata(cls, v: Any) -> dict[str, object]:
        if v is None:
            return {}
        return dict(v)

    @property
    def any_source_selected(self) -> bool:
        return self.use_stm or self.use_ltm or self.use_pdf