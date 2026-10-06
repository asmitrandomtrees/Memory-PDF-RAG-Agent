from pydantic import BaseModel, Field


class RetrievalPlan(BaseModel):
    use_stm: bool = False
    use_ltm: bool = False
    use_pdf: bool = False

    reasoning: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def any_source_selected(self) -> bool:
        return self.use_stm or self.use_ltm or self.use_pdf