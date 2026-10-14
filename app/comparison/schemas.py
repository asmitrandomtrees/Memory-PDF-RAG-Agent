"""Data models for specification comparison results."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator

# baseline: the reference document's own value.
# compliant: meets or exceeds the baseline requirement.
# deviation: conflicts with or falls short of the baseline.
# not_specified: the document does not state this parameter.
# no_requirement: the baseline does not state it, so nothing to check.
Assessment = Literal[
    "baseline",
    "compliant",
    "deviation",
    "not_specified",
    "no_requirement",
]
RowStatus = Literal["match", "deviation", "missing", "not_in_baseline"]


# ---------------------------------------------------------------------------
# What the LLM returns for one aspect. Documents are referred to by short
# labels (D1, D2, ...) so the model never has to copy long IDs.
# ---------------------------------------------------------------------------


class ExtractedCell(BaseModel):
    document: str = Field(description="Document label, e.g. D1.")
    value: str | None = Field(
        default=None,
        description="Concise value with units as stated; null if not stated.",
    )
    quote: str | None = Field(
        default=None,
        description="Shortest phrase copied verbatim from that document's "
        "excerpts that states the value; null if not stated.",
    )
    assessment: Assessment
    note: str | None = Field(
        default=None,
        description="One short reason for a deviation, otherwise null.",
    )


class ExtractedRow(BaseModel):
    parameter: str = Field(description="Short parameter name.")
    cells: list[ExtractedCell] = Field(default_factory=list)

    @field_validator("cells", mode="before")
    @classmethod
    def coerce_cells(cls, v: Any) -> list[Any]:
        return [] if v is None else list(v)


class AspectExtraction(BaseModel):
    rows: list[ExtractedRow] = Field(default_factory=list)
    summary: str = Field(
        default="",
        description="One or two sentences on the key findings.",
    )

    @field_validator("rows", mode="before")
    @classmethod
    def coerce_rows(cls, v: Any) -> list[Any]:
        return [] if v is None else list(v)


# ---------------------------------------------------------------------------
# Final report.
# ---------------------------------------------------------------------------


class ComparedDocument(BaseModel):
    document_id: str
    filename: str


class ComparisonCell(BaseModel):
    document_id: str
    value: str | None = None
    quote: str | None = None
    page_number: int | None = None
    assessment: Assessment
    note: str | None = None
    # False when the model gave a value whose quote is not in the document.
    verified: bool = True


class ComparisonRow(BaseModel):
    aspect: str
    parameter: str
    status: RowStatus
    cells: list[ComparisonCell]


class AspectResult(BaseModel):
    key: str
    title: str
    summary: str = ""
    rows: list[ComparisonRow] = Field(default_factory=list)
    error: str | None = None


class ComparisonReport(BaseModel):
    documents: list[ComparedDocument]
    baseline_document_id: str
    aspects: list[AspectResult]
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def rows(self) -> list[ComparisonRow]:
        return [row for aspect in self.aspects for row in aspect.rows]

    def filename_for(self, document_id: str) -> str:
        return next(
            (d.filename for d in self.documents if d.document_id == document_id),
            document_id,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> dict[str, int]:
        counts = {"match": 0, "deviation": 0, "missing": 0, "not_in_baseline": 0}
        for row in self.rows:
            counts[row.status] += 1
        counts["total"] = sum(counts.values())
        return counts

    @computed_field  # type: ignore[prop-decorator]
    @property
    def missing_specifications(self) -> list[dict[str, Any]]:
        missing: list[dict[str, Any]] = []
        for aspect in self.aspects:
            for row in aspect.rows:
                absent = [
                    self.filename_for(cell.document_id)
                    for cell in row.cells
                    if cell.assessment == "not_specified"
                ]
                if absent:
                    missing.append(
                        {
                            "aspect": aspect.title,
                            "parameter": row.parameter,
                            "missing_in": absent,
                        }
                    )
        return missing