from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


INDIA_TIMEZONE = timezone(timedelta(hours=5, minutes=30))


class QueryAnalysis(BaseModel):
    intent: Literal[
        "episodic_recall",
        "memory_recall",
        "document_question",
        "mixed",
        "general",
    ]
    normalized_query: str = Field(min_length=1)
    date_month: int | None = Field(default=None, ge=1, le=12)
    date_day: int | None = Field(default=None, ge=1, le=31)
    date_year: int | None = Field(default=None, ge=1, le=9999)
    document_ids: list[str] = Field(default_factory=list)

    @field_validator("document_ids", mode="before")
    @classmethod
    def coerce_document_ids(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @model_validator(mode="after")
    def validate_calendar_date(self) -> "QueryAnalysis":
        if (self.date_month is None) != (self.date_day is None):
            raise ValueError("date_month and date_day must be provided together")

        if self.date_month is not None and self.date_day is not None:
            year = self.date_year or 2000
            date(year, self.date_month, self.date_day)

        return self


class AnswerDraft(BaseModel):
    answer: str = Field(
        min_length=1,
        description=(
            "A complete, direct, natural-language reply to the user. "
            "It must never be blank."
        ),
    )
    cited_item_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Internal IDs of retrieved items used for conversation, user, "
            "or document-specific claims. Never include these IDs in answer."
        ),
    )
    insufficient_evidence: bool = False

    @field_validator("cited_item_ids", mode="before")
    @classmethod
    def coerce_cited_item_ids(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @field_validator("insufficient_evidence", mode="before")
    @classmethod
    def coerce_insufficient_evidence(cls, v: Any) -> bool:
        if v is None:
            return False
        return bool(v)


class AnswerValidation(BaseModel):
    is_valid: bool
    needs_retrieval: bool = False
    reason: str | None = None


class RewrittenQuery(BaseModel):
    query: str = Field(min_length=1)


def resolve_calendar_day(
    analysis: QueryAnalysis,
    *,
    now: datetime | None = None,
    local_timezone: tzinfo = INDIA_TIMEZONE,
) -> tuple[datetime | None, datetime | None]:
    if analysis.date_month is None or analysis.date_day is None:
        return None, None

    local_now = (now or datetime.now(timezone.utc)).astimezone(local_timezone)
    if analysis.date_year is not None:
        target_day = date(
            analysis.date_year,
            analysis.date_month,
            analysis.date_day,
        )
    else:
        year = local_now.year
        while True:
            try:
                target_day = date(year, analysis.date_month, analysis.date_day)
            except ValueError:
                year -= 1
                continue
            if target_day <= local_now.date():
                break
            year -= 1

    start_at = datetime.combine(
        target_day,
        time.min,
        tzinfo=local_timezone,
    )
    end_at = start_at + timedelta(days=1)
    return start_at, end_at
