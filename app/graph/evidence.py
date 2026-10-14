"""Turn the answer model's supporting quotes into verified evidence.

The model is asked to copy a verbatim quote from each item it cites, but
models sometimes paraphrase. Every quote is checked against the item's
actual content: exact matches are kept, near matches are replaced by the
closest real sentence, and anything else is dropped. Cited items with no
usable quote fall back to the sentence that best overlaps the answer.
"""
from __future__ import annotations

import re
from typing import Any

from app.contracts.context import AgentContext, ContextItem
from app.graph.schemas import EvidenceQuote

_SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "‐": "-",
        "‑": "-",
        "–": "-",
        "—": "-",
        " ": " ",
    }
)
# Short filler words that do not show a sentence supports a claim.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "shall", "that", "the", "to", "with",
}
# Fraction of a quote's words a sentence must contain to replace it.
_MIN_QUOTE_OVERLAP = 0.6
# Answer words a sentence must share to be used as fallback evidence.
_MIN_ANSWER_OVERLAP = 3
MAX_EVIDENCE_ITEMS = 5


def _normalize(text: str) -> str:
    return " ".join(text.translate(_TRANSLATION).split())


def _words(text: str) -> set[str]:
    return {
        word
        for word in _WORD.findall(text.lower())
        if word not in _STOPWORDS
    }


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(_normalize(text)) if s]


def verify_quote(quote: str, content: str) -> str | None:
    quote = _normalize(quote).strip(" .\"'")
    if not quote:
        return None
    normalized_content = _normalize(content)
    start = normalized_content.lower().find(quote.lower())
    if start >= 0:
        return normalized_content[start : start + len(quote)]

    quote_words = _words(quote)
    if not quote_words:
        return None
    best, best_overlap = None, 0.0
    for sentence in _sentences(content):
        overlap = len(quote_words & _words(sentence)) / len(quote_words)
        if overlap > best_overlap:
            best, best_overlap = sentence, overlap
    return best if best_overlap >= _MIN_QUOTE_OVERLAP else None


def _best_sentence_for_answer(answer: str, content: str) -> str | None:
    answer_words = _words(answer)
    best, best_overlap = None, 0
    for sentence in _sentences(content):
        overlap = len(answer_words & _words(sentence))
        if overlap > best_overlap:
            best, best_overlap = sentence, overlap
    return best if best_overlap >= _MIN_ANSWER_OVERLAP else None


def _evidence_entry(item: ContextItem, quote: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "item_id": item.item_id,
        "source": item.source,
        "quote": quote,
    }
    for key in ("filename", "page_number", "document_id", "role", "timestamp"):
        value = item.metadata.get(key)
        if value not in (None, ""):
            entry[key] = value
    return entry


def extract_evidence(
    *,
    answer: str,
    quotes: list[EvidenceQuote],
    cited_item_ids: list[str],
    context: AgentContext,
) -> list[dict[str, Any]]:
    """Return verified supporting quotes for the items the answer cites."""
    items = {
        item.item_id: item
        for item in context.items
        if item.item_id in cited_item_ids
    }
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    quoted_ids: set[str] = set()

    def add(item: ContextItem, quote: str) -> None:
        key = (item.item_id, quote.lower())
        if key in seen or len(evidence) >= MAX_EVIDENCE_ITEMS:
            return
        seen.add(key)
        quoted_ids.add(item.item_id)
        evidence.append(_evidence_entry(item, quote))

    for quote in quotes:
        item = items.get(quote.item_id)
        if item is None:
            continue
        verified = verify_quote(quote.quote, item.content)
        if verified:
            add(item, verified)

    for item_id in cited_item_ids:
        item = items.get(item_id)
        if item is None or item_id in quoted_ids:
            continue
        fallback = _best_sentence_for_answer(answer, item.content)
        if fallback:
            add(item, fallback)

    return evidence