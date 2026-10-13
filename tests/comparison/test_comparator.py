import csv
import io

import pytest

from app.comparison import (
    ComparedDocument,
    SpecComparator,
    resolve_aspects,
    to_csv,
    to_markdown,
)
from app.comparison.schemas import AspectExtraction
from app.contracts.retrieval import RetrievedItem

SPEC = ComparedDocument(document_id="spec", filename="Spec.pdf")
VENDOR_A = ComparedDocument(document_id="vendor_a", filename="VendorA.pdf")
VENDOR_B = ComparedDocument(document_id="vendor_b", filename="VendorB.pdf")

CHUNKS = {
    "spec": "Motor(s) and Drive: TEFC motor, variable torque. Bearings rated "
    "for L10 (80,000 hr. life). Enclosure: NEMA 4 surface wall mounting.",
    "vendor_a": "Motor: TEFC motor, 30 hp. Fan bearings rated for 50,000 hr "
    "life. Enclosure: NEMA 12 wall mounted enclosure.",
    "vendor_b": "Motor: TEFC motor, 25 hp. Bearings: 80,000 hr life.",
}

EXTRACTION = {
    "summary": "Vendor A has shorter bearing life.",
    "rows": [
        {
            "parameter": "Motor enclosure",
            "cells": [
                {"document": "D1", "value": "TEFC", "quote": "TEFC motor, variable torque", "assessment": "baseline"},
                {"document": "D2", "value": "TEFC", "quote": "TEFC motor, 30 hp", "assessment": "compliant"},
                {"document": "D3", "value": "TEFC", "quote": "TEFC motor, 25 hp", "assessment": "compliant"},
            ],
        },
        {
            "parameter": "Bearing life",
            "cells": [
                {"document": "D1", "value": "80,000 hr", "quote": "L10 (80,000 hr. life)", "assessment": "baseline"},
                {"document": "D2", "value": "50,000 hr", "quote": "rated for 50,000 hr life", "assessment": "deviation", "note": "Shorter than required"},
                {"document": "D3", "value": "80,000 hr", "quote": "80,000 hr life", "assessment": "compliant"},
            ],
        },
        {
            "parameter": "Panel enclosure",
            "cells": [
                {"document": "D1", "value": "NEMA 4", "quote": "NEMA 4 surface wall mounting", "assessment": "baseline"},
                {"document": "D2", "value": "NEMA 12", "quote": "a made-up quote", "assessment": "deviation", "note": "Lower rating"},
                # D3 omitted entirely: treated as not specified.
            ],
        },
        {
            "parameter": "Motor power",
            "cells": [
                {"document": "D1", "value": None, "assessment": "not_specified"},
                {"document": "D2", "value": "30 hp", "quote": "30 hp", "assessment": "compliant"},
                {"document": "D3", "value": "25 hp", "quote": "25 hp", "assessment": "deviation"},
            ],
        },
        {
            "parameter": "Nobody states this",
            "cells": [
                {"document": "D1", "value": None, "assessment": "not_specified"},
                {"document": "D2", "value": None, "assessment": "not_specified"},
            ],
        },
    ],
}


class FakeEmbeddings:
    def embed_query(self, text):
        return [0.1, 0.2]


class FakeStore:
    def __init__(self):
        self.filters = []

    def search(self, *, collection, query_embedding, top_k, filters=None):
        self.filters.append(filters)
        document_id = filters["document_id"]
        return [
            RetrievedItem(
                item_id=f"{document_id}_c0",
                source="pdf",
                content=CHUNKS[document_id],
                metadata={"document_id": document_id, "page_number": 2},
            )
        ]


class FakeLLM:
    def __init__(self, failures=0):
        self.calls = []
        self.failures = failures

    def structured_output(self, messages, output_schema, *, temperature=None):
        self.calls.append(messages)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("malformed tool call")
        return output_schema.model_validate(EXTRACTION)


def _compare(llm=None, store=None, **kwargs):
    comparator = SpecComparator(
        llm=llm or FakeLLM(),
        embeddings=FakeEmbeddings(),
        vector_store=store or FakeStore(),
        collection="pdf_collection",
    )
    return comparator.compare(
        [SPEC, VENDOR_A, VENDOR_B],
        aspects=["fan_motor"],
        **kwargs,
    )


def _row(report, parameter):
    return next(row for row in report.rows if row.parameter == parameter)


def test_builds_matrix_with_statuses_and_verified_quotes() -> None:
    report = _compare()

    assert [r.parameter for r in report.rows] == [
        "Motor enclosure",
        "Bearing life",
        "Panel enclosure",
        "Motor power",
    ]
    assert _row(report, "Motor enclosure").status == "match"

    bearing = _row(report, "Bearing life")
    assert bearing.status == "deviation"
    assert bearing.cells[1].assessment == "deviation"
    assert bearing.cells[1].note == "Shorter than required"
    assert bearing.cells[1].quote == "rated for 50,000 hr life"
    assert bearing.cells[1].page_number == 2


def test_invented_quote_is_flagged_unverified() -> None:
    panel = _row(_compare(), "Panel enclosure")

    assert panel.cells[1].value == "NEMA 12"
    assert panel.cells[1].quote is None
    assert panel.cells[1].verified is False


def test_omitted_document_is_missing() -> None:
    panel = _row(_compare(), "Panel enclosure")

    assert panel.cells[2].assessment == "not_specified"
    # A deviation outranks a missing value.
    assert panel.status == "deviation"


def test_values_without_baseline_requirement_cannot_deviate() -> None:
    power = _row(_compare(), "Motor power")

    assert power.status == "not_in_baseline"
    assert [c.assessment for c in power.cells] == [
        "not_specified",
        "no_requirement",
        "no_requirement",
    ]


def test_summary_and_missing_specifications() -> None:
    report = _compare()

    assert report.summary == {
        "match": 1,
        "deviation": 2,
        "missing": 0,
        "not_in_baseline": 1,
        "total": 4,
    }
    assert {
        "aspect": "Fan motor ratings",
        "parameter": "Panel enclosure",
        "missing_in": ["VendorB.pdf"],
    } in report.missing_specifications


def test_retrieval_is_scoped_to_each_document() -> None:
    store = FakeStore()
    _compare(store=store)

    assert store.filters == [
        {"document_id": "spec"},
        {"document_id": "vendor_a"},
        {"document_id": "vendor_b"},
    ]


def test_chosen_baseline_becomes_reference_column() -> None:
    report = _compare(baseline_document_id="vendor_b")

    assert report.baseline_document_id == "vendor_b"
    assert report.documents[0] == VENDOR_B


def test_retries_once_then_reports_aspect_error() -> None:
    assert _compare(llm=FakeLLM(failures=1)).rows

    report = _compare(llm=FakeLLM(failures=2))
    assert report.rows == []
    assert "Comparison failed" in report.aspects[0].error


def test_requires_two_distinct_documents_and_valid_baseline() -> None:
    comparator = SpecComparator(
        llm=FakeLLM(),
        embeddings=FakeEmbeddings(),
        vector_store=FakeStore(),
        collection="pdf_collection",
    )
    with pytest.raises(ValueError):
        comparator.compare([SPEC])
    with pytest.raises(ValueError):
        comparator.compare([SPEC, SPEC])
    with pytest.raises(ValueError):
        comparator.compare([SPEC, VENDOR_A], baseline_document_id="other")


def test_resolve_aspects_accepts_keys_and_free_text() -> None:
    aspects = resolve_aspects(["fan_motor", "Basin materials", "fan_motor"])

    assert [a.key for a in aspects] == ["fan_motor", "basin_materials"]
    assert len(resolve_aspects(None)) == 4


def test_exports_csv_matrix_and_markdown_report() -> None:
    report = _compare()

    rows = list(csv.reader(io.StringIO(to_csv(report))))
    assert rows[0][:4] == ["Aspect", "Parameter", "Status", "Spec.pdf value"]
    bearing = next(r for r in rows if r[1] == "Bearing life")
    assert bearing[2] == "Deviation"
    assert "50,000 hr" in bearing

    markdown = to_markdown(report)
    assert "## Fan motor ratings" in markdown
    assert "**⚠ 50,000 hr (p.2)**" in markdown
    assert "## Missing specifications" in markdown


def test_report_round_trips_through_json_for_export() -> None:
    report = _compare()
    restored = type(report).model_validate_json(report.model_dump_json())

    assert to_csv(restored) == to_csv(report)
    assert isinstance(AspectExtraction.model_validate(EXTRACTION), AspectExtraction)


def test_document_labels_in_notes_and_summary_become_filenames() -> None:
    extraction = {
        "summary": "D2 has shorter bearing life than D1.",
        "rows": [
            {
                "parameter": "Bearing life",
                "cells": [
                    {"document": "D1", "value": "80,000 hr", "quote": "80,000 hr", "assessment": "baseline"},
                    {"document": "D2", "value": "50,000 hr", "quote": "50,000 hr", "assessment": "deviation", "note": "D2 is below D1."},
                ],
            }
        ],
    }

    class LabelLLM(FakeLLM):
        def structured_output(self, messages, output_schema, *, temperature=None):
            return output_schema.model_validate(extraction)

    comparator = SpecComparator(
        llm=LabelLLM(),
        embeddings=FakeEmbeddings(),
        vector_store=FakeStore(),
        collection="pdf_collection",
    )
    report = comparator.compare([SPEC, VENDOR_A], aspects=["fan_motor"])

    assert report.aspects[0].summary == "VendorA.pdf has shorter bearing life than Spec.pdf."
    assert report.rows[0].cells[1].note == "VendorA.pdf is below Spec.pdf."
