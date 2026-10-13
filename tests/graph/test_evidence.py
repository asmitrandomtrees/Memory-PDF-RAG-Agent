from app.contracts.context import AgentContext, ContextItem
from app.graph.evidence import extract_evidence
from app.graph.schemas import EvidenceQuote


SPEC_TEXT = (
    "Fill & Drift Eliminators: Formed from self-extinguishing PVC with a "
    "flame spread rating of five as per ASTM E84, with a spacing between "
    "fill sheets of 20 mm, shall limit drift loss to a maximum of 0.005 % "
    "of the recirculation rate and suspended from steel tubing. Water "
    "Distribution System: The water distribution system is to distribute "
    "water evenly over the fill."
)


def _context() -> AgentContext:
    return AgentContext(
        items=[
            ContextItem(
                source="pdf",
                item_id="spec_p1_c2",
                content=SPEC_TEXT,
                metadata={"filename": "Spec.pdf", "page_number": 1},
            ),
            ContextItem(
                source="pdf",
                item_id="spec_p2_c0",
                content="Fans shall use L10 bearings rated for 80,000 hours.",
                metadata={"filename": "Spec.pdf", "page_number": 2},
            ),
        ]
    )


def test_keeps_exact_quote_with_source_location() -> None:
    evidence = extract_evidence(
        answer="Drift loss is limited to 0.005% of the recirculation rate.",
        quotes=[
            EvidenceQuote(
                item_id="spec_p1_c2",
                quote="shall limit drift loss to a maximum of 0.005 %",
            )
        ],
        cited_item_ids=["spec_p1_c2"],
        context=_context(),
    )

    assert evidence == [
        {
            "item_id": "spec_p1_c2",
            "source": "pdf",
            "quote": "shall limit drift loss to a maximum of 0.005 %",
            "filename": "Spec.pdf",
            "page_number": 1,
        }
    ]


def test_exact_match_ignores_case_whitespace_and_typographic_dashes() -> None:
    evidence = extract_evidence(
        answer="Drift loss max is 0.005%.",
        quotes=[
            EvidenceQuote(
                item_id="spec_p1_c2",
                quote="SHALL limit   drift loss to a maximum of 0.005 %",
            )
        ],
        cited_item_ids=["spec_p1_c2"],
        context=_context(),
    )

    assert evidence[0]["quote"] == "shall limit drift loss to a maximum of 0.005 %"


def test_paraphrased_quote_is_replaced_by_closest_real_sentence() -> None:
    evidence = extract_evidence(
        answer="Drift loss is limited to 0.005%.",
        quotes=[
            EvidenceQuote(
                item_id="spec_p1_c2",
                quote="Drift eliminators shall limit drift loss to a "
                "maximum of 0.005 % of the recirculation rate.",
            )
        ],
        cited_item_ids=["spec_p1_c2"],
        context=_context(),
    )

    assert len(evidence) == 1
    assert "0.005 % of the recirculation rate" in evidence[0]["quote"]
    assert evidence[0]["quote"] in " ".join(SPEC_TEXT.split())


def test_invented_quote_is_dropped() -> None:
    evidence = extract_evidence(
        answer="The tower is blue.",
        quotes=[
            EvidenceQuote(item_id="spec_p1_c2", quote="The tower shall be painted blue.")
        ],
        cited_item_ids=["spec_p1_c2"],
        context=_context(),
    )

    assert evidence == []


def test_quotes_for_uncited_or_unknown_items_are_ignored() -> None:
    evidence = extract_evidence(
        answer="Bearings last 80,000 hours.",
        quotes=[
            EvidenceQuote(item_id="spec_p2_c0", quote="rated for 80,000 hours"),
            EvidenceQuote(item_id="missing", quote="rated for 80,000 hours"),
        ],
        cited_item_ids=["spec_p1_c2"],
        context=_context(),
    )

    assert all(entry["item_id"] == "spec_p1_c2" for entry in evidence)


def test_cited_item_without_quote_falls_back_to_best_answer_sentence() -> None:
    evidence = extract_evidence(
        answer="The fans use L10 bearings rated for 80,000 hours.",
        quotes=[],
        cited_item_ids=["spec_p2_c0"],
        context=_context(),
    )

    assert evidence[0]["quote"] == (
        "Fans shall use L10 bearings rated for 80,000 hours."
    )
    assert evidence[0]["page_number"] == 2
