from app.contracts.routing import RetrievalPlan
from app.graph.nodes.query import QueryAnalyzer, RetrievalPlanner
from app.graph.schemas import QueryAnalysis
from app.runtime.agent import is_trivial_small_talk


class UnexpectedCallLLM:
    def structured_output(self, messages, output_schema, *, temperature=None):
        raise AssertionError("Fast-path query should not call the LLM")


class CapturingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def structured_output(self, messages, output_schema, *, temperature=None):
        self.calls += 1
        if output_schema is QueryAnalysis:
            return QueryAnalysis(
                intent="memory_recall",
                normalized_query="What is my preferred database?",
            )
        raise AssertionError(f"Unexpected schema: {output_schema}")


def test_named_greeting_skips_query_analysis_and_planning() -> None:
    llm = UnexpectedCallLLM()
    analysis = QueryAnalyzer(llm).analyze("Hi Stan!")

    assert analysis.intent == "general"
    assert analysis.normalized_query == "Hi Stan!"

    plan = RetrievalPlanner(llm).plan(
        query="Hi Stan!",
        analysis=analysis,
    )

    assert plan == RetrievalPlan(use_stm=False, use_ltm=False, use_pdf=False)


def test_short_follow_up_is_not_classified_as_small_talk() -> None:
    assert is_trivial_small_talk("Yes") is False
    assert is_trivial_small_talk("Hi Stan!") is True


def test_standalone_general_query_skips_analysis_and_planning_llm() -> None:
    llm = UnexpectedCallLLM()
    analysis = QueryAnalyzer(llm).analyze("How does OAuth2 work?")

    assert analysis.intent == "general"
    assert analysis.normalized_query == "How does OAuth2 work?"
    assert RetrievalPlanner(llm).plan(
        query="How does OAuth2 work?",
        analysis=analysis,
    ) == RetrievalPlan(use_stm=False, use_ltm=False, use_pdf=False)


def test_standalone_general_query_with_documents_routes_to_pdf() -> None:
    llm = UnexpectedCallLLM()
    analysis = QueryAnalyzer(llm).analyze(
        "How does OAuth2 work?",
        document_ids=["oauth2-guide"],
    )

    assert analysis.intent == "document_question"
    assert analysis.document_ids == ["oauth2-guide"]
    assert RetrievalPlanner(llm).plan(
        query="How does OAuth2 work?",
        analysis=analysis,
        document_ids=["oauth2-guide"],
    ) == RetrievalPlan(
        use_pdf=True,
        metadata={"reason": "selected_documents"},
    )


def test_memory_shaped_query_still_uses_analysis_llm() -> None:
    llm = CapturingLLM()

    analysis = QueryAnalyzer(llm).analyze("What is my preferred database?")

    assert analysis.intent == "memory_recall"
    assert llm.calls == 1
