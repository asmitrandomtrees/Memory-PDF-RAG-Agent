from app.contracts.routing import RetrievalPlan
from app.graph.nodes.query import QueryAnalyzer, RetrievalPlanner
from app.graph.schemas import QueryAnalysis
from app.runtime.text_utils import is_trivial_small_talk


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

    assert plan == RetrievalPlan(
        use_stm=False,
        use_ltm=False,
        use_pdf=False,
        metadata={
            "router": "deterministic",
            "reason": "standalone_general",
        },
    )


def test_short_follow_up_is_not_classified_as_small_talk() -> None:
    assert is_trivial_small_talk("Yes") is False
    assert is_trivial_small_talk("Hi Stan!") is True
    assert is_trivial_small_talk("Hii I am Michael") is True
    assert is_trivial_small_talk("Hii, I am Jim") is True
    assert is_trivial_small_talk("Hi I am building a PDF agent") is False


def test_standalone_general_query_skips_analysis_and_planning_llm() -> None:
    llm = UnexpectedCallLLM()
    analysis = QueryAnalyzer(llm).analyze("How does OAuth2 work?")

    assert analysis.intent == "general"
    assert analysis.normalized_query == "How does OAuth2 work?"
    assert RetrievalPlanner(llm).plan(
        query="How does OAuth2 work?",
        analysis=analysis,
    ) == RetrievalPlan(
        use_stm=False,
        use_ltm=False,
        use_pdf=False,
        metadata={
            "router": "deterministic",
            "reason": "standalone_general",
        },
    )


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
        metadata={
            "router": "deterministic",
            "reason": "selected_documents",
        },
    )


def test_memory_shaped_query_still_uses_analysis_llm() -> None:
    llm = CapturingLLM()

    analysis = QueryAnalyzer(llm).analyze("What is my preferred database?")

    assert analysis.intent == "memory_recall"
    assert llm.calls == 1


def test_memory_recall_uses_deterministic_stm_ltm_plan() -> None:
    llm = UnexpectedCallLLM()

    plan = RetrievalPlanner(llm).plan(
        query="What is my preferred database?",
        analysis=QueryAnalysis(
            intent="memory_recall",
            normalized_query="What is my preferred database?",
        ),
    )

    assert plan == RetrievalPlan(
        use_stm=True,
        use_ltm=True,
        metadata={
            "router": "deterministic",
            "reason": "memory_recall",
        },
    )


def test_contextual_follow_up_uses_deterministic_stm_plan() -> None:
    llm = UnexpectedCallLLM()

    plan = RetrievalPlanner(llm).plan(
        query="with whom?",
        analysis=QueryAnalysis(
            intent="general",
            normalized_query="with whom?",
        ),
    )

    assert plan == RetrievalPlan(
        use_stm=True,
        metadata={
            "router": "deterministic",
            "reason": "contextual_follow_up",
        },
    )


def test_pdf_only_context_uses_deterministic_pdf_plan() -> None:
    llm = UnexpectedCallLLM()

    plan = RetrievalPlanner(llm).plan(
        query="What maximum wind speed is specified?",
        analysis=QueryAnalysis(
            intent="document_question",
            normalized_query="What maximum wind speed is specified?",
        ),
        prefer_pdf=True,
    )

    assert plan == RetrievalPlan(
        use_pdf=True,
        metadata={
            "router": "deterministic",
            "reason": "thread_pdf_context",
        },
    )
