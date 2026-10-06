from app.contracts.context import AgentContext, ContextItem
from app.contracts.errors import LLMError
from app.graph.nodes.answer import AnswerEvidenceValidator, AnswerGenerator
from app.graph.schemas import AnswerDraft
from app.llm.provider import LLMResponse


class PlainFallbackLLM:
    def __init__(self) -> None:
        self.structured_calls = 0
        self.invoke_calls = 0

    def structured_output(self, messages, output_schema, *, temperature=None):
        self.structured_calls += 1
        raise LLMError("The model returned an empty answer field")

    def invoke(self, messages, *, temperature=None):
        self.invoke_calls += 1
        return LLMResponse(content="We were designing the authentication endpoint.")


class PromptCaptureLLM:
    def __init__(self) -> None:
        self.messages = None

    def structured_output(self, messages, output_schema, *, temperature=None):
        self.messages = messages
        return AnswerDraft(answer="Saturday works for a Hindi thriller.")


def test_answer_prompt_limits_follow_up_questions_and_option_menus() -> None:
    llm = PromptCaptureLLM()

    AnswerGenerator(llm).generate(
        query="Saturday, Hindi",
        context=AgentContext(),
    )

    system_prompt = llm.messages[0].content
    assert "Ask at most one follow-up question" in system_prompt
    assert "Do not end routine replies with menus" in system_prompt


def test_answer_generator_falls_back_to_plain_chat_for_invalid_tool_output() -> None:
    llm = PlainFallbackLLM()

    draft = AnswerGenerator(llm).generate(
        query="What endpoint were we just designing?",
        context=AgentContext(),
    )

    assert draft.answer == "We were designing the authentication endpoint."
    assert draft.cited_item_ids == []
    assert draft.insufficient_evidence is False
    assert llm.structured_calls == 1
    assert llm.invoke_calls == 1


def test_validator_accepts_general_answer_when_no_retrieval_was_needed() -> None:
    llm = PlainFallbackLLM()

    validation = AnswerEvidenceValidator(llm).validate(
        query="How does OAuth2 work?",
        draft=AnswerDraft(answer="OAuth2 delegates authorization."),
        context=AgentContext(),
    )

    assert validation.is_valid is True
    assert validation.needs_retrieval is False
    assert llm.structured_calls == 0


def test_validator_keeps_answer_available_when_model_validation_fails() -> None:
    llm = PlainFallbackLLM()
    context = AgentContext(
        items=[
            ContextItem(
                source="stm",
                item_id="message_1",
                content="We discussed authentication.",
            )
        ]
    )

    validation = AnswerEvidenceValidator(llm).validate(
        query="What did we discuss?",
        draft=AnswerDraft(
            answer="We discussed authentication.",
            cited_item_ids=["message_1"],
        ),
        context=context,
    )

    assert validation.is_valid is True
    assert "unavailable" in (validation.reason or "")
