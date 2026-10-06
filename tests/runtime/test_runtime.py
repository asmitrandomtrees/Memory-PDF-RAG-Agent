from app.contracts.runtime import (
    AgentRequest,
)
from app.graph.graph import Phase0Graph
from app.runtime.agent import AgentRuntime
from app.runtime.conversation_store import (
    JsonlConversationStore,
)


def test_runtime_executes_phase0_graph(
    tmp_path,
) -> None:
    store = JsonlConversationStore(tmp_path)

    runtime = AgentRuntime(
        conversation_store=store,
        graph=Phase0Graph(),
    )

    request = AgentRequest(
        user_id="user_001",
        thread_id="thread_001",
        message="Hello",
    )

    response = runtime.handle(request)

    assert response.user_id == "user_001"
    assert response.thread_id == "thread_001"
    assert response.trace_id
    assert "Phase 0 skeleton" in response.answer

    conversation = store.load(
        user_id="user_001",
        thread_id="thread_001",
    )

    assert len(conversation.messages) == 2

    assert conversation.messages[0].role == "user"
    assert conversation.messages[0].content == "Hello"

    assert conversation.messages[1].role == "assistant"