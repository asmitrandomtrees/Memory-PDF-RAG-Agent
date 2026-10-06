from uuid import uuid4

from app.contracts.conversation import ConversationMessage
from app.contracts.runtime import (
    AgentRequest,
    AgentResponse,
    ConversationStore,
    GraphRunner,
)
from app.runtime.session import AgentSession


class AgentRuntime:
    def __init__(
        self,
        conversation_store: ConversationStore,
        graph: GraphRunner,
    ) -> None:
        self.conversation_store = conversation_store
        self.graph = graph

    def handle(self, request: AgentRequest) -> AgentResponse:
        trace_id = str(uuid4())

        conversation = self.conversation_store.load(
            request.user_id,
            request.thread_id,
        )

        session = AgentSession(
            user_id=request.user_id,
            thread_id=request.thread_id,
            conversation=conversation,
        )

        user_message = ConversationMessage(
            user_id=session.user_id,
            thread_id=session.thread_id,
            role="user",
            content=request.message,
            metadata=request.metadata,
        )

        self.conversation_store.append_message(user_message)

        response = self.graph.run(request, trace_id)

        assistant_message = ConversationMessage(
            user_id=session.user_id,
            thread_id=session.thread_id,
            role="assistant",
            content=response.answer,
            metadata={"trace_id": trace_id},
        )

        self.conversation_store.append_message(assistant_message)

        return response