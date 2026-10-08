from uuid import uuid4

from app.contracts.conversation import Conversation, ConversationMessage
from app.contracts.runtime import (
    AgentRequest,
    AgentResponse,
    ConversationStore,
    GraphRunner,
)
from app.memory.ltm.extractor import LTMExtractor
from app.memory.ltm.manager import LTMManager
from app.memory.stm.models import STMMessage
from app.memory.stm.writer import STMWriter
from app.observability.tracker import ObservabilityTracker
from app.runtime.session import AgentSession


class AgentRuntime:
    def __init__(
        self,
        conversation_store: ConversationStore,
        graph: GraphRunner,
        *,
        stm_writer: STMWriter | None = None,
        ltm_extractor: LTMExtractor | None = None,
        ltm_manager: LTMManager | None = None,
        tracker: ObservabilityTracker | None = None,
    ) -> None:
        self.conversation_store = conversation_store
        self.graph = graph
        self.stm_writer = stm_writer
        self.ltm_extractor = ltm_extractor
        self.ltm_manager = ltm_manager
        self.tracker = tracker or ObservabilityTracker()

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
        runtime_errors: dict[str, str] = {}
        self._index_message(user_message, runtime_errors)
        self._capture_long_term_memory(user_message, runtime_errors)

        response = self.graph.run(request, trace_id)

        assistant_message = ConversationMessage(
            user_id=session.user_id,
            thread_id=session.thread_id,
            role="assistant",
            content=response.answer,
            metadata={"trace_id": trace_id, **response.metadata},
        )

        self.conversation_store.append_message(assistant_message)
        self._index_message(assistant_message, runtime_errors)

        if runtime_errors:
            response = response.model_copy(
                update={
                    "metadata": {
                        **response.metadata,
                        "runtime_errors": runtime_errors,
                    }
                }
            )

        return response

    def _index_message(
        self,
        message: ConversationMessage,
        runtime_errors: dict[str, str],
    ) -> None:
        if self.stm_writer is None:
            return

        try:
            with self.tracker.track_latency("stm_indexing"):
                self.stm_writer.write(
                    STMMessage(
                        message_id=message.message_id,
                        user_id=message.user_id,
                        thread_id=message.thread_id,
                        role=message.role,
                        content=message.content,
                        timestamp=message.timestamp,
                        metadata=message.metadata,
                    )
                )
        except Exception as exc:
            runtime_errors["stm_indexing"] = str(exc)

    def _capture_long_term_memory(
        self,
        user_message: ConversationMessage,
        runtime_errors: dict[str, str],
    ) -> None:
        if self.ltm_extractor is None or self.ltm_manager is None:
            return

        conversation = Conversation(
            user_id=user_message.user_id,
            thread_id=user_message.thread_id,
            messages=[user_message],
        )
        try:
            with self.tracker.track_latency("ltm_extraction"):
                candidates = self.ltm_extractor.extract(conversation)
        except Exception as exc:
            runtime_errors["ltm_extraction"] = str(exc)
            return

        for candidate in candidates:
            try:
                with self.tracker.track_latency("ltm_consolidation"):
                    self.ltm_manager.process(candidate)
            except Exception as exc:
                runtime_errors["ltm_consolidation"] = str(exc)