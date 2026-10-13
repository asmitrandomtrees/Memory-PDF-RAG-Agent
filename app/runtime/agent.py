from concurrent.futures import Future, ThreadPoolExecutor
import re
from threading import Lock
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


_GREETING_WORDS = {"hi", "hello", "hey", "hii"}
_ACKNOWLEDGEMENTS = {"thanks", "thank you", "thx"}
_NEUTRAL_ACKNOWLEDGEMENTS = {"ok", "okay"}
_DAY_GREETINGS = {"good morning", "good afternoon", "good evening"}
_CANNED_REPLIES = {
    "hi how can i help you today",
    "you re welcome how can i help",
}
_DURABLE_PROFILE_PATTERNS = (
    re.compile(r"\bmy\s+\w+\s+is\b", re.IGNORECASE),
    re.compile(r"\bi\s+am\b", re.IGNORECASE),
    re.compile(r"\bi'm\b", re.IGNORECASE),
)


def is_trivial_small_talk(text: str) -> bool:
    content = text.strip()
    if not content:
        return True

    words = re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)?", content)
    if not words:
        return False

    normalized_words = [word.lower() for word in words]
    normalized = " ".join(normalized_words)

    if (
        normalized in _ACKNOWLEDGEMENTS
        or normalized in _NEUTRAL_ACKNOWLEDGEMENTS
        or normalized in _DAY_GREETINGS
        or normalized in _CANNED_REPLIES
    ):
        return True

    if normalized_words[0] in _GREETING_WORDS:
        if len(words) == 1:
            return True
        return len(words) == 2 and words[1][0].isupper()

    return False


def small_talk_reply(text: str) -> str:
    normalized = " ".join(
        word.lower()
        for word in re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)?", text)
    )
    if normalized in _ACKNOWLEDGEMENTS:
        return "You're welcome! How can I help?"
    if normalized in _NEUTRAL_ACKNOWLEDGEMENTS:
        return "Okay."
    return "Hi! How can I help you today?"


class AgentTimingSink:
    def emit(
        self,
        *,
        trace_id: str,
        user_id: str,
        thread_id: str,
        timings: dict[str, dict[str, float | int]],
    ) -> None:
        ...


class AgentRuntime:
    _memory_executor = ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="agent-memory",
    )

    def __init__(
        self,
        conversation_store: ConversationStore,
        graph: GraphRunner,
        *,
        stm_writer: STMWriter | None = None,
        ltm_extractor: LTMExtractor | None = None,
        ltm_manager: LTMManager | None = None,
        tracker: ObservabilityTracker | None = None,
        timing_sink: AgentTimingSink | None = None,
        background_memory_writes: bool = True,
    ) -> None:
        self.conversation_store = conversation_store
        self.graph = graph
        self.stm_writer = stm_writer
        self.ltm_extractor = ltm_extractor
        self.ltm_manager = ltm_manager
        self.tracker = tracker or ObservabilityTracker()
        self.timing_sink = timing_sink
        self.background_memory_writes = background_memory_writes
        self._background_futures: list[Future[None]] = []
        self._background_futures_lock = Lock()

    def handle(self, request: AgentRequest) -> AgentResponse:
        with self.tracker.collect_request_timings() as timings:
            with self.tracker.track_latency("agent.total"):
                response = self._handle(request)
        timing_error = None
        if self.timing_sink is not None:
            try:
                self.timing_sink.emit(
                    trace_id=response.trace_id,
                    user_id=request.user_id,
                    thread_id=request.thread_id,
                    timings=timings,
                )
            except Exception as exc:
                timing_error = str(exc)
        metadata = {
            **response.metadata,
            "agent_timings": timings,
        }
        if timing_error is not None:
            metadata["agent_timing_export_error"] = timing_error
        return response.model_copy(
            update={"metadata": metadata}
        )

    def _handle(self, request: AgentRequest) -> AgentResponse:
        trace_id = str(uuid4())

        with self.tracker.track_latency("agent.conversation_load"):
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

        with self.tracker.track_latency("agent.persist_user_message"):
            self.conversation_store.append_message(user_message)

        runtime_errors: dict[str, str] = {}
        if is_trivial_small_talk(request.message):
            return self._handle_small_talk(
                request=request,
                trace_id=trace_id,
                session=session,
                runtime_errors=runtime_errors,
            )

        with self.tracker.track_latency("agent.graph_execution"):
            response = self.graph.run(request, trace_id)

        assistant_message = ConversationMessage(
            user_id=session.user_id,
            thread_id=session.thread_id,
            role="assistant",
            content=response.answer,
            metadata={"trace_id": trace_id, **response.metadata},
        )

        with self.tracker.track_latency("agent.persist_assistant_message"):
            self.conversation_store.append_message(assistant_message)

        self._process_memory_after_response(
            user_message=user_message,
            assistant_message=assistant_message,
            runtime_errors=runtime_errors,
        )

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

    def wait_for_background_tasks(self) -> None:
        while True:
            with self._background_futures_lock:
                futures = self._background_futures
                self._background_futures = []
            if not futures:
                return
            for future in futures:
                future.result()

    def _handle_small_talk(
        self,
        *,
        request: AgentRequest,
        trace_id: str,
        session: AgentSession,
        runtime_errors: dict[str, str],
    ) -> AgentResponse:
        response = AgentResponse(
            user_id=request.user_id,
            thread_id=request.thread_id,
            answer=small_talk_reply(request.message),
            trace_id=trace_id,
            metadata={
                "fast_path": "small_talk",
                "citations": [],
                "insufficient_evidence": False,
            },
        )
        assistant_message = ConversationMessage(
            user_id=session.user_id,
            thread_id=session.thread_id,
            role="assistant",
            content=response.answer,
            metadata={"trace_id": trace_id, **response.metadata},
        )

        with self.tracker.track_latency("agent.persist_assistant_message"):
            self.conversation_store.append_message(assistant_message)

        if runtime_errors:
            return response.model_copy(
                update={
                    "metadata": {
                        **response.metadata,
                        "runtime_errors": runtime_errors,
                    }
                }
            )
        return response

    def _process_memory_after_response(
        self,
        *,
        user_message: ConversationMessage,
        assistant_message: ConversationMessage,
        runtime_errors: dict[str, str],
    ) -> None:
        if self.background_memory_writes:
            future = self._memory_executor.submit(
                self._process_memory_messages,
                user_message,
                assistant_message,
                {},
            )
            with self._background_futures_lock:
                self._background_futures.append(future)
            return

        self._process_memory_messages(
            user_message,
            assistant_message,
            runtime_errors,
        )

    def _process_memory_messages(
        self,
        user_message: ConversationMessage,
        assistant_message: ConversationMessage,
        runtime_errors: dict[str, str],
    ) -> None:
        self._index_message(user_message, runtime_errors)
        self._capture_long_term_memory(user_message, runtime_errors)
        self._index_message(assistant_message, runtime_errors)

    def _index_message(
        self,
        message: ConversationMessage,
        runtime_errors: dict[str, str],
    ) -> None:
        if self.stm_writer is None or is_trivial_small_talk(message.content):
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

    @staticmethod
    def _should_capture_long_term_memory(
        message: ConversationMessage,
    ) -> bool:
        content = message.content.strip()
        if not content:
            return False

        if any(pattern.search(content) for pattern in _DURABLE_PROFILE_PATTERNS):
            return True

        normalized = content.lower()
        tokenized = [token for token in normalized.split() if token]
        if len(tokenized) < 6:
            return False

        if normalized in {
            "hello",
            "hi",
            "hey",
            "thanks",
            "thank you",
            "ok",
            "okay",
        }:
            return False

        trivial_tokens = {
            "yes",
            "no",
            "ok",
            "okay",
            "thanks",
            "thank",
            "hi",
            "hello",
            "hey",
        }
        if all(token in trivial_tokens for token in tokenized):
            return False

        return True

    def _capture_long_term_memory(
        self,
        user_message: ConversationMessage,
        runtime_errors: dict[str, str],
    ) -> None:
        if self.ltm_extractor is None or self.ltm_manager is None:
            return

        if not self._should_capture_long_term_memory(user_message):
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
