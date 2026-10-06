from app.contracts.runtime import (
    AgentRequest,
    AgentResponse,
)


class Phase0Graph:
    def run(
        self,
        request: AgentRequest,
        trace_id: str,
    ) -> AgentResponse:
        return AgentResponse(
            user_id=request.user_id,
            thread_id=request.thread_id,
            answer=(
                "Phase 0 skeleton is working. "
                "The actual agent graph has not been implemented yet."
            ),
            trace_id=trace_id,
        )