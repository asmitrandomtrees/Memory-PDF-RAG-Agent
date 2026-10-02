from app.runtime.agent import AgentRuntime
from app.runtime.factory import create_runtime as _create_runtime


def create_runtime() -> AgentRuntime:
    return _create_runtime()
