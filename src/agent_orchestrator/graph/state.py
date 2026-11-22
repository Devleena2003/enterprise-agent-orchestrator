from typing import Annotated, Any, Literal, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared stateful memory passed between all LangGraph nodes."""

    thread_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    doc_type: str | None

    # Planner outputs
    plan: dict[str, Any] | None

    # Execution loop & safety caps
    step_count: int
    max_steps: int

    # Evidence & tool outputs
    retrieved_evidence: list[dict[str, Any]]
    constructed_context: str
    sufficient_evidence: bool
    tool_outputs: list[dict[str, Any]]

    # Human-in-the-loop (HITL)
    approval_status: Literal["not_required", "pending", "approved", "rejected"]
    approval_request: dict[str, Any] | None

    # Synthesis & Observability
    final_answer: str
    citations: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    token_usage: dict[str, int]
    error: str | None


def extract_token_usage(message: Any) -> dict[str, int]:
    """Safely extract token counts across Groq, Gemini, Ollama, and test stubs."""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        prompt = int(usage.get("input_tokens", 0) or 0)
        completion = int(usage.get("output_tokens", 0) or 0)
        total = int(usage.get("total_tokens", prompt + completion) or 0)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    resp_meta = getattr(message, "response_metadata", None) or {}
    token_usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
    if isinstance(token_usage, dict) and token_usage:
        prompt = int(token_usage.get("prompt_tokens", 0) or 0)
        completion = int(token_usage.get("completion_tokens", 0) or 0)
        total = int(token_usage.get("total_tokens", prompt + completion) or 0)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def merge_token_usage(
    existing: dict[str, int] | None, delta: dict[str, int]
) -> dict[str, int]:
    base = existing or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": base.get("prompt_tokens", 0) + delta.get("prompt_tokens", 0),
        "completion_tokens": base.get("completion_tokens", 0)
        + delta.get("completion_tokens", 0),
        "total_tokens": base.get("total_tokens", 0) + delta.get("total_tokens", 0),
    }