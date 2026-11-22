import json
import logging
import time
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent_orchestrator.graph.state import (
    AgentState,
    extract_token_usage,
    merge_token_usage,
)
from agent_orchestrator.schemas.agent import TraceStep

logger = logging.getLogger(__name__)

_SYNTH_SYSTEM_PROMPT = """You are an enterprise AI assistant that produces strictly grounded, accurate answers.
Rules:
1. Use ONLY the provided Document Context and Tool Outputs. Never invent facts, numbers, or citations.
2. When citing document facts, preserve the exact inline citation markers like [1], [2].
3. When 'table_analyzer' tool output is present, trust its exact row counts and column sums/values instead of recounting manually.
4. When 'external_fx' has is_fallback=true, explicitly tell the user the exchange rate is an approximate fallback because the live FX API was unavailable.
5. Keep the answer concise, clear, and structured."""


def synthesize_node(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    """Generate final grounded response or deterministic refusal without fabricating facts."""
    start = time.perf_counter()
    plan = state.get("plan") or {}
    intent = plan.get("intent", "retrieval")
    approval_status = state.get("approval_status", "not_required")
    tool_outputs = state.get("tool_outputs", [])
    constructed_context = state.get("constructed_context", "")

    # 1. Deterministic HITL rejection
    if approval_status == "rejected":
        answer = (
            "The requested sensitive action was rejected by the human reviewer "
            "and was not executed."
        )
        return _finish(state, answer, [], start, "ok", "HITL rejected deterministically")

    # 2. Deterministic HITL approval summary
    if approval_status == "approved":
        exec_out = next(
            (t for t in reversed(tool_outputs) if t.get("tool") == "governed_action_executor"),
            {},
        )
        answer = (
            f"Action approved by '{exec_out.get('approved_by', 'reviewer')}' "
            f"and executed: {exec_out.get('action', state['user_query'])}."
        )
        return _finish(state, answer, [], start, "ok", "HITL approved and executed")

    # 3. Deterministic upstream error refusal (never guess when Project 1 failed)
    if state.get("error"):
        answer = (
            "Unable to complete this request because an upstream service is unavailable: "
            f"{state['error']}"
        )
        return _finish(state, answer, [], start, "error", "Upstream error refusal")

    # 4. Deterministic insufficient-evidence refusal (0 LLM hallucination risk)
    if plan.get("needs_retrieval") and not state.get("sufficient_evidence", False):
        answer = (
            "The available documents do not contain sufficient relevant information "
            "to answer this query."
        )
        return _finish(
            state, answer, [], start, "ok", "Insufficient evidence refusal (no LLM call)"
        )

    # 5. Out-of-scope handling
    if intent == "out_of_scope":
        answer = (
            "I can help search uploaded enterprise documents, analyze tables/metrics, "
            "convert currencies, or queue governed actions for approval. "
            "Your request appears outside these capabilities."
        )
        return _finish(state, answer, [], start, "ok", "Out-of-scope handled")

    # 6. Grounded LLM Synthesis over retrieved context + tool outputs
    user_prompt = (
        f"User Question: {state['user_query']}\n\n"
        f"Tool Outputs (deterministic & live API results):\n"
        f"{json.dumps(tool_outputs, indent=2) if tool_outputs else 'None'}\n\n"
        f"Document Context (with mandatory citation IDs):\n"
        f"{constructed_context or 'None (external tool query only)'}"
    )

    token_delta = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    status = "ok"
    detail = "Synthesized grounded response"

    try:
        ai_msg = llm.invoke(
            [
                SystemMessage(content=_SYNTH_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        token_delta = extract_token_usage(ai_msg)
        answer = str(ai_msg.content).strip()
        if not answer:
            raise ValueError("Empty LLM synthesis response")
    except Exception as exc:
        logger.warning("Synthesizer LLM call failed (%s); using grounded fallback.", exc)
        status = "fallback"
        detail = f"LLM synthesis failed ({type(exc).__name__}); returned structured fallback"
        answer = _build_deterministic_fallback_answer(tool_outputs, constructed_context)

    citations = state.get("citations", []) if state.get("sufficient_evidence") else []
    return _finish(state, answer, citations, start, status, detail, token_delta)


def _build_deterministic_fallback_answer(
    tool_outputs: list[dict[str, Any]], constructed_context: str
) -> str:
    parts = ["Grounded summary (LLM synthesizer unavailable; returning direct evidence):"]
    if tool_outputs:
        parts.append(f"Tool outputs: {json.dumps(tool_outputs)}")
    if constructed_context:
        parts.append(f"Retrieved context:\n{constructed_context[:1500]}")
    return "\n\n".join(parts)


def _finish(
    state: AgentState,
    answer: str,
    citations: list[dict[str, Any]],
    start_time: float,
    status: str,
    detail: str,
    token_delta: dict[str, int] | None = None,
) -> dict[str, Any]:
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    step = TraceStep(
        node="synthesize",
        status=status,  # type: ignore[arg-type]
        duration_ms=duration_ms,
        detail=detail,
    ).model_dump()

    return {
        "final_answer": answer,
        "citations": citations,
        "messages": [AIMessage(content=answer)],
        "step_count": state.get("step_count", 0) + 1,
        "trace": [*state.get("trace", []), step],
        "token_usage": merge_token_usage(
            state.get("token_usage"),
            token_delta or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        ),
    }