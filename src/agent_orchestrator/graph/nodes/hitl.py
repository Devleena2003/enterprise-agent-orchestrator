import time
from typing import Any
from langgraph.types import interrupt

from agent_orchestrator.graph.state import AgentState
from agent_orchestrator.schemas.agent import TraceStep


def hitl_approval_node(state: AgentState) -> dict[str, Any]:
    """
    Pause graph execution via LangGraph interrupt() for human review of sensitive actions.
    When resumed with Command(resume={"approved": bool, "reviewer": str, "note": str}),
    either executes the governed action record or cancels safely.
    """
    plan = state.get("plan") or {}
    approval_payload = {
        "action": plan.get("rewritten_query") or state["user_query"],
        "intent": plan.get("intent", "sensitive_action"),
        "plan_steps": plan.get("plan_steps", []),
        "reasoning": plan.get("reasoning", "Sensitive action requires human approval."),
    }

    # Execution pauses here on first pass; returns resume dict on second pass
    decision = interrupt(approval_payload)

    start = time.perf_counter()
    approved = bool(isinstance(decision, dict) and decision.get("approved"))
    reviewer = (
        str(decision.get("reviewer", "human_operator"))
        if isinstance(decision, dict)
        else "human_operator"
    )
    note = (
        str(decision.get("note", "")) if isinstance(decision, dict) else ""
    )

    if approved:
        action_result = {
            "tool": "governed_action_executor",
            "executed": True,
            "action": approval_payload["action"],
            "approved_by": reviewer,
            "reviewer_note": note,
        }
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        step = TraceStep(
            node="hitl_approval",
            status="ok",
            duration_ms=duration_ms,
            detail=f"Approved by {reviewer}; governed action executed.",
        ).model_dump()
        return {
            "approval_status": "approved",
            "approval_request": approval_payload,
            "tool_outputs": [*state.get("tool_outputs", []), action_result],
            "step_count": state.get("step_count", 0) + 1,
            "trace": [*state.get("trace", []), step],
        }

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    step = TraceStep(
        node="hitl_approval",
        status="paused",
        duration_ms=duration_ms,
        detail=f"Rejected by {reviewer}; action cancelled.",
    ).model_dump()
    return {
        "approval_status": "rejected",
        "approval_request": approval_payload,
        "step_count": state.get("step_count", 0) + 1,
        "trace": [*state.get("trace", []), step],
    }