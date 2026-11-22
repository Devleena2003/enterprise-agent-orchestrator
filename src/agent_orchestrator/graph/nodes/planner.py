import logging
import time
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent_orchestrator.graph.state import (
    AgentState,
    extract_token_usage,
    merge_token_usage,
)
from agent_orchestrator.schemas.agent import IntentAndPlan, TraceStep

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = """You are an enterprise AI task planner.
Analyze the user's latest request along with any prior conversation history.
Your job is to:
1. Rewrite the query so it is completely self-contained (resolve pronouns like 'it', 'that date', 'the next day').
2. Classify the intent accurately:
   - 'retrieval': single-step question answered from uploaded PDFs/documents.
   - 'multi_step_reasoning': requires document retrieval PLUS counting rows, calculations, or live FX currency conversion.
   - 'external_tool': pure live currency conversion / FX rate question needing NO internal documents.
   - 'sensitive_action': user asks to delete a document, modify records, or send/export an alert, email, or compliance report externally.
   - 'out_of_scope': greetings or requests unrelated to documents, math, or supported tools.
3. Produce 1 to 5 concise execution plan steps.
Be conservative: if an action deletes data or dispatches external notifications/reports, ALWAYS classify as 'sensitive_action' with requires_human_approval=true."""

_SENSITIVE_KEYWORDS = ("delete ", "remove document", "send email", "notify ", "dispatch ", "webhook")


def _format_recent_history(messages: list[Any], max_turns: int = 6) -> str:
    if not messages:
        return "None"
    recent = messages[-max_turns:]
    lines = []
    for msg in recent:
        role = getattr(msg, "type", "user")
        content = str(getattr(msg, "content", ""))[:300]
        lines.append(f"- {role}: {content}")
    return "\n".join(lines)


def _deterministic_fallback_plan(query: str, doc_type: str | None) -> IntentAndPlan:
    """Safe fallback if the LLM rate-limits or returns unparseable structured output."""
    low = query.lower()
    if any(kw in low for kw in _SENSITIVE_KEYWORDS):
        return IntentAndPlan(
            intent="sensitive_action",
            rewritten_query=query,
            doc_type_filter=doc_type,
            needs_retrieval=False,
            requires_human_approval=True,
            plan_steps=["Request human approval before executing sensitive action"],
            reasoning="Fallback heuristic detected a sensitive action keyword.",
        )

    return IntentAndPlan(
        intent="retrieval",
        rewritten_query=query,
        doc_type_filter=doc_type,
        needs_retrieval=True,
        requires_human_approval=False,
        plan_steps=["Retrieve grounded evidence from Project 1"],
        reasoning="Fallback heuristic defaulted to safe grounded retrieval.",
    )


def _apply_safety_invariants(
    plan: IntentAndPlan, user_doc_type: str | None
) -> IntentAndPlan:
    """Enforce mandatory invariants regardless of model quirks."""
    if user_doc_type and not plan.doc_type_filter:
        plan.doc_type_filter = user_doc_type

    if plan.intent == "sensitive_action":
        plan.requires_human_approval = True
    elif plan.requires_human_approval and plan.intent != "sensitive_action":
        plan.intent = "sensitive_action"

    if plan.intent in ("retrieval", "multi_step_reasoning"):
        plan.needs_retrieval = True
    if plan.intent == "external_tool":
        plan.needs_retrieval = False
        plan.needs_external_fx = True

    return plan


def understand_and_plan_node(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    """LangGraph node: classifies intent, resolves context, and emits an execution plan."""
    start = time.perf_counter()
    query = state["user_query"].strip()
    history_text = _format_recent_history(state.get("messages", []))
    user_doc_type = state.get("doc_type")

    prompt_text = (
        f"Conversation History:\n{history_text}\n\n"
        f"Explicit doc_type filter from caller: {user_doc_type or 'None'}\n"
        f"Latest User Query: {query}"
    )

    token_delta = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    status: str = "ok"
    detail: str = ""

    try:
        structured_llm = llm.with_structured_output(IntentAndPlan, include_raw=True)
        result = structured_llm.invoke(
            [
                SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=prompt_text),
            ]
        )
        raw_msg = result.get("raw")
        if raw_msg is not None:
            token_delta = extract_token_usage(raw_msg)

        parsed = result.get("parsed")
        if isinstance(parsed, IntentAndPlan):
            plan = _apply_safety_invariants(parsed, user_doc_type)
            detail = f"intent={plan.intent}, rewritten_query={plan.rewritten_query!r}"
        else:
            parse_err = result.get("parsing_error")
            logger.warning("Planner structured parse failed (%s); using fallback.", parse_err)
            plan = _deterministic_fallback_plan(query, user_doc_type)
            status = "fallback"
            detail = f"Structured output parse failed; fallback intent={plan.intent}"
    except Exception as exc:
        logger.warning("Planner LLM call failed (%s); using deterministic fallback.", exc)
        plan = _deterministic_fallback_plan(query, user_doc_type)
        status = "fallback"
        detail = f"LLM error ({type(exc).__name__}); fallback intent={plan.intent}"

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    step = TraceStep(
        node="planner",
        status=status,  # type: ignore[arg-type]
        duration_ms=duration_ms,
        detail=detail,
    ).model_dump()

    return {
        "plan": plan.model_dump(),
        "step_count": state.get("step_count", 0) + 1,
        "approval_status": "pending" if plan.requires_human_approval else "not_required",
        "trace": [*state.get("trace", []), step],
        "token_usage": merge_token_usage(state.get("token_usage"), token_delta),
    }