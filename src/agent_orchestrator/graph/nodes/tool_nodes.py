import time
from typing import Any

from agent_orchestrator.clients.rag_client import RagEngineClient
from agent_orchestrator.config import Settings
from agent_orchestrator.exceptions import UpstreamServiceError
from agent_orchestrator.graph.state import AgentState
from agent_orchestrator.schemas.agent import TraceStep
from agent_orchestrator.tools.fx_client import FxClient, extract_amount_from_query
from agent_orchestrator.tools.table_analyzer import analyze_markdown_tables


def retrieve_evidence_node(
    state: AgentState, rag_client: RagEngineClient
) -> dict[str, Any]:
    """Call Project 1 public API with fault isolation."""
    start = time.perf_counter()
    plan = state.get("plan") or {}
    query = plan.get("rewritten_query") or state["user_query"]
    doc_type = state.get("doc_type") or plan.get("doc_type_filter")

    try:
        res = rag_client.retrieve(query=query, top_k=5, doc_type=doc_type)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        evidences = [item.model_dump() for item in res.results]
        step = TraceStep(
            node="retrieve",
            status="ok",
            duration_ms=duration_ms,
            detail=f"sufficient_evidence={res.sufficient_evidence}, chunks={len(evidences)}",
        ).model_dump()

        return {
            "retrieved_evidence": evidences,
            "constructed_context": res.constructed_context,
            "sufficient_evidence": res.sufficient_evidence,
            "citations": evidences,
            "step_count": state.get("step_count", 0) + 1,
            "trace": [*state.get("trace", []), step],
        }
    except UpstreamServiceError as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        step = TraceStep(
            node="retrieve",
            status="error",
            duration_ms=duration_ms,
            detail=exc.message,
        ).model_dump()
        return {
            "retrieved_evidence": [],
            "constructed_context": "",
            "sufficient_evidence": False,
            "error": exc.message,
            "step_count": state.get("step_count", 0) + 1,
            "trace": [*state.get("trace", []), step],
        }


def analyze_table_node(state: AgentState) -> dict[str, Any]:
    """Deterministically analyze any retrieved Markdown tables."""
    start = time.perf_counter()
    plan = state.get("plan") or {}
    query = plan.get("rewritten_query") or state["user_query"]
    analysis = analyze_markdown_tables(state.get("retrieved_evidence", []), query)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    step = TraceStep(
        node="analyze_table",
        status="ok",
        duration_ms=duration_ms,
        detail=(
            f"matched_rows={analysis['matched_row_count']}, "
            f"numeric_cols={list(analysis['column_stats'].keys())}"
        ),
    ).model_dump()

    return {
        "tool_outputs": [*state.get("tool_outputs", []), analysis],
        "step_count": state.get("step_count", 0) + 1,
        "trace": [*state.get("trace", []), step],
    }


def external_fx_node(
    state: AgentState, fx_client: FxClient, settings: Settings
) -> dict[str, Any]:
    """Convert an amount from query or table_analyzer output using live FX API + fallback."""
    start = time.perf_counter()
    plan = state.get("plan") or {}
    tool_outputs = state.get("tool_outputs", [])

    # Prefer explicit numeric amount in user query; otherwise use table_analyzer's primary numeric value
    amount = extract_amount_from_query(state["user_query"])
    inferred_base = None

    table_out = next(
        (t for t in reversed(tool_outputs) if t.get("tool") == "table_analyzer"),
        None,
    )
    if table_out:
        inferred_base = table_out.get("inferred_currency")
        if amount is None:
            amount = table_out.get("primary_numeric_value")

    if amount is None:
        amount = 1.0

    base_currency = (
        plan.get("fx_base_currency") or inferred_base or settings.default_currency
    )
    target_currency = plan.get("fx_target_currency") or "USD"

    fx_res = fx_client.convert(
        amount=float(amount),
        base_currency=base_currency,
        target_currency=target_currency,
    )
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    step = TraceStep(
        node="external_fx",
        status="fallback" if fx_res.is_fallback else "ok",
        duration_ms=duration_ms,
        detail=f"{fx_res.amount} {fx_res.base_currency} -> {fx_res.converted_amount} {fx_res.target_currency} ({fx_res.source})",
    ).model_dump()

    return {
        "tool_outputs": [*tool_outputs, fx_res.model_dump()],
        "step_count": state.get("step_count", 0) + 1,
        "trace": [*state.get("trace", []), step],
    }