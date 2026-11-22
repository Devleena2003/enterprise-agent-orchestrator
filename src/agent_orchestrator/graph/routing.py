from agent_orchestrator.graph.state import AgentState


def route_after_planner(state: AgentState) -> str:
    if state.get("step_count", 0) >= state.get("max_steps", 6):
        return "synthesize"
    if state.get("approval_status") == "pending":
        return "hitl_approval"

    plan = state.get("plan") or {}
    intent = plan.get("intent", "retrieval")

    if intent == "out_of_scope":
        return "synthesize"
    if plan.get("needs_retrieval"):
        return "retrieve"
    if plan.get("needs_external_fx"):
        return "external_fx"
    return "synthesize"


def route_after_retrieve(state: AgentState) -> str:
    if state.get("step_count", 0) >= state.get("max_steps", 6):
        return "synthesize"
    if state.get("error") or not state.get("sufficient_evidence", False):
        return "synthesize"

    plan = state.get("plan") or {}
    if plan.get("needs_table_analysis"):
        return "analyze_table"
    if plan.get("needs_external_fx"):
        return "external_fx"
    return "synthesize"


def route_after_table(state: AgentState) -> str:
    if state.get("step_count", 0) >= state.get("max_steps", 6):
        return "synthesize"

    plan = state.get("plan") or {}
    if plan.get("needs_external_fx"):
        return "external_fx"
    return "synthesize"