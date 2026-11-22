import sqlite3
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agent_orchestrator.clients.rag_client import RagEngineClient
from agent_orchestrator.config import Settings
from agent_orchestrator.graph.nodes.hitl import hitl_approval_node
from agent_orchestrator.graph.nodes.planner import understand_and_plan_node
from agent_orchestrator.graph.nodes.synthesizer import synthesize_node
from agent_orchestrator.graph.nodes.tool_nodes import (
    analyze_table_node,
    external_fx_node,
    retrieve_evidence_node,
)
from agent_orchestrator.graph.routing import (
    route_after_planner,
    route_after_retrieve,
    route_after_table,
)
from agent_orchestrator.graph.state import AgentState
from agent_orchestrator.tools.fx_client import FxClient


def create_sqlite_checkpointer(db_path: str) -> SqliteSaver:
    """Create a persistent SQLite checkpointer safe for FastAPI worker threads."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


def new_turn_input(
    *,
    thread_id: str,
    user_query: str,
    doc_type: str | None = None,
    max_steps: int = 6,
) -> dict[str, Any]:
    """
    Build clean per-turn graph input while preserving prior `messages` in the thread
    via LangGraph's `add_messages` reducer.
    """
    return {
        "thread_id": thread_id,
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "doc_type": doc_type,
        "plan": None,
        "step_count": 0,
        "max_steps": max_steps,
        "retrieved_evidence": [],
        "constructed_context": "",
        "sufficient_evidence": False,
        "tool_outputs": [],
        "approval_status": "not_required",
        "approval_request": None,
        "final_answer": "",
        "citations": [],
        "trace": [],
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "error": None,
    }


def build_agent_graph(
    *,
    llm: BaseChatModel,
    rag_client: RagEngineClient,
    fx_client: FxClient,
    settings: Settings,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Assemble and compile the complete stateful LangGraph agent."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", lambda state: understand_and_plan_node(state, llm))
    graph.add_node("retrieve", lambda state: retrieve_evidence_node(state, rag_client))
    graph.add_node("analyze_table", analyze_table_node)
    graph.add_node(
        "external_fx", lambda state: external_fx_node(state, fx_client, settings)
    )
    graph.add_node("hitl_approval", hitl_approval_node)
    graph.add_node("synthesize", lambda state: synthesize_node(state, llm))

    graph.add_edge(START, "planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retrieve": "retrieve",
            "external_fx": "external_fx",
            "hitl_approval": "hitl_approval",
            "synthesize": "synthesize",
        },
    )

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "analyze_table": "analyze_table",
            "external_fx": "external_fx",
            "synthesize": "synthesize",
        },
    )

    graph.add_conditional_edges(
        "analyze_table",
        route_after_table,
        {
            "external_fx": "external_fx",
            "synthesize": "synthesize",
        },
    )

    graph.add_edge("external_fx", "synthesize")
    graph.add_edge("hitl_approval", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer)