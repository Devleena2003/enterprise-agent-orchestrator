import httpx

from agent_orchestrator.clients.rag_client import RagEngineClient
from agent_orchestrator.config import Settings
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
from agent_orchestrator.tools.table_analyzer import analyze_markdown_tables

_INSURANCE_TABLE_MD = """| Tier | Deductible | Annual Limit |
| --- | --- | --- |
| Standard | $1,000 | $50,000 |
| Platinum | $250 | $250,000 |"""

_REGULATION_TABLE_MD = """| Violation Class | Minimum Fine | Maximum Penalty |
| --- | --- | --- |
| Class A | €500 | €2,000 |
| Class B | €2,500 | €10,000 |"""

_LEDGER_TABLE_MD = """| Txn Date | Description | Debit | Credit | Balance |
| --- | --- | --- | --- | --- |
| 01-08-23 | Wire In | - | ₹10,000.00 | ₹15,000.00 |
| 01-08-23 | Fee | ₹500.00 | - | ₹14,500.00 |
| 02-08-23 | Purchase | ₹1,000.00 | - | ₹13,500.00 |"""


def _state(**overrides) -> AgentState:
    base: AgentState = {
        "thread_id": "t1",
        "messages": [],
        "user_query": "Convert Platinum Annual Limit to EUR",
        "doc_type": None,
        "plan": {
            "intent": "multi_step_reasoning",
            "rewritten_query": "Platinum Annual Limit EUR",
            "doc_type_filter": None,
            "needs_retrieval": True,
            "needs_table_analysis": True,
            "needs_external_fx": True,
            "fx_base_currency": None,
            "fx_target_currency": "EUR",
            "requires_human_approval": False,
        },
        "step_count": 1,
        "max_steps": 6,
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
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def test_table_analyzer_works_for_insurance_regulation_and_financial_pdfs() -> None:
    # 1. Insurance policy table (keyword filter "Platinum")
    ins = analyze_markdown_tables(
        [{"chunk_type": "table", "content": _INSURANCE_TABLE_MD}],
        "What is the Platinum deductible and annual limit?",
    )
    assert ins["matched_row_count"] == 1
    assert ins["column_stats"]["Deductible"]["first"] == 250.0
    assert ins["column_stats"]["Annual Limit"]["last"] == 250000.0
    assert ins["inferred_currency"] == "USD"

    # 2. Government regulation table (full-table summary)
    reg = analyze_markdown_tables(
        [{"chunk_type": "table", "content": _REGULATION_TABLE_MD}],
        "Summarize all penalty caps",
    )
    assert reg["matched_row_count"] == 2
    assert reg["column_stats"]["Maximum Penalty"]["sum"] == 12000.0
    assert reg["inferred_currency"] == "EUR"

    # 3. Date-filtered financial table
    fin = analyze_markdown_tables(
        [{"chunk_type": "table", "content": _LEDGER_TABLE_MD}],
        "How many rows on 1st August 2023 and what is the final balance?",
    )
    assert fin["matched_row_count"] == 2
    assert fin["column_stats"]["Balance"]["last"] == 14500.0
    assert fin["inferred_currency"] == "INR"


def test_fx_client_live_success_and_timeout_fallback() -> None:
    live_client = FxClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"date": "2026-11-14", "rates": {"EUR": 230000.0}}
            )
        )
    )
    ok = live_client.convert(amount=250000.0, base_currency="USD", target_currency="EUR")
    assert ok.is_fallback is False
    assert ok.converted_amount == 230000.0

    def fail_timeout(req):
        raise httpx.ReadTimeout("timeout")

    fallback_client = FxClient(
        max_retries=0, transport=httpx.MockTransport(fail_timeout)
    )
    fb = fallback_client.convert(amount=1000.0, base_currency="USD", target_currency="EUR")
    assert fb.is_fallback is True
    assert fb.converted_amount > 0


def test_retrieve_node_handles_upstream_failure_gracefully() -> None:
    rag = RagEngineClient(
        base_url="http://down",
        max_retries=0,
        transport=httpx.MockTransport(lambda req: httpx.Response(503)),
    )
    out = retrieve_evidence_node(_state(), rag)
    assert out["sufficient_evidence"] is False
    assert out["error"] is not None
    assert out["trace"][0]["status"] == "error"


def test_multi_step_insurance_table_to_fx_and_routing_guards() -> None:
    s1 = _state(
        retrieved_evidence=[{"chunk_type": "table", "content": _INSURANCE_TABLE_MD}],
        sufficient_evidence=True,
    )
    assert route_after_planner(s1) == "retrieve"
    assert route_after_retrieve(s1) == "analyze_table"

    tbl_out = analyze_table_node(s1)
    s2 = {**s1, **tbl_out}
    assert route_after_table(s2) == "external_fx"

    fx_client = FxClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"date": "2026-11-14", "rates": {"EUR": 231480.0}})
        )
    )
    fx_out = external_fx_node(s2, fx_client, Settings())
    assert fx_out["tool_outputs"][-1]["amount"] == 250000.0
    assert fx_out["tool_outputs"][-1]["base_currency"] == "USD"

    s_no_ev = _state(sufficient_evidence=False)
    assert route_after_retrieve(s_no_ev) == "synthesize"

    s_loop = _state(step_count=6, max_steps=6)
    assert route_after_planner(s_loop) == "synthesize"
    assert route_after_retrieve(s_loop) == "synthesize"