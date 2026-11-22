from pathlib import Path
import httpx
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent_orchestrator.clients.rag_client import RagEngineClient
from agent_orchestrator.config import Settings
from agent_orchestrator.graph.workflow import (
    build_agent_graph,
    create_sqlite_checkpointer,
    new_turn_input,
)
from agent_orchestrator.schemas.agent import IntentAndPlan
from agent_orchestrator.tools.fx_client import FxClient

_INSURANCE_MD = """| Tier | Deductible | Annual Limit |
| --- | --- | --- |
| Standard | $1,000 | $50,000 |
| Platinum | $250 | $250,000 |"""


class _StubLLM:
    def __init__(self, plans: list[IntentAndPlan], synth_text: str = "Grounded answer [1].") -> None:
        self._plans = list(plans)
        self._synth_text = synth_text
        self.planner_prompts: list[str] = []

    def with_structured_output(self, schema, include_raw: bool = False):
        outer = self

        class _Runner:
            def invoke(self, messages):
                outer.planner_prompts.append(messages[1].content)
                plan = outer._plans.pop(0)
                raw = AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 50, "output_tokens": 25, "total_tokens": 75},
                )
                return {"raw": raw, "parsed": plan, "parsing_error": None}

        return _Runner()

    def invoke(self, messages):
        return AIMessage(
            content=self._synth_text,
            usage_metadata={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
        )


def test_full_multi_step_workflow_execution() -> None:
    plan = IntentAndPlan(
        intent="multi_step_reasoning",
        rewritten_query="Platinum Annual Limit in EUR",
        needs_retrieval=True,
        needs_table_analysis=True,
        needs_external_fx=True,
        fx_base_currency="USD",
        fx_target_currency="EUR",
        plan_steps=["Retrieve table", "Extract limit", "Convert USD to EUR"],
        reasoning="Needs retrieval, table extraction, and FX.",
    )
    llm = _StubLLM([plan], synth_text="The Platinum annual limit is $250,000 (~€231,480) [1].")

    rag = RagEngineClient(
        base_url="http://rag",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "query": "Platinum Annual Limit in EUR",
                    "sufficient_evidence": True,
                    "message": "ok",
                    "reranker_used": False,
                    "total_candidates_evaluated": 1,
                    "results": [
                        {
                            "citation_id": "[1]",
                            "chunk_id": "c1",
                            "document_id": "d1",
                            "document_name": "policy.pdf",
                            "doc_type": "insurance_policy",
                            "page_number": 2,
                            "chunk_type": "table",
                            "content": _INSURANCE_MD,
                            "relevance_score": 0.91,
                        }
                    ],
                    "constructed_context": f"[1] Document: policy.pdf | Page: 2\n{_INSURANCE_MD}",
                },
            )
        ),
    )
    fx = FxClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"date": "2026-11-14", "rates": {"EUR": 231480.0}})
        )
    )

    graph = build_agent_graph(
        llm=llm,  # type: ignore[arg-type]
        rag_client=rag,
        fx_client=fx,
        settings=Settings(),
        checkpointer=MemorySaver(),
    )

    out = graph.invoke(
        new_turn_input(thread_id="t_multi", user_query="What is Platinum Annual Limit in EUR?"),
        config={"configurable": {"thread_id": "t_multi"}},
    )

    assert [t["node"] for t in out["trace"]] == [
        "planner",
        "retrieve",
        "analyze_table",
        "external_fx",
        "synthesize",
    ]
    assert out["token_usage"]["total_tokens"] == 205
    assert len(out["citations"]) == 1
    assert "231,480" in out["final_answer"]


def test_workflow_refuses_without_calling_synth_llm_when_no_evidence() -> None:
    plan = IntentAndPlan(
        intent="retrieval",
        rewritten_query="Mars colony tax rate",
        needs_retrieval=True,
        plan_steps=["Retrieve"],
        reasoning="Document lookup.",
    )
    llm = _StubLLM([plan], synth_text="SHOULD NEVER BE CALLED")
    rag = RagEngineClient(
        base_url="http://rag",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "query": "Mars colony tax rate",
                    "sufficient_evidence": False,
                    "message": "insufficient",
                    "reranker_used": False,
                    "total_candidates_evaluated": 5,
                    "results": [],
                    "constructed_context": "",
                },
            )
        ),
    )
    graph = build_agent_graph(
        llm=llm,  # type: ignore[arg-type]
        rag_client=rag,
        fx_client=FxClient(),
        settings=Settings(),
        checkpointer=MemorySaver(),
    )

    out = graph.invoke(
        new_turn_input(thread_id="t_refuse", user_query="What is Mars tax rate?"),
        config={"configurable": {"thread_id": "t_refuse"}},
    )

    assert "do not contain sufficient relevant information" in out["final_answer"]
    assert out["citations"] == []
    assert out["token_usage"]["total_tokens"] == 75  # Only planner ran, synth LLM skipped


def test_hitl_interrupt_approve_and_reject_with_sqlite(tmp_path: Path) -> None:
    db_file = tmp_path / "checkpoints.db"
    checkpointer = create_sqlite_checkpointer(str(db_file))
    try:
        sensitive_plan = IntentAndPlan(
            intent="sensitive_action",
            rewritten_query="Send compliance breach alert to regulator",
            needs_retrieval=False,
            requires_human_approval=True,
            plan_steps=["Await approval", "Dispatch alert"],
            reasoning="External dispatch requires approval.",
        )
        llm = _StubLLM([sensitive_plan, sensitive_plan])
        graph = build_agent_graph(
            llm=llm,  # type: ignore[arg-type]
            rag_client=RagEngineClient(base_url="http://unused"),
            fx_client=FxClient(),
            settings=Settings(),
            checkpointer=checkpointer,
        )

        # 1. Approve flow
        cfg_approve = {"configurable": {"thread_id": "t_hitl_approve"}}
        paused = graph.invoke(
            new_turn_input(thread_id="t_hitl_approve", user_query="Send compliance alert"),
            config=cfg_approve,
        )
        assert "__interrupt__" in paused
        assert paused["approval_status"] == "pending"

        resumed = graph.invoke(
            Command(resume={"approved": True, "reviewer": "compliance_lead"}),
            config=cfg_approve,
        )
        assert resumed["approval_status"] == "approved"
        assert "Action approved by 'compliance_lead'" in resumed["final_answer"]

        # 2. Reject flow
        cfg_reject = {"configurable": {"thread_id": "t_hitl_reject"}}
        paused_2 = graph.invoke(
            new_turn_input(thread_id="t_hitl_reject", user_query="Send compliance alert"),
            config=cfg_reject,
        )
        assert "__interrupt__" in paused_2

        resumed_2 = graph.invoke(
            Command(resume={"approved": False, "reviewer": "compliance_lead"}),
            config=cfg_reject,
        )
        assert resumed_2["approval_status"] == "rejected"
        assert "rejected" in resumed_2["final_answer"].lower()
    finally:
        checkpointer.conn.close()


def test_multi_turn_thread_preserves_history_and_resets_per_turn_fields() -> None:
    p1 = IntentAndPlan(
        intent="retrieval",
        rewritten_query="Standard tier deductible",
        needs_retrieval=True,
        plan_steps=["Retrieve"],
        reasoning="Lookup.",
    )
    p2 = IntentAndPlan(
        intent="retrieval",
        rewritten_query="Platinum tier deductible",
        needs_retrieval=True,
        plan_steps=["Retrieve"],
        reasoning="Follow-up lookup.",
    )
    llm = _StubLLM([p1, p2], synth_text="Deductible info [1].")
    rag = RagEngineClient(
        base_url="http://rag",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "query": "q",
                    "sufficient_evidence": True,
                    "message": "ok",
                    "reranker_used": False,
                    "total_candidates_evaluated": 1,
                    "results": [
                        {
                            "citation_id": "[1]",
                            "chunk_id": "c1",
                            "document_id": "d1",
                            "document_name": "policy.pdf",
                            "doc_type": "insurance_policy",
                            "page_number": 1,
                            "chunk_type": "text",
                            "content": "Standard $1,000. Platinum $250.",
                            "relevance_score": 0.9,
                        }
                    ],
                    "constructed_context": "[1] Standard $1,000. Platinum $250.",
                },
            )
        ),
    )
    graph = build_agent_graph(
        llm=llm,  # type: ignore[arg-type]
        rag_client=rag,
        fx_client=FxClient(),
        settings=Settings(),
        checkpointer=MemorySaver(),
    )

    cfg = {"configurable": {"thread_id": "t_convo"}}
    out1 = graph.invoke(
        new_turn_input(thread_id="t_convo", user_query="What is Standard deductible?"),
        config=cfg,
    )
    out2 = graph.invoke(
        new_turn_input(thread_id="t_convo", user_query="And what about Platinum?"),
        config=cfg,
    )

    # Turn 2 planner saw Turn 1's question & answer in Conversation History
    assert "What is Standard deductible?" in llm.planner_prompts[1]
    assert "Deductible info [1]." in llm.planner_prompts[1]
    # Per-turn trace & step_count stayed clean (did not double-accumulate)
    assert [t["node"] for t in out2["trace"]] == ["planner", "retrieve", "synthesize"]
    assert out2["step_count"] == 3
    assert len(out1["messages"]) == 2
    assert len(out2["messages"]) == 4