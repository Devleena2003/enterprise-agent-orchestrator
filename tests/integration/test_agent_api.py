from pathlib import Path
import httpx
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agent_orchestrator.api.agent import get_agent_service
from agent_orchestrator.clients.rag_client import RagEngineClient
from agent_orchestrator.config import Settings
from agent_orchestrator.graph.workflow import build_agent_graph, create_sqlite_checkpointer
from agent_orchestrator.main import create_app
from agent_orchestrator.schemas.agent import IntentAndPlan
from agent_orchestrator.service import AgentOrchestrationService
from agent_orchestrator.tools.fx_client import FxClient


class _ApiStubLLM:
    def __init__(self, plans: list[IntentAndPlan], synth_text: str = "Answer [1].") -> None:
        self._plans = list(plans)
        self._synth_text = synth_text

    def with_structured_output(self, schema, include_raw: bool = False):
        outer = self

        class _Runner:
            def invoke(self, messages):
                raw = AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 40, "output_tokens": 20, "total_tokens": 60},
                )
                return {"raw": raw, "parsed": outer._plans.pop(0), "parsing_error": None}

        return _Runner()

    def invoke(self, messages):
        return AIMessage(
            content=self._synth_text,
            usage_metadata={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
        )


def _build_test_client(
    tmp_path: Path,
    plans: list[IntentAndPlan],
    rag_transport: httpx.BaseTransport,
):
    db_path = tmp_path / "api_checkpoints.db"
    checkpointer = create_sqlite_checkpointer(str(db_path))
    settings = Settings(llm_api_key="test_key")
    compiled = build_agent_graph(
        llm=_ApiStubLLM(plans),  # type: ignore[arg-type]
        rag_client=RagEngineClient(base_url="http://rag", max_retries=0, transport=rag_transport),
        fx_client=FxClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json={"date": "2026-11-14", "rates": {"EUR": 90.0}})
            )
        ),
        settings=settings,
        checkpointer=checkpointer,
    )
    svc = AgentOrchestrationService(compiled, settings)
    app = create_app()
    app.dependency_overrides[get_agent_service] = lambda: svc
    return TestClient(app), checkpointer


def test_agent_run_hitl_approve_and_thread_history(tmp_path: Path) -> None:
    plans = [
        IntentAndPlan(
            intent="retrieval",
            rewritten_query="Clause 14B deductible",
            needs_retrieval=True,
            plan_steps=["Retrieve"],
            reasoning="Lookup.",
        ),
        IntentAndPlan(
            intent="sensitive_action",
            rewritten_query="Dispatch compliance alert",
            needs_retrieval=False,
            requires_human_approval=True,
            plan_steps=["Await approval", "Send alert"],
            reasoning="External alert needs approval.",
        ),
    ]
    rag_ok = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "query": "Clause 14B deductible",
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
                        "chunk_type": "text",
                        "content": "Clause 14B requires $1,000 deductible.",
                        "relevance_score": 0.9,
                    }
                ],
                "constructed_context": "[1] Clause 14B requires $1,000 deductible.",
            },
        )
    )
    client, cp = _build_test_client(tmp_path, plans, rag_ok)
    try:
        # 1. Grounded retrieval turn
        r1 = client.post(
            "/api/v1/agent/run",
            json={"query": "What does Clause 14B require?", "thread_id": "thr_demo"},
        )
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["status"] == "completed"
        assert b1["cost_summary"]["total_tokens"] == 160
        assert b1["cost_summary"]["billing_tier"] == "free_tier"
        assert len(b1["citations"]) == 1

        # 2. Sensitive action pauses for HITL
        r2 = client.post(
            "/api/v1/agent/run",
            json={"query": "Dispatch compliance alert", "thread_id": "thr_demo"},
        )
        assert r2.status_code == 200
        assert r2.json()["status"] == "awaiting_approval"
        assert r2.json()["approval_request"] is not None

        # 3. Human approves
        r3 = client.post(
            "/api/v1/agent/approve",
            json={"thread_id": "thr_demo", "approved": True, "reviewer": "lead_fde"},
        )
        assert r3.status_code == 200
        assert r3.json()["status"] == "completed"
        assert "lead_fde" in r3.json()["final_answer"]

        # 4. Approving again returns 409 conflict (no longer pending)
        r_dup = client.post(
            "/api/v1/agent/approve",
            json={"thread_id": "thr_demo", "approved": True},
        )
        assert r_dup.status_code == 409
        assert r_dup.json()["error"]["code"] == "APPROVAL_NOT_PENDING"

        # 5. Thread history preserved
        hist = client.get("/api/v1/agent/threads/thr_demo")
        assert hist.status_code == 200
        assert len(hist.json()["messages"]) == 4

        # 6. Unknown thread returns 404
        missing = client.get("/api/v1/agent/threads/does_not_exist")
        assert missing.status_code == 404
    finally:
        cp.conn.close()


def test_agent_handles_project1_outage_gracefully(tmp_path: Path) -> None:
    plans = [
        IntentAndPlan(
            intent="retrieval",
            rewritten_query="Policy deductible",
            needs_retrieval=True,
            plan_steps=["Retrieve"],
            reasoning="Lookup.",
        )
    ]
    rag_down = httpx.MockTransport(lambda req: httpx.Response(503))
    client, cp = _build_test_client(tmp_path, plans, rag_down)
    try:
        res = client.post("/api/v1/agent/run", json={"query": "What is the deductible?"})
        assert res.status_code == 200
        assert res.json()["status"] == "failed"
        assert "unavailable" in res.json()["final_answer"].lower()
    finally:
        cp.conn.close()