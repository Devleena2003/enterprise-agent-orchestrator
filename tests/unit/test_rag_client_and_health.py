import httpx
import pytest
from fastapi.testclient import TestClient

from agent_orchestrator.clients.rag_client import RagEngineClient
from agent_orchestrator.exceptions import UpstreamServiceError
from agent_orchestrator.logging import request_id_ctx
from agent_orchestrator.main import create_app


def test_rag_client_retries_transient_503_and_propagates_request_id() -> None:
    attempts = 0
    seen_req_id = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts, seen_req_id
        attempts += 1
        seen_req_id = request.headers.get("X-Request-ID")
        if attempts == 1:
            return httpx.Response(503, json={"detail": "temporarily busy"})
        return httpx.Response(
            200,
            json={
                "query": "test",
                "sufficient_evidence": True,
                "message": "ok",
                "reranker_used": False,
                "total_candidates_evaluated": 1,
                "results": [],
                "constructed_context": "[1] ok",
            },
        )

    client = RagEngineClient(
        base_url="http://test-rag",
        max_retries=2,
        transport=httpx.MockTransport(handler),
    )

    token = request_id_ctx.set("req_trace_99")
    try:
        res = client.retrieve(query="test")
    finally:
        request_id_ctx.reset(token)

    assert attempts == 2
    assert seen_req_id == "req_trace_99"
    assert res.sufficient_evidence is True


def test_rag_client_raises_after_timeout_retries_exhausted() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("simulated timeout")

    client = RagEngineClient(
        base_url="http://test-rag",
        max_retries=1,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(UpstreamServiceError) as exc_info:
        client.retrieve(query="test")

    assert attempts == 2
    assert exc_info.value.code == "UPSTREAM_SERVICE_ERROR"


def test_rag_client_does_not_retry_400_bad_request() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": {"code": "INVALID_INPUT"}})

    client = RagEngineClient(
        base_url="http://test-rag",
        max_retries=2,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(UpstreamServiceError):
        client.retrieve(query="x")

    assert attempts == 1


def test_health_live() -> None:
    client = TestClient(create_app())
    res = client.get("/health/live")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"