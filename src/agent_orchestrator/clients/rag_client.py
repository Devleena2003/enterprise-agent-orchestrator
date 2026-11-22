import logging
import time
from typing import Any
import httpx
from pydantic import BaseModel, Field

from agent_orchestrator.config import Settings, get_settings
from agent_orchestrator.exceptions import UpstreamServiceError
from agent_orchestrator.logging import request_id_ctx

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES = {502, 503, 504}


class RetrievalEvidenceItem(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    document_name: str
    doc_type: str
    page_number: int
    page_end: int | None = None
    section_title: str | None = None
    chunk_type: str
    content: str
    relevance_score: float


class RagRetrieveResult(BaseModel):
    query: str
    sufficient_evidence: bool
    message: str
    reranker_used: bool
    total_candidates_evaluated: int
    results: list[RetrievalEvidenceItem] = Field(default_factory=list)
    constructed_context: str = ""


class RagEngineClient:
    """
    HTTP client for Project 1 (enterprise-rag-engine) with:
    - Strict timeouts
    - Exponential backoff retries on transient 5xx/network errors
    - X-Request-ID propagation for cross-service log correlation
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)
        self._max_retries = max_retries
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        req_id = request_id_ctx.get()
        headers = {"Content-Type": "application/json"}
        if req_id and req_id != "-":
            headers["X-Request-ID"] = req_id
        return headers

    def check_ready(self) -> tuple[bool, str]:
        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=httpx.Timeout(3.0),
                transport=self._transport,
            ) as client:
                resp = client.get("/health/ready", headers=self._headers())
            if resp.status_code == 200:
                return True, "ok"
            return False, f"http_{resp.status_code}"
        except httpx.HTTPError as exc:
            logger.warning("Project 1 readiness check failed: %s", exc)
            return False, "unreachable"

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 5,
        doc_type: str | None = None,
        use_reranker: bool | None = None,
    ) -> RagRetrieveResult:
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if doc_type:
            payload["doc_type"] = doc_type
        if use_reranker is not None:
            payload["use_reranker"] = use_reranker

        last_error: str = "unknown_error"

        for attempt in range(self._max_retries + 1):
            try:
                with httpx.Client(
                    base_url=self._base_url,
                    timeout=self._timeout,
                    transport=self._transport,
                ) as client:
                    resp = client.post(
                        "/api/v1/retrieve",
                        json=payload,
                        headers=self._headers(),
                    )

                if resp.status_code == 200:
                    return RagRetrieveResult.model_validate(resp.json())

                if resp.status_code in _TRANSIENT_STATUS_CODES:
                    last_error = f"HTTP {resp.status_code}"
                    logger.warning(
                        "Project 1 transient failure (attempt %d/%d): %s",
                        attempt + 1,
                        self._max_retries + 1,
                        last_error,
                    )
                else:
                    # 4xx errors are non-retryable client/contract mistakes
                    raise UpstreamServiceError(
                        "rag-engine",
                        f"Non-retryable HTTP {resp.status_code}: {resp.text[:200]}",
                    )

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Project 1 network/timeout error (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    last_error,
                )

            if attempt < self._max_retries:
                time.sleep(0.2 * (2**attempt))

        raise UpstreamServiceError("rag-engine", last_error)


def get_rag_client(settings: Settings | None = None) -> RagEngineClient:
    cfg = settings or get_settings()
    return RagEngineClient(
        base_url=cfg.rag_engine_base_url,
        timeout_seconds=cfg.rag_engine_timeout_seconds,
        max_retries=cfg.rag_engine_max_retries,
    )