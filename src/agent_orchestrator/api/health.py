from fastapi import APIRouter, Response, status

from agent_orchestrator.clients.rag_client import get_rag_client
from agent_orchestrator.config import get_settings
from agent_orchestrator.schemas.common import LivenessResponse, ReadinessResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live", response_model=LivenessResponse)
def health_live() -> LivenessResponse:
    settings = get_settings()
    return LivenessResponse(status="ok", service=settings.app_name)


@router.get("/ready", response_model=ReadinessResponse)
def health_ready(response: Response) -> ReadinessResponse:
    settings = get_settings()
    rag_ok, rag_detail = get_rag_client(settings).check_ready()
    llm_configured = (
        bool(settings.llm_api_key.strip())
        and settings.llm_api_key.strip() != "your_free_api_key_here"
    )

    checks = {
        "rag_engine": rag_detail,
        "llm_config": "ok" if llm_configured else "missing_api_key",
    }

    if not rag_ok or not llm_configured:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="degraded",
            service=settings.app_name,
            checks=checks,
        )

    return ReadinessResponse(
        status="ready",
        service=settings.app_name,
        checks=checks,
    )