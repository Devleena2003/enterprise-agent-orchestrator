from fastapi import APIRouter, Depends, Request

from agent_orchestrator.schemas.agent import (
    AgentApprovalRequest,
    AgentRunRequest,
    AgentRunResponse,
    ThreadHistoryResponse,
)
from agent_orchestrator.service import AgentOrchestrationService

router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])


def get_agent_service(request: Request) -> AgentOrchestrationService:
    return request.app.state.agent_service


@router.post("/run", response_model=AgentRunResponse)
def run_agent(
    payload: AgentRunRequest,
    service: AgentOrchestrationService = Depends(get_agent_service),
) -> AgentRunResponse:
    return service.run(payload)


@router.post("/approve", response_model=AgentRunResponse)
def approve_action(
    payload: AgentApprovalRequest,
    service: AgentOrchestrationService = Depends(get_agent_service),
) -> AgentRunResponse:
    return service.approve(payload)


@router.get("/threads/{thread_id}", response_model=ThreadHistoryResponse)
def get_thread_history(
    thread_id: str,
    service: AgentOrchestrationService = Depends(get_agent_service),
) -> ThreadHistoryResponse:
    return service.get_thread(thread_id)