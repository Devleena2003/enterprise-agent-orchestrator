import time
from typing import Any
import uuid
from langgraph.types import Command

from agent_orchestrator.config import Settings
from agent_orchestrator.exceptions import ApprovalNotPendingError, ThreadNotFoundError
from agent_orchestrator.graph.workflow import new_turn_input
from agent_orchestrator.schemas.agent import (
    AgentApprovalRequest,
    AgentRunRequest,
    AgentRunResponse,
    CostSummary,
    ThreadHistoryResponse,
    ThreadMessage,
)


class AgentOrchestrationService:
    def __init__(self, compiled_graph: Any, settings: Settings) -> None:
        self._graph = compiled_graph
        self._settings = settings

    def _build_cost_summary(self, token_usage: dict[str, int]) -> CostSummary:
        prompt = int(token_usage.get("prompt_tokens", 0))
        completion = int(token_usage.get("completion_tokens", 0))
        total = int(token_usage.get("total_tokens", prompt + completion))

        in_rate = self._settings.llm_cost_per_1m_input_tokens
        out_rate = self._settings.llm_cost_per_1m_output_tokens
        cost = (prompt / 1_000_000) * in_rate + (completion / 1_000_000) * out_rate
        tier = "free_tier" if (in_rate == 0.0 and out_rate == 0.0) else "metered"

        return CostSummary(
            provider=self._settings.llm_provider_name,
            model=self._settings.llm_model,
            billing_tier=tier,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            estimated_cost_usd=round(cost, 6),
        )

    def _to_response(
        self, thread_id: str, state_out: dict[str, Any], total_duration_ms: float
    ) -> AgentRunResponse:
        plan = state_out.get("plan") or {}
        interrupts = state_out.get("__interrupt__")

        if interrupts:
            first_interrupt = interrupts[0]
            payload = getattr(first_interrupt, "value", first_interrupt)
            return AgentRunResponse(
                thread_id=thread_id,
                status="awaiting_approval",
                intent=plan.get("intent"),
                final_answer=(
                    "Execution paused: human approval is required before proceeding."
                ),
                approval_request=payload if isinstance(payload, dict) else {"detail": payload},
                trace=state_out.get("trace", []),
                total_duration_ms=total_duration_ms,
                cost_summary=self._build_cost_summary(state_out.get("token_usage") or {}),
            )

        if state_out.get("error"):
            status = "failed"
        elif plan.get("needs_retrieval") and not state_out.get("sufficient_evidence", False):
            status = "refused"
        else:
            status = "completed"

        return AgentRunResponse(
            thread_id=thread_id,
            status=status,  # type: ignore[arg-type]
            intent=plan.get("intent"),
            final_answer=state_out.get("final_answer", ""),
            citations=state_out.get("citations", []),
            tool_outputs=state_out.get("tool_outputs", []),
            approval_request=state_out.get("approval_request"),
            trace=state_out.get("trace", []),
            total_duration_ms=total_duration_ms,
            cost_summary=self._build_cost_summary(state_out.get("token_usage") or {}),
        )

    def run(self, req: AgentRunRequest) -> AgentRunResponse:
        thread_id = req.thread_id or f"thr_{uuid.uuid4().hex[:12]}"
        start = time.perf_counter()
        result = self._graph.invoke(
            new_turn_input(
                thread_id=thread_id,
                user_query=req.query,
                doc_type=req.doc_type,
                max_steps=self._settings.agent_max_steps,
            ),
            config={"configurable": {"thread_id": thread_id}},
        )
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return self._to_response(thread_id, result, duration_ms)

    def approve(self, req: AgentApprovalRequest) -> AgentRunResponse:
        config = {"configurable": {"thread_id": req.thread_id}}
        snapshot = self._graph.get_state(config)
        if not snapshot or not snapshot.values:
            raise ThreadNotFoundError(req.thread_id)
        if not snapshot.next or snapshot.values.get("approval_status") != "pending":
            raise ApprovalNotPendingError(req.thread_id)

        start = time.perf_counter()
        result = self._graph.invoke(
            Command(
                resume={
                    "approved": req.approved,
                    "reviewer": req.reviewer,
                    "note": req.note,
                }
            ),
            config=config,
        )
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return self._to_response(req.thread_id, result, duration_ms)

    def get_thread(self, thread_id: str) -> ThreadHistoryResponse:
        snapshot = self._graph.get_state({"configurable": {"thread_id": thread_id}})
        if not snapshot or not snapshot.values:
            raise ThreadNotFoundError(thread_id)

        msgs = [
            ThreadMessage(
                role=getattr(m, "type", "unknown"),
                content=str(getattr(m, "content", "")),
            )
            for m in snapshot.values.get("messages", [])
        ]
        return ThreadHistoryResponse(
            thread_id=thread_id,
            approval_status=snapshot.values.get("approval_status", "not_required"),
            messages=msgs,
        )