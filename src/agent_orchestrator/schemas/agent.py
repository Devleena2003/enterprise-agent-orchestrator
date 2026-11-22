from typing import Literal
from pydantic import BaseModel, Field
from typing import Any  


IntentType = Literal[
    "retrieval",
    "multi_step_reasoning",
    "external_tool",
    "sensitive_action",
    "out_of_scope",
]


class IntentAndPlan(BaseModel):
    """Structured query understanding, intent classification, and execution plan."""

    intent: IntentType = Field(
        description=(
            "Choose: 'retrieval' for document lookup; 'multi_step_reasoning' for document "
            "lookup combined with counting/math/currency conversion; 'external_tool' for "
            "live currency/rate queries needing no documents; 'sensitive_action' for actions "
            "that modify data, delete documents, or dispatch external emails/alerts/reports; "
            "'out_of_scope' for casual chat or unrelated unsupported requests."
        )
    )
    rewritten_query: str = Field(
        min_length=2,
        description=(
            "Self-contained search query resolving any pronouns from conversation history "
            "(e.g., 'What about the next day?' -> 'Transactions on 2nd August 2023')."
        ),
    )
    doc_type_filter: str | None = Field(
        default=None,
        description="Optional document category if explicitly implied (e.g., 'statement', 'insurance_policy', 'regulation').",
    )
    needs_retrieval: bool = Field(
        description="True if internal uploaded documents from Project 1 should be searched."
    )
    needs_table_analysis: bool = Field(
        default=False,
        description="True if retrieved Markdown tables need deterministic row counting or numeric aggregation.",
    )
    needs_external_fx: bool = Field(
        default=False,
        description="True if live currency exchange conversion is needed.",
    )
    fx_base_currency: str | None = Field(
        default=None,
        description="3-letter ISO source currency code if FX conversion is requested (e.g., 'INR', 'USD').",
    )
    fx_target_currency: str | None = Field(
        default=None,
        description="3-letter ISO target currency code if FX conversion is requested (e.g., 'USD', 'EUR').",
    )
    requires_human_approval: bool = Field(
        default=False,
        description="True ONLY for sensitive actions (deleting data, sending external alerts/reports).",
    )
    plan_steps: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Ordered concise steps the agent will execute.",
    )
    reasoning: str = Field(
        description="One-sentence justification for the chosen intent and plan."
    )


class TraceStep(BaseModel):
    node: str
    status: Literal["ok", "fallback", "paused", "error"]
    duration_ms: float
    detail: str


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class CostSummary(BaseModel):
    provider: str
    model: str
    billing_tier: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class AgentRunRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Provide an existing thread_id to continue a multi-turn conversation.",
    )
    doc_type: str | None = Field(default=None)


class AgentApprovalRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=100)
    approved: bool
    reviewer: str = Field(default="human_operator", min_length=1, max_length=100)
    note: str = Field(default="", max_length=500)


class AgentRunResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "awaiting_approval", "refused", "failed"]
    intent: str | None = None
    final_answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    approval_request: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    total_duration_ms: float
    cost_summary: CostSummary


class ThreadMessage(BaseModel):
    role: str
    content: str


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    approval_status: str
    messages: list[ThreadMessage] = Field(default_factory=list)