from typing import Any


class AgentError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        http_status: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details or {}
        super().__init__(message)


class UpstreamServiceError(AgentError):
    """Raised when Project 1 or an external API fails/times out after retries."""

    def __init__(self, service: str, detail: str) -> None:
        super().__init__(
            code="UPSTREAM_SERVICE_ERROR",
            message=f"Upstream service '{service}' failed: {detail}",
            http_status=503,
            details={"service": service},
        )


class LLMConfigurationError(AgentError):
    def __init__(self, detail: str) -> None:
        super().__init__(
            code="LLM_NOT_CONFIGURED",
            message=detail,
            http_status=503,
        )

class ThreadNotFoundError(AgentError):
    def __init__(self, thread_id: str) -> None:
        super().__init__(
            code="THREAD_NOT_FOUND",
            message=f"Conversation thread '{thread_id}' was not found.",
            http_status=404,
        )


class ApprovalNotPendingError(AgentError):
    def __init__(self, thread_id: str) -> None:
        super().__init__(
           code="APPROVAL_NOT_PENDING",
            message=f"Thread '{thread_id}' has no pending approval request.",
            http_status=409,
        )