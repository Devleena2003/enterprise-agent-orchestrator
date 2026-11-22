from contextlib import asynccontextmanager
import logging
import time
import uuid
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from agent_orchestrator.api.agent import router as agent_router
from agent_orchestrator.api.health import router as health_router
from agent_orchestrator.clients.rag_client import get_rag_client
from agent_orchestrator.config import get_settings
from agent_orchestrator.exceptions import AgentError
from agent_orchestrator.graph.workflow import build_agent_graph, create_sqlite_checkpointer
from agent_orchestrator.llm.provider import get_chat_model
from agent_orchestrator.logging import configure_logging, request_id_ctx
from agent_orchestrator.schemas.common import ErrorDetail, ErrorResponse
from agent_orchestrator.service import AgentOrchestrationService
from agent_orchestrator.tools.fx_client import get_fx_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    checkpointer = create_sqlite_checkpointer(settings.checkpoint_db_path)

    # Lazy wrapper isn't needed if key is valid; defer LLM config check gracefully if missing during unit tests
    try:
        llm = get_chat_model()
    except AgentError:
        llm = None  # Tests override agent_service; /health/ready still reports degraded

    if llm is not None:
        compiled = build_agent_graph(
            llm=llm,
            rag_client=get_rag_client(settings),
            fx_client=get_fx_client(settings),
            settings=settings,
            checkpointer=checkpointer,
        )
        app.state.agent_service = AgentOrchestrationService(compiled, settings)
    else:
        app.state.agent_service = None

    try:
        yield
    finally:
        checkpointer.conn.close()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Enterprise Agent Orchestration Service",
        version="0.1.0",
        description="Stateful, fault-tolerant LangGraph agent layer built on Project 1.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        token = request_id_ctx.set(req_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "Unhandled exception processing %s %s", request.method, request.url.path
            )
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "HTTP request completed: method=%s path=%s duration_ms=%s",
                request.method,
                request.url.path,
                duration_ms,
            )
            request_id_ctx.reset(token)

        response.headers["X-Request-ID"] = req_id
        return response

    @app.exception_handler(AgentError)
    async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
        req_id = request_id_ctx.get()
        payload = ErrorResponse(
            error=ErrorDetail(
                code=exc.code,
                message=exc.message,
                request_id=req_id,
                details=exc.details,
            )
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=payload.model_dump(),
            headers={"X-Request-ID": req_id},
        )

    app.include_router(health_router)
    app.include_router(agent_router)
    return app


app = create_app()