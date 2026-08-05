import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from findoc_rag import __version__
from findoc_rag.config import AppSettings, load_settings
from findoc_rag.indexing import IndexManifest
from findoc_rag.observability import RetrievalMetrics, RetrievalTrace
from findoc_rag.service import (
    RetrievalService,
    SearchRequest,
    SearchResponse,
    TracedSearchError,
)

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        trace_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.trace_id = trace_id


def create_app(settings: AppSettings | None = None) -> FastAPI:
    resolved_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.retrieval_service = RetrievalService(
            resolved_settings.retrieval,
            resolved_settings.observability,
            resolved_settings.reranker,
            scope_settings=resolved_settings.scope_routing,
        )
        yield

    app = FastAPI(
        title="FinDocRAG Retrieval API",
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_identity(request: Request, call_next) -> Response:
        supplied_request_id = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied_request_id
            if REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else uuid4().hex
        )
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={
                "request_id": request.state.request_id,
                "trace_id": error.trace_id,
                "error": {"code": error.code, "message": error.message},
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        details = [
            {"location": list(item["loc"]), "message": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request.state.request_id,
                "error": {
                    "code": "request_validation_error",
                    "message": "Request validation failed",
                    "details": details,
                },
            },
        )

    def service(request: Request) -> RetrievalService:
        return request.app.state.retrieval_service

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready(request: Request) -> dict[str, str]:
        retrieval = service(request)
        return {"status": "ready", "index_id": retrieval.manifest.index_id}

    @app.get("/v1/index", response_model=IndexManifest)
    def index_info(request: Request) -> IndexManifest:
        return service(request).manifest

    @app.get("/v1/traces/{trace_id}", response_model=RetrievalTrace)
    def trace_info(trace_id: str, request: Request) -> RetrievalTrace:
        store = service(request).trace_store
        if store is None:
            raise ApiError(404, "tracing_disabled", "Retrieval tracing is disabled")
        trace = store.get(trace_id)
        if trace is None:
            raise ApiError(404, "trace_not_found", f"Trace {trace_id!r} was not found")
        return trace

    @app.get("/v1/metrics", response_model=RetrievalMetrics)
    def metrics(request: Request) -> RetrievalMetrics:
        store = service(request).trace_store
        if store is None:
            raise ApiError(404, "tracing_disabled", "Retrieval tracing is disabled")
        return store.metrics()

    @app.post("/v1/search", response_model=SearchResponse)
    def search(payload: SearchRequest, request: Request) -> SearchResponse:
        try:
            return service(request).search(payload, request.state.request_id)
        except TracedSearchError as exc:
            raise ApiError(
                400 if exc.client_error else 500,
                "invalid_search_request" if exc.client_error else "retrieval_failure",
                str(exc),
                trace_id=exc.trace_id,
            ) from exc

    return app
