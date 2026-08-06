import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from findoc_rag import __version__
from findoc_rag.answer_generation import GeneratedAnswer, GroundedAnswerGenerator
from findoc_rag.config import AppSettings, load_settings
from findoc_rag.indexing import IndexManifest, SearchFilters
from findoc_rag.observability import RetrievalMetrics, RetrievalTrace
from findoc_rag.service import (
    RetrievalService,
    SearchRequest,
    SearchResponse,
    TracedSearchError,
)

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


class UploadJob(BaseModel):
    job_id: str
    filename: str
    status: str
    bytes_written: int = 0
    message: str = ""


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
    upload_root = Path("data/uploads")
    upload_jobs: dict[str, UploadJob] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.retrieval_service = RetrievalService(
            resolved_settings.retrieval,
            resolved_settings.observability,
            resolved_settings.reranker,
            scope_settings=resolved_settings.scope_routing,
        )
        app.state.answer_generator = GroundedAnswerGenerator(
            resolved_settings.answer_generation.model,
            resolved_settings.answer_generation.endpoint,
            resolved_settings.answer_generation.enabled,
        )
        yield

    app = FastAPI(
        title="FinDocRAG Retrieval API",
        version=__version__,
        lifespan=lifespan,
    )
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    if docs_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=docs_dir), name="ui")

    @app.get("/", include_in_schema=False)
    def ui_home() -> RedirectResponse:
        return RedirectResponse("/ui/workspace-v3.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> RedirectResponse:
        return RedirectResponse("/ui/favicon.svg")

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

    @app.post("/v1/uploads", response_model=UploadJob, status_code=202)
    async def upload_pdf(request: Request) -> UploadJob:
        filename = Path(request.headers.get("X-Filename", "upload.pdf")).name
        if not filename.lower().endswith(".pdf"):
            raise ApiError(415, "unsupported_file_type", "Only PDF uploads are supported")
        job_id = uuid4().hex
        target_dir = upload_root / job_id
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / filename
        written = 0
        with target.open("wb") as handle:
            async for chunk in request.stream():
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    target.unlink(missing_ok=True)
                    target_dir.rmdir()
                    raise ApiError(413, "file_too_large", "PDF exceeds the 100 MB limit")
                handle.write(chunk)
        job = UploadJob(job_id=job_id, filename=filename, status="uploaded", bytes_written=written, message="Upload received; indexing can be started from this job")
        upload_jobs[job_id] = job
        return job

    @app.get("/v1/uploads/{job_id}", response_model=UploadJob)
    def upload_status(job_id: str) -> UploadJob:
        job = upload_jobs.get(job_id)
        if job is None:
            raise ApiError(404, "upload_not_found", f"Upload job {job_id!r} was not found")
        return job

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

    @app.post("/v1/query", response_model=GeneratedAnswer)
    def query(payload: SearchRequest, request: Request) -> GeneratedAnswer:
        try:
            # 财务问答默认按问题中的公司和报告年份路由，避免跨公司串文档。
            inferred_company = next(
                (name for name in ("贵州茅台", "伊利股份") if name in payload.query),
                None,
            )
            year_match = re.search(r"20\d{2}", payload.query)
            current = payload.filters or SearchFilters()
            if inferred_company or year_match:
                payload.filters = SearchFilters(
                    document_keys=current.document_keys,
                    company_names=current.company_names or ([inferred_company] if inferred_company else []),
                    report_years=current.report_years or ([int(year_match.group())] if year_match else []),
                    document_types=current.document_types,
                )
            result = service(request).search(payload, request.state.request_id)
            return request.app.state.answer_generator.generate(payload.query, result.hits)
        except TracedSearchError as exc:
            raise ApiError(400 if exc.client_error else 500, "query_failure", str(exc), trace_id=exc.trace_id) from exc

    return app
