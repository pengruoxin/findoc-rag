import hashlib
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from findoc_rag import __version__
from findoc_rag.answer_generation import (
    MAX_GENERATION_CONTEXTS,
    GeneratedAnswer,
    GroundedAnswerGenerator,
)
from findoc_rag.config import AppSettings, load_settings
from findoc_rag.corpus import build_active_corpus_index
from findoc_rag.documents.models import DocumentChunk
from findoc_rag.documents.quality import PdfQualityConfig
from findoc_rag.indexing import IndexManifest, SearchFilters
from findoc_rag.ingestion import ingest_pdf
from findoc_rag.observability import RetrievalMetrics, RetrievalTrace
from findoc_rag.query_gating import select_best_query
from findoc_rag.query_rewriting import LLMQueryRewriter
from findoc_rag.query_routing import (
    FinanceQueryRoute,
    prepare_finance_query,
    route_finance_query,
)
from findoc_rag.registry import DocumentRegistry
from findoc_rag.service import (
    QueryRequest,
    RetrievalService,
    SearchRequest,
    SearchResponse,
    TracedSearchError,
)
from findoc_rag.time_utils import resolve_relative_time
from findoc_rag.upload_jobs import (
    StartUploadProcessingRequest,
    UploadJob,
    UploadJobStore,
)

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
logger = logging.getLogger(__name__)


class QueryResponse(GeneratedAnswer):
    """Stable Agent contract layered over the existing generated answer."""

    contract_version: Literal["1.0"] = "1.0"
    request_id: str
    trace_id: str
    index_id: str
    original_query: str
    resolved_query: str
    applied_filters: SearchFilters
    route: FinanceQueryRoute
    outcome: Literal["answer", "abstain", "clarify", "evidence_only"]
    rewrite_mode: str
    rewrite_gate: str


class CapabilityFeatures(BaseModel):
    answer: bool
    rewrite: bool
    reranker: bool
    scope: bool
    tracing: bool
    deterministic_tables: bool
    structured_table_artifacts: bool
    ingestion_jobs: bool
    claim_citations: bool
    evidence_resolve: bool


class CapabilityLimit(BaseModel):
    default: int
    minimum: int
    maximum: int


class CapabilityLimits(BaseModel):
    top_k: CapabilityLimit
    candidate_k: CapabilityLimit
    max_generation_contexts: int
    max_evidence_chunk_ids: int = 50


class CapabilitiesResponse(BaseModel):
    contract_version: Literal["1.0"] = "1.0"
    index_id: str
    modes: list[Literal["lexical", "dense", "hybrid"]]
    features: CapabilityFeatures
    filter_fields: list[str]
    limits: CapabilityLimits


class EvidenceResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index_id: str = Field(min_length=1)
    chunk_ids: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1, max_length=50
    )


class ResolvedEvidence(BaseModel):
    chunk: DocumentChunk
    sha256: str


class EvidenceResolveResponse(BaseModel):
    contract_version: Literal["1.0"] = "1.0"
    index_id: str
    evidence: list[ResolvedEvidence]


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        trace_id: str | None = None,
        details: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.trace_id = trace_id
        self.details = details


def _answer_outcome(answer: GeneratedAnswer) -> Literal[
    "answer", "abstain", "clarify", "evidence_only"
]:
    if answer.provider == "clarification":
        return "clarify"
    if answer.provider == "evidence-only":
        return "evidence_only"
    if not answer.grounded:
        return "abstain"
    return "answer"


def _chunk_sha256(chunk: DocumentChunk) -> str:
    return hashlib.sha256(chunk.model_dump_json().encode("utf-8")).hexdigest()


def create_app(settings: AppSettings | None = None) -> FastAPI:
    resolved_settings = settings or load_settings()
    upload_root = resolved_settings.ingestion.upload_root
    upload_jobs = UploadJobStore(upload_root)
    rewrite_mode = os.getenv("FINDOC_RAG_QUERY_REWRITE", "deterministic")
    rewrite_cache = Path(
        os.getenv("FINDOC_RAG_REWRITE_CACHE", "data/cache/rewrites.json")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        interrupted = upload_jobs.fail_interrupted_jobs()
        if interrupted:
            logger.warning("Marked %d interrupted upload jobs as failed", interrupted)
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
        app.state.query_rewriter = (
            LLMQueryRewriter(cache_path=rewrite_cache)
            if rewrite_mode == "llm"
            else None
        )
        app.state.query_rewrite_mode = rewrite_mode
        yield

    app = FastAPI(
        title="FinDocRAG Retrieval API",
        version=__version__,
        lifespan=lifespan,
    )
    ui_dir = Path(__file__).resolve().parents[2] / "docs" / "ui"
    if ui_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=ui_dir), name="ui")

    @app.get("/", include_in_schema=False)
    def ui_home() -> RedirectResponse:
        return RedirectResponse("/ui/workspace-v4.html")

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
                "error": {
                    "code": error.code,
                    "message": error.message,
                    **({"details": error.details} if error.details is not None else {}),
                },
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

    @app.get("/health/live", operation_id="health_live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", operation_id="health_ready")
    def ready(request: Request) -> dict[str, str]:
        retrieval = service(request)
        return {"status": "ready", "index_id": retrieval.manifest.index_id}

    @app.get("/v1/index", response_model=IndexManifest, operation_id="get_index")
    def index_info(request: Request) -> IndexManifest:
        return service(request).manifest

    @app.get(
        "/v1/traces/{trace_id}",
        response_model=RetrievalTrace,
        operation_id="get_retrieval_trace",
    )
    def trace_info(trace_id: str, request: Request) -> RetrievalTrace:
        store = service(request).trace_store
        if store is None:
            raise ApiError(404, "tracing_disabled", "Retrieval tracing is disabled")
        trace = store.get(trace_id)
        if trace is None:
            raise ApiError(404, "trace_not_found", f"Trace {trace_id!r} was not found")
        return trace

    @app.get(
        "/v1/metrics",
        response_model=RetrievalMetrics,
        operation_id="get_retrieval_metrics",
    )
    def metrics(request: Request) -> RetrievalMetrics:
        store = service(request).trace_store
        if store is None:
            raise ApiError(404, "tracing_disabled", "Retrieval tracing is disabled")
        return store.metrics()

    @app.post(
        "/v1/uploads",
        response_model=UploadJob,
        status_code=202,
        operation_id="create_pdf_upload",
    )
    async def upload_pdf(request: Request) -> UploadJob:
        if not resolved_settings.ingestion.enabled:
            raise ApiError(503, "ingestion_disabled", "PDF ingestion is disabled")
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
        return upload_jobs.create(
            job_id=job_id, filename=filename, bytes_written=written
        )

    @app.get(
        "/v1/uploads/{job_id}",
        response_model=UploadJob,
        operation_id="get_pdf_upload",
    )
    def upload_status(job_id: str) -> UploadJob:
        try:
            job = upload_jobs.get(job_id)
        except ValueError as exc:
            raise ApiError(404, "upload_not_found", "Upload job was not found") from exc
        if job is None:
            raise ApiError(404, "upload_not_found", f"Upload job {job_id!r} was not found")
        return job

    def process_upload_job(
        job: UploadJob,
        payload: StartUploadProcessingRequest,
        application: FastAPI,
    ) -> UploadJob:
        try:
            job = upload_jobs.update(
                job, status="ingesting", message="Parsing and versioning document"
            )
            source_path = upload_jobs.source_path(job)
            if source_path.stat().st_size == 0:
                raise ValueError("Uploaded PDF is empty")
            with source_path.open("rb") as source:
                if source.read(5) != b"%PDF-":
                    raise ValueError("Uploaded content does not have a PDF signature")
            result = ingest_pdf(
                source_path,
                payload.document_key,
                DocumentRegistry(resolved_settings.ingestion.registry_path),
                resolved_settings.ingestion.storage_dir,
                metadata=payload.metadata,
                pdf_quality_config=PdfQualityConfig(policy="strict"),
            )
            job = upload_jobs.update(
                job,
                document_version_id=result.version.version_id,
                message="Document version is active",
            )
            index_id = None
            if resolved_settings.ingestion.rebuild_index:
                job = upload_jobs.update(
                    job, status="indexing", message="Building immutable corpus index"
                )
                indexed = build_active_corpus_index(
                    DocumentRegistry(resolved_settings.ingestion.registry_path),
                    resolved_settings.ingestion.index_root,
                )
                index_id = indexed.pointer.index_id
                application.state.retrieval_service = RetrievalService(
                    resolved_settings.retrieval.model_copy(
                        update={"index_dir": resolved_settings.ingestion.index_root}
                    ),
                    resolved_settings.observability,
                    resolved_settings.reranker,
                    scope_settings=resolved_settings.scope_routing,
                )
            return upload_jobs.update(
                job,
                status="ready",
                index_id=index_id,
                message=(
                    "Document is indexed and ready"
                    if index_id
                    else "Document version is active; index rebuild is disabled"
                ),
            )
        except Exception as exc:
            logger.exception("Upload job %s failed", job.job_id)
            return upload_jobs.update(
                job,
                status="failed",
                error_code=type(exc).__name__,
                message=str(exc),
            )

    @app.post(
        "/v1/uploads/{job_id}:process",
        response_model=UploadJob,
        status_code=202,
        operation_id="process_pdf_upload",
    )
    async def process_upload(
        job_id: str,
        payload: StartUploadProcessingRequest,
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> UploadJob:
        if not resolved_settings.ingestion.enabled:
            raise ApiError(503, "ingestion_disabled", "PDF ingestion is disabled")
        try:
            job = upload_jobs.claim(job_id, payload)
        except ValueError as exc:
            raise ApiError(404, "upload_not_found", "Upload job was not found") from exc
        except RuntimeError as exc:
            raise ApiError(
                409,
                "upload_job_not_startable",
                f"Upload job in status {exc.args[0]!r} cannot be started",
            ) from exc
        if job is None:
            raise ApiError(404, "upload_not_found", f"Upload job {job_id!r} was not found")
        background_tasks.add_task(
            process_upload_job, job, payload, request.app
        )
        return job

    @app.post(
        "/v1/search", response_model=SearchResponse, operation_id="search_evidence"
    )
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

    @app.get(
        "/v1/capabilities",
        response_model=CapabilitiesResponse,
        operation_id="get_capabilities",
    )
    def capabilities(request: Request) -> CapabilitiesResponse:
        retrieval = service(request)
        modes: list[Literal["lexical", "dense", "hybrid"]] = ["lexical"]
        if retrieval.manifest.dense_model:
            modes.extend(["dense", "hybrid"])
        return CapabilitiesResponse(
            index_id=retrieval.manifest.index_id,
            modes=modes,
            features=CapabilityFeatures(
                answer=True,
                rewrite=request.app.state.query_rewrite_mode != "none",
                reranker=retrieval.reranker is not None,
                scope=retrieval.scope_settings.enabled,
                tracing=retrieval.trace_store is not None,
                deterministic_tables=(
                    os.getenv("FINDOC_RAG_DISABLE_DETERMINISTIC_TABLES") != "1"
                ),
                structured_table_artifacts=(
                    retrieval.manifest.structured_table_count > 0
                ),
                ingestion_jobs=resolved_settings.ingestion.enabled,
                claim_citations=True,
                evidence_resolve=True,
            ),
            filter_fields=list(SearchFilters.model_fields),
            limits=CapabilityLimits(
                top_k=CapabilityLimit(
                    default=retrieval.settings.top_k, minimum=1, maximum=100
                ),
                candidate_k=CapabilityLimit(
                    default=retrieval.settings.candidate_k, minimum=1, maximum=1000
                ),
                max_generation_contexts=MAX_GENERATION_CONTEXTS,
            ),
        )

    @app.post(
        "/v1/evidence:resolve",
        response_model=EvidenceResolveResponse,
        operation_id="resolve_evidence",
    )
    def resolve_evidence(
        payload: EvidenceResolveRequest, request: Request
    ) -> EvidenceResolveResponse:
        retrieval = service(request)
        if payload.index_id != retrieval.manifest.index_id:
            raise ApiError(
                409,
                "index_id_mismatch",
                "Requested evidence is not bound to the active index",
                details={
                    "requested_index_id": payload.index_id,
                    "active_index_id": retrieval.manifest.index_id,
                },
            )
        chunks = retrieval.index.resolve_chunks(payload.chunk_ids)
        missing = [
            chunk_id
            for chunk_id, chunk in zip(payload.chunk_ids, chunks, strict=True)
            if chunk is None
        ]
        if missing:
            raise ApiError(
                404,
                "evidence_not_found",
                "One or more evidence chunks were not found in the requested index",
                details={"missing_chunk_ids": list(dict.fromkeys(missing))},
            )
        resolved_chunks = [chunk for chunk in chunks if chunk is not None]
        return EvidenceResolveResponse(
            index_id=retrieval.manifest.index_id,
            evidence=[
                ResolvedEvidence(chunk=chunk, sha256=_chunk_sha256(chunk))
                for chunk in resolved_chunks
            ],
        )

    @app.post(
        "/v1/query", response_model=QueryResponse, operation_id="query_documents"
    )
    def query(payload: QueryRequest, request: Request) -> QueryResponse:
        try:
            # 生产环境使用当前日期作为相对时间锚点（评测链路禁用系统时钟，
            # 见 time_utils 文档）；公司别名/代码归一化后参与 metadata 路由。
            original_query = payload.query
            as_of = datetime.now(UTC).date()
            resolved_base, _ = resolve_relative_time(original_query, as_of)
            route = route_finance_query(resolved_base)
            companies = route.company_names
            years = route.report_years
            current = payload.filters or SearchFilters()
            if companies or years:
                payload.filters = SearchFilters(
                    document_keys=current.document_keys,
                    company_names=current.company_names or companies,
                    report_years=current.report_years or years,
                    document_types=current.document_types,
                )
            mode = request.app.state.query_rewrite_mode
            if mode == "none":
                resolved, gate = resolved_base, "none"
            else:
                deterministic_query = prepare_finance_query(
                    resolved_base,
                    as_of_date=as_of,
                    rewrite_mode="deterministic",
                )
                resolved, gate = deterministic_query, "deterministic"
            if mode == "llm":
                llm_query = prepare_finance_query(
                    resolved_base,
                    as_of_date=as_of,
                    rewrite_mode="llm",
                    rewriter=request.app.state.query_rewriter,
                )
                if os.getenv("FINDOC_RAG_QUERY_GATE", "1") == "1":
                    resolved, gate = select_best_query(
                        request.app.state.retrieval_service.index,
                        llm_query,
                        deterministic_query,
                        payload.filters,
                    )
                else:
                    resolved, gate = llm_query, "llm"
            payload.query = resolved
            logger.info(
                "query_route request_id=%s companies=%s report_years=%s "
                "fact_periods=%s forecast_target_years=%s year_filter_policy=%s "
                "rewrite_mode=%s gate=%s",
                request.state.request_id,
                companies,
                years,
                route.fact_periods,
                route.forecast_target_years,
                route.year_filter_policy,
                mode,
                gate,
            )
            result = service(request).search(payload, request.state.request_id)
            generated = request.app.state.answer_generator.generate(resolved, result.hits)
            return QueryResponse(
                **generated.model_dump(),
                request_id=request.state.request_id,
                trace_id=result.trace_id,
                index_id=result.index_id,
                original_query=original_query,
                resolved_query=resolved,
                applied_filters=payload.filters or SearchFilters(),
                route=route,
                outcome=_answer_outcome(generated),
                rewrite_mode=mode,
                rewrite_gate=gate,
            )
        except TracedSearchError as exc:
            raise ApiError(400 if exc.client_error else 500, "query_failure", str(exc), trace_id=exc.trace_id) from exc

    return app
