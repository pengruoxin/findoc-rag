import hashlib
import logging
from datetime import UTC, datetime
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from findoc_rag.config import (
    ObservabilitySettings,
    RerankerSettings,
    RetrievalSettings,
    ScopeRoutingSettings,
)
from findoc_rag.corpus import resolve_current_index
from findoc_rag.indexing import (
    IndexManifest,
    PersistentIndex,
    SearchFilters,
    SearchHit,
    reciprocal_rank_fusion,
)
from findoc_rag.observability import (
    RankedHitSnapshot,
    RetrievalStageTrace,
    RetrievalTrace,
    TraceStore,
)
from findoc_rag.reranking import CrossEncoderReranker, Reranker
from findoc_rag.scope_routing import (
    QueryScope,
    plan_candidate_budget,
    route_by_scope,
    route_structured_evidence,
)

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["lexical", "dense", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    candidate_k: int | None = Field(default=None, ge=1, le=1000)
    rerank: bool | None = None
    filters: SearchFilters | None = None
    scope_routing: bool | None = None
    adaptive_candidate_budget: bool | None = None

    @model_validator(mode="after")
    def strip_and_validate_query(self) -> "SearchRequest":
        self.query = self.query.strip()
        if not self.query:
            raise ValueError("query must contain non-whitespace characters")
        if (
            self.top_k is not None
            and self.candidate_k is not None
            and self.candidate_k < self.top_k
        ):
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class QueryRequest(SearchRequest):
    """Agent-facing query contract, kept distinct from raw evidence search."""


class SearchResponse(BaseModel):
    request_id: str
    trace_id: str
    index_id: str
    mode: Literal["lexical", "dense", "hybrid"]
    reranked: bool
    reranker_model: str | None = None
    inferred_scope: str | None = None
    scope_confidence: str | None = None
    base_candidate_k: int
    effective_candidate_k: int
    candidate_budget_reason: str
    took_ms: float
    hits: list[SearchHit]


class TracedSearchError(ValueError):
    def __init__(
        self,
        message: str,
        trace_id: str,
        *,
        client_error: bool,
    ) -> None:
        super().__init__(message)
        self.trace_id = trace_id
        self.client_error = client_error


class RetrievalService:
    """Long-lived retrieval boundary with persistent, privacy-aware tracing."""

    def __init__(
        self,
        settings: RetrievalSettings,
        observability: ObservabilitySettings | None = None,
        reranker_settings: RerankerSettings | None = None,
        reranker: Reranker | None = None,
        scope_settings: ScopeRoutingSettings | None = None,
    ) -> None:
        self.settings = settings
        self.observability = observability or ObservabilitySettings(enabled=False)
        self.reranker_settings = reranker_settings or RerankerSettings()
        self.scope_settings = scope_settings or ScopeRoutingSettings()
        self.reranker = reranker or (
            CrossEncoderReranker(
                self.reranker_settings.model,
                self.reranker_settings.batch_size,
            )
            if self.reranker_settings.enabled
            else None
        )
        self.index = (
            resolve_current_index(settings.index_dir)
            if (settings.index_dir / "current.json").is_file()
            else PersistentIndex(settings.index_dir)
        )
        self.trace_store = (
            TraceStore(self.observability.trace_db) if self.observability.enabled else None
        )

    @property
    def manifest(self) -> IndexManifest:
        return self.index.manifest

    def _snapshot(self, hits: list[SearchHit]) -> list[RankedHitSnapshot]:
        return [
            RankedHitSnapshot(
                rank=hit.rank,
                chunk_id=hit.chunk.chunk_id,
                score=hit.score,
                lexical_rank=hit.lexical_rank,
                dense_rank=hit.dense_rank,
                original_rank=hit.original_rank,
                rerank_score=hit.rerank_score,
                rank_delta=hit.rank_delta,
                retrieval_rank=hit.retrieval_rank,
                scope_score=hit.scope_score,
                scope_rank_delta=hit.scope_rank_delta,
            )
            for hit in hits[: self.observability.max_recorded_hits]
        ]

    def _record_trace(self, trace: RetrievalTrace) -> None:
        if self.trace_store is None:
            return
        try:
            self.trace_store.record(trace)
        except Exception:
            logger.exception("Failed to persist retrieval trace %s", trace.trace_id)

    def search(self, request: SearchRequest, request_id: str) -> SearchResponse:
        mode = request.mode or self.settings.default_mode
        top_k = request.top_k or self.settings.top_k
        base_candidate_k = request.candidate_k or self.settings.candidate_k
        trace_id = uuid4().hex
        started_at = datetime.now(UTC)
        started = perf_counter()
        stages: list[RetrievalStageTrace] = []
        hits: list[SearchHit] = []

        def run_stage(
            stage: Literal["lexical", "dense", "rrf", "scope", "structured", "rerank"],
            operation,
        ) -> list[SearchHit]:
            stage_started = perf_counter()
            stage_hits = operation()
            stages.append(
                RetrievalStageTrace(
                    stage=stage,
                    duration_ms=(perf_counter() - stage_started) * 1000,
                    candidate_count=len(stage_hits),
                    hits=self._snapshot(stage_hits),
                )
            )
            return stage_hits

        error: Exception | None = None
        rerank_requested = request.rerank if request.rerank is not None else self.reranker is not None
        scope_requested = (
            request.scope_routing
            if request.scope_routing is not None
            else self.scope_settings.enabled
        )
        adaptive_budget = (
            request.adaptive_candidate_budget
            if request.adaptive_candidate_budget is not None
            else self.scope_settings.adaptive_candidate_budget
        )
        _, budget_plan = plan_candidate_budget(
            request.query,
            base_candidate_k,
            maximum_candidate_k=self.scope_settings.max_candidate_k,
            enabled=adaptive_budget,
        )
        candidate_k = budget_plan.effective_candidate_k
        inferred_scope: QueryScope | None = None
        # The schema-aware router needs a bounded candidate pool even when no
        # neural reranker or broad section router is enabled.
        expand_candidates = True
        try:
            if candidate_k < top_k:
                raise ValueError("candidate_k must be greater than or equal to top_k")
            if mode in {"dense", "hybrid"} and not self.manifest.dense_model:
                raise ValueError(f"Search mode {mode!r} requires an index with dense embeddings")

            if mode == "lexical":
                hits = run_stage(
                    "lexical",
                    lambda: self.index.search_lexical(
                        request.query,
                        candidate_k if expand_candidates else top_k,
                        request.filters,
                    ),
                )
            elif mode == "dense":
                hits = run_stage(
                    "dense",
                    lambda: self.index.search_dense(
                        request.query,
                        candidate_k if expand_candidates else top_k,
                        request.filters,
                    ),
                )
            else:
                lexical = run_stage(
                    "lexical",
                    lambda: self.index.search_lexical(
                        request.query, candidate_k, request.filters
                    ),
                )
                dense = run_stage(
                    "dense",
                    lambda: self.index.search_dense(request.query, candidate_k, request.filters),
                )
                hits = run_stage(
                    "rrf",
                    lambda: reciprocal_rank_fusion(
                        lexical,
                        dense,
                        top_k=(
                            2 * candidate_k
                            if scope_requested
                            else candidate_k if expand_candidates else top_k
                        ),
                        rrf_k=self.settings.rrf_k,
                        lexical_weight=self.settings.lexical_weight,
                        dense_weight=self.settings.dense_weight,
                    ),
                )
            if scope_requested:
                scope_holder: list[QueryScope] = []

                def apply_scope() -> list[SearchHit]:
                    scope, scoped_hits = route_by_scope(
                        request.query, hits, candidate_k
                    )
                    scope_holder.append(scope)
                    return scoped_hits

                hits = run_stage("scope", apply_scope)
                inferred_scope = scope_holder[0]
            if rerank_requested:
                if self.reranker is None:
                    raise ValueError("Reranking was requested but no reranker is configured")
                hits = run_stage(
                    "rerank", lambda: self.reranker.rerank(request.query, hits, top_k)
                )
            else:
                hits = run_stage(
                    "structured",
                    lambda: route_structured_evidence(request.query, hits, top_k),
                )
        # Trace unexpected backend/model failures before re-raising them with a trace ID.
        except Exception as exc:  # noqa: BLE001
            error = exc

        completed_at = datetime.now(UTC)
        total_duration_ms = (perf_counter() - started) * 1000
        trace = RetrievalTrace(
            trace_id=trace_id,
            request_id=request_id,
            index_id=self.manifest.index_id,
            started_at=started_at,
            completed_at=completed_at,
            query_sha256=hashlib.sha256(request.query.encode()).hexdigest(),
            query_text=request.query if self.observability.capture_query_text else None,
            mode=mode,
            requested_top_k=top_k,
            candidate_k=candidate_k,
            base_candidate_k=base_candidate_k,
            candidate_budget_reason=budget_plan.reason,
            reranked=rerank_requested and error is None,
            reranker_model=self.reranker.model_name if rerank_requested and self.reranker else None,
            inferred_scope=inferred_scope.name if inferred_scope else None,
            scope_confidence=inferred_scope.confidence if inferred_scope else None,
            status="error" if error else "success",
            total_duration_ms=total_duration_ms,
            result_count=len(hits),
            stages=stages,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
        )
        self._record_trace(trace)

        if error:
            raise TracedSearchError(
                str(error),
                trace_id,
                client_error=isinstance(error, ValueError),
            ) from error
        return SearchResponse(
            request_id=request_id,
            trace_id=trace_id,
            index_id=self.manifest.index_id,
            mode=mode,
            reranked=rerank_requested,
            reranker_model=self.reranker.model_name if rerank_requested and self.reranker else None,
            inferred_scope=inferred_scope.name if inferred_scope else None,
            scope_confidence=inferred_scope.confidence if inferred_scope else None,
            base_candidate_k=base_candidate_k,
            effective_candidate_k=candidate_k,
            candidate_budget_reason=budget_plan.reason,
            took_ms=total_duration_ms,
            hits=hits,
        )
