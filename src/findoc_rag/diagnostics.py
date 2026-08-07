import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from findoc_rag.documents.models import DocumentChunk
from findoc_rag.indexing import PersistentIndex, SearchFilters, SearchHit, reciprocal_rank_fusion
from findoc_rag.io import read_jsonl
from findoc_rag.registry import DocumentRegistry
from findoc_rag.scope_routing import plan_candidate_budget, route_by_scope


class DocumentProfile(BaseModel):
    document_key: str
    company: str
    year: int = Field(ge=1990, le=2100)


class EvidenceJudgment(BaseModel):
    chunk_id: str
    label: Literal["relevant", "irrelevant", "needs_review"]
    reason: str
    page_start: int
    page_end: int
    section_path: list[str]
    excerpt: str
    source: Literal["structural_rule", "retrieval_candidate"]
    negative_type: Literal[
        "wrong_company", "wrong_period", "wrong_scope", "mention_only", "other"
    ] | None = None


class RankingDiagnostic(BaseModel):
    query_id: str
    query: str
    company: str
    year: int
    metric: str
    scope: str
    status: Literal["accepted", "needs_review"]
    judgments: list[EvidenceJudgment]

    @model_validator(mode="after")
    def validate_judgments(self) -> "RankingDiagnostic":
        if not any(item.label == "relevant" for item in self.judgments):
            raise ValueError("A diagnostic query requires at least one relevant judgment")
        return self


class DiagnosticDataset(BaseModel):
    schema_version: int = 1
    dataset_id: str
    index_id: str
    generator: str = "structure-anchored-v1"
    query_count: int
    accepted_count: int
    needs_review_count: int
    queries: list[RankingDiagnostic]

    @model_validator(mode="after")
    def validate_dataset(self) -> "DiagnosticDataset":
        if self.query_count != len(self.queries):
            raise ValueError("query_count does not match queries")
        if len({item.query_id for item in self.queries}) != len(self.queries):
            raise ValueError("query IDs must be unique")
        return self


class DiagnosticQueryResult(BaseModel):
    query_id: str
    first_relevant_rank: int | None
    hit_at_k: bool
    reciprocal_rank: float
    ranked_chunk_ids: list[str]
    inferred_scope: str | None = None
    effective_candidate_k: int
    candidate_first_rank: int | None = None
    candidate_recall: bool = False
    candidate_pool_size: int = 0
    relevant_count: int = 0
    retrieved_relevant_count: int = 0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0


class DiagnosticEvaluation(BaseModel):
    dataset_id: str
    index_id: str
    mode: Literal["lexical", "dense", "hybrid"]
    reranker_model: str | None = None
    evaluated_query_count: int
    top_k: int
    candidate_k: int
    metadata_filters: bool = False
    scope_routing: bool = False
    adaptive_candidate_budget: bool = False
    average_effective_candidate_k: float
    candidate_recall_rate: float
    hit_at_k: float
    mrr_at_k: float
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    results: list[DiagnosticQueryResult]


class RecallFailure(BaseModel):
    query_id: str
    query: str
    relevant_chunk_ids: list[str]
    effective_candidate_k: int
    lexical_first_rank: int | None
    dense_first_rank: int | None
    fused_first_rank: int | None
    failure_type: Literal[
        "metadata_filter_mismatch",
        "candidate_budget_too_small",
        "fusion_displacement",
        "downstream_ranking",
        "unresolved",
    ]
    explanation: str


class RecallFailureReport(BaseModel):
    dataset_id: str
    evaluation_index_id: str
    failure_count: int
    failures_by_type: dict[str, int]
    failures: list[RecallFailure]


class DiagnosticSpec(BaseModel):
    metric: str
    scope: str
    query_template: str
    positive_cues: tuple[str, ...]
    conflicting_cues: tuple[str, ...] = ()


SPECS = (
    DiagnosticSpec(
        metric="营业收入",
        scope="年度主要财务指标",
        query_template="{company}{year}年营业收入是多少",
        positive_cues=("主要会计数据", "主要财务指标"),
        conflicting_cues=("分季度", "分行业", "分产品", "收入确认"),
    ),
    DiagnosticSpec(
        metric="归属于上市公司股东的净利润",
        scope="年度主要财务指标",
        query_template="{company}{year}年归属于上市公司股东的净利润是多少",
        positive_cues=("主要会计数据", "主要财务指标"),
        conflicting_cues=("分季度",),
    ),
    DiagnosticSpec(
        metric="经营活动产生的现金流量净额",
        scope="年度主要财务指标",
        query_template="{company}{year}年经营活动产生的现金流量净额是多少",
        positive_cues=("主要会计数据", "主要财务指标"),
    ),
    DiagnosticSpec(
        metric="基本每股收益",
        scope="年度主要财务指标",
        query_template="{company}{year}年基本每股收益是多少",
        positive_cues=("主要会计数据", "主要财务指标"),
    ),
    DiagnosticSpec(
        metric="营业收入",
        scope="季度财务数据",
        query_template="{company}{year}年分季度营业收入是多少",
        positive_cues=("分季度",),
        conflicting_cues=("主要会计数据", "分行业", "分产品", "收入确认"),
    ),
    DiagnosticSpec(
        metric="营业收入",
        scope="主营业务分部",
        query_template="{company}{year}年主营业务分行业或分产品营业收入",
        positive_cues=("主营业务", "分行业", "分产品"),
        conflicting_cues=("分季度", "主要会计数据", "收入确认"),
    ),
)


def _first_relevant_rank(hits: list[SearchHit], relevant: set[str]) -> int | None:
    return min(
        (hit.rank for hit in hits if hit.chunk.chunk_id in relevant),
        default=None,
    )


def _cue_match(chunk: DocumentChunk, cues: tuple[str, ...]) -> int:
    context = " ".join(chunk.section_path) + " " + chunk.text[:500]
    return sum(cue in context for cue in cues)


def _judgment(chunk: DocumentChunk, label: str, reason: str, source: str, **extra) -> EvidenceJudgment:
    return EvidenceJudgment(
        chunk_id=chunk.chunk_id,
        label=label,
        reason=reason,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_path=chunk.section_path,
        excerpt=chunk.text[:500],
        source=source,
        **extra,
    )


def generate_diagnostic_dataset(
    registry: DocumentRegistry,
    index: PersistentIndex,
    profiles: list[DocumentProfile],
    candidate_k: int = 20,
) -> DiagnosticDataset:
    active = {version.document_key: version for version in registry.active_versions()}
    document_chunks: dict[str, list[DocumentChunk]] = {}
    chunk_owner: dict[str, str] = {}
    for profile in profiles:
        version = active.get(profile.document_key)
        if version is None or not version.chunks_path:
            raise ValueError(f"Profile is not an active document: {profile.document_key}")
        chunks = read_jsonl(Path(version.chunks_path), DocumentChunk)
        document_chunks[profile.document_key] = chunks
        chunk_owner.update((chunk.chunk_id, profile.document_key) for chunk in chunks)

    diagnostics: list[RankingDiagnostic] = []
    for profile in profiles:
        chunks = document_chunks[profile.document_key]
        for spec in SPECS:
            matches = [chunk for chunk in chunks if spec.metric in chunk.text]
            anchored = [chunk for chunk in matches if _cue_match(chunk, spec.positive_cues)]
            if not anchored:
                continue
            anchored.sort(key=lambda chunk: (-_cue_match(chunk, spec.positive_cues), chunk.chunk_index))
            positive = anchored[0]
            query = spec.query_template.format(company=profile.company, year=profile.year)
            judgments = [
                _judgment(
                    positive,
                    "relevant",
                    f"metric text and structural cues matched: {', '.join(spec.positive_cues)}",
                    "structural_rule",
                )
            ]
            seen = {positive.chunk_id}
            hits = index.search(query, top_k=candidate_k, mode="hybrid", candidate_k=candidate_k)
            for hit in hits:
                if hit.chunk.chunk_id in seen:
                    continue
                seen.add(hit.chunk.chunk_id)
                owner = chunk_owner.get(hit.chunk.chunk_id)
                if owner and owner != profile.document_key:
                    judgments.append(
                        _judgment(
                            hit.chunk,
                            "irrelevant",
                            "candidate belongs to a different profiled company",
                            "retrieval_candidate",
                            negative_type="wrong_company",
                        )
                    )
                elif spec.metric in hit.chunk.text and _cue_match(hit.chunk, spec.conflicting_cues):
                    context = " ".join(hit.chunk.section_path) + " " + hit.chunk.text[:500]
                    negative_type = (
                        "wrong_period"
                        if "分季度" in context and spec.scope != "季度财务数据"
                        else "wrong_scope"
                    )
                    judgments.append(
                        _judgment(
                            hit.chunk,
                            "irrelevant",
                            "metric matches but section indicates a conflicting period or scope",
                            "retrieval_candidate",
                            negative_type=negative_type,
                        )
                    )
                else:
                    judgments.append(
                        _judgment(
                            hit.chunk,
                            "needs_review",
                            "retrieved candidate is not safely classifiable by structural rules",
                            "retrieval_candidate",
                        )
                    )
            status = "accepted" if any(j.label == "irrelevant" for j in judgments) else "needs_review"
            seed = f"{profile.document_key}:{spec.scope}:{spec.metric}"
            diagnostics.append(
                RankingDiagnostic(
                    query_id=hashlib.sha256(seed.encode()).hexdigest()[:16],
                    query=query,
                    company=profile.company,
                    year=profile.year,
                    metric=spec.metric,
                    scope=spec.scope,
                    status=status,
                    judgments=judgments,
                )
            )

    counts = Counter(item.status for item in diagnostics)
    content_seed = "\n".join(item.model_dump_json() for item in diagnostics)
    return DiagnosticDataset(
        dataset_id=hashlib.sha256(content_seed.encode()).hexdigest()[:20],
        index_id=index.manifest.index_id,
        query_count=len(diagnostics),
        accepted_count=counts["accepted"],
        needs_review_count=counts["needs_review"],
        queries=diagnostics,
    )


def evaluate_diagnostic_dataset(
    dataset: DiagnosticDataset,
    index: PersistentIndex,
    mode: Literal["lexical", "dense", "hybrid"] = "lexical",
    top_k: int = 5,
    candidate_k: int = 20,
    reranker=None,
    use_metadata_filters: bool = False,
    use_scope_routing: bool = False,
    adaptive_candidate_budget: bool = False,
    max_candidate_k: int = 100,
) -> DiagnosticEvaluation:
    results: list[DiagnosticQueryResult] = []
    accepted = [item for item in dataset.queries if item.status == "accepted"]
    plans = [
        plan_candidate_budget(
            item.query,
            candidate_k,
            maximum_candidate_k=max_candidate_k,
            enabled=adaptive_candidate_budget,
        )[1]
        for item in accepted
    ]
    batch_filters = [
        (
            SearchFilters(company_names=[item.company], report_years=[item.year])
            if use_metadata_filters
            else None
        )
        for item in accepted
    ]
    dense_batch = None
    if mode in {"dense", "hybrid"}:
        dense_batch = index.search_dense_batch(
            [item.query for item in accepted],
            top_k=max(plan.effective_candidate_k for plan in plans),
            filters=batch_filters,
        )
    for position, diagnostic in enumerate(accepted):
        relevant = {
            item.chunk_id for item in diagnostic.judgments if item.label == "relevant"
        }
        effective_candidate_k = plans[position].effective_candidate_k
        retrieval_k = (
            effective_candidate_k if reranker or use_scope_routing else top_k
        )
        if mode == "lexical":
            hits = index.search_lexical(diagnostic.query, retrieval_k, batch_filters[position])
        elif mode == "dense":
            hits = dense_batch[position][:retrieval_k]
        else:
            lexical = index.search_lexical(
                diagnostic.query, effective_candidate_k, batch_filters[position]
            )
            dense = dense_batch[position][:effective_candidate_k]
            hits = reciprocal_rank_fusion(
                lexical,
                dense,
                top_k=(
                    2 * effective_candidate_k
                    if use_scope_routing
                    else retrieval_k
                ),
                rrf_k=60,
            )
        candidate_ranks = [
            hit.rank for hit in hits if hit.chunk.chunk_id in relevant
        ]
        candidate_first_rank = min(candidate_ranks, default=None)
        candidate_pool_size = len(hits)
        inferred_scope = None
        if use_scope_routing:
            scope, hits = route_by_scope(
                diagnostic.query, hits, effective_candidate_k if reranker else top_k
            )
            inferred_scope = scope.name
        if reranker:
            hits = reranker.rerank(diagnostic.query, hits, top_k)
        ranks = [hit.rank for hit in hits if hit.chunk.chunk_id in relevant]
        first_rank = min(ranks, default=None)
        top_hits = hits[:top_k]
        retrieved_relevant = [
            hit for hit in top_hits if hit.chunk.chunk_id in relevant
        ]
        precision_at_k = len(retrieved_relevant) / top_k
        recall_at_k = len(retrieved_relevant) / len(relevant)
        dcg = sum(
            1 / math.log2(rank + 1)
            for rank, hit in enumerate(top_hits, start=1)
            if hit.chunk.chunk_id in relevant
        )
        ideal_count = min(len(relevant), top_k)
        ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg_at_k = dcg / ideal_dcg if ideal_dcg else 0.0
        results.append(
            DiagnosticQueryResult(
                query_id=diagnostic.query_id,
                first_relevant_rank=first_rank,
                hit_at_k=first_rank is not None and first_rank <= top_k,
                reciprocal_rank=1 / first_rank if first_rank is not None else 0,
                ranked_chunk_ids=[hit.chunk.chunk_id for hit in top_hits],
                inferred_scope=inferred_scope,
                effective_candidate_k=effective_candidate_k,
                candidate_first_rank=candidate_first_rank,
                candidate_recall=candidate_first_rank is not None,
                candidate_pool_size=candidate_pool_size,
                relevant_count=len(relevant),
                retrieved_relevant_count=len(retrieved_relevant),
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                ndcg_at_k=ndcg_at_k,
            )
        )
    if not results:
        raise ValueError("The dataset contains no accepted diagnostic queries")
    return DiagnosticEvaluation(
        dataset_id=dataset.dataset_id,
        index_id=index.manifest.index_id,
        mode=mode,
        reranker_model=reranker.model_name if reranker else None,
        evaluated_query_count=len(results),
        top_k=top_k,
        candidate_k=candidate_k,
        metadata_filters=use_metadata_filters,
        scope_routing=use_scope_routing,
        adaptive_candidate_budget=adaptive_candidate_budget,
        average_effective_candidate_k=(
            sum(item.effective_candidate_k for item in results) / len(results)
        ),
        candidate_recall_rate=sum(item.candidate_recall for item in results) / len(results),
        hit_at_k=sum(item.hit_at_k for item in results) / len(results),
        mrr_at_k=sum(item.reciprocal_rank for item in results) / len(results),
        precision_at_k=sum(item.precision_at_k for item in results) / len(results),
        recall_at_k=sum(item.recall_at_k for item in results) / len(results),
        ndcg_at_k=sum(item.ndcg_at_k for item in results) / len(results),
        results=results,
    )


def analyze_recall_failures(
    dataset: DiagnosticDataset,
    evaluation: DiagnosticEvaluation,
    index: PersistentIndex,
) -> RecallFailureReport:
    diagnostics = {item.query_id: item for item in dataset.queries}
    missed = [item for item in evaluation.results if not item.candidate_recall]
    filters = [
        (
            SearchFilters(company_names=[diagnostics[item.query_id].company], report_years=[diagnostics[item.query_id].year])
            if evaluation.metadata_filters
            else None
        )
        for item in missed
    ]
    dense_results = (
        index.search_dense_batch(
            [diagnostics[item.query_id].query for item in missed],
            top_k=index.manifest.chunk_count,
            filters=filters,
        )
        if missed and evaluation.mode in {"dense", "hybrid"}
        else [[] for _ in missed]
    )
    failures: list[RecallFailure] = []
    for position, result in enumerate(missed):
        diagnostic = diagnostics[result.query_id]
        relevant = {
            item.chunk_id for item in diagnostic.judgments if item.label == "relevant"
        }
        lexical = (
            index.search_lexical(
                diagnostic.query, index.manifest.chunk_count, filters[position]
            )
            if evaluation.mode in {"lexical", "hybrid"}
            else []
        )
        dense = dense_results[position]
        fused = (
            reciprocal_rank_fusion(
                lexical,
                dense,
                top_k=index.manifest.chunk_count,
                rrf_k=60,
            )
            if evaluation.mode == "hybrid"
            else lexical if evaluation.mode == "lexical" else dense
        )

        lexical_rank = _first_relevant_rank(lexical, relevant)
        dense_rank = _first_relevant_rank(dense, relevant)
        fused_rank = _first_relevant_rank(fused, relevant)
        gold = index._load_chunks(list(relevant))
        if len(gold) != len(relevant):
            failure_type = "metadata_filter_mismatch"
            explanation = "One or more gold chunk IDs are absent from the active index"
        elif fused_rank is not None and fused_rank > result.effective_candidate_k:
            if (
                lexical_rank is not None
                and lexical_rank <= result.effective_candidate_k
            ) or (dense_rank is not None and dense_rank <= result.effective_candidate_k):
                failure_type = "fusion_displacement"
                explanation = "A component retrieved gold within budget, but RRF pushed it out"
            else:
                failure_type = "candidate_budget_too_small"
                explanation = "Gold exists in the filtered corpus but ranks below the candidate budget"
        elif fused_rank is not None:
            failure_type = "downstream_ranking"
            explanation = "Gold entered the candidate pool but was lost downstream"
        else:
            failure_type = "unresolved"
            explanation = "Gold received no rank from the enabled retrievers"
        failures.append(
            RecallFailure(
                query_id=result.query_id,
                query=diagnostic.query,
                relevant_chunk_ids=sorted(relevant),
                effective_candidate_k=result.effective_candidate_k,
                lexical_first_rank=lexical_rank,
                dense_first_rank=dense_rank,
                fused_first_rank=fused_rank,
                failure_type=failure_type,
                explanation=explanation,
            )
        )
    counts = Counter(item.failure_type for item in failures)
    return RecallFailureReport(
        dataset_id=dataset.dataset_id,
        evaluation_index_id=evaluation.index_id,
        failure_count=len(failures),
        failures_by_type=dict(counts),
        failures=failures,
    )
