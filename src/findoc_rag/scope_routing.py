from typing import Literal

from pydantic import BaseModel

from findoc_rag.indexing import SearchHit

ScopeName = Literal[
    "annual_summary",
    "quarterly",
    "segment",
    "consolidated_statement",
    "parent_statement",
    "notes",
    "audit",
    "unspecified",
]


class QueryScope(BaseModel):
    name: ScopeName
    confidence: Literal["explicit", "default", "unknown"]
    matched_cues: list[str]


class CandidateBudgetPlan(BaseModel):
    base_candidate_k: int
    effective_candidate_k: int
    inferred_scope: ScopeName
    expanded: bool
    reason: str


SCOPE_CUES: dict[ScopeName, tuple[str, ...]] = {
    "quarterly": ("分季度", "季度", "第一季度", "第二季度", "第三季度", "第四季度"),
    "segment": ("主营业务", "分行业", "分产品", "分地区", "分部"),
    "parent_statement": ("母公司",),
    "consolidated_statement": ("合并利润表", "合并报表", "合并口径"),
    "notes": ("附注", "注释",),
    "audit": ("审计", "关键审计事项", "收入确认"),
}

FINANCIAL_METRICS = (
    "营业收入",
    "营业成本",
    "净利润",
    "每股收益",
    "现金流量净额",
    "总资产",
    "净资产",
)

POSITIVE_SECTION_CUES: dict[ScopeName, tuple[str, ...]] = {
    "annual_summary": ("主要会计数据", "主要财务指标"),
    "quarterly": ("分季度", "季度主要财务数据"),
    "segment": ("主营业务", "分行业", "分产品", "分地区", "分部"),
    "consolidated_statement": ("合并利润表", "合并现金流量表", "合并资产负债表"),
    "parent_statement": ("母公司财务报表", "母公司财务报表主要项目注释"),
    "notes": ("项目注释", "财务报表附注", "营业收入和营业成本"),
    "audit": ("关键审计事项", "审计报告", "收入确认"),
    "unspecified": (),
}

CONFLICTING_SECTION_CUES: dict[ScopeName, tuple[str, ...]] = {
    "annual_summary": ("分季度", "分行业", "分产品", "关键审计事项", "母公司", "项目注释"),
    "quarterly": ("分行业", "分产品", "关键审计事项", "母公司", "项目注释"),
    "segment": ("分季度", "关键审计事项", "母公司", "项目注释"),
    "consolidated_statement": ("母公司", "关键审计事项"),
    "parent_statement": ("合并利润表", "关键审计事项"),
    "notes": ("关键审计事项",),
    "audit": ("主要会计数据", "分季度", "分行业", "分产品"),
    "unspecified": (),
}


def infer_query_scope(query: str) -> QueryScope:
    for scope, cues in SCOPE_CUES.items():
        matches = [cue for cue in cues if cue in query]
        if matches:
            return QueryScope(name=scope, confidence="explicit", matched_cues=matches)
    matches = [metric for metric in FINANCIAL_METRICS if metric in query]
    if matches:
        return QueryScope(name="annual_summary", confidence="default", matched_cues=matches)
    return QueryScope(name="unspecified", confidence="unknown", matched_cues=[])


DEFAULT_SCOPE_BUDGETS: dict[ScopeName, int] = {
    "annual_summary": 100,
    "quarterly": 40,
    "segment": 40,
    "consolidated_statement": 60,
    "parent_statement": 60,
    "notes": 60,
    "audit": 60,
    "unspecified": 20,
}


def plan_candidate_budget(
    query: str,
    base_candidate_k: int,
    maximum_candidate_k: int = 100,
    enabled: bool = True,
) -> tuple[QueryScope, CandidateBudgetPlan]:
    scope = infer_query_scope(query)
    target = DEFAULT_SCOPE_BUDGETS[scope.name] if enabled else base_candidate_k
    effective = min(maximum_candidate_k, max(base_candidate_k, target))
    expanded = effective > base_candidate_k
    reason = (
        f"expanded for {scope.name} scope"
        if expanded
        else "base budget retained"
    )
    return scope, CandidateBudgetPlan(
        base_candidate_k=base_candidate_k,
        effective_candidate_k=effective,
        inferred_scope=scope.name,
        expanded=expanded,
        reason=reason,
    )


def score_scope(hit: SearchHit, scope: QueryScope) -> int:
    if scope.name == "unspecified":
        return 0
    context = " ".join(hit.chunk.section_path)
    positive = sum(cue in context for cue in POSITIVE_SECTION_CUES[scope.name])
    conflicting = sum(cue in context for cue in CONFLICTING_SECTION_CUES[scope.name])
    return 2 * positive - 2 * conflicting


def score_structured_evidence(query: str, hit: SearchHit) -> int:
    """Promote coordinate-verified table families required by the query."""
    table_types = {table.table_type for table in hit.chunk.structured_tables}
    score = 0
    if "季度" in query:
        if "quarterly" in table_types:
            score += 20
        if ("合计" in query or "全年" in query) and "annual_data" in table_types:
            score += 16
    if "毛利率" in query and "segment" in table_types:
        score += 20
    if (
        "主营业务" in query
        and any(cue in query for cue in ("分产品", "分行业", "毛利率"))
        and "segment" in table_types
    ):
        score += 20
    if (
        "合并" in query
        and "母公司" in query
        and "营业收入" in query
        and "note_cost" in table_types
    ):
        score += 20
    if (
        "成本" in query
        and any(cue in query for cue in ("差额", "不同", "为什么", "附注"))
        and "note_cost" in table_types
    ):
        score += 20
    if (
        "营业收入" in query
        and "annual_data" in table_types
        and not any(cue in query for cue in ("合并", "母公司", "主营业务"))
    ):
        score += 20
    if (
        ("归母净利润" in query or "扣非归母净利润" in query)
        and "annual_data" in table_types
    ):
        score += 20
    return score


def route_structured_evidence(
    query: str, hits: list[SearchHit], top_k: int
) -> list[SearchHit]:
    """Rerank a bounded candidate set using only verified table semantics."""
    ranked = sorted(
        hits,
        key=lambda hit: (-score_structured_evidence(query, hit), hit.rank),
    )[:top_k]
    return [
        hit.model_copy(
            update={
                "rank": rank,
                "retrieval_rank": hit.rank,
                "scope_score": score_structured_evidence(query, hit),
                "scope_rank_delta": hit.rank - rank,
            }
        )
        for rank, hit in enumerate(ranked, start=1)
    ]


def route_by_scope(query: str, hits: list[SearchHit], top_k: int) -> tuple[QueryScope, list[SearchHit]]:
    scope = infer_query_scope(query)
    scored = [(hit, score_scope(hit, scope)) for hit in hits]
    ranked = sorted(scored, key=lambda item: (-item[1], item[0].rank))[:top_k]
    return scope, [
        hit.model_copy(
            update={
                "rank": rank,
                "retrieval_rank": hit.rank,
                "scope_score": score,
                "scope_rank_delta": hit.rank - rank,
            }
        )
        for rank, (hit, score) in enumerate(ranked, start=1)
    ]
