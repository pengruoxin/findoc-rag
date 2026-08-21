"""Provider-neutral, auditable task runtime for FinDocRAG agents."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field

from findoc_rag.answer_generation import (
    CITATION_ORDINAL_PATTERN,
    AnswerGenerator,
    Citation,
    ClaimCitation,
    GeneratedAnswer,
)
from findoc_rag.indexing import IndexManifest, SearchFilters, SearchHit
from findoc_rag.query_routing import route_finance_query
from findoc_rag.service import SearchRequest, SearchResponse
from findoc_rag.table_cell_proof import (
    TableCellGeometryProof,
    build_table_cell_proofs,
)

YEAR_PATTERN = re.compile(r"20\d{2}")
DOCUMENT_YEAR_PATTERN = re.compile(
    r"(?P<year>20\d{2})\s*年?\s*(?:年报|年度报告)"
)
TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
PLANNER_REVISION = "deterministic-document-tasks-p3a-atomic-fact-ledger"
StatementScope = Literal["consolidated", "parent", "unspecified"]
ValueVersion = Literal["as_reported", "restated", "current", "unspecified"]
FactSubjectScope = Literal["group", "business_segment", "document", "unspecified"]
FactEvidenceType = Literal[
    "table_value",
    "narrative",
    "audit_risk",
    "audit_response",
    "accounting_policy",
    "other",
]
DocumentScopeResolution = Literal[
    "explicit_document_year",
    "latest_index_document_for_forecast",
    "mentioned_available_year",
    "unique_index_document",
]
StopReason = Literal[
    "sufficient_evidence",
    "needs_clarification",
    "no_new_evidence",
    "tool_budget_exhausted",
    "max_rounds",
    "tool_error",
    "model_budget_exhausted",
    "invalid_model_output",
    "model_error",
    "evidence_verifier_rejected",
    "evidence_verifier_error",
    "evidence_verifier_manual_review",
    "claim_risk_gate_rejected",
]


class AgentTaskRequest(BaseModel):
    """Bounded user task accepted by the first agent runtime."""

    model_config = ConfigDict(extra="forbid")

    task_type: Literal["compare", "extract", "calculate"] = "compare"
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["lexical", "dense", "hybrid"] = "lexical"
    top_k: int = Field(default=3, ge=1, le=10)
    max_rounds: int = Field(default=3, ge=1, le=4)
    max_tool_calls: int = Field(default=8, ge=1, le=8)
    min_evidence_per_target: int = Field(default=1, ge=1, le=3)


class ComparisonTarget(BaseModel):
    target_id: str
    label: str
    filters: SearchFilters
    fact_year: int | None = None
    document_year: int | None = None
    statement_scope: StatementScope = "unspecified"
    value_version: ValueVersion = "unspecified"
    retrieval_hint: str | None = None
    cross_target_group: str | None = None


class AtomicFactRequirement(BaseModel):
    """One answer obligation produced before extraction finalization."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(pattern=r"^r[1-9]\d*$")
    description: str = Field(min_length=1, max_length=300)
    subject: str | None = Field(default=None, max_length=100)
    subject_scope: FactSubjectScope = "unspecified"
    fact_period: str | None = Field(default=None, max_length=50)
    evidence_type: FactEvidenceType = "other"
    candidate_evidence_chunk_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
    )


class AgentTaskPlan(BaseModel):
    planner_revision: str = PLANNER_REVISION
    task_type: Literal["compare", "extract", "calculate"] = "compare"
    targets: list[ComparisonTarget] = Field(default_factory=list)
    document_scope: SearchFilters | None = None
    document_year: int | None = None
    fact_periods: list[int] = Field(default_factory=list)
    required_metrics: list[str] = Field(default_factory=list)
    fact_requirements: list[AtomicFactRequirement] = Field(default_factory=list)
    document_scope_resolution: DocumentScopeResolution | None = None
    retrieval_hint: str | None = None
    clarification: str | None = None
    corpus_unavailable_reason: str | None = None


class AgentToolCall(BaseModel):
    call_id: str
    round_number: int = Field(ge=1)
    tool: Literal[
        "search_evidence",
        "search_authoritative_source",
        "get_page_window",
        "reconstruct_page_layout",
        "inspect_page_region",
        "calculate",
    ] = "search_evidence"
    target_id: str = ""
    query: str
    filters: SearchFilters
    status: Literal["success", "error"]
    duration_ms: float = Field(ge=0)
    retrieval_trace_id: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class AgentEvidence(BaseModel):
    chunk_id: str
    content_sha256: str
    target_ids: list[str]
    document_id: str
    document_key: str | None = None
    company_name: str | None = None
    report_year: int | None = None
    page_start: int
    page_end: int
    section_path: list[str]
    excerpt: str
    table_cell_proofs: list[TableCellGeometryProof] = Field(default_factory=list)


class EvidenceMemory(BaseModel):
    index_id: str
    items: list[AgentEvidence] = Field(default_factory=list)


class SufficiencyDecision(BaseModel):
    status: Literal["sufficient", "incomplete"]
    evidence_count_by_target: dict[str, int]
    gaps: list[str]
    metric_gaps_by_target: dict[str, list[str]] = Field(default_factory=dict)
    requirement_gaps: list[str] = Field(default_factory=list)


class AgentTaskResult(BaseModel):
    outcome: Literal["answer", "evidence_only", "clarify", "abstain"]
    answer: GeneratedAnswer
    target_evidence: dict[str, list[str]]
    target_answers: dict[str, GeneratedAnswer] = Field(default_factory=dict)
    requirement_claims: dict[str, list[str]] = Field(default_factory=dict)
    requirement_evidence: dict[str, list[str]] = Field(default_factory=dict)
    requirement_scope_validated: dict[str, bool] = Field(default_factory=dict)


class AgentModelTurn(BaseModel):
    turn_number: int = Field(ge=1)
    finish_reason: str | None = None
    function_names: list[str]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    elapsed_ms: float = Field(ge=0)
    validation_errors: list[str] = Field(default_factory=list)


class AgentModelTrace(BaseModel):
    provider: str
    model: str
    endpoint: str
    prompt_revision: str
    prompt_sha256: str
    turns: list[AgentModelTurn]

    @computed_field
    @property
    def request_count(self) -> int:
        return len(self.turns)

    @computed_field
    @property
    def input_tokens(self) -> int | None:
        values = [turn.input_tokens for turn in self.turns]
        return (
            sum(value for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )

    @computed_field
    @property
    def output_tokens(self) -> int | None:
        values = [turn.output_tokens for turn in self.turns]
        return (
            sum(value for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )


class EvidenceVerificationFinding(BaseModel):
    """One structured verifier judgment without free-form reasoning traces."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(pattern=r"^r[1-9]\d*$")
    verdict: Literal[
        "supported",
        "incomplete",
        "contradicted",
        "insufficient_evidence",
    ]
    feedback: str = Field(min_length=1, max_length=500)
    evidence_chunk_ids: list[str] = Field(default_factory=list, max_length=5)
    missing_supported_details: list[str] = Field(default_factory=list, max_length=8)


class EvidenceSupportQuote(BaseModel):
    """One verbatim evidence span used by a verifier support proof."""

    model_config = ConfigDict(extra="forbid")

    evidence_chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=320)


class EvidenceSupportProof(BaseModel):
    """Auditable requirement-to-claim-to-evidence support, without hidden reasoning."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(pattern=r"^r[1-9]\d*$")
    claim: str = Field(min_length=1, max_length=1000)
    evidence_quotes: list[EvidenceSupportQuote] = Field(
        min_length=1,
        max_length=3,
    )


class EvidenceVerificationTurn(BaseModel):
    """Auditable metadata for one verifier or bounded repair request."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal[
        "initial_verification",
        "support_proof_retry",
        "support_proof_challenge",
        "repair",
        "post_repair_verification",
    ]
    role: Literal["verifier", "optimizer"]
    provider: str
    model: str
    endpoint: str
    decision: Literal[
        "accept",
        "revise",
        "abstain",
        "manual_review",
        "error",
    ]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    elapsed_ms: float = Field(ge=0)
    supported_requirement_ids: list[str] = Field(default_factory=list)
    support_proofs: list[EvidenceSupportProof] = Field(default_factory=list)
    challenge_requirement_ids: list[str] = Field(default_factory=list)
    findings: list[EvidenceVerificationFinding] = Field(default_factory=list)
    manual_review_reasons: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


class EvidenceVerificationTrace(BaseModel):
    """Separate-context evidence-verifier workflow attached to a task trace."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    prompt_revision: str
    prompt_sha256: str
    routed: bool
    route_reason: str
    repair_attempted: bool = False
    verification_retry_attempted: bool = False
    final_decision: Literal[
        "not_routed",
        "accept_original",
        "accept_repaired",
        "abstain",
        "manual_review",
        "error",
    ]
    human_review_required: bool = False
    human_review_reasons: list[str] = Field(default_factory=list)
    candidate_result: AgentTaskResult | None = None
    turns: list[EvidenceVerificationTurn] = Field(default_factory=list)

    @computed_field
    @property
    def request_count(self) -> int:
        return len(self.turns)

    @computed_field
    @property
    def input_tokens(self) -> int | None:
        values = [turn.input_tokens for turn in self.turns]
        return (
            sum(value for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )

    @computed_field
    @property
    def output_tokens(self) -> int | None:
        values = [turn.output_tokens for turn in self.turns]
        return (
            sum(value for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )


class ClaimRiskFinding(BaseModel):
    """One deterministic claim/evidence conflict found without a model call."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(pattern=r"^r[1-9]\d*$")
    check: Literal[
        "subject_conflict",
        "period_conflict",
        "unsupported_numeric",
        "unsupported_unit",
        "citation_scope_conflict",
        "low_evidence_language_coverage",
        "requirement_claim_divergence",
        "accounting_sign_conflict",
        "missing_requirement_numeric",
    ]
    claim: str = Field(min_length=1, max_length=1000)
    detail: str = Field(min_length=1, max_length=500)
    evidence_chunk_ids: list[str] = Field(default_factory=list, max_length=5)


class ClaimRiskGateTrace(BaseModel):
    """Trace for the zero-token deterministic claim risk gate."""

    model_config = ConfigDict(extra="forbid")

    revision: str
    status: Literal["pass", "review", "reject", "not_applicable"]
    checked_requirement_count: int = Field(ge=0)
    findings: list[ClaimRiskFinding] = Field(default_factory=list)


class AgentTaskTrace(BaseModel):
    schema_version: Literal["1"] = "1"
    task_id: str
    task_type: Literal["compare", "extract", "calculate"] = "compare"
    runtime: Literal["deterministic_baseline", "deepseek_tool_calling"] = "deterministic_baseline"
    status: Literal["completed", "needs_clarification", "failed"]
    stop_reason: StopReason
    query: str
    index_id: str
    created_at: datetime
    completed_at: datetime
    rounds_completed: int = Field(ge=0)
    plan: AgentTaskPlan
    tool_calls: list[AgentToolCall]
    evidence_memory: EvidenceMemory
    sufficiency: SufficiencyDecision
    result: AgentTaskResult
    model_trace: AgentModelTrace | None = None
    claim_risk_gate: ClaimRiskGateTrace | None = None
    evidence_verification: EvidenceVerificationTrace | None = None


class AgentSearchBackend(Protocol):
    @property
    def manifest(self) -> IndexManifest: ...

    def search(self, request: SearchRequest, request_id: str) -> SearchResponse: ...


def _ordered_unique[UniqueValue: (str, int)](
    values: list[UniqueValue],
) -> list[UniqueValue]:
    return list(dict.fromkeys(values))


def _mentioned_companies(query: str, available_companies: list[str]) -> list[str]:
    exact = sorted(
        (company for company in available_companies if company in query),
        key=query.find,
    )
    routed = route_finance_query(query).company_names
    return _ordered_unique(exact + routed)


def _document_years(query: str) -> list[int]:
    return _ordered_unique(
        [int(match.group("year")) for match in DOCUMENT_YEAR_PATTERN.finditer(query)]
    )


def _metric_slug(query: str) -> str:
    metrics = {
        "营业收入": "revenue",
        "营业成本": "operating_cost",
        "净利润": "net_profit",
        "总资产": "total_assets",
        "净资产": "net_assets",
        "现金流量净额": "cash_flow",
    }
    return next((slug for cue, slug in metrics.items() if cue in query), "metric")


def _metric_label(query: str) -> str:
    return next(
        (
            cue
            for cue in (
                "营业收入",
                "营业成本",
                "净利润",
                "总资产",
                "净资产",
                "现金流量净额",
            )
            if cue in query
        ),
        "主要会计数据",
    )


METRIC_REQUIREMENTS: tuple[
    tuple[str, tuple[str, ...], tuple[str, ...]], ...
] = (
    (
        "归母营运利润",
        ("归母营运利润", "归属于母公司股东的营运利润"),
        ("归母营运利润", "归属于母公司股东的营运利润"),
    ),
    (
        "归母净利润",
        ("归母净利润", "归属于母公司股东的净利润", "归属于上市公司股东的净利润"),
        ("归母净利润", "归属于母公司股东的净利润", "归属于上市公司股东的净利润"),
    ),
    (
        "经营活动现金流量净额",
        ("经营活动现金流量净额", "经营活动产生的现金流量净额"),
        ("经营活动现金流量净额", "经营活动产生的现金流量净额"),
    ),
    (
        "客户贷款及垫款总额",
        ("客户贷款及垫款总额", "客户贷款及垫款"),
        ("客户贷款及垫款总额", "客户贷款及垫款"),
    ),
    ("研发投入金额", ("研发投入金额", "研发投入"), ("研发投入金额", "研发投入")),
    (
        "研发投入强度",
        ("研发投入强度", "研发投入占营业收入比例"),
        ("研发投入强度", "研发投入占营业收入比例"),
    ),
    ("研发人员数量", ("研发人员数量", "研发人员"), ("研发人员数量", "研发人员")),
    ("个人客户数", ("个人客户数", "个人客户"), ("个人客户数", "个人客户")),
    ("客均合同数", ("客均合同数",), ("客均合同数",)),
    ("不良贷款率", ("不良贷款率",), ("不良贷款率",)),
    ("拨备覆盖率", ("拨备覆盖率",), ("拨备覆盖率",)),
    ("资本充足率", ("资本充足率",), ("资本充足率",)),
    ("成本收入比", ("成本收入比",), ("成本收入比",)),
    ("资产总额", ("资产总额", "总资产"), ("资产总额", "总资产")),
    ("客户存款", ("客户存款",), ("客户存款",)),
    ("新业务价值", ("寿险新业务价值", "新业务价值"), ("新业务价值",)),
    ("内含价值", ("内含价值",), ("内含价值",)),
    ("留存率", ("留存率",), ("留存率",)),
    ("净利息收益率", ("净利息收益率",), ("净利息收益率",)),
    ("净利息收入", ("净利息收入", "利息净收入"), ("净利息收入", "利息净收入")),
    ("营业收入", ("营业收入",), ("营业收入",)),
    ("毛利率", ("毛利率",), ("毛利率",)),
    ("净利润", ("净利润",), ("净利润",)),
    ("分部收入", ("收入",), ("营业收入", "收入")),
)
METRIC_EVIDENCE_ALIASES = {
    canonical: aliases for canonical, _, aliases in METRIC_REQUIREMENTS
}


def extract_metric_requirements(query: str) -> list[str]:
    """Extract non-overlapping financial metric concepts in query order."""

    candidates: list[tuple[int, int, str]] = []
    for canonical, query_aliases, _ in METRIC_REQUIREMENTS:
        for alias in query_aliases:
            start = query.find(alias)
            while start >= 0:
                candidates.append((start, start + len(alias), canonical))
                start = query.find(alias, start + 1)
    occupied: list[tuple[int, int]] = []
    selected: list[tuple[int, str]] = []
    seen: set[str] = set()
    for start, end, canonical in sorted(
        candidates, key=lambda item: (item[0], -(item[1] - item[0]))
    ):
        if canonical in seen or any(start < right and end > left for left, right in occupied):
            continue
        occupied.append((start, end))
        selected.append((start, canonical))
        seen.add(canonical)
    return [canonical for _, canonical in sorted(selected)]


def missing_required_metrics(
    plan: AgentTaskPlan,
    memory: EvidenceMemory,
    target_id: str,
) -> list[str]:
    excerpts = " ".join(
        evidence.excerpt
        for evidence in memory.items
        if target_id in evidence.target_ids
    )
    compact = re.sub(r"\s+", "", excerpts)
    return [
        metric
        for metric in plan.required_metrics
        if not any(
            re.sub(r"\s+", "", alias) in compact
            for alias in METRIC_EVIDENCE_ALIASES.get(metric, (metric,))
        )
    ]


def _retrieval_hint(
    query: str,
    *,
    fact_year: int | None,
    document_year: int | None,
    statement_scope: StatementScope = "unspecified",
    value_version: ValueVersion = "unspecified",
) -> str:
    required_metrics = extract_metric_requirements(query)
    metric = " ".join(required_metrics) or _metric_label(query)
    years = " ".join(
        f"{year}年" for year in _ordered_unique([document_year, fact_year]) if year is not None
    )
    if statement_scope == "parent":
        return f"母公司财务报表主要项目注释 {metric}和营业成本 本期发生额 合计 单位 元"
    section_hints = ["近三年主要会计数据", "主要财务指标"]
    if any(
        item in required_metrics
        for item in ("归母营运利润", "新业务价值", "内含价值")
    ):
        section_hints.extend(["主要经营指标", "寿险及健康险", "可比口径"])
    if any(item in required_metrics for item in ("个人客户数", "客均合同数", "留存率")):
        section_hints.append("个人客户")
    if "毛利率" in required_metrics:
        section_hints.extend(["主营业务分析", "分产品"])
    parts = [*section_hints, metric, years]
    if (
        "调整后" in query
        or "调整前" in query
        or value_version
        in {
            "as_reported",
            "restated",
        }
        or (fact_year is not None and document_year is not None and fact_year != document_year)
    ):
        parts.extend(["调整后", "调整前"])
    parts.extend(["单位", "元"])
    return " ".join(part for part in parts if part)


def _scope_comparison_plan(
    request: AgentTaskRequest,
    *,
    companies: list[str],
    years: list[int],
) -> AgentTaskPlan | None:
    if not ("合并" in request.query and "母公司" in request.query):
        return None
    if len(companies) != 1 or len(years) != 1:
        return AgentTaskPlan(
            targets=[], clarification="请明确一个公司和一个报告年度后再比较报表口径。"
        )
    company = companies[0]
    year = years[0]
    targets = [
        ComparisonTarget(
            target_id=f"scope:{company}:consolidated:year:{year}",
            label=f"{company}-{year}-合并口径",
            filters=SearchFilters(company_names=[company], report_years=[year]),
            fact_year=year,
            document_year=year,
            statement_scope="consolidated",
            value_version="current",
            retrieval_hint=_retrieval_hint(
                request.query,
                fact_year=year,
                document_year=year,
                statement_scope="consolidated",
                value_version="current",
            ),
            cross_target_group=f"scope:{company}:year:{year}",
        ),
        ComparisonTarget(
            target_id=f"scope:{company}:parent:year:{year}",
            label=f"{company}-{year}-母公司口径",
            filters=SearchFilters(company_names=[company], report_years=[year]),
            fact_year=year,
            document_year=year,
            statement_scope="parent",
            value_version="current",
            retrieval_hint=_retrieval_hint(
                request.query,
                fact_year=year,
                document_year=year,
                statement_scope="parent",
                value_version="current",
            ),
            cross_target_group=f"scope:{company}:year:{year}",
        ),
    ]
    return AgentTaskPlan(targets=targets)


def _restatement_version_plan(
    request: AgentTaskRequest,
    *,
    companies: list[str],
    years: list[int],
    document_years: list[int],
) -> AgentTaskPlan | None:
    if not ("调整前" in request.query and "调整后" in request.query):
        return None
    if len(companies) != 1 or not document_years:
        return None
    document_year = document_years[-1]
    fact_years = [year for year in years if year != document_year]
    if len(fact_years) != 1:
        return None
    company = companies[0]
    fact_year = fact_years[0]
    metric = _metric_slug(request.query)
    filters = SearchFilters(company_names=[company], report_years=[document_year])
    return AgentTaskPlan(
        targets=[
            ComparisonTarget(
                target_id=f"fact:{company}:{metric}:year:{fact_year}:as_reported",
                label=f"{company}-{fact_year}-调整前",
                filters=filters,
                fact_year=fact_year,
                document_year=document_year,
                value_version="as_reported",
                retrieval_hint=_retrieval_hint(
                    request.query,
                    fact_year=fact_year,
                    document_year=document_year,
                    value_version="as_reported",
                ),
                cross_target_group=(
                    f"fact:{company}:{metric}:year:{fact_year}:document:{document_year}"
                ),
            ),
            ComparisonTarget(
                target_id=(f"fact:{company}:{metric}:year:{fact_year}:restated_in_{document_year}"),
                label=f"{company}-{fact_year}-调整后（{document_year}年报）",
                filters=filters,
                fact_year=fact_year,
                document_year=document_year,
                value_version="restated",
                retrieval_hint=_retrieval_hint(
                    request.query,
                    fact_year=fact_year,
                    document_year=document_year,
                    value_version="restated",
                ),
                cross_target_group=(
                    f"fact:{company}:{metric}:year:{fact_year}:document:{document_year}"
                ),
            ),
        ]
    )


def plan_compare_task(
    request: AgentTaskRequest,
    *,
    available_companies: list[str],
) -> AgentTaskPlan:
    """Build a bounded comparison plan using only query and index metadata."""

    companies = _mentioned_companies(request.query, available_companies)
    years = _ordered_unique([int(value) for value in YEAR_PATTERN.findall(request.query)])
    document_years = _document_years(request.query)
    targets: list[ComparisonTarget] = []

    scope_plan = _scope_comparison_plan(request, companies=companies, years=years)
    if scope_plan is not None:
        return scope_plan
    restatement_plan = _restatement_version_plan(
        request,
        companies=companies,
        years=years,
        document_years=document_years,
    )
    if restatement_plan is not None:
        return restatement_plan

    adjusted_comparative = "调整后" in request.query and bool(years)
    comparative_document_year = (
        document_years[-1] if adjusted_comparative and document_years else None
    )
    grouped_comparison = any(cue in request.query for cue in ("调整", "同比", "增长率"))

    if len(companies) >= 2:
        target_years: list[int | None] = years if len(years) >= 2 else [years[0] if years else None]
        for company in companies:
            for year in target_years:
                label = (
                    f"{company}-{year}" if year is not None and len(target_years) > 1 else company
                )
                document_year = (
                    comparative_document_year
                    if year is not None
                    and comparative_document_year is not None
                    and year <= comparative_document_year
                    else year
                )
                value_version: ValueVersion = (
                    "restated"
                    if year is not None
                    and comparative_document_year is not None
                    and year < comparative_document_year
                    else "current"
                )
                targets.append(
                    ComparisonTarget(
                        target_id=f"company:{company}:year:{year or 'any'}",
                        label=label,
                        filters=SearchFilters(
                            company_names=[company],
                            report_years=[document_year] if document_year is not None else [],
                        ),
                        fact_year=year,
                        document_year=document_year,
                        value_version=value_version,
                        retrieval_hint=_retrieval_hint(
                            request.query,
                            fact_year=year,
                            document_year=document_year,
                            value_version=value_version,
                        ),
                        cross_target_group=(
                            f"company:{company}:comparative-table" if grouped_comparison else None
                        ),
                    )
                )
    elif len(years) >= 2:
        company = companies[0] if companies else None
        for year in years:
            label = f"{company}-{year}" if company else str(year)
            document_year = (
                comparative_document_year
                if comparative_document_year is not None and year <= comparative_document_year
                else year
            )
            value_version = (
                "restated"
                if comparative_document_year is not None and year < comparative_document_year
                else "current"
            )
            targets.append(
                ComparisonTarget(
                    target_id=f"company:{company or 'any'}:year:{year}",
                    label=label,
                    filters=SearchFilters(
                        company_names=[company] if company else [],
                        report_years=[document_year],
                    ),
                    fact_year=year,
                    document_year=document_year,
                    value_version=value_version,
                    retrieval_hint=_retrieval_hint(
                        request.query,
                        fact_year=year,
                        document_year=document_year,
                        value_version=value_version,
                    ),
                    cross_target_group=(
                        f"company:{company or 'any'}:comparative-table"
                        if grouped_comparison
                        else None
                    ),
                )
            )
    else:
        return AgentTaskPlan(
            targets=[],
            clarification="请至少明确两个比较对象，例如两家公司或两个报告年度。",
        )

    if len(targets) > request.max_tool_calls:
        return AgentTaskPlan(
            targets=[],
            clarification=(
                f"当前比较会展开为 {len(targets)} 个对象，超过本次 "
                f"{request.max_tool_calls} 次工具调用预算；请缩小公司或年度范围。"
            ),
        )
    return AgentTaskPlan(
        task_type="compare",
        targets=targets,
        required_metrics=extract_metric_requirements(request.query),
    )


def plan_document_task(
    request: AgentTaskRequest,
    *,
    available_companies: list[str],
    available_report_years_by_company: dict[str, list[int]] | None = None,
) -> AgentTaskPlan:
    """Plan a single-document extraction without inventing comparison targets."""

    if request.task_type not in {"extract", "calculate"}:
        raise ValueError("plan_document_task only accepts extract or calculate tasks")
    companies = _mentioned_companies(request.query, available_companies)
    years = _ordered_unique([int(value) for value in YEAR_PATTERN.findall(request.query)])
    explicit_document_years = _document_years(request.query)
    if len(companies) != 1:
        return AgentTaskPlan(
            task_type=request.task_type,
            clarification="请明确一家公司和一个报告年度，以限定文档范围。",
        )
    company = companies[0]
    available_years = _ordered_unique(
        (available_report_years_by_company or {}).get(company, [])
    )
    forecast_cues = ("预计", "预测", "目标", "计划", "指引", "展望", "力争")
    is_forecast_request = any(cue in request.query for cue in forecast_cues)
    if (
        len(years) == 1
        and available_years
        and years[0] > max(available_years)
        and not explicit_document_years
        and not is_forecast_request
    ):
        requested_year = years[0]
        return AgentTaskPlan(
            task_type=request.task_type,
            fact_periods=[requested_year],
            corpus_unavailable_reason=(
                f"当前索引中{company}仅包含报告年度"
                f"{','.join(str(year) for year in available_years)}，"
                f"没有{requested_year}年实际事实的可验证年报证据。"
            ),
        )
    document_year: int | None = None
    fact_periods: list[int] = []
    resolution: DocumentScopeResolution | None = None

    if len(explicit_document_years) == 1:
        document_year = explicit_document_years[0]
        fact_periods = [year for year in years if year != document_year]
        if not fact_periods:
            fact_periods = [document_year]
        resolution = "explicit_document_year"
    elif len(explicit_document_years) > 1:
        document_year = None
    elif len(years) == 1 and (not available_years or years[0] in available_years):
        document_year = years[0]
        fact_periods = list(years)
        resolution = "mentioned_available_year"
    elif len(years) == 1 and is_forecast_request and available_years:
        document_year = max(available_years)
        fact_periods = list(years)
        resolution = "latest_index_document_for_forecast"
    elif len(available_years) == 1:
        document_year = available_years[0]
        fact_periods = [year for year in years if year != document_year]
        resolution = "unique_index_document"
    elif len(years) > 1:
        matching_years = [year for year in years if year in available_years]
        if len(matching_years) == 1:
            document_year = matching_years[0]
            fact_periods = [year for year in years if year != document_year]
            resolution = "mentioned_available_year"

    if document_year is None:
        return AgentTaskPlan(
            task_type=request.task_type,
            fact_periods=years,
            clarification=(
                "无法从问题和当前索引唯一确定报告文档；请明确公司和报告年度。"
            ),
        )

    retrieval_hint = request.query
    if "使用权资产" in request.query and "累计折旧" in request.query:
        retrieval_hint = (
            "使用权资产 二、累计折旧 5.期末余额 合计 "
            "房屋及建筑物 机器设备 运输工具 办公设备 其他"
        )
    return AgentTaskPlan(
        task_type=request.task_type,
        targets=[],
        document_scope=SearchFilters(
            company_names=companies,
            report_years=[document_year],
        ),
        document_year=document_year,
        fact_periods=fact_periods,
        required_metrics=extract_metric_requirements(request.query),
        document_scope_resolution=resolution,
        retrieval_hint=retrieval_hint,
    )


def _evidence_digest(hit: SearchHit) -> str:
    return hashlib.sha256(hit.chunk.model_dump_json().encode("utf-8")).hexdigest()


def add_evidence(
    memory: EvidenceMemory,
    target_id: str,
    hits: list[SearchHit],
) -> int:
    existing = {item.chunk_id: item for item in memory.items}
    added = 0
    for hit in hits:
        item = existing.get(hit.chunk.chunk_id)
        if item is not None:
            if target_id not in item.target_ids:
                item.target_ids.append(target_id)
            continue
        evidence = AgentEvidence(
            chunk_id=hit.chunk.chunk_id,
            content_sha256=_evidence_digest(hit),
            target_ids=[target_id],
            document_id=hit.chunk.document_id,
            document_key=hit.chunk.document_key,
            company_name=hit.chunk.company_name,
            report_year=hit.chunk.report_year,
            page_start=hit.chunk.page_start,
            page_end=hit.chunk.page_end,
            section_path=hit.chunk.section_path,
            excerpt=hit.chunk.text[:1600],
            table_cell_proofs=build_table_cell_proofs(hit.chunk),
        )
        memory.items.append(evidence)
        existing[evidence.chunk_id] = evidence
        added += 1
    return added


def judge_sufficiency(
    plan: AgentTaskPlan,
    memory: EvidenceMemory,
    *,
    minimum_per_target: int,
) -> SufficiencyDecision:
    counts = {
        target.target_id: sum(target.target_id in evidence.target_ids for evidence in memory.items)
        for target in plan.targets
    }
    metric_gaps = {
        target.target_id: missing_required_metrics(plan, memory, target.target_id)
        for target in plan.targets
    }
    metric_gaps = {
        target_id: metrics for target_id, metrics in metric_gaps.items() if metrics
    }
    gaps = [
        target_id
        for target_id, count in counts.items()
        if count < minimum_per_target or target_id in metric_gaps
    ]
    return SufficiencyDecision(
        status="incomplete" if gaps else "sufficient",
        evidence_count_by_target=counts,
        gaps=gaps,
        metric_gaps_by_target=metric_gaps,
    )


def _round_query(query: str, target: ComparisonTarget, round_number: int) -> str:
    if round_number == 1:
        return query
    if round_number == 2:
        return f"{target.label} {query} 年度报告 财务数据"
    return f"{target.label} {query} 财务报表附注"


def target_evidence(plan: AgentTaskPlan, memory: EvidenceMemory) -> dict[str, list[str]]:
    return {
        target.target_id: [
            evidence.chunk_id
            for evidence in memory.items
            if target.target_id in evidence.target_ids
        ]
        for target in plan.targets
    }


def _target_answer_query(
    original_query: str,
    target: ComparisonTarget,
    plan: AgentTaskPlan,
) -> str:
    query = original_query
    target_companies = set(target.filters.company_names)
    target_years = (
        {target.fact_year} if target.fact_year is not None else set(target.filters.report_years)
    )
    for other in plan.targets:
        for company in other.filters.company_names:
            if company not in target_companies:
                query = query.replace(company, "")
    if target_years:
        query = YEAR_PATTERN.sub(
            lambda match: match.group() if int(match.group()) in target_years else "",
            query,
        )
    query = query.replace("比较", "").replace("对比", "").strip(" ，,、和与")
    return f"{target.label} {query}".strip()


def _compose_target_answers(
    plan: AgentTaskPlan,
    target_answers: dict[str, GeneratedAnswer],
) -> tuple[GeneratedAnswer, Literal["answer", "evidence_only", "clarify", "abstain"]]:
    if all(answer.grounded for answer in target_answers.values()):
        citations: list[Citation] = []
        citation_ordinals: dict[str, int] = {}
        claims: list[ClaimCitation] = []
        lines: list[str] = []
        for target in plan.targets:
            target_answer = target_answers[target.target_id]
            local_to_global: dict[int, int] = {}
            for citation in target_answer.citations:
                ordinal = citation_ordinals.get(citation.chunk_id)
                if ordinal is None:
                    ordinal = len(citations) + 1
                    citation_ordinals[citation.chunk_id] = ordinal
                    citations.append(citation.model_copy(update={"ordinal": ordinal}))
                local_to_global[citation.ordinal] = ordinal

            def replace_ordinal(
                match: re.Match[str], mapping: dict[int, int] = local_to_global
            ) -> str:
                local = int(match.group(1))
                return f"[{mapping.get(local, local)}]"

            rewritten = CITATION_ORDINAL_PATTERN.sub(replace_ordinal, target_answer.answer)
            lines.append(f"{target.label}：{rewritten}")
            for claim in target_answer.claim_citations:
                claims.append(
                    ClaimCitation(
                        claim=f"{target.label}：{claim.claim}",
                        citation_ordinals=[
                            local_to_global[ordinal]
                            for ordinal in claim.citation_ordinals
                            if ordinal in local_to_global
                        ],
                    )
                )
        return (
            GeneratedAnswer(
                answer="\n".join(lines),
                citations=citations,
                provider="agent-comparison",
                grounded=True,
                claim_citations=claims,
            ),
            "answer",
        )

    citations = []
    seen_chunks: set[str] = set()
    for target in plan.targets:
        for citation in target_answers[target.target_id].citations:
            if citation.chunk_id in seen_chunks:
                continue
            seen_chunks.add(citation.chunk_id)
            citations.append(citation.model_copy(update={"ordinal": len(citations) + 1}))
    evidence_only_or_grounded = all(
        answer.grounded or answer.provider == "evidence-only" for answer in target_answers.values()
    )
    if evidence_only_or_grounded and any(
        answer.provider == "evidence-only" for answer in target_answers.values()
    ):
        return (
            GeneratedAnswer(
                answer=(
                    "已找到各比较对象的证据，但部分目标尚未形成结构化答案；请查看逐目标答案和证据。"
                ),
                citations=citations,
                provider="agent-comparison-evidence-only",
                grounded=False,
            ),
            "evidence_only",
        )
    return (
        GeneratedAnswer(
            answer="部分比较对象未形成可验证答案，系统拒绝输出不完整比较。",
            citations=citations,
            provider="agent-comparison-incomplete",
            grounded=False,
        ),
        "abstain",
    )


class CompareTaskController:
    """Execute a bounded compare task over an immutable retrieval index."""

    def __init__(
        self,
        search_backend: AgentSearchBackend,
        answer_generator: AnswerGenerator,
        *,
        available_companies: list[str],
    ) -> None:
        self.search_backend = search_backend
        self.answer_generator = answer_generator
        self.available_companies = available_companies

    def run(self, request: AgentTaskRequest) -> AgentTaskTrace:
        task_id = uuid4().hex
        created_at = datetime.now(UTC)
        index_id = self.search_backend.manifest.index_id
        plan = plan_compare_task(request, available_companies=self.available_companies)
        memory = EvidenceMemory(index_id=index_id)
        empty_sufficiency = SufficiencyDecision(
            status="incomplete", evidence_count_by_target={}, gaps=[]
        )
        if plan.clarification:
            answer = GeneratedAnswer(
                answer=plan.clarification,
                citations=[],
                provider="agent-clarification",
                grounded=False,
            )
            return AgentTaskTrace(
                task_id=task_id,
                status="needs_clarification",
                stop_reason="needs_clarification",
                query=request.query,
                index_id=index_id,
                created_at=created_at,
                completed_at=datetime.now(UTC),
                rounds_completed=0,
                plan=plan,
                tool_calls=[],
                evidence_memory=memory,
                sufficiency=empty_sufficiency,
                result=AgentTaskResult(outcome="clarify", answer=answer, target_evidence={}),
            )

        calls: list[AgentToolCall] = []
        hits_by_chunk: dict[str, SearchHit] = {}
        sufficiency = judge_sufficiency(
            plan, memory, minimum_per_target=request.min_evidence_per_target
        )
        stop_reason: StopReason = "max_rounds"
        rounds_completed = 0

        for round_number in range(1, request.max_rounds + 1):
            rounds_completed = round_number
            added_this_round = 0
            for target in plan.targets:
                if target.target_id not in sufficiency.gaps:
                    continue
                if len(calls) >= request.max_tool_calls:
                    stop_reason = "tool_budget_exhausted"
                    break
                query = _round_query(request.query, target, round_number)
                call_id = uuid4().hex
                started = perf_counter()
                try:
                    response = self.search_backend.search(
                        SearchRequest(
                            query=query,
                            mode=request.mode,
                            top_k=request.top_k,
                            filters=target.filters,
                        ),
                        f"{task_id}:{call_id}",
                    )
                    duration_ms = (perf_counter() - started) * 1000
                    calls.append(
                        AgentToolCall(
                            call_id=call_id,
                            round_number=round_number,
                            target_id=target.target_id,
                            query=query,
                            filters=target.filters,
                            status="success",
                            duration_ms=duration_ms,
                            retrieval_trace_id=response.trace_id,
                            evidence_chunk_ids=[hit.chunk.chunk_id for hit in response.hits],
                        )
                    )
                    for hit in response.hits:
                        hits_by_chunk.setdefault(hit.chunk.chunk_id, hit)
                    added_this_round += add_evidence(memory, target.target_id, response.hits)
                except Exception as exc:  # noqa: BLE001 - persist the tool failure
                    calls.append(
                        AgentToolCall(
                            call_id=call_id,
                            round_number=round_number,
                            target_id=target.target_id,
                            query=query,
                            filters=target.filters,
                            status="error",
                            duration_ms=(perf_counter() - started) * 1000,
                            retrieval_trace_id=getattr(exc, "trace_id", None),
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    )
                    stop_reason = "tool_error"
                    sufficiency = judge_sufficiency(
                        plan,
                        memory,
                        minimum_per_target=request.min_evidence_per_target,
                    )
                    answer = GeneratedAnswer(
                        answer="证据工具执行失败，任务已安全停止。",
                        citations=[],
                        provider="agent-tool-error",
                        grounded=False,
                    )
                    return AgentTaskTrace(
                        task_id=task_id,
                        status="failed",
                        stop_reason=stop_reason,
                        query=request.query,
                        index_id=index_id,
                        created_at=created_at,
                        completed_at=datetime.now(UTC),
                        rounds_completed=rounds_completed,
                        plan=plan,
                        tool_calls=calls,
                        evidence_memory=memory,
                        sufficiency=sufficiency,
                        result=AgentTaskResult(
                            outcome="abstain",
                            answer=answer,
                            target_evidence=target_evidence(plan, memory),
                        ),
                    )

            sufficiency = judge_sufficiency(
                plan, memory, minimum_per_target=request.min_evidence_per_target
            )
            if sufficiency.status == "sufficient":
                stop_reason = "sufficient_evidence"
                break
            if len(calls) >= request.max_tool_calls:
                stop_reason = "tool_budget_exhausted"
                break
            if added_this_round == 0:
                stop_reason = "no_new_evidence"
                break

        if sufficiency.status == "sufficient":
            target_answers = {}
            for target in plan.targets:
                target_hits = [
                    hits_by_chunk[evidence.chunk_id]
                    for evidence in memory.items
                    if target.target_id in evidence.target_ids
                    and evidence.chunk_id in hits_by_chunk
                ]
                target_answers[target.target_id] = self.answer_generator.generate(
                    _target_answer_query(request.query, target, plan), target_hits
                )
            answer, outcome = _compose_target_answers(plan, target_answers)
        else:
            missing_labels = [
                target.label for target in plan.targets if target.target_id in sufficiency.gaps
            ]
            answer = GeneratedAnswer(
                answer=f"以下比较对象证据不足：{', '.join(missing_labels)}。",
                citations=[],
                provider="agent-evidence-gap",
                grounded=False,
            )
            outcome = "abstain"
            target_answers = {}

        return AgentTaskTrace(
            task_id=task_id,
            status="completed",
            stop_reason=stop_reason,
            query=request.query,
            index_id=index_id,
            created_at=created_at,
            completed_at=datetime.now(UTC),
            rounds_completed=rounds_completed,
            plan=plan,
            tool_calls=calls,
            evidence_memory=memory,
            sufficiency=sufficiency,
            result=AgentTaskResult(
                outcome=outcome,
                answer=answer,
                target_evidence=target_evidence(plan, memory),
                target_answers=target_answers,
            ),
        )


class AgentTaskStore:
    """Small durable store for replayable task traces."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()

    def path_for(self, task_id: str) -> Path:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError("Invalid task ID")
        return self.directory / f"{task_id}.json"

    def save(self, trace: AgentTaskTrace) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.path_for(trace.task_id)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            trace.model_dump_json(indent=2, exclude_computed_fields=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    def load(self, task_id: str) -> AgentTaskTrace:
        path = self.path_for(task_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("model_trace", "evidence_verification"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                for computed_key in ("request_count", "input_tokens", "output_tokens"):
                    nested.pop(computed_key, None)
        return AgentTaskTrace.model_validate(payload)
