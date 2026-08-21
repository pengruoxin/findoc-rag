"""Governance checks for benchmark credibility and document-blind evaluation.

These checks intentionally live outside the frozen benchmark integrity gate.
The integrity gate proves that committed evidence is internally consistent;
this module answers the harder question: whether a benchmark is independent
and broad enough to support external quality claims.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class GovernancePolicy(BaseModel):
    policy_id: str
    split_strategy: str
    document_blind_required: bool = True
    company_blind_required: bool = False
    required_human_reviewers: int = Field(default=2, ge=1)
    minimum_companies: int = Field(default=4, ge=1)
    minimum_report_years: int = Field(default=2, ge=1)
    minimum_frozen_test_items: int = Field(default=24, ge=1)
    require_variant_review: bool = True


class SplitOverlap(BaseModel):
    unit_id: str
    splits: list[str]
    query_ids: list[str]


class GovernanceStats(BaseModel):
    item_count: int
    company_count: int
    report_year_count: int
    frozen_test_item_count: int
    answerable_item_count: int
    fully_reviewed_item_count: int
    review_status_counts: dict[str, int]
    unique_document_count: int


class GovernanceReport(BaseModel):
    policy_id: str
    dataset_id: str
    split_strategy: str
    ready_for_external_claims: bool
    blockers: list[str]
    warnings: list[str]
    stats: GovernanceStats
    document_split_overlaps: list[SplitOverlap]
    company_split_overlaps: list[SplitOverlap]
    entity_period_split_overlaps: list[SplitOverlap]
    family_split_overlaps: list[SplitOverlap]


class PlannedDocument(BaseModel):
    """A source document allocated before question authoring or rule tuning."""

    document_key: str = Field(min_length=1)
    security_code: str = Field(pattern=r"^\d{6}$")
    company_name: str = Field(min_length=1)
    report_year: int = Field(ge=2000, le=2100)
    split: Literal["calibration", "dev", "frozen_test"]
    source: Literal["cninfo"] = "cninfo"
    question_target: int = Field(default=0, ge=0)


class BenchmarkCorpusPlan(BaseModel):
    """Immutable allocation plan for constructing a document-blind benchmark."""

    schema_version: Literal["1"] = "1"
    plan_id: str = Field(min_length=1)
    sealed_at: datetime
    split_strategy: Literal["document_blind"] = "document_blind"
    allocation_unit: Literal["company"] = "company"
    selection_protocol: Literal["allocated_before_question_authoring_or_rule_tuning"]
    documents: list[PlannedDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_documents(self) -> BenchmarkCorpusPlan:
        document_keys = [item.document_key for item in self.documents]
        if len(document_keys) != len(set(document_keys)):
            raise ValueError("document_key values must be unique")
        entity_periods = [
            (item.security_code, item.report_year) for item in self.documents
        ]
        if len(entity_periods) != len(set(entity_periods)):
            raise ValueError("security_code/report_year allocations must be unique")
        return self


class CorpusPlanStats(BaseModel):
    document_count: int
    company_count: int
    report_year_count: int
    split_document_counts: dict[str, int]
    split_question_targets: dict[str, int]


class CorpusPlanAudit(BaseModel):
    policy_id: str
    plan_id: str
    ready_for_acquisition: bool
    blockers: list[str]
    stats: CorpusPlanStats
    company_split_overlaps: list[SplitOverlap]


class PreparedCorpusAudit(BaseModel):
    plan_id: str
    ready_for_annotation: bool
    blockers: list[str]
    document_count: int
    source_hash_count: int
    index_document_counts: dict[str, int]


def _split_overlaps(unit_queries: dict[str, list[tuple[str, str]]]) -> list[SplitOverlap]:
    overlaps: list[SplitOverlap] = []
    for unit_id, observations in sorted(unit_queries.items()):
        splits = sorted({split for split, _ in observations})
        if len(splits) <= 1:
            continue
        overlaps.append(
            SplitOverlap(
                unit_id=unit_id,
                splits=splits,
                query_ids=sorted({query_id for _, query_id in observations}),
            )
        )
    return overlaps


def _approved_reviewers(annotation: dict[str, Any], require_variants: bool) -> set[str]:
    reviewers: set[str] = set()
    for review in annotation.get("human_reviews") or []:
        if review.get("verdict") != "approve":
            continue
        required_checks = (
            review.get("query_semantics_verified") is True,
            review.get("evidence_verified") is True,
            review.get("reference_answer_verified") is True,
            not require_variants or review.get("variants_verified") is True,
        )
        reviewer_id = str(review.get("reviewer_id") or "").strip()
        if reviewer_id and all(required_checks):
            reviewers.add(reviewer_id)
    return reviewers


def _atomic_company_ids(values: list[Any]) -> list[str]:
    """Normalize legacy composite IDs such as ``600519+600887`` for auditing."""
    return sorted(
        {
            company_id.strip()
            for value in values
            for company_id in str(value).split("+")
            if company_id.strip()
        }
    )


def audit_benchmark_governance(
    benchmark: dict[str, Any],
    policy: GovernancePolicy,
) -> GovernanceReport:
    """Audit split isolation, review independence, and minimum coverage."""
    items = benchmark.get("items") or []
    document_queries: dict[str, list[tuple[str, str]]] = defaultdict(list)
    entity_period_queries: dict[str, list[tuple[str, str]]] = defaultdict(list)
    family_queries: dict[str, list[tuple[str, str]]] = defaultdict(list)
    company_queries: dict[str, list[tuple[str, str]]] = defaultdict(list)
    companies: set[str] = set()
    report_years: set[int] = set()
    document_ids: set[str] = set()
    review_status_counts: Counter[str] = Counter()
    fully_reviewed = 0
    answerable = 0

    for item in items:
        query_id = str(item.get("query_id") or "")
        split = str(item.get("split") or "unassigned")
        family_id = str(item.get("family_id") or query_id)
        family_queries[family_id].append((split, query_id))
        company_ids = _atomic_company_ids(item.get("company_ids") or [])
        years = [int(value) for value in item.get("report_years") or []]
        companies.update(company_ids)
        for company_id in company_ids:
            company_queries[company_id].append((split, query_id))
        for company_id in company_ids or ["unspecified"]:
            for year in years or [0]:
                entity_period_queries[f"{company_id}:{year}"].append((split, query_id))

        evidence_documents = {
            str(evidence.get("document_version_id"))
            for evidence in item.get("gold_evidence") or []
            if evidence.get("document_version_id")
        }
        document_ids.update(evidence_documents)
        for document_id in evidence_documents:
            document_queries[document_id].append((split, query_id))

        if item.get("answerability") == "answerable":
            answerable += 1
            report_years.update(years)
        annotation = item.get("annotation") or {}
        status = str(annotation.get("review_status") or "missing")
        review_status_counts[status] += 1
        if len(_approved_reviewers(annotation, policy.require_variant_review)) >= (
            policy.required_human_reviewers
        ):
            fully_reviewed += 1

    document_overlaps = _split_overlaps(document_queries)
    company_overlaps = _split_overlaps(company_queries)
    entity_period_overlaps = _split_overlaps(entity_period_queries)
    family_overlaps = _split_overlaps(family_queries)
    frozen_count = sum(item.get("split") == "frozen_test" for item in items)
    stats = GovernanceStats(
        item_count=len(items),
        company_count=len(companies),
        report_year_count=len(report_years),
        frozen_test_item_count=frozen_count,
        answerable_item_count=answerable,
        fully_reviewed_item_count=fully_reviewed,
        review_status_counts=dict(sorted(review_status_counts.items())),
        unique_document_count=len(document_ids),
    )

    blockers: list[str] = []
    warnings: list[str] = []
    if policy.document_blind_required and policy.split_strategy != "document_blind":
        blockers.append(
            f"split_strategy {policy.split_strategy!r} is not document_blind"
        )
    if policy.document_blind_required and document_overlaps:
        blockers.append(
            f"{len(document_overlaps)} source documents occur in multiple splits"
        )
    if policy.company_blind_required and company_overlaps:
        blockers.append(f"{len(company_overlaps)} companies occur in multiple splits")
    if policy.document_blind_required and entity_period_overlaps:
        blockers.append(
            f"{len(entity_period_overlaps)} company/report-year units occur in multiple splits"
        )
    if family_overlaps:
        blockers.append(f"{len(family_overlaps)} question families occur in multiple splits")
    if len(companies) < policy.minimum_companies:
        blockers.append(
            f"company coverage {len(companies)} is below required {policy.minimum_companies}"
        )
    if len(report_years) < policy.minimum_report_years:
        blockers.append(
            f"report-year coverage {len(report_years)} is below required "
            f"{policy.minimum_report_years}"
        )
    if frozen_count < policy.minimum_frozen_test_items:
        blockers.append(
            f"frozen_test size {frozen_count} is below required "
            f"{policy.minimum_frozen_test_items}"
        )
    if fully_reviewed < len(items):
        blockers.append(
            f"double-reviewed items {fully_reviewed}/{len(items)}; "
            f"{policy.required_human_reviewers} independent approvals are required"
        )
    if benchmark.get("independent_gold") is not True:
        blockers.append("dataset declares independent_gold=false")
    if benchmark.get("status") != "human_frozen":
        blockers.append("dataset status is not human_frozen")
    if not document_ids:
        warnings.append("no document_version_id values were found in gold evidence")

    return GovernanceReport(
        policy_id=policy.policy_id,
        dataset_id=str(benchmark.get("dataset_id") or "unknown"),
        split_strategy=policy.split_strategy,
        ready_for_external_claims=not blockers,
        blockers=blockers,
        warnings=warnings,
        stats=stats,
        document_split_overlaps=document_overlaps,
        company_split_overlaps=company_overlaps,
        entity_period_split_overlaps=entity_period_overlaps,
        family_split_overlaps=family_overlaps,
    )


def audit_corpus_plan(
    plan: BenchmarkCorpusPlan,
    policy: GovernancePolicy,
) -> CorpusPlanAudit:
    """Check that a sealed acquisition plan can support the P0 benchmark claims."""
    company_documents: dict[str, list[tuple[str, str]]] = defaultdict(list)
    split_document_counts: Counter[str] = Counter()
    split_question_targets: Counter[str] = Counter()
    companies: set[str] = set()
    report_years: set[int] = set()

    for document in plan.documents:
        companies.add(document.security_code)
        report_years.add(document.report_year)
        split_document_counts[document.split] += 1
        split_question_targets[document.split] += document.question_target
        company_documents[document.security_code].append(
            (document.split, document.document_key)
        )

    company_overlaps = _split_overlaps(company_documents)
    blockers: list[str] = []
    if policy.document_blind_required and plan.split_strategy != "document_blind":
        blockers.append("corpus plan is not document_blind")
    if plan.allocation_unit == "company" and company_overlaps:
        blockers.append(
            f"{len(company_overlaps)} companies have documents allocated to multiple splits"
        )
    if len(companies) < policy.minimum_companies:
        blockers.append(
            f"company coverage {len(companies)} is below required {policy.minimum_companies}"
        )
    if len(report_years) < policy.minimum_report_years:
        blockers.append(
            f"report-year coverage {len(report_years)} is below required "
            f"{policy.minimum_report_years}"
        )
    frozen_target = split_question_targets["frozen_test"]
    if frozen_target < policy.minimum_frozen_test_items:
        blockers.append(
            f"frozen_test question target {frozen_target} is below required "
            f"{policy.minimum_frozen_test_items}"
        )
    for required_split in ("calibration", "dev", "frozen_test"):
        if split_document_counts[required_split] == 0:
            blockers.append(f"split {required_split!r} has no allocated documents")

    return CorpusPlanAudit(
        policy_id=policy.policy_id,
        plan_id=plan.plan_id,
        ready_for_acquisition=not blockers,
        blockers=blockers,
        stats=CorpusPlanStats(
            document_count=len(plan.documents),
            company_count=len(companies),
            report_year_count=len(report_years),
            split_document_counts=dict(sorted(split_document_counts.items())),
            split_question_targets=dict(sorted(split_question_targets.items())),
        ),
        company_split_overlaps=company_overlaps,
    )


def audit_prepared_corpus(
    plan: BenchmarkCorpusPlan,
    source_manifest: dict[str, Any],
    version_manifest: dict[str, Any],
) -> PreparedCorpusAudit:
    """Verify source hashes, active versions, and split-isolated index membership."""
    blockers: list[str] = []
    if source_manifest.get("plan_id") != plan.plan_id:
        blockers.append("source manifest plan_id does not match the sealed plan")
    if version_manifest.get("plan_id") != plan.plan_id:
        blockers.append("version manifest plan_id does not match the sealed plan")

    planned = {item.document_key: item for item in plan.documents}
    sources = {
        str(item.get("document_key")): item
        for item in source_manifest.get("documents") or []
    }
    versions = {
        str(item.get("document_key")): item
        for item in version_manifest.get("documents") or []
    }
    if set(sources) != set(planned):
        blockers.append("source manifest document keys do not exactly match the plan")
    if set(versions) != set(planned):
        blockers.append("version manifest document keys do not exactly match the plan")

    for document_key, document in planned.items():
        source = sources.get(document_key) or {}
        version = versions.get(document_key) or {}
        if source.get("security_code") != document.security_code:
            blockers.append(f"source security mismatch for {document_key}")
        if source.get("report_year") != document.report_year:
            blockers.append(f"source report-year mismatch for {document_key}")
        if source.get("split") != document.split or version.get("split") != document.split:
            blockers.append(f"split mismatch for {document_key}")
        if source.get("sha256") != version.get("content_sha256"):
            blockers.append(f"source/version hash mismatch for {document_key}")
        if version.get("status") != "active":
            blockers.append(f"version is not active for {document_key}")

    expected_memberships = {
        "calibration": {"calibration"},
        "development": {"calibration", "dev"},
        "frozen_test": {"frozen_test"},
    }
    indexes = version_manifest.get("indexes") or {}
    index_document_counts: dict[str, int] = {}
    for index_name, allowed_splits in expected_memberships.items():
        index = indexes.get(index_name) or {}
        actual_ids = set(index.get("active_version_ids") or [])
        expected_ids = {
            str(versions[item.document_key].get("version_id"))
            for item in plan.documents
            if item.split in allowed_splits and item.document_key in versions
        }
        index_document_counts[index_name] = len(actual_ids)
        if not index.get("index_id"):
            blockers.append(f"index {index_name!r} has no index_id")
        if actual_ids != expected_ids:
            blockers.append(f"index {index_name!r} membership violates split isolation")

    source_hashes = {
        str(item.get("sha256")) for item in sources.values() if item.get("sha256")
    }
    if len(source_hashes) != len(plan.documents):
        blockers.append("source documents do not have unique, complete SHA-256 identities")
    return PreparedCorpusAudit(
        plan_id=plan.plan_id,
        ready_for_annotation=not blockers,
        blockers=blockers,
        document_count=len(versions),
        source_hash_count=len(source_hashes),
        index_document_counts=index_document_counts,
    )
