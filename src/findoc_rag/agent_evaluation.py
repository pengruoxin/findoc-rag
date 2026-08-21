"""Gold models and local hard-scoring rules for document-agent evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from findoc_rag.agent_tasks import AgentTaskTrace


class AgentHardExpectedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str
    description: str
    acceptable_values: list[str] = Field(min_length=1)
    unit: str | None = None


class AgentHardEvidenceSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_key: str
    local_file: str
    pages: list[int]


class AgentHardCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    task_type: Literal["compare", "extract", "calculate"]
    query: str
    challenge_types: list[str] = Field(min_length=1)
    expected_behavior: Literal["answer", "abstain", "clarify"]
    expected_target_ids: list[str]
    expected_facts: list[AgentHardExpectedFact]
    evidence_sources: list[AgentHardEvidenceSource] = Field(default_factory=list)
    gold_rationale: str
    annotation_status: str


class AgentHardDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    dataset_id: str
    description: str
    index_scope: str
    source_manifest: str
    gold_policy: str
    cases: list[AgentHardCase] = Field(min_length=1)


class AgentHardFactScore(BaseModel):
    fact_id: str
    matched: bool
    matched_value: str | None = None


class AgentHardCaseScore(BaseModel):
    behavior_correct: bool
    plan_target_exact: bool
    fact_scores: list[AgentHardFactScore]
    fact_accuracy: float = Field(ge=0, le=1)
    citation_integrity: bool
    citation_source_coverage: float = Field(ge=0, le=1)
    case_pass: bool


class AgentHardSourceValidation(BaseModel):
    valid: bool
    referenced_document_count: int
    verified_document_count: int
    errors: list[str]


class AgentRequirementDiagnostics(BaseModel):
    applicable: bool
    planned_requirement_count: int = Field(ge=0)
    covered_requirement_count: int = Field(ge=0)
    evidence_bound_requirement_count: int = Field(ge=0)
    scoped_requirement_count: int = Field(ge=0)
    scope_validated_requirement_count: int = Field(ge=0)
    task_requirement_coverage: float | None = Field(default=None, ge=0, le=1)
    requirement_evidence_coverage: float | None = Field(default=None, ge=0, le=1)
    scope_validation_rate: float | None = Field(default=None, ge=0, le=1)
    claim_citation_completeness: float | None = Field(default=None, ge=0, le=1)


_NUMERIC_VALUE = re.compile(r"^-?\d+(?:\.\d+)?%?$")


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[\s,，]", "", normalized)


def _match_value(answer: str, acceptable_values: list[str]) -> str | None:
    normalized_answer = _normalized_text(answer)
    for value in acceptable_values:
        normalized_value = _normalized_text(value)
        if _NUMERIC_VALUE.fullmatch(normalized_value):
            trailing_boundary = "" if normalized_value.endswith("%") else r"(?!\d)"
            pattern = (
                rf"(?<![\d.]){re.escape(normalized_value)}{trailing_boundary}"
            )
            if re.search(pattern, normalized_answer):
                return value
        elif normalized_value in normalized_answer:
            return value
    return None


def _overlaps_source(
    *,
    document_key: str | None,
    page_start: int,
    page_end: int,
    source: AgentHardEvidenceSource,
) -> bool:
    if document_key != source.document_key:
        return False
    if not source.pages:
        return True
    return any(page_start <= page <= page_end for page in source.pages)


def score_agent_hard_case(
    case: AgentHardCase,
    trace: AgentTaskTrace,
) -> AgentHardCaseScore:
    """Score without an LLM judge; gold values never enter the agent prompt."""

    answer = trace.result.answer.answer
    fact_scores = []
    for fact in case.expected_facts:
        matched_value = _match_value(answer, fact.acceptable_values)
        fact_scores.append(
            AgentHardFactScore(
                fact_id=fact.fact_id,
                matched=matched_value is not None,
                matched_value=matched_value,
            )
        )
    fact_accuracy = (
        sum(score.matched for score in fact_scores) / len(fact_scores)
        if fact_scores
        else 1.0
    )
    if case.expected_behavior == "answer":
        behavior_correct = trace.result.outcome == "answer"
    elif case.expected_behavior == "abstain":
        behavior_correct = (
            trace.result.outcome == "abstain"
            and not trace.result.answer.grounded
            and trace.stop_reason == "no_new_evidence"
        )
    else:
        behavior_correct = (
            trace.result.outcome == "clarify"
            and not trace.result.answer.grounded
            and trace.stop_reason == "needs_clarification"
        )
    actual_targets = {target.target_id for target in trace.plan.targets}
    plan_target_exact = actual_targets == set(case.expected_target_ids)

    evidence_by_chunk = {
        evidence.chunk_id: evidence for evidence in trace.evidence_memory.items
    }
    citations = trace.result.answer.citations
    citation_integrity = all(
        citation.chunk_id in evidence_by_chunk for citation in citations
    )
    sources_with_pages = [source for source in case.evidence_sources if source.pages]
    covered_sources = 0
    for source in sources_with_pages:
        if any(
            citation.chunk_id in evidence_by_chunk
            and _overlaps_source(
                document_key=evidence_by_chunk[citation.chunk_id].document_key,
                page_start=citation.page_start,
                page_end=citation.page_end,
                source=source,
            )
            for citation in citations
        ):
            covered_sources += 1
    citation_source_coverage = (
        covered_sources / len(sources_with_pages) if sources_with_pages else 1.0
    )
    citation_gate = (
        citation_integrity and citation_source_coverage == 1.0
        if case.expected_behavior == "answer"
        else True
    )
    return AgentHardCaseScore(
        behavior_correct=behavior_correct,
        plan_target_exact=plan_target_exact,
        fact_scores=fact_scores,
        fact_accuracy=fact_accuracy,
        citation_integrity=citation_integrity,
        citation_source_coverage=citation_source_coverage,
        case_pass=(
            behavior_correct
            and plan_target_exact
            and fact_accuracy == 1.0
            and citation_gate
        ),
    )


def diagnose_agent_requirements(
    trace: AgentTaskTrace,
) -> AgentRequirementDiagnostics:
    """Report ledger coverage without consulting hard-set Gold answers."""

    requirements = trace.plan.fact_requirements
    if not requirements:
        return AgentRequirementDiagnostics(
            applicable=False,
            planned_requirement_count=0,
            covered_requirement_count=0,
            evidence_bound_requirement_count=0,
            scoped_requirement_count=0,
            scope_validated_requirement_count=0,
        )
    requirement_ids = {
        requirement.requirement_id for requirement in requirements
    }
    covered = {
        requirement_id
        for requirement_id, claims in trace.result.requirement_claims.items()
        if requirement_id in requirement_ids and claims
    }
    evidence_bound = {
        requirement_id
        for requirement_id, chunks in trace.result.requirement_evidence.items()
        if requirement_id in requirement_ids and chunks
    }
    scoped = {
        requirement.requirement_id
        for requirement in requirements
        if requirement.subject_scope != "unspecified"
    }
    scope_validated = {
        requirement_id
        for requirement_id, valid in trace.result.requirement_scope_validated.items()
        if requirement_id in scoped and valid
    }
    claims = trace.result.answer.claim_citations
    citation_completeness = (
        sum(bool(claim.citation_ordinals) for claim in claims) / len(claims)
        if claims
        else 0.0
    )
    planned_count = len(requirements)
    return AgentRequirementDiagnostics(
        applicable=True,
        planned_requirement_count=planned_count,
        covered_requirement_count=len(covered),
        evidence_bound_requirement_count=len(evidence_bound),
        scoped_requirement_count=len(scoped),
        scope_validated_requirement_count=len(scope_validated),
        task_requirement_coverage=len(covered) / planned_count,
        requirement_evidence_coverage=len(evidence_bound) / planned_count,
        scope_validation_rate=(
            len(scope_validated) / len(scoped) if scoped else 1.0
        ),
        claim_citation_completeness=citation_completeness,
    )


def validate_agent_hard_sources(
    dataset: AgentHardDataset,
    *,
    workspace: Path,
) -> AgentHardSourceValidation:
    """Bind every hard case to an exact source-manifest PDF and SHA-256."""

    manifest_path = (workspace / dataset.source_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = {
        document["document_key"]: document for document in manifest["documents"]
    }
    referenced = {
        source.document_key: source
        for case in dataset.cases
        for source in case.evidence_sources
    }
    errors: list[str] = []
    verified = 0
    for document_key, source in referenced.items():
        document = documents.get(document_key)
        if document is None:
            errors.append(f"source manifest is missing {document_key}")
            continue
        if document["local_file"] != source.local_file:
            errors.append(f"local_file mismatch for {document_key}")
            continue
        source_path = (workspace / source.local_file).resolve()
        if not source_path.is_file():
            errors.append(f"source PDF is missing for {document_key}")
            continue
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest != document["sha256"]:
            errors.append(f"source PDF SHA-256 mismatch for {document_key}")
            continue
        verified += 1
    return AgentHardSourceValidation(
        valid=not errors and verified == len(referenced),
        referenced_document_count=len(referenced),
        verified_document_count=verified,
        errors=errors,
    )
