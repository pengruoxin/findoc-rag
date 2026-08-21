"""Append-only human-review queue for evidence-gated Agent tasks."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field

from findoc_rag.agent_tasks import AgentTaskResult, AgentTaskStore, AgentTaskTrace
from findoc_rag.table_cell_proof import TableCellGeometryProof
from findoc_rag.table_reconstruction import normalize_label, normalize_value
from findoc_rag.visual_inspection import PdfRegionInspector, TableCellRegionProof

REVIEW_ID_PATTERN = r"^[0-9a-f]{32}$"
TRACE_SHA256_PATTERN = r"^[0-9a-f]{64}$"
ReviewDecision = Literal["approve", "correct", "reject"]
ReviewStatus = Literal["pending", "resolved"]


def agent_task_trace_sha256(trace: AgentTaskTrace) -> str:
    """Hash the semantic trace content independently of JSON indentation."""

    canonical = json.dumps(
        trace.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class HumanReviewEvidence(BaseModel):
    """Evidence excerpt shown to a reviewer without another retrieval run."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    content_sha256: str = Field(pattern=TRACE_SHA256_PATTERN)
    document_id: str
    document_key: str | None = None
    company_name: str | None = None
    report_year: int | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: list[str] = Field(default_factory=list)
    excerpt: str


class HumanReviewRequirement(BaseModel):
    """One atomic answer obligation and its exact candidate evidence."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(pattern=r"^r[1-9]\d*$")
    description: str
    claims: list[str] = Field(default_factory=list)
    evidence: list[HumanReviewEvidence] = Field(default_factory=list)
    table_cell_proofs: list[TableCellGeometryProof] = Field(default_factory=list)
    region_proofs: list[TableCellRegionProof] = Field(default_factory=list)
    region_proof_errors: list[str] = Field(default_factory=list)


class HumanReviewPacket(BaseModel):
    """Immutable packet created when an automated answer is withheld."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    review_id: str = Field(pattern=REVIEW_ID_PATTERN)
    task_id: str = Field(pattern=REVIEW_ID_PATTERN)
    task_trace_sha256: str = Field(pattern=TRACE_SHA256_PATTERN)
    created_at: datetime
    index_id: str
    query: str
    reasons: list[str] = Field(min_length=1)
    candidate_result: AgentTaskResult | None = None
    requirements: list[HumanReviewRequirement] = Field(default_factory=list)


class HumanReviewResolution(BaseModel):
    """One immutable decision bound to the exact source task trace."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    resolution_id: str = Field(pattern=REVIEW_ID_PATTERN)
    review_id: str = Field(pattern=REVIEW_ID_PATTERN)
    task_id: str = Field(pattern=REVIEW_ID_PATTERN)
    task_trace_sha256: str = Field(pattern=TRACE_SHA256_PATTERN)
    decision: ReviewDecision
    reviewer: str = Field(min_length=1, max_length=100)
    comment: str = Field(default="", max_length=1000)
    resolved_at: datetime
    final_outcome: Literal["answer", "abstain"]
    final_answer: str = Field(min_length=1, max_length=8000)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class HumanReviewQueueItem(BaseModel):
    """Materialized queue view; packet and resolution remain separate on disk."""

    packet: HumanReviewPacket
    resolution: HumanReviewResolution | None = None

    @computed_field
    @property
    def status(self) -> ReviewStatus:
        return "resolved" if self.resolution is not None else "pending"


def _review_packet(
    trace: AgentTaskTrace,
    *,
    region_inspector: PdfRegionInspector | None = None,
    region_directory: Path | None = None,
) -> HumanReviewPacket:
    verification = trace.evidence_verification
    if (
        trace.stop_reason != "evidence_verifier_manual_review"
        or verification is None
        or not verification.human_review_required
    ):
        raise ValueError("Task trace is not awaiting evidence-verifier human review")
    trace_sha256 = agent_task_trace_sha256(trace)
    review_id = hashlib.sha256(f"human-review:{trace.task_id}:{trace_sha256}".encode()).hexdigest()[
        :32
    ]
    candidate = verification.candidate_result
    evidence_by_id = {item.chunk_id: item for item in trace.evidence_memory.items}
    proof_evidence_by_requirement: dict[str, list[str]] = {}
    for turn in verification.turns:
        for proof in turn.support_proofs:
            proof_evidence_by_requirement.setdefault(proof.requirement_id, []).extend(
                quote.evidence_chunk_id for quote in proof.evidence_quotes
            )

    requirements: list[HumanReviewRequirement] = []
    for requirement in trace.plan.fact_requirements:
        requirement_id = requirement.requirement_id
        evidence_ids = candidate.requirement_evidence.get(requirement_id, []) if candidate else []
        evidence_ids = list(
            dict.fromkeys(
                [
                    *evidence_ids,
                    *proof_evidence_by_requirement.get(requirement_id, []),
                ]
            )
        )
        claims = candidate.requirement_claims.get(requirement_id, []) if candidate else []
        evidence_items = [
            evidence_by_id[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
        ]
        candidate_cell_proofs = [
            proof for evidence in evidence_items for proof in evidence.table_cell_proofs
        ]
        context = normalize_label(" ".join([requirement.description, *claims]))
        numeric_context = normalize_value(context)
        value_counts: dict[str, int] = {}
        for proof in candidate_cell_proofs:
            value = normalize_value(proof.value)
            value_counts[value] = value_counts.get(value, 0) + 1
        matched_cell_proofs = [
            proof
            for proof in candidate_cell_proofs
            if normalize_value(proof.value) in numeric_context
            and (
                normalize_label(proof.row) in context
                or normalize_label(proof.column) in context
                or value_counts[normalize_value(proof.value)] == 1
            )
        ]
        unique_cell_proofs = list(
            {proof.binding_sha256: proof for proof in matched_cell_proofs}.values()
        )
        region_proofs: list[TableCellRegionProof] = []
        region_proof_errors: list[str] = []
        if region_inspector is not None:
            if region_directory is None:
                raise ValueError("region_directory is required when region_inspector is configured")
            document_key_by_chunk = {
                evidence.chunk_id: evidence.document_key
                for evidence in evidence_items
                if evidence.document_key
            }
            for proof in unique_cell_proofs:
                document_key = document_key_by_chunk.get(proof.chunk_id)
                if document_key is None:
                    region_proof_errors.append(f"{proof.binding_sha256[:16]}:missing_document_key")
                    continue
                try:
                    region_proofs.append(
                        region_inspector.render_table_cell_region(
                            document_key,
                            proof,
                            output_directory=region_directory,
                        )
                    )
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    region_proof_errors.append(
                        f"{proof.binding_sha256[:16]}:{type(exc).__name__}:{exc}"
                    )
        requirements.append(
            HumanReviewRequirement(
                requirement_id=requirement_id,
                description=requirement.description,
                claims=claims,
                evidence=[
                    HumanReviewEvidence.model_validate(
                        evidence.model_dump(exclude={"target_ids", "table_cell_proofs"})
                    )
                    for evidence in evidence_items
                ],
                table_cell_proofs=unique_cell_proofs,
                region_proofs=region_proofs,
                region_proof_errors=region_proof_errors,
            )
        )
    if not any(requirement.evidence for requirement in requirements):
        fallback = [
            HumanReviewEvidence.model_validate(
                item.model_dump(exclude={"target_ids", "table_cell_proofs"})
            )
            for item in trace.evidence_memory.items[:5]
        ]
        if requirements:
            requirements[0].evidence = fallback

    return HumanReviewPacket(
        review_id=review_id,
        task_id=trace.task_id,
        task_trace_sha256=trace_sha256,
        created_at=trace.completed_at,
        index_id=trace.index_id,
        query=trace.query,
        reasons=(
            list(dict.fromkeys(verification.human_review_reasons))
            or ["evidence verifier requested human review"]
        ),
        candidate_result=candidate,
        requirements=requirements,
    )


class HumanReviewStore:
    """Filesystem review queue with immutable packets and one-shot resolutions."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.packet_directory = self.directory / "packets"
        self.resolution_directory = self.directory / "resolutions"

    @staticmethod
    def _validate_review_id(review_id: str) -> None:
        if len(review_id) != 32 or any(
            character not in "0123456789abcdef" for character in review_id
        ):
            raise ValueError("Invalid review ID")

    def packet_path(self, review_id: str) -> Path:
        self._validate_review_id(review_id)
        return self.packet_directory / f"{review_id}.json"

    def resolution_path(self, review_id: str) -> Path:
        self._validate_review_id(review_id)
        return self.resolution_directory / f"{review_id}.json"

    @staticmethod
    def _write_once(path: Path, value: BaseModel) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(value.model_dump_json(indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError(f"Audit record already exists: {path.name}") from exc
        finally:
            temporary.unlink(missing_ok=True)

    def create(
        self,
        trace: AgentTaskTrace,
        *,
        region_inspector: PdfRegionInspector | None = None,
        region_directory: Path | None = None,
    ) -> tuple[HumanReviewPacket, Path]:
        packet = _review_packet(
            trace,
            region_inspector=region_inspector,
            region_directory=region_directory,
        )
        path = self.packet_path(packet.review_id)
        if path.is_file():
            existing = self.load_packet(packet.review_id)
            if existing != packet:
                raise ValueError("Existing review packet does not match current task trace")
            return existing, path
        try:
            self._write_once(path, packet)
        except ValueError:
            existing = self.load_packet(packet.review_id)
            if existing != packet:
                raise
            return existing, path
        return packet, path

    def load_packet(self, review_id: str) -> HumanReviewPacket:
        path = self.packet_path(review_id)
        return HumanReviewPacket.model_validate_json(path.read_text(encoding="utf-8"))

    def load_resolution(self, review_id: str) -> HumanReviewResolution | None:
        path = self.resolution_path(review_id)
        if not path.is_file():
            return None
        return HumanReviewResolution.model_validate_json(path.read_text(encoding="utf-8"))

    def inspect(self, review_id: str) -> HumanReviewQueueItem:
        return HumanReviewQueueItem(
            packet=self.load_packet(review_id),
            resolution=self.load_resolution(review_id),
        )

    def list(
        self, *, status: Literal["all", "pending", "resolved"] = "all"
    ) -> list[HumanReviewQueueItem]:
        if status not in {"all", "pending", "resolved"}:
            raise ValueError("status must be all, pending, or resolved")
        if not self.packet_directory.is_dir():
            return []
        items = [self.inspect(path.stem) for path in self.packet_directory.glob("*.json")]
        if status != "all":
            items = [item for item in items if item.status == status]
        return sorted(items, key=lambda item: item.packet.created_at, reverse=True)

    def resolve(
        self,
        review_id: str,
        *,
        task_store: AgentTaskStore,
        decision: ReviewDecision,
        reviewer: str,
        comment: str = "",
        corrected_answer: str | None = None,
        evidence_chunk_ids: list[str] | None = None,
    ) -> tuple[HumanReviewResolution, Path]:
        packet = self.load_packet(review_id)
        if self.load_resolution(review_id) is not None:
            raise ValueError("Review is already resolved")
        current_trace = task_store.load(packet.task_id)
        current_sha256 = agent_task_trace_sha256(current_trace)
        if current_sha256 != packet.task_trace_sha256:
            raise ValueError("Task trace changed after review creation; create a new review packet")

        known_evidence_ids = {
            evidence.chunk_id
            for requirement in packet.requirements
            for evidence in requirement.evidence
        }
        selected_evidence_ids = list(dict.fromkeys(evidence_chunk_ids or []))
        unknown_evidence_ids = sorted(set(selected_evidence_ids) - known_evidence_ids)
        if unknown_evidence_ids:
            raise ValueError(
                "Review resolution references evidence outside the immutable packet: "
                + ", ".join(unknown_evidence_ids)
            )

        if decision == "approve":
            if packet.candidate_result is None:
                raise ValueError("Legacy review packet has no candidate result to approve")
            if packet.candidate_result.outcome != "answer":
                raise ValueError("Review packet candidate is not an answer")
            final_outcome: Literal["answer", "abstain"] = "answer"
            final_answer = packet.candidate_result.answer.answer
            candidate_evidence_ids = list(
                dict.fromkeys(
                    citation.chunk_id for citation in packet.candidate_result.answer.citations
                )
            )
            if not candidate_evidence_ids:
                raise ValueError("Candidate answer has no evidence bound to the review packet")
            outside_candidate_evidence = sorted(set(candidate_evidence_ids) - known_evidence_ids)
            if outside_candidate_evidence:
                raise ValueError(
                    "Candidate answer references evidence outside the immutable packet: "
                    + ", ".join(outside_candidate_evidence)
                )
            selected_evidence_ids = candidate_evidence_ids
        elif decision == "correct":
            if corrected_answer is None or not corrected_answer.strip():
                raise ValueError("correct decision requires a non-empty corrected answer")
            if not selected_evidence_ids:
                raise ValueError("correct decision requires at least one packet evidence chunk ID")
            final_outcome = "answer"
            final_answer = corrected_answer.strip()
        elif decision == "reject":
            final_outcome = "abstain"
            final_answer = "人工复核未批准候选答案，系统保持拒答。"
            selected_evidence_ids = []
        else:
            raise ValueError("decision must be approve, correct, or reject")

        resolution = HumanReviewResolution(
            resolution_id=uuid4().hex,
            review_id=packet.review_id,
            task_id=packet.task_id,
            task_trace_sha256=packet.task_trace_sha256,
            decision=decision,
            reviewer=reviewer.strip(),
            comment=comment.strip(),
            resolved_at=datetime.now(UTC),
            final_outcome=final_outcome,
            final_answer=final_answer,
            evidence_chunk_ids=selected_evidence_ids,
        )
        path = self.resolution_path(review_id)
        self._write_once(path, resolution)
        return resolution, path
