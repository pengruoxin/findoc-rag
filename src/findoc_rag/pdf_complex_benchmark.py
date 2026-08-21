"""Versioned contracts for complex-PDF table evidence and evaluation inventory.

The models in this module deliberately sit beside the production
``StructuredTable`` sidecar.  They are the richer interchange contract used to
compare native extraction, OCR, table-structure models and visual fallbacks
before any candidate is promoted into production evidence.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from findoc_rag.documents.models import BoundingBox

SHA256_PATTERN = r"^[0-9a-f]{64}$"

PdfChallengeStratum = Literal[
    "native_control",
    "genuine_scan",
    "degraded_text_layer",
    "borderless_table",
    "merged_hierarchical_header",
    "cross_page_continuation",
    "rotated_or_mixed_layout",
]
PdfBenchmarkSplit = Literal["legacy_regression", "calibration", "development", "frozen"]
PdfSourceKind = Literal[
    "official_native_pdf",
    "genuine_scanned_pdf",
    "controlled_rasterization",
    "controlled_mixed_overlay",
    "controlled_text_layer_corruption",
]
EvidenceSource = Literal["native", "ocr", "table_model", "vlm", "human"]


class PdfStratumTarget(BaseModel):
    stratum: PdfChallengeStratum
    calibration_tables: int = Field(default=3, ge=0)
    development_tables: int = Field(default=3, ge=0)
    frozen_tables: int = Field(default=4, ge=0)


class PdfBenchmarkSource(BaseModel):
    source_id: str
    source_format: Literal[
        "legacy_extraction_benchmark", "development_candidate_manifest"
    ] = "legacy_extraction_benchmark"
    benchmark_path: str
    benchmark_sha256: str = Field(pattern=SHA256_PATTERN)
    pdf_path: str
    pdf_sha256: str = Field(pattern=SHA256_PATTERN)
    description: str = ""


class PdfComplexCase(BaseModel):
    case_id: str
    source_id: str
    source_case_id: str | None = None
    source_group_id: str | None = None
    page_numbers: list[int] = Field(min_length=1)
    split: PdfBenchmarkSplit
    primary_stratum: PdfChallengeStratum
    additional_strata: list[PdfChallengeStratum] = Field(default_factory=list)
    source_kind: PdfSourceKind
    table_count: int = Field(default=1, ge=1)
    question_count: int = Field(default=0, ge=0)
    annotation_status: Literal[
        "unannotated", "assistant_curated_provisional", "human_verified"
    ]
    counts_toward_target: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_case(self) -> PdfComplexCase:
        if len(set(self.page_numbers)) != len(self.page_numbers):
            raise ValueError("Complex PDF case page numbers must be unique")
        if self.page_numbers != sorted(self.page_numbers):
            raise ValueError("Complex PDF case page numbers must be ascending")
        if len(set(self.additional_strata)) != len(self.additional_strata):
            raise ValueError("Complex PDF case additional strata must be unique")
        if self.primary_stratum in self.additional_strata:
            raise ValueError("Primary stratum must not be repeated in additional_strata")
        if bool(self.source_case_id) == bool(self.source_group_id):
            raise ValueError("Set exactly one of source_case_id or source_group_id")
        if self.split == "legacy_regression" and self.counts_toward_target:
            raise ValueError("Legacy regression cases cannot fill new benchmark quotas")
        if self.counts_toward_target and self.annotation_status != "human_verified":
            raise ValueError("Quota-eligible cases must be human verified")
        if self.primary_stratum == "genuine_scan" and self.source_kind != "genuine_scanned_pdf":
            raise ValueError("Genuine-scan cases must come from a genuine scanned PDF")
        return self


class PdfGoldPolicy(BaseModel):
    evaluation_unit: Literal["table"] = "table"
    gold_fields: list[str] = Field(min_length=1)
    independent_frozen_reviewer_required: bool = True
    candidate_verification_is_blind_annotation: Literal[False] = False
    frozen_gold_must_be_separate: bool = True


class PdfHardBenchmarkManifest(BaseModel):
    schema_version: Literal["1"] = "1"
    dataset_id: str
    status: Literal["seed_inventory", "development_ready", "frozen_ready"]
    targets: list[PdfStratumTarget]
    sources: list[PdfBenchmarkSource]
    cases: list[PdfComplexCase]
    gold_policy: PdfGoldPolicy

    @model_validator(mode="after")
    def validate_manifest(self) -> PdfHardBenchmarkManifest:
        target_names = [target.stratum for target in self.targets]
        if len(set(target_names)) != len(target_names):
            raise ValueError("Benchmark strata targets must be unique")
        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Benchmark source IDs must be unique")
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Complex PDF case IDs must be unique")
        unknown_sources = {case.source_id for case in self.cases} - set(source_ids)
        if unknown_sources:
            raise ValueError(f"Cases reference unknown sources: {sorted(unknown_sources)}")
        missing_targets = {case.primary_stratum for case in self.cases} - set(target_names)
        if missing_targets:
            raise ValueError(f"Cases reference untargeted strata: {sorted(missing_targets)}")
        return self

    def quota_report(self) -> dict[str, dict[str, int]]:
        """Return table counts and remaining quotas for every formal split."""

        counts: Counter[tuple[PdfChallengeStratum, str]] = Counter()
        for case in self.cases:
            if case.counts_toward_target and case.split != "legacy_regression":
                counts[(case.primary_stratum, case.split)] += case.table_count

        report: dict[str, dict[str, int]] = {}
        for target in self.targets:
            values: dict[str, int] = {}
            for split in ("calibration", "development", "frozen"):
                target_count = getattr(target, f"{split}_tables")
                actual_count = counts[(target.stratum, split)]
                values[f"{split}_target"] = target_count
                values[f"{split}_actual"] = actual_count
                values[f"{split}_remaining"] = max(0, target_count - actual_count)
            report[target.stratum] = values
        return report


class TableRegionEvidence(BaseModel):
    page_number: int = Field(ge=1)
    bbox: BoundingBox
    coordinate_space: Literal["pymupdf_unrotated_page", "rendered_pixel"]
    region_sha256: str = Field(pattern=SHA256_PATTERN)


class TableCellEvidence(BaseModel):
    cell_id: str
    row_index: int = Field(ge=1)
    column_index: int = Field(ge=1)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    raw_text: str = ""
    normalized_value: str = ""
    row_header_path: list[str] = Field(default_factory=list)
    column_header_path: list[str] = Field(default_factory=list)
    page_number: int | None = Field(default=None, ge=1)
    bbox: BoundingBox | None = None
    coordinate_space: Literal["pymupdf_unrotated_page", "rendered_pixel"] | None = None
    source: EvidenceSource
    backend_name: str
    backend_version: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    source_region_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_cell(self) -> TableCellEvidence:
        geometry = (self.page_number, self.bbox, self.coordinate_space, self.source_region_sha256)
        if any(value is not None for value in geometry) and not all(
            value is not None for value in geometry
        ):
            raise ValueError(
                "page_number, bbox, coordinate_space and source_region_sha256 must be set together"
            )
        if not self.row_header_path and not self.column_header_path and not self.raw_text:
            raise ValueError("A cell must contain text or a row/column header path")
        return self


class TableSegmentEvidence(BaseModel):
    segment_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    regions: list[TableRegionEvidence] = Field(min_length=1)
    title: str = ""
    unit: str = ""
    cells: list[TableCellEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_segment(self) -> TableSegmentEvidence:
        if self.page_end < self.page_start:
            raise ValueError("Table segment page_end must not precede page_start")
        if any(not self.page_start <= region.page_number <= self.page_end for region in self.regions):
            raise ValueError("Table region page must fall within the segment page range")
        if any(
            cell.page_number is not None
            and not self.page_start <= cell.page_number <= self.page_end
            for cell in self.cells
        ):
            raise ValueError("Table cell page must fall within the segment page range")
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("Table cell IDs must be unique within a segment")
        return self


class TableContinuationEdge(BaseModel):
    from_segment_id: str
    to_segment_id: str
    direction: Literal["vertical", "horizontal", "both"]
    reason_codes: list[
        Literal[
            "repeated_header",
            "matching_column_geometry",
            "matching_title",
            "matching_unit",
            "continuation_marker",
            "section_carryover",
        ]
    ] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_edge(self) -> TableContinuationEdge:
        if self.from_segment_id == self.to_segment_id:
            raise ValueError("A continuation edge cannot point to the same segment")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("Continuation edge reason codes must be unique")
        return self


class TableEvidenceCandidate(BaseModel):
    schema_version: Literal["1"] = "1"
    candidate_id: str
    logical_table_id: str
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    backend_name: str
    backend_version: str
    segments: list[TableSegmentEvidence] = Field(min_length=1)
    continuation_edges: list[TableContinuationEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_candidate(self) -> TableEvidenceCandidate:
        segment_ids = [segment.segment_id for segment in self.segments]
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("Table segment IDs must be unique")
        known_segments = set(segment_ids)
        for edge in self.continuation_edges:
            if edge.from_segment_id not in known_segments or edge.to_segment_id not in known_segments:
                raise ValueError("Continuation edge references an unknown segment")
            source = next(segment for segment in self.segments if segment.segment_id == edge.from_segment_id)
            target = next(segment for segment in self.segments if segment.segment_id == edge.to_segment_id)
            if source.page_start > target.page_start:
                raise ValueError("Continuation edges must follow document page order")
        return self


class TableExtractionDecision(BaseModel):
    decision_id: str
    candidate_ids: list[str] = Field(min_length=1)
    status: Literal["accepted", "manual_review", "rejected"]
    selected_candidate_id: str | None = None
    reason_codes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision(self) -> TableExtractionDecision:
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("Decision candidate IDs must be unique")
        if self.status == "accepted" and self.selected_candidate_id is None:
            raise ValueError("Accepted decisions require a selected candidate")
        if self.status != "accepted" and self.selected_candidate_id is not None:
            raise ValueError("Only accepted decisions may select a candidate")
        if self.selected_candidate_id not in {None, *self.candidate_ids}:
            raise ValueError("Selected candidate must be included in candidate_ids")
        return self
