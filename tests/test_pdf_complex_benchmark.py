from pathlib import Path

import pytest
from pydantic import ValidationError

from findoc_rag.documents.models import BoundingBox
from findoc_rag.pdf_complex_benchmark import (
    PdfComplexCase,
    PdfHardBenchmarkManifest,
    TableCellEvidence,
    TableContinuationEdge,
    TableEvidenceCandidate,
    TableExtractionDecision,
    TableRegionEvidence,
    TableSegmentEvidence,
)


def _region(page_number: int = 1) -> TableRegionEvidence:
    return TableRegionEvidence(
        page_number=page_number,
        bbox=BoundingBox(x0=10, y0=20, x1=500, y1=700),
        coordinate_space="pymupdf_unrotated_page",
        region_sha256="a" * 64,
    )


def _cell(cell_id: str = "c1", page_number: int = 1) -> TableCellEvidence:
    return TableCellEvidence(
        cell_id=cell_id,
        row_index=2,
        column_index=3,
        row_span=1,
        column_span=2,
        raw_text="100.00",
        normalized_value="100.00",
        row_header_path=["营业收入"],
        column_header_path=["2024年", "调整后"],
        page_number=page_number,
        bbox=BoundingBox(x0=200, y0=120, x1=260, y1=135),
        coordinate_space="pymupdf_unrotated_page",
        source="table_model",
        backend_name="test-table-model",
        backend_version="1",
        confidence=0.95,
        source_region_sha256="a" * 64,
    )


def _segment(segment_id: str, page_number: int) -> TableSegmentEvidence:
    return TableSegmentEvidence(
        segment_id=segment_id,
        page_start=page_number,
        page_end=page_number,
        regions=[_region(page_number)],
        title="主要会计数据",
        unit="元",
        cells=[_cell(f"{segment_id}:c1", page_number)],
    )


def test_evidence_contract_preserves_merged_hierarchical_header() -> None:
    cell = _cell()

    assert cell.column_span == 2
    assert cell.column_header_path == ["2024年", "调整后"]
    assert cell.bbox == BoundingBox(x0=200, y0=120, x1=260, y1=135)


def test_evidence_cell_rejects_partial_geometry() -> None:
    with pytest.raises(ValidationError, match="must be set together"):
        TableCellEvidence(
            cell_id="broken",
            row_index=1,
            column_index=1,
            raw_text="100",
            page_number=1,
            source="ocr",
            backend_name="rapidocr",
            backend_version="3",
        )


def test_candidate_validates_cross_page_continuation_edges() -> None:
    candidate = TableEvidenceCandidate(
        candidate_id="candidate-1",
        logical_table_id="table-1",
        document_sha256="b" * 64,
        backend_name="test",
        backend_version="1",
        segments=[_segment("page-1", 1), _segment("page-2", 2)],
        continuation_edges=[
            TableContinuationEdge(
                from_segment_id="page-1",
                to_segment_id="page-2",
                direction="horizontal",
                reason_codes=["repeated_header", "matching_unit"],
                confidence=0.9,
            )
        ],
    )

    assert candidate.continuation_edges[0].to_segment_id == "page-2"

    with pytest.raises(ValidationError, match="unknown segment"):
        TableEvidenceCandidate(
            candidate_id="candidate-broken",
            logical_table_id="table-1",
            document_sha256="b" * 64,
            backend_name="test",
            backend_version="1",
            segments=[_segment("page-1", 1)],
            continuation_edges=[
                TableContinuationEdge(
                    from_segment_id="page-1",
                    to_segment_id="missing",
                    direction="vertical",
                    reason_codes=["section_carryover"],
                    confidence=0.5,
                )
            ],
        )


def test_decision_fails_closed_when_no_candidate_is_safe() -> None:
    review = TableExtractionDecision(
        decision_id="decision-1",
        candidate_ids=["native", "ocr"],
        status="manual_review",
        reason_codes=["candidate_value_conflict"],
    )

    assert review.selected_candidate_id is None
    with pytest.raises(ValidationError, match="require a selected candidate"):
        TableExtractionDecision(
            decision_id="decision-2",
            candidate_ids=["native"],
            status="accepted",
            reason_codes=["native_complete"],
        )


def test_manifest_is_seed_only_and_keeps_unverified_scans_out_of_formal_quota() -> None:
    manifest_path = Path("data/evaluation/pdf-hard-v2/manifest.json")
    manifest = PdfHardBenchmarkManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest.status == "seed_inventory"
    assert all(not case.counts_toward_target for case in manifest.cases)
    assert any(case.source_kind == "controlled_rasterization" for case in manifest.cases)
    genuine_cases = [
        case for case in manifest.cases if case.source_kind == "genuine_scanned_pdf"
    ]
    assert len(genuine_cases) == 4
    assert all(case.annotation_status == "unannotated" for case in genuine_cases)
    genuine_scan = manifest.quota_report()["genuine_scan"]
    assert genuine_scan["calibration_actual"] == 0
    assert genuine_scan["frozen_remaining"] == 4


def test_genuine_scan_label_cannot_be_assigned_to_synthetic_rasterization() -> None:
    with pytest.raises(ValidationError, match="genuine scanned PDF"):
        PdfComplexCase(
            case_id="fake-scan",
            source_id="source",
            source_case_id="page-1",
            page_numbers=[1],
            split="development",
            primary_stratum="genuine_scan",
            source_kind="controlled_rasterization",
            annotation_status="human_verified",
            counts_toward_target=True,
        )
