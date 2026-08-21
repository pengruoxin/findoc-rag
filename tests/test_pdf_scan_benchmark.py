import json
from pathlib import Path

import pymupdf

from findoc_rag.pdf_evaluation import file_sha256
from findoc_rag.pdf_scan_benchmark import (
    GenuineScanRegistry,
    audit_genuine_scan_registry,
    build_genuine_scan_candidate_pdf,
)


def _image_pdf(path: Path, *, pages: int = 1, hidden_text: bool = False) -> None:
    pdf = pymupdf.open()
    for _ in range(pages):
        page = pdf.new_page(width=500, height=700)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
        page.insert_image(page.rect, pixmap=pixmap)
        if hidden_text:
            page.insert_text((20, 20), "usable native text " * 20)
    pdf.save(path)
    pdf.close()


def _registry(tmp_path: Path, accepted: Path, rejected: Path) -> Path:
    payload = {
        "schema_version": "1",
        "dataset_id": "test-scans",
        "review_status": "assistant_visual_review_provisional",
        "thresholds": {
            "max_native_characters": 5,
            "min_image_coverage": 0.85,
            "text_layer_rejection_min_characters": 80,
        },
        "sources": [
            {
                "source_id": "accepted",
                "security_code": "000001",
                "company_name": "测试公司",
                "announcement_id": "1",
                "announcement_title": "扫描件",
                "official_url": "https://static.cninfo.com.cn/finalpage/2026-01-01/1.PDF",
                "local_file": accepted.as_posix(),
                "expected_sha256": file_sha256(accepted),
                "expected_size_bytes": accepted.stat().st_size,
                "disposition": "accepted_genuine_scan",
                "disposition_reason": "zero text",
                "selections": [
                    {
                        "candidate_id": "scan-page",
                        "source_page_number": 1,
                        "assigned_split": "calibration",
                        "primary_stratum": "genuine_scan",
                        "additional_strata": [],
                        "table_title": "测试表",
                        "visual_review_notes": ["visually checked"],
                    }
                ],
            },
            {
                "source_id": "rejected",
                "security_code": "000002",
                "company_name": "测试公司二",
                "announcement_id": "2",
                "announcement_title": "扫描件",
                "official_url": "https://static.cninfo.com.cn/finalpage/2026-01-01/2.PDF",
                "local_file": rejected.as_posix(),
                "expected_sha256": file_sha256(rejected),
                "expected_size_bytes": rejected.stat().st_size,
                "disposition": "rejected_usable_text_layer",
                "disposition_reason": "usable text",
                "selections": [],
            },
        ],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_scan_audit_requires_image_dominance_and_zero_native_text(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted.pdf"
    rejected = tmp_path / "rejected.pdf"
    _image_pdf(accepted)
    _image_pdf(rejected, hidden_text=True)
    registry = _registry(tmp_path, accepted, rejected)

    report = audit_genuine_scan_registry(registry, tmp_path)

    assert report["all_sources_valid"] is True
    assert report["selected_page_count"] == 1
    assert report["sources"][0]["selected_pages_eligible"] is True
    assert report["sources"][1]["minimum_native_character_count"] >= 80


def test_scan_builder_copies_only_accepted_selected_pages(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted.pdf"
    rejected = tmp_path / "rejected.pdf"
    _image_pdf(accepted, pages=2)
    _image_pdf(rejected, hidden_text=True)
    registry = _registry(tmp_path, accepted, rejected)
    output = tmp_path / "candidates.pdf"

    manifest, audit = build_genuine_scan_candidate_pdf(
        registry, output, workspace=tmp_path
    )

    assert audit["all_sources_valid"] is True
    assert manifest["page_count"] == 1
    assert manifest["pages"][0]["candidate_id"] == "scan-page"
    assert manifest["pages"][0]["source_page_profile"]["native_character_count"] == 0
    assert manifest["counts_toward_target"] is False


def test_registry_rejects_non_official_download_host() -> None:
    payload = {
        "schema_version": "1",
        "dataset_id": "bad",
        "review_status": "assistant_visual_review_provisional",
        "sources": [
            {
                "source_id": "bad",
                "security_code": "1",
                "company_name": "bad",
                "announcement_id": "1",
                "announcement_title": "bad",
                "official_url": "https://example.com/file.pdf",
                "local_file": "bad.pdf",
                "expected_sha256": "a" * 64,
                "expected_size_bytes": 1,
                "disposition": "accepted_genuine_scan",
                "disposition_reason": "bad",
                "selections": [
                    {
                        "candidate_id": "bad",
                        "source_page_number": 1,
                        "assigned_split": "calibration",
                        "table_title": "bad",
                        "visual_review_notes": ["bad"],
                    }
                ],
            }
        ],
    }

    try:
        GenuineScanRegistry.model_validate(payload)
    except ValueError as exc:
        assert "static.cninfo.com.cn" in str(exc)
    else:
        raise AssertionError("Non-official scan host was accepted")
