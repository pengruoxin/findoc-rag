from findoc_rag.documents.models import BoundingBox, DocumentElement, DocumentPage
from findoc_rag.pdf_scan_evaluation import (
    ScanProbePage,
    ScanSemanticProbe,
    evaluate_page_probes,
    summarize_scan_lane,
)


def _element(text: str, x0: float, y0: float, x1: float, y1: float) -> DocumentElement:
    return DocumentElement(
        element_id=f"e:{text}",
        element_type="text",
        text=text,
        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
        reading_order=0,
        extraction_source="ocr",
        confidence=0.9,
    )


def test_row_value_probe_requires_label_and_value_on_same_geometry_line() -> None:
    page = DocumentPage(
        page_number=1,
        width=500,
        height=700,
        elements=[
            _element("合并资产负债表", 100, 20, 250, 40),
            _element("货币资金", 30, 100, 100, 115),
            _element("105,904,442.39", 300, 101, 400, 116),
            _element("139,954,977.08", 410, 101, 490, 116),
            _element("22,842,501.76", 300, 160, 400, 175),
        ],
        extracted_character_count=40,
        image_count=1,
        needs_ocr=False,
        extraction_route="full_ocr",
        native_character_count=0,
        ocr_character_count=40,
        ocr_attempted=True,
        ocr_succeeded=True,
        ocr_backend="fake",
    )
    expected = ScanProbePage(
        candidate_id="case",
        page_number=1,
        probes=[
            ScanSemanticProbe(
                probe_id="title",
                probe_type="text",
                category="title",
                expected_text="合并资产负债表",
                annotation_note="test",
            ),
            ScanSemanticProbe(
                probe_id="cash",
                probe_type="row_value",
                category="cell",
                row_label="货币资金",
                column_label="年末余额",
                expected_value="105,904,442.39",
                annotation_note="test",
            ),
            ScanSemanticProbe(
                probe_id="wrong-row",
                probe_type="row_value",
                category="cell",
                row_label="货币资金",
                column_label="年末余额",
                expected_value="22,842,501.76",
                annotation_note="test",
            ),
        ],
    )

    report = evaluate_page_probes(page, expected)

    assert report["predictions"][0]["success"] is True
    assert report["predictions"][1]["same_row_association"] is True
    assert report["predictions"][2]["value_found"] is True
    assert report["predictions"][2]["same_row_association"] is False


def test_scan_summary_separates_value_recovery_from_row_association() -> None:
    page_report = {
        "route_correct": True,
        "ocr_attempted": True,
        "ocr_succeeded": True,
        "native_character_count": 0,
        "ocr_character_count": 100,
        "ocr_element_count": 4,
        "ocr_coordinate_in_bounds_count": 3,
        "ocr_coordinate_bounds_rate": 0.75,
        "predictions": [
            {"probe_type": "text", "success": True},
            {
                "probe_type": "row_value",
                "success": False,
                "label_found": True,
                "value_found": True,
                "same_row_association": False,
                "column_header_found": True,
                "column_aligned": False,
                "structured_cell_recoverable": False,
            },
        ],
    }

    metrics = summarize_scan_lane([page_report], 50.0)

    assert metrics["row_value_exact_recall"] == 1.0
    assert metrics["same_row_association_recall"] == 0.0
    assert metrics["column_header_exact_recall"] == 1.0
    assert metrics["structured_cell_recall"] == 0.0
    assert metrics["strict_probe_recall"] == 0.5
    assert metrics["ocr_coordinate_bounds_rate"] == 0.75


def test_legacy_geometry_mode_reproduces_rotated_row_failure() -> None:
    page = DocumentPage(
        page_number=1,
        width=600,
        height=800,
        elements=[
            _element("货币资金", 100, 650, 115, 720),
            _element("105,904,442.39", 101, 300, 116, 450),
        ],
        extracted_character_count=20,
        image_count=1,
        needs_ocr=False,
        extraction_route="full_ocr",
        rotation=90,
        native_character_count=0,
        ocr_character_count=20,
        ocr_attempted=True,
        ocr_succeeded=True,
        ocr_backend="fake",
    )
    expected = ScanProbePage(
        candidate_id="rotated",
        page_number=1,
        probes=[
            ScanSemanticProbe(
                probe_id="cash",
                probe_type="row_value",
                category="cell",
                row_label="货币资金",
                column_label="年末余额",
                expected_value="105,904,442.39",
                annotation_note="test",
            )
        ],
    )

    fixed = evaluate_page_probes(page, expected)
    legacy = evaluate_page_probes(page, expected, use_display_geometry=False)

    assert fixed["predictions"][0]["same_row_association"] is True
    assert legacy["predictions"][0]["same_row_association"] is False
