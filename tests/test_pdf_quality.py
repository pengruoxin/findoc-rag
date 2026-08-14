import importlib.util
import json
import sys
from pathlib import Path

import pymupdf
import pytest

from findoc_rag.documents.models import (
    BoundingBox,
    DocumentElement,
    DocumentPage,
    ParsedDocument,
)
from findoc_rag.documents.quality import (
    PdfQualityConfig,
    PdfQualityError,
    evaluate_pdf_quality,
    write_pdf_quality_report,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_pdf_pipeline", ROOT / "scripts" / "audit_pdf_pipeline.py"
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
audit_module = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = audit_module
AUDIT_SPEC.loader.exec_module(audit_module)
audit_document = audit_module.audit_document
chunk_stats = audit_module.chunk_stats
parse_pdf_specs = audit_module.parse_pdf_specs


def _page(
    page_number: int,
    text: str = "",
    *,
    image_count: int = 0,
    needs_ocr: bool = True,
) -> DocumentPage:
    elements = []
    if text:
        elements.append(
            DocumentElement(
                element_id=f"p{page_number}:text",
                element_type="text",
                text=text,
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=20),
                reading_order=0,
            )
        )
    if image_count:
        elements.append(
            DocumentElement(
                element_id=f"p{page_number}:image",
                element_type="image",
                bbox=BoundingBox(x0=0, y0=0, x1=500, y1=700),
                reading_order=len(elements),
            )
        )
    return DocumentPage(
        page_number=page_number,
        width=500,
        height=700,
        elements=elements,
        extracted_character_count=len(text),
        image_count=image_count,
        needs_ocr=needs_ocr,
    )


def _document(pages: list[DocumentPage]) -> ParsedDocument:
    return ParsedDocument(
        document_id="sha256:test",
        source_path="test.pdf",
        filename="test.pdf",
        content_sha256="test",
        page_count=len(pages),
        pages=pages,
        parser="test",
        parser_version="1",
    )


def test_quality_report_distinguishes_short_text_from_unresolved_ocr(tmp_path: Path) -> None:
    short_legal_page = _page(1, "目录", needs_ocr=True)
    image_only_page = _page(2, image_count=1, needs_ocr=True)
    empty_page = _page(3, needs_ocr=True)
    text_with_replacement = _page(4, "正常正文包含�替换字符" * 3, needs_ocr=False)
    report = evaluate_pdf_quality(
        _document([short_legal_page, image_only_page, empty_page, text_with_replacement])
    )

    assert report.policy == "warning"
    assert report.unresolved_ocr_pages == [2, 3]
    assert report.empty_pages == [3]
    assert report.low_text_pages == [1, 2, 3]
    assert report.replacement_char_pages == [4]
    assert report.page_text_coverage == 0.5
    assert report.pages[0].needs_ocr is True
    assert report.pages[0].unresolved_ocr is False
    assert report.pages[1].likely_image_only is True
    assert report.pages[3].text_element_count == 1
    assert report.pages[3].density > 0

    target = tmp_path / "quality" / "test.quality.json"
    write_pdf_quality_report(report, target)
    assert json.loads(target.read_text(encoding="utf-8"))["unresolved_ocr_pages"] == [2, 3]


def test_strict_quality_gate_uses_explicit_unresolved_page_limit() -> None:
    document = _document([_page(1, image_count=1), _page(2, "签字")])

    allowed = evaluate_pdf_quality(
        document,
        PdfQualityConfig(policy="strict", max_unresolved_ocr_pages=1),
    )
    assert allowed.unresolved_ocr_pages == [1]

    with pytest.raises(PdfQualityError, match="exceeds limit 0") as error:
        evaluate_pdf_quality(document, PdfQualityConfig(policy="strict"))
    assert error.value.report.unresolved_ocr_pages == [1]


def test_unresolved_detection_does_not_depend_on_legacy_needs_ocr_value() -> None:
    report = evaluate_pdf_quality(_document([_page(1, image_count=1, needs_ocr=False)]))

    assert report.pages[0].needs_ocr is False
    assert report.pages[0].unresolved_ocr is True
    assert report.unresolved_ocr_pages == [1]


def test_audit_helpers_handle_posix_chunk_paths_and_empty_text_pdf(tmp_path: Path) -> None:
    chunks_root = tmp_path / "versions"
    version_dir = chunks_root / "version-posix"
    version_dir.mkdir(parents=True)
    (version_dir / "chunks.jsonl").write_text(
        json.dumps({"page_start": 1, "page_end": 2, "is_continuation": True}) + "\n",
        encoding="utf-8",
    )
    assert chunk_stats(chunks_root) == {
        "version-posix": {"chunks": 1, "cross_page": 1, "continuation": 1}
    }

    source = tmp_path / "empty-and-image.pdf"
    pdf = pymupdf.open()
    pdf.new_page()
    image_page = pdf.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2, 2), False)
    image_page.insert_image(pymupdf.Rect(0, 0, 100, 100), pixmap=pixmap)
    pdf.save(source)
    pdf.close()

    audit = audit_document(source)
    assert audit["page_count"] == 2
    assert audit["chars_per_page_min"] == 0
    assert audit["chars_per_span_max"] == 0
    assert audit["multi_span_line_ratio"] == 0


def test_parse_pdf_specs_accepts_repeatable_named_paths() -> None:
    assert parse_pdf_specs(["first=/tmp/a.pdf", "second=/tmp/b.pdf"]) == {
        "first": Path("/tmp/a.pdf"),
        "second": Path("/tmp/b.pdf"),
    }
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_pdf_specs(["missing-separator"])
