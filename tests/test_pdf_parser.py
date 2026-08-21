from pathlib import Path

import pymupdf

from findoc_rag.documents.models import BoundingBox, ParsedDocument
from findoc_rag.documents.pdf import _covering_box, _safe_bounding_box, parse_pdf


def test_invalid_pdf_geometry_can_be_recovered_from_child_boxes() -> None:
    assert _safe_bounding_box((10, 20, 0, 0)) is None
    recovered = _covering_box(
        [
            BoundingBox(x0=10, y0=20, x1=30, y1=40),
            BoundingBox(x0=5, y0=25, x1=35, y1=45),
        ]
    )

    assert recovered == BoundingBox(x0=5, y0=20, x1=35, y1=45)


def test_parse_pdf_preserves_pages_text_and_coordinates(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    pdf = pymupdf.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "Revenue increased to 100 million in 2024.")
    pdf.new_page()
    pdf.save(source)
    pdf.close()

    document = parse_pdf(source)

    assert document.page_count == 2
    assert document.content_sha256 in document.document_id
    assert "Revenue increased" in document.pages[0].elements[0].text
    assert document.pages[0].elements[0].bbox.x0 > 0
    assert document.pages[0].elements[0].lines[0].bbox.x0 > 0
    assert document.pages[0].elements[0].lines[0].direction == (1.0, 0.0)
    assert document.pages[0].elements[0].lines[0].spans[0].text.startswith("Revenue")
    assert document.pages[0].elements[0].lines[0].spans[0].font
    assert document.pages[0].elements[0].lines[0].spans[0].size > 0
    assert document.pages[0].needs_ocr is False
    assert document.pages[1].needs_ocr is True

    restored = ParsedDocument.model_validate_json(document.model_dump_json())
    assert restored == document


def test_parse_pdf_records_unrotated_geometry_for_rotated_pages(tmp_path: Path) -> None:
    source = tmp_path / "rotated.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page(width=600, height=800)
    page.insert_text((72, 72), "Rotated page geometry")
    page.set_rotation(90)
    pdf.save(source)
    pdf.close()

    parsed_page = parse_pdf(source).pages[0]

    assert parsed_page.rotation == 90
    assert parsed_page.width == 600
    assert parsed_page.height == 800
    assert parsed_page.media_box is not None
    assert parsed_page.media_box.x1 == 600
    assert parsed_page.media_box.y1 == 800
    assert parsed_page.crop_box == parsed_page.media_box
    assert parsed_page.coordinate_space == "pymupdf_unrotated_page"
    assert parsed_page.elements[0].lines[0].bbox.x0 == 72
    assert parsed_page.elements[0].lines[0].direction == (1.0, 0.0)


def test_old_parser_json_remains_readable(tmp_path: Path) -> None:
    source = tmp_path / "legacy.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Legacy JSON")
    pdf.save(source)
    pdf.close()

    current = parse_pdf(source)
    legacy = current.model_dump()
    for page_data in legacy["pages"]:
        page_data.pop("rotation")
        page_data.pop("media_box")
        page_data.pop("crop_box")
        page_data.pop("coordinate_space")
        for element_data in page_data["elements"]:
            element_data.pop("lines")

    restored = ParsedDocument.model_validate(legacy)

    assert restored.pages[0].rotation == 0
    assert restored.pages[0].media_box is None
    assert restored.pages[0].crop_box is None
    assert restored.pages[0].coordinate_space == "pymupdf_unrotated_page"
    assert restored.pages[0].elements[0].lines == []
