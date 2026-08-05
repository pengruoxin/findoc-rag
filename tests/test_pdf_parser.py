from pathlib import Path

import pymupdf

from findoc_rag.documents.pdf import parse_pdf


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
    assert document.pages[0].needs_ocr is False
    assert document.pages[1].needs_ocr is True

