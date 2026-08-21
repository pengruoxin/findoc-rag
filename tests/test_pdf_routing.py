from pathlib import Path

import pymupdf

from findoc_rag.documents.models import BoundingBox, DocumentElement, DocumentPage
from findoc_rag.documents.ocr import OcrPageResult, OcrRegion
from findoc_rag.documents.pdf import (
    PdfExtractionConfig,
    parse_pdf,
    replace_parsed_document_pages,
)
from findoc_rag.documents.routing import profile_page


def _page(text: str = "", *, image: bool = False) -> DocumentPage:
    elements: list[DocumentElement] = []
    if image:
        elements.append(
            DocumentElement(
                element_id="image",
                element_type="image",
                bbox=BoundingBox(x0=0, y0=0, x1=500, y1=700),
                reading_order=0,
            )
        )
    if text:
        elements.append(
            DocumentElement(
                element_id="text",
                element_type="text",
                text=text,
                bbox=BoundingBox(x0=10, y0=10, x1=200, y1=30),
                reading_order=len(elements),
            )
        )
    return DocumentPage(
        page_number=1,
        width=500,
        height=700,
        elements=elements,
        extracted_character_count=len(text),
        image_count=int(image),
        needs_ocr=not bool(text),
    )


def test_page_profiler_routes_native_scanned_mixed_and_empty_pages() -> None:
    native = profile_page(_page("Reliable annual report text with revenue 100 million."))
    scanned = profile_page(_page(image=True))
    mixed = profile_page(_page("MIXED PAGE NATIVE MARKER", image=True))
    empty = profile_page(_page())

    assert native.recommended_route == "native"
    assert scanned.recommended_route == "full_ocr"
    assert scanned.image_area_ratio == 1.0
    assert mixed.recommended_route == "partial_ocr"
    assert empty.recommended_route == "manual_review"


class _FakeOcrBackend:
    name = "fake-ocr"

    def extract(self, image: bytes, *, width: int, height: int) -> OcrPageResult:
        assert image.startswith(b"\x89PNG")
        return OcrPageResult(
            backend=self.name,
            image_width=width,
            image_height=height,
            regions=[
                OcrRegion(
                    text="Revenue 100 million",
                    pixel_bbox=BoundingBox(
                        x0=width * 0.1,
                        y0=height * 0.1,
                        x1=width * 0.6,
                        y1=height * 0.2,
                    ),
                    confidence=0.95,
                )
            ],
        )


def test_auto_extraction_uses_injected_ocr_and_preserves_provenance(tmp_path: Path) -> None:
    source = tmp_path / "scan.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page(width=500, height=700)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
    page.insert_image(page.rect, pixmap=pixmap)
    pdf.save(source)
    pdf.close()

    document = parse_pdf(
        source,
        PdfExtractionConfig(mode="auto", ocr_backend="fake"),
        ocr_backend=_FakeOcrBackend(),
    )
    parsed = document.pages[0]

    assert parsed.extraction_route == "full_ocr"
    assert parsed.ocr_attempted is True
    assert parsed.ocr_succeeded is True
    assert parsed.ocr_backend == "fake-ocr"
    assert parsed.needs_ocr is False
    assert parsed.native_character_count == 0
    assert parsed.ocr_character_count == len("Revenue 100 million")
    assert any(element.extraction_source == "ocr" for element in parsed.elements)
    assert "Revenue 100 million" in "\n".join(
        element.text for element in parsed.elements
    )


def test_rotated_page_dimensions_match_unrotated_coordinate_contract(tmp_path: Path) -> None:
    source = tmp_path / "rotated.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page(width=500, height=700)
    page.set_rotation(90)
    page.insert_text((50, 100), "unrotated coordinates remain authoritative")
    pdf.save(source)
    pdf.close()

    parsed = parse_pdf(source).pages[0]

    assert parsed.rotation == 90
    assert parsed.width == 500
    assert parsed.height == 700
    assert parsed.crop_box == BoundingBox(x0=0, y0=0, x1=500, y1=700)


def test_ocr_page_selection_and_retry_merge_are_bounded(tmp_path: Path) -> None:
    source = tmp_path / "two-pages.pdf"
    pdf = pymupdf.open()
    for _ in range(2):
        page = pdf.new_page(width=500, height=700)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        page.insert_image(page.rect, pixmap=pixmap)
    pdf.save(source)
    pdf.close()

    base = parse_pdf(source)
    retry = parse_pdf(
        source,
        PdfExtractionConfig(mode="force", ocr_page_numbers=[2]),
        ocr_backend=_FakeOcrBackend(),
    )
    merged = replace_parsed_document_pages(base, retry, {2})

    assert retry.pages[0].ocr_attempted is False
    assert retry.pages[1].ocr_succeeded is True
    assert merged.pages[0] == base.pages[0]
    assert merged.pages[1].ocr_backend == "fake-ocr"
