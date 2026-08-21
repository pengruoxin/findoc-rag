"""Page-level quality signals for the existing PDF intermediate representation.

The report is deliberately separate from parsing and chunking.  It can therefore
be added to an ingestion workflow without changing document or chunk identity.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from findoc_rag.documents.models import DocumentPage, ParsedDocument
from findoc_rag.documents.routing import ExtractionRoute, profile_page


class PdfQualityConfig(BaseModel):
    """Thresholds for reporting PDF text-layer quality.

    Warning mode always returns a report.  Strict mode rejects a document only
    when the number of unresolved OCR pages exceeds the configured limit.
    """

    policy: Literal["warning", "strict"] = "warning"
    low_text_character_threshold: int = Field(default=20, ge=1)
    replacement_character_ratio_threshold: float = Field(default=0.01, ge=0, le=1)
    max_unresolved_ocr_pages: int = Field(default=0, ge=0)


class PdfPageQuality(BaseModel):
    page_number: int = Field(ge=1)
    char_count: int = Field(ge=0)
    density: float = Field(ge=0)
    image_count: int = Field(ge=0)
    replacement_char_ratio: float = Field(ge=0, le=1)
    text_element_count: int = Field(ge=0)
    empty: bool
    low_text: bool
    likely_image_only: bool
    needs_ocr: bool
    unresolved_ocr: bool
    image_area_ratio: float = Field(default=0, ge=0, le=1)
    suspicious_character_ratio: float = Field(default=0, ge=0, le=1)
    duplicate_text_ratio: float = Field(default=0, ge=0, le=1)
    extraction_route: ExtractionRoute = "native"
    native_char_count: int = Field(default=0, ge=0)
    ocr_char_count: int = Field(default=0, ge=0)
    ocr_attempted: bool = False
    ocr_succeeded: bool = False
    ocr_backend: str | None = None


class PdfQualityReport(BaseModel):
    document_id: str
    filename: str
    page_count: int = Field(ge=1)
    policy: Literal["warning", "strict"]
    low_text_character_threshold: int = Field(ge=1)
    replacement_character_ratio_threshold: float = Field(ge=0, le=1)
    max_unresolved_ocr_pages: int = Field(ge=0)
    pages: list[PdfPageQuality]
    unresolved_ocr_pages: list[int]
    empty_pages: list[int]
    low_text_pages: list[int]
    replacement_char_pages: list[int]
    page_text_coverage: float = Field(ge=0, le=1)


class PdfQualityError(ValueError):
    """Raised when a strict PDF quality policy rejects a parsed document."""

    def __init__(self, report: PdfQualityReport) -> None:
        self.report = report
        count = len(report.unresolved_ocr_pages)
        limit = report.max_unresolved_ocr_pages
        super().__init__(
            f"PDF quality gate rejected {report.filename}: {count} unresolved OCR "
            f"page(s) exceeds limit {limit}"
        )


def _page_quality(page: DocumentPage, config: PdfQualityConfig) -> PdfPageQuality:
    text_elements = [element for element in page.elements if element.element_type == "text"]
    extracted_text = "".join(element.text for element in text_elements)
    character_count = page.extracted_character_count
    replacement_count = extracted_text.count("\ufffd")
    replacement_ratio = replacement_count / character_count if character_count else 0.0
    page_area = page.width * page.height

    low_text = character_count < config.low_text_character_threshold
    empty = character_count == 0 and page.image_count == 0
    likely_image_only = low_text and page.image_count > 0
    profile = profile_page(page)

    # A short title, signature, or divider page may retain the parser's legacy
    # needs_ocr=True signal, but is not unresolved unless it is blank or image-led.
    unresolved_ocr = (low_text and (empty or likely_image_only)) or (
        page.needs_ocr and page.extraction_route in {"partial_ocr", "full_ocr"}
    )

    return PdfPageQuality(
        page_number=page.page_number,
        char_count=character_count,
        density=character_count / page_area,
        image_count=page.image_count,
        replacement_char_ratio=replacement_ratio,
        text_element_count=len(text_elements),
        empty=empty,
        low_text=low_text,
        likely_image_only=likely_image_only,
        needs_ocr=page.needs_ocr,
        unresolved_ocr=unresolved_ocr,
        image_area_ratio=profile.image_area_ratio,
        suspicious_character_ratio=profile.suspicious_character_ratio,
        duplicate_text_ratio=profile.duplicate_text_ratio,
        extraction_route=page.extraction_route,
        native_char_count=(
            page.native_character_count
            if page.native_character_count is not None
            else character_count
        ),
        ocr_char_count=page.ocr_character_count,
        ocr_attempted=page.ocr_attempted,
        ocr_succeeded=page.ocr_succeeded,
        ocr_backend=page.ocr_backend,
    )


def evaluate_pdf_quality(
    document: ParsedDocument,
    config: PdfQualityConfig | None = None,
) -> PdfQualityReport:
    """Build a quality report and apply the optional strict gate."""

    effective_config = config or PdfQualityConfig()
    pages = [_page_quality(page, effective_config) for page in document.pages]
    replacement_pages = [
        page.page_number
        for page in pages
        if page.replacement_char_ratio
        > effective_config.replacement_character_ratio_threshold
    ]
    report = PdfQualityReport(
        document_id=document.document_id,
        filename=document.filename,
        page_count=document.page_count,
        policy=effective_config.policy,
        low_text_character_threshold=effective_config.low_text_character_threshold,
        replacement_character_ratio_threshold=(
            effective_config.replacement_character_ratio_threshold
        ),
        max_unresolved_ocr_pages=effective_config.max_unresolved_ocr_pages,
        pages=pages,
        unresolved_ocr_pages=[page.page_number for page in pages if page.unresolved_ocr],
        empty_pages=[page.page_number for page in pages if page.empty],
        low_text_pages=[page.page_number for page in pages if page.low_text],
        replacement_char_pages=replacement_pages,
        page_text_coverage=(
            sum(page.char_count > 0 for page in pages) / document.page_count
        ),
    )
    if (
        effective_config.policy == "strict"
        and len(report.unresolved_ocr_pages) > effective_config.max_unresolved_ocr_pages
    ):
        raise PdfQualityError(report)
    return report


def write_pdf_quality_report(report: PdfQualityReport, path: Path) -> None:
    """Write a human-readable JSON sidecar for a parsed PDF."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
