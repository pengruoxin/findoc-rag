import hashlib
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel, Field

from findoc_rag.documents.models import (
    BoundingBox,
    DocumentElement,
    DocumentPage,
    ParsedDocument,
    PdfLine,
    PdfSpan,
)
from findoc_rag.documents.ocr import OcrBackend, OcrPageResult, create_ocr_backend
from findoc_rag.documents.routing import PdfRoutingConfig, profile_page

MIN_TEXT_CHARACTERS = 20


class PdfExtractionConfig(BaseModel):
    """Versionable native-first extraction policy.

    ``auto`` keeps reliable native text and only invokes OCR for pages selected
    by the cheap page profiler. ``force`` is intended for evaluation and
    diagnostics, not normal ingestion.
    """

    mode: Literal["disabled", "auto", "force"] = "disabled"
    ocr_backend: str = "rapidocr"
    ocr_dpi: int = Field(default=180, ge=72, le=600)
    ocr_error_policy: Literal["raise", "mark"] = "raise"
    ocr_page_numbers: list[int] | None = None
    routing: PdfRoutingConfig = Field(default_factory=PdfRoutingConfig)


def _bounding_box(value: object) -> BoundingBox:
    return BoundingBox.model_validate(
        dict(zip(("x0", "y0", "x1", "y1"), value, strict=True))  # type: ignore[arg-type]
    )


def _safe_bounding_box(value: object) -> BoundingBox | None:
    """Return usable PDF geometry without allowing one corrupt block to abort parsing."""
    try:
        coordinates = tuple(float(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    if len(coordinates) != 4 or not all(math.isfinite(item) for item in coordinates):
        return None
    x0, y0, x1, y1 = coordinates
    if x1 < x0 or y1 < y0:
        return None
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _covering_box(boxes: list[BoundingBox]) -> BoundingBox | None:
    if not boxes:
        return None
    return BoundingBox(
        x0=min(box.x0 for box in boxes),
        y0=min(box.y0 for box in boxes),
        x1=max(box.x1 for box in boxes),
        y1=max(box.y1 for box in boxes),
    )


def _is_bold_span(span: dict) -> bool:
    return bool(
        int(span.get("flags", 0)) & 16
        or any(
            marker in str(span.get("font", "")).lower()
            for marker in ("bold", "black", "heavy")
        )
    )


def _pdf_line(line: dict) -> tuple[PdfLine | None, int]:
    warnings = 0
    spans: list[PdfSpan] = []
    for span in line.get("spans", []):
        span_box = _safe_bounding_box(span.get("bbox"))
        if span_box is None:
            warnings += 1
            continue
        spans.append(
            PdfSpan(
                text=str(span.get("text", "")),
                bbox=span_box,
                font=str(span.get("font", "")),
                size=float(span.get("size", 0)),
                flags=int(span.get("flags", 0)),
                bold=_is_bold_span(span),
            )
        )
    line_box = _safe_bounding_box(line.get("bbox"))
    if line_box is None:
        warnings += 1
        line_box = _covering_box([span.bbox for span in spans])
    if line_box is None:
        return None, warnings
    direction = line.get("dir", (1.0, 0.0))
    return (
        PdfLine(
            bbox=line_box,
            direction=(float(direction[0]), float(direction[1])),
            spans=spans,
        ),
        warnings,
    )


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_pdf_datetime(value: str | None) -> datetime | None:
    if not value or not value.startswith("D:"):
        return None
    digits = "".join(character for character in value[2:16] if character.isdigit())
    if len(digits) < 4:
        return None
    padded = (digits + "0101000000")[:14]
    try:
        return datetime.strptime(padded, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _intersection_ratio(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    area = max(0.0, (first.x1 - first.x0) * (first.y1 - first.y0))
    return width * height / area if area else 0.0


def _ocr_elements(
    result: OcrPageResult,
    *,
    content_hash: str,
    page_index: int,
    page: pymupdf.Page,
) -> list[DocumentElement]:
    x_scale = page.rect.width / result.image_width
    y_scale = page.rect.height / result.image_height
    elements: list[DocumentElement] = []
    for index, region in enumerate(result.regions):
        rotated_rect = pymupdf.Rect(
            region.pixel_bbox.x0 * x_scale,
            region.pixel_bbox.y0 * y_scale,
            region.pixel_bbox.x1 * x_scale,
            region.pixel_bbox.y1 * y_scale,
        )
        rect = rotated_rect * page.derotation_matrix if page.rotation else rotated_rect
        bbox = _safe_bounding_box((rect.x0, rect.y0, rect.x1, rect.y1))
        if bbox is None:
            continue
        elements.append(
            DocumentElement(
                element_id=f"{content_hash[:12]}:p{page_index}:ocr{index}",
                element_type="text",
                text=region.text,
                bbox=bbox,
                reading_order=index,
                extraction_source="ocr",
                confidence=region.confidence,
            )
        )
    return elements


def _merge_partial_ocr(
    native_elements: list[DocumentElement],
    ocr_elements: list[DocumentElement],
) -> list[DocumentElement]:
    native_text = [element for element in native_elements if element.element_type == "text"]
    native_compact = {"".join(element.text.split()) for element in native_text}
    accepted: list[DocumentElement] = []
    for candidate in ocr_elements:
        compact = "".join(candidate.text.split())
        if compact and compact in native_compact:
            continue
        if any(_intersection_ratio(candidate.bbox, item.bbox) >= 0.6 for item in native_text):
            continue
        accepted.append(candidate)
    merged = [*native_elements, *accepted]
    merged.sort(key=lambda element: (element.bbox.y0, element.bbox.x0, element.element_type))
    for reading_order, element in enumerate(merged):
        element.reading_order = reading_order
    return merged


def parse_pdf(
    path: Path,
    extraction_config: PdfExtractionConfig | None = None,
    *,
    ocr_backend: OcrBackend | None = None,
) -> ParsedDocument:
    """Parse a PDF into a native-first, coordinate-preserving page IR."""
    resolved = path.resolve(strict=True)
    if resolved.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {resolved}")

    content_hash = _sha256(resolved)
    config = extraction_config or PdfExtractionConfig()
    pages: list[DocumentPage] = []
    active_ocr_backend = ocr_backend

    with pymupdf.open(resolved) as pdf:
        if pdf.page_count == 0:
            raise ValueError("PDF contains no pages")

        for page_index, page in enumerate(pdf, start=1):
            raw = page.get_text("dict", sort=True)
            elements: list[DocumentElement] = []
            image_count = 0
            geometry_warning_count = 0
            discarded_block_count = 0

            for block in raw.get("blocks", []):
                block_type = int(block.get("type", -1))
                bbox = _safe_bounding_box(block.get("bbox"))
                pdf_lines: list[PdfLine] = []
                if block_type == 0:
                    lines = []
                    spans = []
                    for line in block.get("lines", []):
                        line_spans = line.get("spans", [])
                        pdf_line, warning_count = _pdf_line(line)
                        geometry_warning_count += warning_count
                        if pdf_line is not None:
                            pdf_lines.append(pdf_line)
                        spans.extend(line_spans)
                        line_text = "".join(span.get("text", "") for span in line_spans)
                        if line_text.strip():
                            lines.append(line_text.strip())
                    text = "\n".join(lines).strip()
                    if not text:
                        continue
                    element_type = "text"
                    font_size = max((float(span.get("size", 0)) for span in spans), default=None)
                    font_name = max(
                        (str(span.get("font", "")) for span in spans),
                        key=len,
                        default="",
                    )
                    is_bold = any(_is_bold_span(span) for span in spans)
                elif block_type == 1:
                    image_count += 1
                    text = ""
                    element_type = "image"
                    font_size = None
                    font_name = ""
                    is_bold = False
                else:
                    continue

                if bbox is None:
                    geometry_warning_count += 1
                    bbox = _covering_box([line.bbox for line in pdf_lines])
                if bbox is None:
                    discarded_block_count += 1
                    continue

                reading_order = len(elements)
                elements.append(
                    DocumentElement(
                        element_id=f"{content_hash[:12]}:p{page_index}:e{reading_order}",
                        element_type=element_type,
                        text=text,
                        bbox=bbox,
                        reading_order=reading_order,
                        font_size=font_size,
                        font_name=font_name,
                        is_bold=is_bold,
                        lines=pdf_lines,
                    )
                )

            native_character_count = sum(len(element.text) for element in elements)
            native_page = DocumentPage(
                page_number=page_index,
                # All element boxes are persisted in PyMuPDF's unrotated page
                # coordinate space.  page.rect follows display rotation, while
                # cropbox retains the matching unrotated dimensions.
                width=float(page.cropbox.width),
                height=float(page.cropbox.height),
                elements=elements,
                extracted_character_count=native_character_count,
                image_count=image_count,
                needs_ocr=native_character_count < MIN_TEXT_CHARACTERS,
                rotation=int(page.rotation),
                media_box=_bounding_box(page.mediabox),
                crop_box=_bounding_box(page.cropbox),
                coordinate_space="pymupdf_unrotated_page",
                geometry_warning_count=geometry_warning_count,
                discarded_block_count=discarded_block_count,
                native_character_count=native_character_count,
            )
            profile = profile_page(native_page, config.routing)
            route = "full_ocr" if config.mode == "force" else profile.recommended_route
            final_elements = elements
            ocr_character_count = 0
            ocr_attempted = False
            ocr_succeeded = False
            warnings: list[str] = []

            page_selected = (
                config.ocr_page_numbers is None
                or page_index in config.ocr_page_numbers
            )
            should_ocr = page_selected and (
                config.mode == "force" or (
                config.mode == "auto" and route in {"partial_ocr", "full_ocr"}
                )
            )
            if should_ocr:
                ocr_attempted = True
                try:
                    if active_ocr_backend is None:
                        active_ocr_backend = create_ocr_backend(config.ocr_backend)
                    pixmap = page.get_pixmap(dpi=config.ocr_dpi, alpha=False)
                    ocr_result = active_ocr_backend.extract(
                        pixmap.tobytes("png"), width=pixmap.width, height=pixmap.height
                    )
                    recognized = _ocr_elements(
                        ocr_result,
                        content_hash=content_hash,
                        page_index=page_index,
                        page=page,
                    )
                    ocr_character_count = sum(len(element.text) for element in recognized)
                    ocr_succeeded = bool(ocr_character_count)
                    if ocr_succeeded and route == "full_ocr":
                        final_elements = [
                            element for element in elements if element.element_type == "image"
                        ] + recognized
                    elif ocr_succeeded:
                        final_elements = _merge_partial_ocr(elements, recognized)
                    else:
                        warnings.append("ocr_returned_no_text")
                except Exception as exc:
                    if config.ocr_error_policy == "raise":
                        raise
                    warnings.append(f"ocr_failed:{type(exc).__name__}:{exc}")

            final_character_count = sum(
                len(element.text)
                for element in final_elements
                if element.element_type == "text"
            )
            pages.append(
                native_page.model_copy(
                    update={
                        "elements": final_elements,
                        "extracted_character_count": final_character_count,
                        "needs_ocr": route != "native" and not ocr_succeeded,
                        "extraction_route": route,
                        "ocr_character_count": ocr_character_count,
                        "ocr_attempted": ocr_attempted,
                        "ocr_succeeded": ocr_succeeded,
                        "ocr_backend": (
                            active_ocr_backend.name if ocr_attempted and active_ocr_backend else None
                        ),
                        "extraction_warnings": warnings,
                    }
                )
            )

        metadata = pdf.metadata or {}
        return ParsedDocument(
            document_id=f"sha256:{content_hash}",
            source_path=resolved.as_posix(),
            filename=resolved.name,
            content_sha256=content_hash,
            page_count=pdf.page_count,
            pages=pages,
            title=metadata.get("title") or "",
            author=metadata.get("author") or "",
            created_at=_parse_pdf_datetime(metadata.get("creationDate")),
            parser="pymupdf",
            parser_version=pymupdf.VersionBind,
        )


def replace_parsed_document_pages(
    base: ParsedDocument,
    retry: ParsedDocument,
    page_numbers: set[int],
) -> ParsedDocument:
    """Replace selected pages after a bounded second-pass extraction."""

    if base.content_sha256 != retry.content_sha256 or base.page_count != retry.page_count:
        raise ValueError("Cannot merge extraction passes from different PDFs")
    if not page_numbers <= set(range(1, base.page_count + 1)):
        raise ValueError("Retry page numbers fall outside the parsed document")
    pages = [
        retry.pages[index] if index + 1 in page_numbers else page
        for index, page in enumerate(base.pages)
    ]
    return base.model_copy(update={"pages": pages})
