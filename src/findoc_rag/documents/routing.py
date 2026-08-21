"""Auditable page profiling and OCR routing for PDF extraction.

The router deliberately uses cheap signals available in the persisted document
IR.  It does not claim that character presence implies text correctness: image
coverage, suspicious Unicode, and duplicated text all contribute to the route.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from findoc_rag.documents.models import DocumentPage

ExtractionRoute = Literal["native", "partial_ocr", "full_ocr", "manual_review"]


class PdfRoutingConfig(BaseModel):
    min_native_characters: int = Field(default=20, ge=1)
    mixed_page_native_character_limit: int = Field(default=100, ge=1)
    full_page_image_ratio: float = Field(default=0.65, ge=0, le=1)
    suspicious_character_ratio: float = Field(default=0.02, ge=0, le=1)
    duplicate_text_ratio: float = Field(default=0.5, ge=0, le=1)


class PdfPageProfile(BaseModel):
    page_number: int = Field(ge=1)
    native_character_count: int = Field(ge=0)
    text_element_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    image_area_ratio: float = Field(ge=0, le=1)
    text_area_ratio: float = Field(ge=0, le=1)
    replacement_character_ratio: float = Field(ge=0, le=1)
    suspicious_character_ratio: float = Field(ge=0, le=1)
    duplicate_text_ratio: float = Field(ge=0, le=1)
    recommended_route: ExtractionRoute
    reasons: list[str]


def _area_ratio(page: DocumentPage, element_type: str) -> float:
    page_area = page.width * page.height
    if page_area <= 0:
        return 0.0
    area = sum(
        max(0.0, element.bbox.x1 - element.bbox.x0)
        * max(0.0, element.bbox.y1 - element.bbox.y0)
        for element in page.elements
        if element.element_type == element_type
    )
    return min(1.0, area / page_area)


def _duplicate_text_ratio(texts: list[str]) -> float:
    normalized = ["".join(text.split()) for text in texts if text.strip()]
    total = sum(len(text) for text in normalized)
    if not total:
        return 0.0
    counts = Counter(normalized)
    duplicate_characters = sum((count - 1) * len(text) for text, count in counts.items())
    return min(1.0, duplicate_characters / total)


def _suspicious_ratio(text: str) -> tuple[float, float]:
    compact = "".join(text.split())
    if not compact:
        return 0.0, 0.0
    replacement_count = compact.count("\ufffd")
    suspicious_count = replacement_count + sum(
        unicodedata.category(character) in {"Cc", "Cs", "Co"}
        for character in compact
        if character != "\ufffd"
    )
    return replacement_count / len(compact), min(1.0, suspicious_count / len(compact))


def profile_page(
    page: DocumentPage,
    config: PdfRoutingConfig | None = None,
) -> PdfPageProfile:
    """Profile one native extraction and recommend an explicit extraction route."""

    effective = config or PdfRoutingConfig()
    text_elements = [element for element in page.elements if element.element_type == "text"]
    texts = [element.text for element in text_elements]
    text = "\n".join(texts)
    character_count = sum(len(value) for value in texts)
    image_ratio = _area_ratio(page, "image")
    replacement_ratio, suspicious_ratio = _suspicious_ratio(text)
    duplicate_ratio = _duplicate_text_ratio(texts)
    reasons: list[str] = []

    if character_count == 0:
        if page.image_count:
            route: ExtractionRoute = "full_ocr"
            reasons.append("no_native_text_with_image")
        else:
            route = "manual_review"
            reasons.append("empty_page_without_image")
    elif suspicious_ratio > effective.suspicious_character_ratio:
        route = "full_ocr" if image_ratio >= effective.full_page_image_ratio else "manual_review"
        reasons.append("suspicious_unicode")
    elif duplicate_ratio > effective.duplicate_text_ratio:
        route = "manual_review"
        reasons.append("duplicated_native_text")
    elif (
        page.image_count
        and image_ratio >= effective.full_page_image_ratio
        and character_count < effective.mixed_page_native_character_limit
    ):
        route = "partial_ocr"
        reasons.append("large_image_with_sparse_native_text")
    elif character_count < effective.min_native_characters and page.image_count:
        route = "partial_ocr"
        reasons.append("low_native_text_with_image")
    else:
        route = "native"
        reasons.append("native_text_accepted")

    return PdfPageProfile(
        page_number=page.page_number,
        native_character_count=character_count,
        text_element_count=len(text_elements),
        image_count=page.image_count,
        image_area_ratio=image_ratio,
        text_area_ratio=_area_ratio(page, "text"),
        replacement_character_ratio=replacement_ratio,
        suspicious_character_ratio=suspicious_ratio,
        duplicate_text_ratio=duplicate_ratio,
        recommended_route=route,
        reasons=reasons,
    )
