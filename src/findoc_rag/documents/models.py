from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Bounding box maximum coordinates must not precede minimums")
        return self


class DocumentElement(BaseModel):
    element_id: str
    element_type: Literal["text", "image"]
    text: str = ""
    bbox: BoundingBox
    reading_order: int
    font_size: float | None = None
    font_name: str = ""
    is_bold: bool = False


class ElementReference(BaseModel):
    element_id: str
    page_number: int = Field(ge=1)
    bbox: BoundingBox


class DocumentPage(BaseModel):
    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    elements: list[DocumentElement]
    extracted_character_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    needs_ocr: bool


class ParsedDocument(BaseModel):
    document_id: str
    source_path: str
    filename: str
    content_sha256: str
    page_count: int = Field(ge=1)
    pages: list[DocumentPage]
    title: str = ""
    author: str = ""
    created_at: datetime | None = None
    parser: str
    parser_version: str

    @model_validator(mode="after")
    def validate_page_count(self) -> "ParsedDocument":
        if self.page_count != len(self.pages):
            raise ValueError("page_count must equal the number of parsed pages")
        return self


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int = Field(ge=0)
    text: str
    section_path: list[str]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    element_references: list[ElementReference]
    character_count: int = Field(ge=1)
    estimated_token_count: int = Field(ge=1)
    is_continuation: bool = False
    document_key: str | None = None
    company_name: str | None = None
    report_year: int | None = None
    document_type: str | None = None

    @model_validator(mode="after")
    def validate_page_range(self) -> "DocumentChunk":
        if self.page_end < self.page_start:
            raise ValueError("page_end must not precede page_start")
        if not self.element_references:
            raise ValueError("A chunk must reference at least one source element")
        return self
