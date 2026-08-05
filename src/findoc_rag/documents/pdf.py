import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from findoc_rag.documents.models import (
    BoundingBox,
    DocumentElement,
    DocumentPage,
    ParsedDocument,
)

MIN_TEXT_CHARACTERS = 20


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


def parse_pdf(path: Path) -> ParsedDocument:
    """Parse a text-capable PDF into a page-level, coordinate-preserving IR."""
    resolved = path.resolve(strict=True)
    if resolved.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {resolved}")

    content_hash = _sha256(resolved)
    pages: list[DocumentPage] = []

    with pymupdf.open(resolved) as pdf:
        if pdf.page_count == 0:
            raise ValueError("PDF contains no pages")

        for page_index, page in enumerate(pdf, start=1):
            raw = page.get_text("dict", sort=True)
            elements: list[DocumentElement] = []
            image_count = 0

            for block in raw.get("blocks", []):
                block_type = int(block.get("type", -1))
                bbox = BoundingBox.model_validate(
                    dict(zip(("x0", "y0", "x1", "y1"), block["bbox"], strict=True))
                )
                if block_type == 0:
                    lines = []
                    for line in block.get("lines", []):
                        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                        if line_text.strip():
                            lines.append(line_text.strip())
                    text = "\n".join(lines).strip()
                    if not text:
                        continue
                    element_type = "text"
                elif block_type == 1:
                    image_count += 1
                    text = ""
                    element_type = "image"
                else:
                    continue

                reading_order = len(elements)
                elements.append(
                    DocumentElement(
                        element_id=f"{content_hash[:12]}:p{page_index}:e{reading_order}",
                        element_type=element_type,
                        text=text,
                        bbox=bbox,
                        reading_order=reading_order,
                    )
                )

            character_count = sum(len(element.text) for element in elements)
            pages.append(
                DocumentPage(
                    page_number=page_index,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    elements=elements,
                    extracted_character_count=character_count,
                    image_count=image_count,
                    needs_ocr=character_count < MIN_TEXT_CHARACTERS,
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
