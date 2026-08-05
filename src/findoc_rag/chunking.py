import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from pydantic import BaseModel, Field, model_validator

from findoc_rag.documents.models import (
    DocumentChunk,
    DocumentElement,
    ElementReference,
    ParsedDocument,
)

CJK_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._%/-][A-Za-z0-9]+)*")
WHITESPACE = re.compile(r"\s+")
VARIABLE_NUMBER = re.compile(r"\d+")
PAGE_NUMBER_ONLY = re.compile(r"^(?:第\s*)?\d+(?:\s*页)?(?:\s*/\s*\d+)?$")
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])|\n+")

HEADING_PATTERNS: tuple[tuple[int, re.Pattern[str]], ...] = (
    (1, re.compile(r"^第[一二三四五六七八九十百千万零〇\d]+[编章节]\s*\S+")),
    (2, re.compile(r"^[一二三四五六七八九十百千万零〇]+、\s*\S+")),
    (3, re.compile(r"^[（(][一二三四五六七八九十百千万零〇\d]+[）)]\s*\S+")),
    (4, re.compile(r"^\d+(?:\.\d+)+[.、]?\s*\S+")),
    (4, re.compile(r"^\d+[.、]\s*\S+")),
)


class ChunkingConfig(BaseModel):
    target_tokens: int = Field(default=450, ge=50)
    max_tokens: int = Field(default=650, ge=100)
    min_tokens: int = Field(default=100, ge=1)
    overlap_tokens: int = Field(default=60, ge=0)
    repeated_margin_ratio: float = Field(default=0.12, gt=0, lt=0.5)
    repeated_page_ratio: float = Field(default=0.2, gt=0, le=1)

    @model_validator(mode="after")
    def validate_budgets(self) -> "ChunkingConfig":
        if self.min_tokens > self.target_tokens or self.target_tokens > self.max_tokens:
            raise ValueError("Expected min_tokens <= target_tokens <= max_tokens")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        return self


class ChunkingReport(BaseModel):
    document_id: str
    page_count: int
    chunk_count: int
    input_text_element_count: int
    repeated_margin_element_count: int
    referenced_text_element_count: int
    source_element_coverage: float
    chunks_below_min_tokens: int
    chunks_above_max_tokens: int
    cross_page_chunk_count: int
    continuation_chunk_count: int
    section_context_chunk_count: int
    token_min: int
    token_p50: int
    token_p95: int
    token_max: int
    config: ChunkingConfig


@dataclass(frozen=True)
class _Unit:
    text: str
    tokens: int
    reference: ElementReference
    section_path: tuple[str, ...]
    heading_level: int | None
    is_continuation: bool


def estimate_tokens(text: str) -> int:
    """Estimate tokens without tying chunking to a specific embedding tokenizer."""
    cjk_count = len(CJK_CHARACTER.findall(text))
    latin_count = len(LATIN_TOKEN.findall(CJK_CHARACTER.sub(" ", text)))
    return max(1, cjk_count + math.ceil(latin_count * 1.3))


def _normalize_repeated_text(text: str) -> str:
    return VARIABLE_NUMBER.sub("#", WHITESPACE.sub("", text)).lower()


def detect_repeated_margin_elements(
    document: ParsedDocument, config: ChunkingConfig
) -> set[str]:
    """Detect repeated headers/footers using both page position and repetition."""
    occurrences: dict[str, set[int]] = defaultdict(set)
    elements_by_key: dict[str, list[DocumentElement]] = defaultdict(list)

    for page in document.pages:
        for element in page.elements:
            if element.element_type != "text" or len(element.text) > 160:
                continue
            normalized = _normalize_repeated_text(element.text)
            if not normalized:
                continue
            at_margin = (
                element.bbox.y1 <= page.height * config.repeated_margin_ratio
                or element.bbox.y0 >= page.height * (1 - config.repeated_margin_ratio)
            )
            if at_margin:
                occurrences[normalized].add(page.page_number)
                elements_by_key[normalized].append(element)

    threshold = max(3, math.ceil(document.page_count * config.repeated_page_ratio))
    repeated_keys = {key for key, pages in occurrences.items() if len(pages) >= threshold}
    excluded = {
        element.element_id
        for key in repeated_keys
        for element in elements_by_key[key]
    }

    for page in document.pages:
        for element in page.elements:
            if PAGE_NUMBER_ONLY.fullmatch(WHITESPACE.sub("", element.text)):
                excluded.add(element.element_id)
    return excluded


def _body_font_size(document: ParsedDocument, excluded_ids: set[str]) -> float:
    sizes = [
        element.font_size
        for page in document.pages
        for element in page.elements
        if element.element_id not in excluded_ids
        and element.element_type == "text"
        and element.font_size is not None
        and len(element.text) >= 20
    ]
    return float(median(sizes)) if sizes else 10.0


def detect_heading_level(element: DocumentElement, body_font_size: float) -> int | None:
    text = WHITESPACE.sub(" ", element.text).strip()
    if not text or len(text) > 100 or text.count("\n") > 2:
        return None
    numeric_groups = re.findall(r"[-+]?\d[\d,.]*", text)
    looks_like_table_row = len(numeric_groups) >= 2
    looks_like_toc_entry = bool(re.search(r"\.{5,}|…{3,}", text))
    if looks_like_table_row or looks_like_toc_entry:
        return None
    looks_like_sentence = text.endswith(("。", "；", ";", "，", ",", "：", ":"))
    for level, pattern in HEADING_PATTERNS:
        if pattern.match(text) and (level == 1 or (len(text) <= 60 and not looks_like_sentence)):
            return level

    is_large = element.font_size is not None and element.font_size >= body_font_size * 1.18
    is_prominent_bold = (
        element.is_bold
        and element.font_size is not None
        and element.font_size >= body_font_size * 1.05
    )
    is_visually_prominent = is_large or is_prominent_bold
    if is_visually_prominent and not looks_like_sentence and len(text) <= 60:
        return 2
    return None


def _hard_split(text: str, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    remaining = text.strip()
    while estimate_tokens(remaining) > max_tokens:
        low, high = 1, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if estimate_tokens(remaining[:middle]) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        split_at = max(low, 1)
        whitespace_at = remaining.rfind(" ", 0, split_at)
        if whitespace_at >= split_at // 2:
            split_at = whitespace_at
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def split_element_text(text: str, max_tokens: int) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text.strip()]

    sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if sentence_tokens > max_tokens:
            if current:
                pieces.append("".join(current))
                current = []
                current_tokens = 0
            pieces.extend(_hard_split(sentence, max_tokens))
        elif current and current_tokens + sentence_tokens > max_tokens:
            pieces.append("".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens
    if current:
        pieces.append("".join(current))
    return pieces


def _build_units(document: ParsedDocument, config: ChunkingConfig) -> list[_Unit]:
    excluded = detect_repeated_margin_elements(document, config)
    body_font_size = _body_font_size(document, excluded)
    section_stack: list[str] = []
    units: list[_Unit] = []

    for page in document.pages:
        for element in page.elements:
            if element.element_id in excluded or element.element_type != "text":
                continue
            text = element.text.strip()
            if not text:
                continue
            heading_level = detect_heading_level(element, body_font_size)
            if heading_level is not None:
                section_stack = section_stack[: heading_level - 1]
                section_stack.append(WHITESPACE.sub(" ", text))

            reference = ElementReference(
                element_id=element.element_id,
                page_number=page.page_number,
                bbox=element.bbox,
            )
            pieces = split_element_text(text, config.target_tokens)
            for piece_index, piece in enumerate(pieces):
                units.append(
                    _Unit(
                        text=piece,
                        tokens=estimate_tokens(piece),
                        reference=reference,
                        section_path=tuple(section_stack),
                        heading_level=heading_level if piece_index == 0 else None,
                        is_continuation=piece_index > 0,
                    )
                )
    return units


def chunk_document(
    document: ParsedDocument, config: ChunkingConfig | None = None
) -> list[DocumentChunk]:
    config = config or ChunkingConfig()
    units = _build_units(document, config)
    chunks: list[DocumentChunk] = []
    current: list[_Unit] = []
    current_tokens = 0

    def flush(retain_overlap: bool) -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = "\n\n".join(unit.text for unit in current)
        references = list(
            {
                (unit.reference.element_id, unit.reference.page_number): unit.reference
                for unit in current
            }.values()
        )
        page_numbers = [reference.page_number for reference in references]
        index = len(chunks)
        digest = hashlib.sha256(
            f"{document.document_id}:{index}:{text}".encode()
        ).hexdigest()[:16]
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.content_sha256[:12]}:c{index}:{digest}",
                document_id=document.document_id,
                chunk_index=index,
                text=text,
                section_path=list(current[-1].section_path),
                page_start=min(page_numbers),
                page_end=max(page_numbers),
                element_references=references,
                character_count=len(text),
                estimated_token_count=estimate_tokens(text),
                is_continuation=any(unit.is_continuation for unit in current),
            )
        )

        overlap: list[_Unit] = []
        overlap_total = 0
        if retain_overlap and config.overlap_tokens:
            for unit in reversed(current):
                if unit.heading_level is not None or overlap_total + unit.tokens > config.overlap_tokens:
                    break
                overlap.append(unit)
                overlap_total += unit.tokens
        current = list(reversed(overlap))
        current_tokens = overlap_total

    for unit in units:
        starts_new_section = unit.heading_level is not None and current
        if starts_new_section and current_tokens >= config.min_tokens:
            flush(retain_overlap=False)

        exceeds_max = current and current_tokens + unit.tokens > config.max_tokens
        reaches_target = current and current_tokens >= config.target_tokens
        if exceeds_max or reaches_target:
            flush(retain_overlap=not starts_new_section)
            if current and current_tokens + unit.tokens > config.max_tokens:
                flush(retain_overlap=False)

        current.append(unit)
        current_tokens += unit.tokens

    flush(retain_overlap=False)
    return chunks


def build_chunking_report(
    document: ParsedDocument,
    chunks: list[DocumentChunk],
    config: ChunkingConfig,
) -> ChunkingReport:
    if not chunks:
        raise ValueError("Cannot report on an empty chunk collection")
    text_elements = {
        element.element_id
        for page in document.pages
        for element in page.elements
        if element.element_type == "text" and element.text.strip()
    }
    excluded = detect_repeated_margin_elements(document, config) & text_elements
    eligible = text_elements - excluded
    referenced = {
        reference.element_id for chunk in chunks for reference in chunk.element_references
    }
    token_counts = sorted(chunk.estimated_token_count for chunk in chunks)
    p50_index = len(token_counts) // 2
    p95_index = min(len(token_counts) - 1, math.ceil(len(token_counts) * 0.95) - 1)
    coverage = len(referenced & eligible) / len(eligible) if eligible else 1.0

    return ChunkingReport(
        document_id=document.document_id,
        page_count=document.page_count,
        chunk_count=len(chunks),
        input_text_element_count=len(text_elements),
        repeated_margin_element_count=len(excluded),
        referenced_text_element_count=len(referenced & eligible),
        source_element_coverage=coverage,
        chunks_below_min_tokens=sum(
            chunk.estimated_token_count < config.min_tokens for chunk in chunks
        ),
        chunks_above_max_tokens=sum(
            chunk.estimated_token_count > config.max_tokens for chunk in chunks
        ),
        cross_page_chunk_count=sum(chunk.page_start != chunk.page_end for chunk in chunks),
        continuation_chunk_count=sum(chunk.is_continuation for chunk in chunks),
        section_context_chunk_count=sum(bool(chunk.section_path) for chunk in chunks),
        token_min=token_counts[0],
        token_p50=token_counts[p50_index],
        token_p95=token_counts[p95_index],
        token_max=token_counts[-1],
        config=config,
    )
