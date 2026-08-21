"""Layout-preserving PDF table evidence and optional DeepSeek interpretation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, Field, field_validator

from findoc_rag.documents.geometry import element_display_bbox
from findoc_rag.documents.models import BoundingBox, DocumentPage
from findoc_rag.provider_credentials import resolve_provider_api_key

TABLE_INTERPRETATION_PROMPT_REVISION = "layout-table-json-v1"
SECTION_HEADING = re.compile(r"^[一二三四五六七八九十]+[、.]")


class TableQuestion(BaseModel):
    """A hard-labelled table fact used by the PDF extraction benchmark."""

    question_id: str
    question: str = Field(min_length=1)
    expected_value: str = Field(min_length=1)
    expected_unit: str = ""
    row_label: str = Field(min_length=1)
    column_label: str = Field(min_length=1)
    section_label: str = ""
    target_page_offset: int = Field(default=0, ge=0)
    requires_previous_page_context: bool = False
    annotation_status: Literal[
        "assistant_curated_provisional", "human_verified"
    ] = "assistant_curated_provisional"


class InterpretedTableAnswer(BaseModel):
    question_id: str
    status: Literal["answered", "insufficient_evidence"]
    value: str = ""
    unit: str = ""
    row_label: str = ""
    column_label: str = ""
    section_label: str = ""
    evidence: str = ""

    @field_validator(
        "value",
        "unit",
        "row_label",
        "column_label",
        "section_label",
        "evidence",
        mode="before",
    )
    @classmethod
    def normalize_nullable_provider_fields(cls, value: object) -> object:
        return "" if value is None else value


class TableInterpretationResponse(BaseModel):
    answers: list[InterpretedTableAnswer]


class TableInterpretationBatch(BaseModel):
    answers: list[InterpretedTableAnswer]
    elapsed_ms: float = Field(ge=0)
    prompt_sha256: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class TableInterpreter(Protocol):
    provider: str
    model: str
    endpoint: str
    prompt_revision: str

    def interpret_page(
        self, questions: Sequence[TableQuestion], evidence: str
    ) -> TableInterpretationBatch: ...


@dataclass(frozen=True)
class LayoutToken:
    text: str
    bbox: BoundingBox

    @property
    def x_center(self) -> float:
        return (self.bbox.x0 + self.bbox.x1) / 2

    @property
    def y_center(self) -> float:
        return (self.bbox.y0 + self.bbox.y1) / 2

    @property
    def height(self) -> float:
        return max(0.01, self.bbox.y1 - self.bbox.y0)

    @property
    def width(self) -> float:
        return max(0.01, self.bbox.x1 - self.bbox.x0)


@dataclass(frozen=True)
class LayoutRow:
    tokens: tuple[LayoutToken, ...]

    @property
    def text(self) -> str:
        return " | ".join(token.text.strip() for token in self.tokens if token.text.strip())

    @property
    def y_center(self) -> float:
        return sum(token.y_center for token in self.tokens) / len(self.tokens)

    @property
    def height(self) -> float:
        return max(token.height for token in self.tokens)


def _page_tokens(page: DocumentPage) -> list[LayoutToken]:
    tokens: list[LayoutToken] = []
    for element in page.elements:
        if element.element_type != "text" or not element.text.strip():
            continue
        if element.lines:
            for line in element.lines:
                for span in line.spans:
                    if span.text.strip():
                        tokens.append(
                            LayoutToken(
                                text=span.text,
                                bbox=element_display_bbox(page, span.bbox),
                            )
                        )
        else:
            tokens.append(
                LayoutToken(
                    text=element.text,
                    bbox=element_display_bbox(page, element.bbox),
                )
            )
    return sorted(tokens, key=lambda token: (token.y_center, token.bbox.x0))


def layout_rows(page: DocumentPage) -> list[LayoutRow]:
    """Cluster positioned native/OCR tokens into compact visual rows."""

    rows: list[list[LayoutToken]] = []
    for token in _page_tokens(page):
        best_index: int | None = None
        best_distance = float("inf")
        for index in range(max(0, len(rows) - 3), len(rows)):
            candidate = LayoutRow(tuple(rows[index]))
            distance = abs(token.y_center - candidate.y_center)
            tolerance = max(2.0, 0.55 * max(token.height, candidate.height))
            if distance <= tolerance and distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            rows.append([token])
        else:
            rows[best_index].append(token)
    clustered = [
        LayoutRow(tuple(sorted(row, key=lambda token: token.bbox.x0))) for row in rows
    ]
    return sorted(clustered, key=lambda row: row.y_center)


def serialize_layout_page(page: DocumentPage, *, max_characters: int = 12_000) -> str:
    """Serialize one page as row-preserving evidence without embedding the PDF/image."""

    lines: list[str] = []
    length = 0
    for index, row in enumerate(layout_rows(page), start=1):
        line = f"L{index:03d}: {row.text}"
        if length + len(line) + 1 > max_characters:
            break
        lines.append(line)
        length += len(line) + 1
    return "\n".join(lines)


def serialize_layout_pages(
    pages: Sequence[DocumentPage], *, max_characters_per_page: int = 12_000
) -> str:
    """Serialize a bounded page group while preserving explicit page boundaries."""

    return "\n\n".join(
        f"[PAGE {offset + 1}]\n"
        f"{serialize_layout_page(page, max_characters=max_characters_per_page)}"
        for offset, page in enumerate(pages)
    )


def normalize_table_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"[\s|:：]", "", normalized)


def normalize_table_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"[\s,，￥¥元]", "", normalized)


def table_values_equal(expected: str, actual: str) -> bool:
    left = normalize_table_value(expected)
    right = normalize_table_value(actual)
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return left == right


def _row_contains(row: LayoutRow, label: str) -> bool:
    expected = normalize_table_label(label)
    if expected in normalize_table_label(row.text):
        return True
    return any(_financial_period_label_match(token.text, label) for token in row.tokens)


def _canonical_financial_period_label(value: str) -> str:
    normalized = normalize_table_label(value)
    normalized = re.sub(r"^[一二三四五六七八九十\d、.]+", "", normalized)
    return normalized.replace("年初余额", "期初余额").replace("年末余额", "期末余额")


def _financial_period_label_match(candidate: str, expected: str) -> bool:
    left = _canonical_financial_period_label(candidate)
    right = _canonical_financial_period_label(expected)
    suffixes = ("期初余额", "期末余额")
    shared_suffix = next(
        (suffix for suffix in suffixes if left.endswith(suffix) and right.endswith(suffix)),
        None,
    )
    if shared_suffix is None or min(len(left), len(right)) < 5:
        return False
    if left == right:
        return True
    return abs(len(left) - len(right)) == 1 and (
        left.endswith(right) or right.endswith(left)
    )


def _matching_tokens(row: LayoutRow, value: str, *, numeric: bool = False) -> list[LayoutToken]:
    normalize = normalize_table_value if numeric else normalize_table_label
    needle = normalize(value)
    return [token for token in row.tokens if needle in normalize(token.text)]


def _column_bounds(header: LayoutRow, column_token: LayoutToken) -> tuple[float, float]:
    centers = sorted(token.x_center for token in header.tokens)
    position = centers.index(column_token.x_center)
    left = -float("inf") if position == 0 else (centers[position - 1] + centers[position]) / 2
    right = (
        float("inf")
        if position == len(centers) - 1
        else (centers[position] + centers[position + 1]) / 2
    )
    return left, right


def _horizontally_related(first: LayoutToken, second: LayoutToken) -> bool:
    overlap = min(first.bbox.x1, second.bbox.x1) - max(
        first.bbox.x0, second.bbox.x0
    )
    center_distance = abs(first.x_center - second.x_center)
    return overlap > 0 or center_distance <= 0.45 * max(first.width, second.width)


def _hierarchical_column_tokens(
    rows: Sequence[LayoutRow], data_row: LayoutRow, column_label: str
) -> list[LayoutToken]:
    """Reconstruct a column label split across vertically stacked header rows."""

    expected = normalize_table_label(column_label)
    if not expected:
        return []
    header_tokens = [
        token
        for row in rows
        if row.y_center < data_row.y_center
        for token in row.tokens
        if len(normalize_table_label(token.text)) >= 2
        and normalize_table_label(token.text) in expected
    ]
    matches: list[LayoutToken] = []
    for seed in header_tokens:
        related = [
            token for token in header_tokens if _horizontally_related(seed, token)
        ]
        related.sort(key=lambda token: (token.y_center, token.bbox.x0))
        combined = "".join(normalize_table_label(token.text) for token in related)
        if expected not in combined:
            continue
        matches.append(
            LayoutToken(
                text=column_label,
                bbox=BoundingBox(
                    x0=min(token.bbox.x0 for token in related),
                    y0=min(token.bbox.y0 for token in related),
                    x1=max(token.bbox.x1 for token in related),
                    y1=max(token.bbox.y1 for token in related),
                ),
            )
        )
    unique: dict[tuple[float, float], LayoutToken] = {}
    for token in matches:
        unique[(round(token.bbox.x0, 3), round(token.bbox.x1, 3))] = token
    return list(unique.values())


def _wrapped_label_rows(
    rows: Sequence[LayoutRow], row_label: str, expected_value: str
) -> list[LayoutRow]:
    """Join a wrapped left-hand row label without merging neighboring data rows."""

    expected_label = normalize_table_label(row_label)
    value_rows = [
        row
        for row in rows
        if normalize_table_value(expected_value) in normalize_table_value(row.text)
    ]
    matches: list[LayoutRow] = []
    for value_row in value_rows:
        value_tokens = _matching_tokens(value_row, expected_value, numeric=True)
        if not value_tokens:
            continue
        value_x = min(token.x_center for token in value_tokens)
        nearby_tokens = [
            token
            for row in rows
            if abs(row.y_center - value_row.y_center)
            <= 2.5 * max(row.height, value_row.height)
            for token in row.tokens
            if token.x_center < value_x
            and normalize_table_label(token.text)
            and normalize_table_label(token.text) in expected_label
        ]
        nearby_tokens.sort(key=lambda token: (token.y_center, token.bbox.x0))
        combined = "".join(normalize_table_label(token.text) for token in nearby_tokens)
        if expected_label not in combined:
            continue
        token_map = {
            (token.text, token.bbox.x0, token.bbox.y0, token.bbox.x1, token.bbox.y1): token
            for token in [*value_row.tokens, *nearby_tokens]
        }
        matches.append(
            LayoutRow(
                tuple(
                    sorted(
                        token_map.values(),
                        key=lambda token: (token.y_center, token.bbox.x0),
                    )
                )
            )
        )
    return matches


def _section_context(
    pages: Sequence[DocumentPage], target_offset: int, data_row: LayoutRow, label: str
) -> tuple[bool, bool, bool]:
    if not label:
        return True, True, False
    context: list[tuple[int, LayoutRow]] = []
    for page_offset, page in enumerate(pages[: target_offset + 1]):
        for row in layout_rows(page):
            if page_offset < target_offset or row.y_center < data_row.y_center:
                context.append((page_offset, row))
    matches = [
        index for index, (_, row) in enumerate(context) if _row_contains(row, label)
    ]
    if not matches:
        return False, False, False
    section_index = matches[-1]
    later_heading = any(
        SECTION_HEADING.match(normalize_table_label(row.text))
        for _, row in context[section_index + 1 :]
    )
    source_page_offset = context[section_index][0]
    return True, not later_heading, source_page_offset < target_offset


def score_table_fact_pages(
    pages: Sequence[DocumentPage],
    question: TableQuestion,
    *,
    allow_hierarchical_headers: bool = True,
    allow_wrapped_row_labels: bool = True,
) -> dict:
    """Score a table fact, including section carry-over across page boundaries."""

    if question.target_page_offset >= len(pages):
        raise ValueError(
            f"Question {question.question_id} targets page offset "
            f"{question.target_page_offset}, but the group has {len(pages)} pages"
        )
    target_page = pages[question.target_page_offset]
    rows = layout_rows(target_page)
    row_matches = [row for row in rows if _row_contains(row, question.row_label)]
    value_rows = [
        row
        for row in rows
        if normalize_table_value(question.expected_value) in normalize_table_value(row.text)
    ]
    associated_rows = [
        row
        for row in row_matches
        if normalize_table_value(question.expected_value) in normalize_table_value(row.text)
    ]
    wrapped_rows = (
        _wrapped_label_rows(rows, question.row_label, question.expected_value)
        if allow_wrapped_row_labels
        else []
    )
    if wrapped_rows:
        row_matches = [*row_matches, *wrapped_rows]
        associated_rows = [*associated_rows, *wrapped_rows]

    section_contexts = [
        _section_context(pages, question.target_page_offset, row, question.section_label)
        for row in associated_rows
    ]
    section_found = any(item[0] for item in section_contexts)
    section_active = any(item[1] for item in section_contexts)
    section_carried = any(item[1] and item[2] for item in section_contexts)
    section_carryover_correct = (
        not question.requires_previous_page_context or section_carried
    )

    header_found = False
    column_aligned = False
    for data_row, (_, active, _) in zip(
        associated_rows, section_contexts, strict=True
    ):
        if not active:
            continue
        header_candidates = [
            row
            for row in rows
            if row.y_center < data_row.y_center
            and _matching_tokens(row, question.column_label)
        ]
        header = header_candidates[-1] if header_candidates else None
        if header is not None:
            header_found = True
            column_tokens = _matching_tokens(header, question.column_label)
            value_tokens = _matching_tokens(data_row, question.expected_value, numeric=True)
            for column_token in column_tokens:
                left, right = _column_bounds(header, column_token)
                if any(left <= token.x_center < right for token in value_tokens):
                    column_aligned = True
                    break
        elif allow_hierarchical_headers:
            hierarchical_tokens = _hierarchical_column_tokens(
                rows, data_row, question.column_label
            )
            if hierarchical_tokens:
                header_found = True
                value_tokens = _matching_tokens(
                    data_row, question.expected_value, numeric=True
                )
                if any(
                    header_token.bbox.x0 <= value_token.x_center <= header_token.bbox.x1
                    for header_token in hierarchical_tokens
                    for value_token in value_tokens
                ):
                    column_aligned = True
        if column_aligned:
            break

    return {
        "question_id": question.question_id,
        "target_page_offset": question.target_page_offset,
        "section_found": section_found,
        "section_active": section_active,
        "section_carried_from_previous_page": section_carried,
        "section_carryover_correct": section_carryover_correct,
        "row_found": bool(row_matches),
        "value_found": bool(value_rows),
        "row_value_same_row": bool(associated_rows),
        "column_header_found": header_found,
        "column_aligned": column_aligned,
        "recoverable": (
            section_active
            and section_carryover_correct
            and bool(associated_rows)
            and header_found
            and column_aligned
        ),
    }


def score_table_fact(
    page: DocumentPage,
    question: TableQuestion,
    *,
    allow_hierarchical_headers: bool = True,
    allow_wrapped_row_labels: bool = True,
) -> dict:
    """Backward-compatible single-page table association scorer."""

    return score_table_fact_pages(
        [page],
        question,
        allow_hierarchical_headers=allow_hierarchical_headers,
        allow_wrapped_row_labels=allow_wrapped_row_labels,
    )


class DeepSeekTableInterpreter:
    """Batch table questions against bounded page evidence using a text-only API."""

    provider = "deepseek-text"
    prompt_revision = TABLE_INTERPRETATION_PROMPT_REVISION

    def __init__(
        self,
        *,
        model: str = "",
        endpoint: str = "",
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model or os.getenv("FINDOC_RAG_ANSWER_MODEL", "deepseek-chat")
        self.endpoint = endpoint or os.getenv(
            "FINDOC_RAG_ANSWER_ENDPOINT", "https://api.deepseek.com/chat/completions"
        )
        self.api_key = resolve_provider_api_key(self.endpoint, api_key)
        self._client = client

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _json_content(content: str) -> dict:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE
            )
        value = json.loads(stripped)
        if not isinstance(value, dict):
            raise TypeError("Table interpreter response must be a JSON object")
        return value

    def interpret_page(
        self, questions: Sequence[TableQuestion], evidence: str
    ) -> TableInterpretationBatch:
        if not self.api_key:
            raise RuntimeError("DeepSeek API key is unavailable for the configured endpoint")
        question_payload = [
            {"question_id": item.question_id, "question": item.question}
            for item in questions
        ]
        user_content = (
            "请依据下面限定页组的表格证据回答所有问题。问题：\n"
            f"{json.dumps(question_payload, ensure_ascii=False)}\n\n"
            "页面证据（PAGE 表示页边界；每个 L 行代表视觉行，竖线分隔同一行内按横坐标排序的单元格）：\n"
            f"{evidence}"
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是严格的财务表格证据解释器。只能使用给定页面，不得凭常识补值。"
                        "返回 JSON 对象，唯一顶层字段为 answers。每个答案必须包含 "
                        "question_id、status(answered 或 insufficient_evidence)、value、unit、"
                        "row_label、column_label、section_label、evidence。value 只填表中原值；"
                        "evidence 填最短证据行。无法确认行列关系时必须 insufficient_evidence。"
                    ),
                },
                {"role": "user", "content": user_content},
            ],
        }
        prompt_bytes = json.dumps(payload["messages"], ensure_ascii=False).encode("utf-8")
        prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        started = time.perf_counter()
        response: httpx.Response | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                if self._client is None:
                    response = httpx.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                        timeout=httpx.Timeout(120.0, connect=30.0),
                    )
                else:
                    response = self._client.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                        timeout=httpx.Timeout(120.0, connect=30.0),
                    )
                response.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
        else:
            raise RuntimeError(f"DeepSeek table interpretation failed: {last_error}") from last_error
        assert response is not None
        body = response.json()
        parsed = TableInterpretationResponse.model_validate(
            self._json_content(body["choices"][0]["message"]["content"])
        )
        usage = body.get("usage", {})
        return TableInterpretationBatch(
            answers=parsed.answers,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            prompt_sha256=prompt_sha256,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )
