"""Conservative discovery heuristics for complex table-page annotation candidates."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field

CandidateKind = Literal[
    "native_control",
    "borderless_table",
    "merged_hierarchical_header",
    "cross_page_continuation",
    "rotated_or_mixed_layout",
]

NUMBER = re.compile(r"[-+]?\d[\d,，]*(?:\.\d+)?%?")
CELL_NUMBER = re.compile(r"^\(?[-+]?\d[\d,，]*(?:\.\d+)?%?\)?$")
TABLE_TERMS = ("项目", "合计", "期末余额", "期初余额", "本期", "上期", "单位")
HIERARCHY_TERMS = (
    "调整前",
    "调整后",
    "本期发生额",
    "上期发生额",
    "本年发生额",
    "上年发生额",
    "本年数",
    "上年数",
)
CONTINUATION_TERMS = ("续表", "接上表", "续上表")
GROUP_HEADERS = {
    "本期发生额",
    "上期发生额",
    "本年发生额",
    "上年发生额",
}
CHILD_HEADERS = {"收入", "成本", "调整前", "调整后", "期初余额", "期末余额"}
YEAR_HEADER = re.compile(r"^20\d{2}年$")


class PdfPageCandidateSignals(BaseModel):
    page_number: int = Field(ge=1)
    native_characters: int = Field(ge=0)
    numeric_token_count: int = Field(ge=0)
    financial_numeric_span_count: int = Field(ge=0)
    repeated_numeric_column_count: int = Field(ge=0)
    max_numeric_column_support: int = Field(ge=0)
    table_term_count: int = Field(ge=0)
    hierarchy_term_count: int = Field(ge=0)
    hierarchical_header_pair_count: int = Field(ge=0)
    continuation_term_count: int = Field(ge=0)
    drawing_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    image_coverage_max: float = Field(ge=0)
    rotation: int


class PdfPageCandidate(BaseModel):
    kind: CandidateKind
    score: float
    reason_codes: list[str]


def compact_text(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def build_page_signals(
    *,
    page_number: int,
    text: str,
    span_texts: list[str] | None = None,
    span_items: list[tuple[str, float, float]] | None = None,
    drawing_count: int,
    image_count: int,
    image_coverage_max: float,
    rotation: int,
) -> PdfPageCandidateSignals:
    compact = compact_text(text)
    financial_spans = 0
    column_rows: dict[int, set[int]] = {}
    items = span_items or [(value, 0.0, float(index)) for index, value in enumerate(span_texts or [])]
    for raw_span, x_center, y_center in items:
        span = compact_text(raw_span).replace("（", "(").replace("）", ")")
        digits = "".join(character for character in span if character.isdigit())
        if CELL_NUMBER.fullmatch(span) and (
            "," in span or "，" in span or "." in span or "%" in span or len(digits) >= 5
        ):
            financial_spans += 1
            x_bucket = round(x_center / 15)
            column_rows.setdefault(x_bucket, set()).add(round(y_center / 4))
    column_supports = [len(rows) for rows in column_rows.values()]
    normalized_items = [
        (compact_text(raw), x_center, y_center)
        for raw, x_center, y_center in items
        if compact_text(raw)
    ]
    hierarchical_pairs = 0
    for group, group_x, group_y in normalized_items:
        if group not in GROUP_HEADERS:
            continue
        hierarchical_pairs += sum(
            child in CHILD_HEADERS
            and 0 < child_y - group_y <= 45
            and abs(child_x - group_x) <= 110
            for child, child_x, child_y in normalized_items
        )
    for year, year_x, year_y in normalized_items:
        if not YEAR_HEADER.fullmatch(year):
            continue
        hierarchical_pairs += any(
            not NUMBER.fullmatch(parent)
            and parent not in GROUP_HEADERS
            and 0 < year_y - parent_y <= 45
            and abs(year_x - parent_x) <= 60
            and len(parent) <= 12
            for parent, parent_x, parent_y in normalized_items
        )
    return PdfPageCandidateSignals(
        page_number=page_number,
        native_characters=len(compact),
        numeric_token_count=len(NUMBER.findall(text)),
        financial_numeric_span_count=financial_spans,
        repeated_numeric_column_count=sum(support >= 3 for support in column_supports),
        max_numeric_column_support=max(column_supports, default=0),
        table_term_count=sum(compact.count(term) for term in TABLE_TERMS),
        hierarchy_term_count=sum(compact.count(term) for term in HIERARCHY_TERMS),
        hierarchical_header_pair_count=hierarchical_pairs,
        continuation_term_count=sum(compact.count(term) for term in CONTINUATION_TERMS),
        drawing_count=drawing_count,
        image_count=image_count,
        image_coverage_max=image_coverage_max,
        rotation=rotation,
    )


def classify_page(signals: PdfPageCandidateSignals) -> list[PdfPageCandidate]:
    """Return annotation candidates, never automatic gold labels."""

    candidates: list[PdfPageCandidate] = []
    table_like = (
        signals.financial_numeric_span_count >= 8
        and signals.repeated_numeric_column_count >= 2
        and signals.table_term_count >= 2
    )
    if table_like and signals.native_characters >= 500:
        candidates.append(
            PdfPageCandidate(
                kind="native_control",
                score=(
                    signals.numeric_token_count
                    + signals.table_term_count * 3
                    + min(20.0, signals.native_characters / 100)
                ),
                reason_codes=["dense_native_text", "table_terms", "numeric_density"],
            )
        )
    borderless_grid = (
        signals.repeated_numeric_column_count >= 3
        and signals.max_numeric_column_support >= 4
    )
    if table_like and borderless_grid and signals.drawing_count <= 6:
        candidates.append(
            PdfPageCandidate(
                kind="borderless_table",
                score=(
                    signals.numeric_token_count
                    + signals.table_term_count * 4
                    + signals.financial_numeric_span_count
                    + signals.repeated_numeric_column_count * 15
                    + signals.max_numeric_column_support * 3
                    - signals.drawing_count * 2
                ),
                reason_codes=[
                    "table_like_text",
                    "repeated_numeric_columns",
                    "few_vector_rules",
                ],
            )
        )
    if table_like and signals.hierarchical_header_pair_count >= 2:
        candidates.append(
            PdfPageCandidate(
                kind="merged_hierarchical_header",
                score=(
                    signals.hierarchical_header_pair_count * 20
                    + signals.hierarchy_term_count * 4
                    + signals.numeric_token_count
                    + signals.table_term_count
                ),
                reason_codes=["geometric_header_parent_child_pairs", "table_like_text"],
            )
        )
    if table_like and signals.continuation_term_count:
        candidates.append(
            PdfPageCandidate(
                kind="cross_page_continuation",
                score=(
                    signals.continuation_term_count * 20
                    + signals.numeric_token_count
                    + signals.table_term_count
                ),
                reason_codes=["explicit_continuation_marker", "table_like_text"],
            )
        )
    mixed_layout = (
        signals.image_count > 0
        and signals.image_coverage_max >= 0.1
        and signals.native_characters >= 250
        and signals.numeric_token_count >= 10
    )
    if signals.rotation % 360 or mixed_layout:
        reasons = []
        if signals.rotation % 360:
            reasons.append("rotated_page")
        if mixed_layout:
            reasons.append("image_and_table_text_overlap")
        candidates.append(
            PdfPageCandidate(
                kind="rotated_or_mixed_layout",
                score=(
                    abs(signals.rotation % 360)
                    + signals.image_coverage_max * 30
                    + signals.numeric_token_count
                ),
                reason_codes=reasons,
            )
        )
    return candidates
