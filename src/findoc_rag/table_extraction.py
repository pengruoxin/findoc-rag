"""Table extraction interface and deterministic baselines.

The interface converts linearized chunk text into cell triples
(row label, column header, normalized value). Table reconstruction work
(B phase) must implement the same interface so table-eval-v1 can regress it
against the same annotations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

TableType = Literal["quarterly", "note_cost", "segment", "annual_data"]

NUMBER_PATTERN = re.compile(r"[-−]?\d[\d,]*(?:\.\d+)?")
QUARTER_RANGE_PATTERN = re.compile(r"[（(]\s*\d+\s*[-—]\s*\d+\s*月份?[）)]")
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ExtractedCell:
    row: str
    column: str
    value: str
    section: str = ""


@dataclass(frozen=True)
class AnnualRow:
    """One annual-data row with the yoy delta kept alongside the year values."""

    label: str
    value_2024: str
    value_2023: str
    yoy: str | None
    value_2022: str


def normalize_value(raw: str) -> str:
    """Normalize a numeric string for comparison (commas, plus sign)."""
    return raw.replace(",", "").replace("+", "").replace("−", "-")


def normalize_label(raw: str) -> str:
    return WHITESPACE.sub("", raw)


QUARTER_COLUMNS = ("第一季度", "第二季度", "第三季度", "第四季度")
QUARTER_METRICS = (
    ("营业收入", "营业收入"),
    ("归属于上市公司股东的净利润", "归属于上市公司股东的净利润"),
    (
        "归属于上市公司股东的扣除非经常性损益后的净利润",
        "归属于上市公司股东的扣除非经常性损益后的净利润",
    ),
    ("经营活动产生的现金流量净额", "经营活动产生的现金流量净额"),
)

NOTE_COST_ROWS = ("主营业务", "其他业务", "合计")
NOTE_COST_COLUMNS = ("本期收入", "本期成本", "上期收入", "上期成本")

SEGMENT_MARKERS = (
    "主营业务分行业情况",
    "主营业务分产品情况",
    "主营业务分地区情况",
    "主营业务分销售模式情况",
)
SEGMENT_COLUMNS = ("营业收入", "营业成本", "毛利率")
SEGMENT_HEADER_TERMS = (
    "分行业",
    "分产品",
    "分地区",
    "销售模式",
    "营业收入",
    "营业成本",
    "毛利率",
    "比上年增减",
    "上年增减",
    "（%）",
    "(%)",
    "百分点",
)
SEGMENT_ROW_JUNK = ("营业收入", "营业成本", "毛利率", "百分点", "增减")

ANNUAL_COLUMNS = ("2024年", "2023年", "2022年")
ANNUAL_ROW_LABELS = frozenset(
    {
        "营业收入",
        "归属于上市公司股东的净利润",
        "归属于上市公司股东的扣除非经常性损益的净利润",
        "经营活动产生的现金流量净额",
        "归属于上市公司股东的净资产",
        "总资产",
        "股本",
    }
)
ANNUAL_HEADER_TERMS = (
    "2022年",
    "2022年末",
    "年末",
    "年",
    "（%）",
    "(%)",
    "比上年",
    "同期末",
    "本期",
    "增减",
)


def _compact(text: str) -> str:
    return WHITESPACE.sub("", text)


def _normalize(text: str) -> str:
    """Collapse whitespace to single spaces so adjacent numbers stay separate."""
    return WHITESPACE.sub(" ", text).strip()


def extract_quarterly(text: str) -> list[ExtractedCell]:
    """Extract the regulated four-row quarterly table.

    The CSRC quarterly table always has one numeric row per metric in the
    order 营业收入 / 归母净利润 / 扣非净利润 / 经营现金流净额. Reading order
    from the PDF can place a wrapped label *after* its numbers (Moutai
    deducted-profit row), so rows are aligned by four-number groups mapped to
    the metric labels that actually occur in the text, not by label position.
    """
    normalized = QUARTER_RANGE_PATTERN.sub("", _normalize(text))
    table_start = normalized.find("第一季度")
    if table_start >= 0:
        normalized = normalized[table_start:]
    compact = _compact(normalized)
    present = [label for label, _ in QUARTER_METRICS if label in compact]
    numbers = NUMBER_PATTERN.findall(normalized)
    if not present or len(numbers) != 4 * len(present):
        return []
    cells: list[ExtractedCell] = []
    for index, label in enumerate(present):
        values = numbers[index * 4 : (index + 1) * 4]
        for column, value in zip(QUARTER_COLUMNS, values, strict=True):
            cells.append(
                ExtractedCell(row=label, column=column, value=normalize_value(value))
            )
    return cells


def _strip_header(label: str, terms: tuple[str, ...]) -> str:
    """Remove leading table-header fragments, keeping the trailing row label."""
    best_end = -1
    for term in terms:
        position = label.rfind(term)
        if position >= 0 and position + len(term) > best_end:
            best_end = position + len(term)
    return label[best_end:] if best_end >= 0 else label


def _tokenize(text: str) -> list[str]:
    """Split into text / number tokens, keeping whitespace as a separator."""
    normalized = _normalize(text)
    return [
        token
        for token in re.split(rf"\s+|({NUMBER_PATTERN.pattern})", normalized)
        if token and not token.isspace()
    ]


def _number_group_flush(
    group: list[str],
    label_buf: str,
    header_terms: tuple[str, ...],
    row_labels: frozenset[str] | None,
    columns: tuple[str, ...],
    min_group: int,
    max_group: int,
    junk: tuple[str, ...],
    section: str = "",
) -> tuple[list[ExtractedCell], str]:
    """Emit cells for a completed numeric row when the label matches."""
    if not (min_group <= len(group) <= max_group):
        return [], ""
    label = _compact(_strip_header(label_buf, header_terms)).strip(
        " ，,。、：:；;（()）"
    )
    if not label or len(label) > 24:
        return [], ""
    if any(marker in label for marker in junk):
        return [], ""
    if row_labels is not None and label not in row_labels:
        return [], ""
    if len(group) == 4:
        values = (group[0], group[1], group[3])
    else:
        values = tuple(group[: len(columns)])
    cells = [
        ExtractedCell(row=label, column=column, value=normalize_value(value), section=section)
        for column, value in zip(columns, values, strict=True)
    ]
    return cells, ""


def extract_note_cost(text: str) -> list[ExtractedCell]:
    """Extract 主营业务/其他业务/合计 rows from the note revenue-cost table."""
    normalized = _normalize(text)
    cells: list[ExtractedCell] = []
    for row in NOTE_COST_ROWS:
        position = normalized.find(row)
        if position < 0:
            continue
        values = NUMBER_PATTERN.findall(normalized[position + len(row) :])
        if len(values) < len(NOTE_COST_COLUMNS):
            continue
        for column, value in zip(NOTE_COST_COLUMNS, values[:4], strict=True):
            cells.append(
                ExtractedCell(row=row, column=column, value=normalize_value(value))
            )
    return cells


def _parse_segment_region(region: str, section: str) -> list[ExtractedCell]:
    cells: list[ExtractedCell] = []
    label_buf = ""
    group: list[str] = []

    def flush() -> None:
        nonlocal group, label_buf
        if not group:
            return
        emitted, label_buf = _number_group_flush(
            group,
            label_buf,
            SEGMENT_HEADER_TERMS,
            None,
            SEGMENT_COLUMNS,
            min_group=5,
            max_group=6,
            junk=SEGMENT_ROW_JUNK,
            section=section,
        )
        cells.extend(emitted)
        group = []

    for token in _tokenize(region):
        if NUMBER_PATTERN.fullmatch(token):
            group.append(token)
        else:
            flush()
            label_buf += token
    flush()
    return cells


def extract_segment(text: str) -> list[ExtractedCell]:
    """Extract 营业收入/营业成本/毛利率 rows from segment sub-tables."""
    normalized = _normalize(text)
    cells: list[ExtractedCell] = []
    for marker in SEGMENT_MARKERS:
        start = normalized.find(marker)
        if start < 0:
            continue
        ends = [
            position
            for other in SEGMENT_MARKERS
            if other != marker
            for position in [normalized.find(other, start + len(marker))]
            if position > start
        ]
        end = min(ends) if ends else len(normalized)
        cells.extend(
            _parse_segment_region(normalized[start + len(marker) : end], marker)
        )
    return cells


def _parse_annual_block(block: str) -> list[AnnualRow]:
    rows: list[AnnualRow] = []
    label_buf = ""
    group: list[str] = []

    def flush() -> None:
        nonlocal group, label_buf
        if not group:
            return
        label = _compact(_strip_header(label_buf, ANNUAL_HEADER_TERMS)).strip(
            " ，,。、：:；;（()）"
        )
        label_buf = ""
        if label in ANNUAL_ROW_LABELS and len(group) in (3, 4):
            if len(group) == 4:
                rows.append(
                    AnnualRow(
                        label=label,
                        value_2024=normalize_value(group[0]),
                        value_2023=normalize_value(group[1]),
                        yoy=normalize_value(group[2]),
                        value_2022=normalize_value(group[3]),
                    )
                )
            else:
                rows.append(
                    AnnualRow(
                        label=label,
                        value_2024=normalize_value(group[0]),
                        value_2023=normalize_value(group[1]),
                        yoy=None,
                        value_2022=normalize_value(group[2]),
                    )
                )
        group = []

    for token in _tokenize(block):
        if NUMBER_PATTERN.fullmatch(token):
            group.append(token)
        else:
            flush()
            label_buf += token
    flush()
    return rows


def extract_annual_rows(text: str) -> list[AnnualRow]:
    """Extract annual accounting-data rows including the yoy delta column."""
    normalized = _normalize(text)
    balance_match = re.search(r"2024\s*年末", normalized)
    balance_position = balance_match.start() if balance_match else -1
    if balance_position >= 0:
        blocks = (normalized[:balance_position], normalized[balance_position:])
    else:
        blocks = (normalized,)
    rows: list[AnnualRow] = []
    for block in blocks:
        rows.extend(_parse_annual_block(block))
    return rows


def extract_annual_data(text: str) -> list[ExtractedCell]:
    """Extract the annual accounting-data table (both 年报 and 年末 blocks)."""
    cells: list[ExtractedCell] = []
    for row in extract_annual_rows(text):
        values = (row.value_2024, row.value_2023, row.value_2022)
        for column, value in zip(ANNUAL_COLUMNS, values, strict=True):
            cells.append(ExtractedCell(row=row.label, column=column, value=value))
    return cells


def extract_cells(text: str, table_type: TableType) -> list[ExtractedCell]:
    """Extract cell triples from linearized chunk text."""
    if table_type == "quarterly":
        return extract_quarterly(text)
    if table_type == "note_cost":
        return extract_note_cost(text)
    if table_type == "segment":
        return extract_segment(text)
    if table_type == "annual_data":
        return extract_annual_data(text)
    return []
