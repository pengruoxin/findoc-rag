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

NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ExtractedCell:
    row: str
    column: str
    value: str


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


def _compact(text: str) -> str:
    return WHITESPACE.sub("", text)


def extract_quarterly(text: str) -> list[ExtractedCell]:
    """Deterministic baseline: metric label followed by four numbers.

    Known limitation reproduced on purpose: when a label appears *after* its
    values (Moutai quarterly deducted profit), the baseline reads the wrong
    row. This is what table-eval-v1 is designed to catch.
    """
    compact = _compact(text)
    cells: list[ExtractedCell] = []
    for query_label, text_label in QUARTER_METRICS:
        if query_label not in compact:
            continue
        pattern = r"\s*".join(re.escape(char) for char in text_label)
        match = re.search(pattern, text)
        if match is None:
            continue
        values = NUMBER_PATTERN.findall(text[match.end() :])
        if len(values) < 4:
            continue
        for column, value in zip(QUARTER_COLUMNS, values[:4], strict=True):
            cells.append(
                ExtractedCell(row=query_label, column=column, value=normalize_value(value))
            )
    return cells


def extract_cells(text: str, table_type: TableType) -> list[ExtractedCell]:
    """Extract cell triples from linearized chunk text.

    Only ``quarterly`` has a baseline implementation in v1; other table types
    are intentionally unimplemented until the B-phase reconstruction lands.
    """
    if table_type == "quarterly":
        return extract_quarterly(text)
    return []
