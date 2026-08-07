"""Tests for table extraction interface and quarterly baseline."""

from __future__ import annotations

from findoc_rag.table_extraction import extract_cells, extract_quarterly

QUARTERLY_TEXT = """九、2024 年分季度主要财务数据
单位：元
第一季度
第二季度
第三季度
第四季度
营业收入
45,775,517,043.29
36,155,460,624.46
38,845,154,206.94
50,123,020,401.65
归属于上市公司股东
的净利润
24,065,262,374.15
17,630,348,609.22
19,131,941,135.14
25,400,594,303.11
经营活动产生的现金
流量净额
9,187,422,415.09
27,434,411,397.54
7,799,552,404.82
48,042,305,950.98
"""


def test_quarterly_extracts_three_rows() -> None:
    cells = extract_quarterly(QUARTERLY_TEXT)
    rows = {cell.row for cell in cells}
    assert rows == {
        "营业收入",
        "归属于上市公司股东的净利润",
        "经营活动产生的现金流量净额",
    }


def test_quarterly_values_normalized() -> None:
    cells = extract_quarterly(QUARTERLY_TEXT)
    revenue = [cell for cell in cells if cell.row == "营业收入"]
    assert [cell.value for cell in revenue] == [
        "45775517043.29",
        "36155460624.46",
        "38845154206.94",
        "50123020401.65",
    ]


def test_unimplemented_table_types_return_empty() -> None:
    assert extract_cells("任意文本", "segment") == []
    assert extract_cells("任意文本", "note_cost") == []
    assert extract_cells("任意文本", "annual_data") == []
