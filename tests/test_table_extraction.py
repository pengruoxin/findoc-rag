"""Tests for table extraction interface and deterministic extractors."""

from __future__ import annotations

from findoc_rag.table_extraction import (
    extract_annual_data,
    extract_cells,
    extract_note_cost,
    extract_quarterly,
    extract_segment,
)

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


def test_quarterly_handles_label_after_values() -> None:
    text = """九、 2024 年分季度主要财务数据
单位：元  币种：人民币
第一季度 （1-3 月份） 第二季度 （4-6 月份） 第三季度 （7-9 月份） 第四季度 （10-12 月份）
营业收入
45,775,517,043.29 36,155,460,624.46 38,845,154,206.94 50,123,020,401.65
归属于上市公司股东
的净利润
24,065,262,374.15 17,630,348,609.22 19,131,941,135.14 25,400,594,303.11
24,051,471,185.69 17,618,626,634.30 19,108,543,634.77 25,462,264,522.66
归属于上市公司股东
的扣除非经常性损益
后的净利润
经营活动产生的现金
流量净额
9,187,422,415.09 27,434,411,397.54 7,799,552,404.82 48,042,305,950.98
"""
    cells = extract_quarterly(text)
    assert len(cells) == 16
    deducted = [cell for cell in cells if "扣除非经常性损益" in cell.row]
    assert [cell.value for cell in deducted] == [
        "24051471185.69",
        "17618626634.30",
        "19108543634.77",
        "25462264522.66",
    ]
    assert {cell.row for cell in cells} == {
        "营业收入",
        "归属于上市公司股东的净利润",
        "归属于上市公司股东的扣除非经常性损益后的净利润",
        "经营活动产生的现金流量净额",
    }


def test_note_cost_extracts_rows_and_columns() -> None:
    text = """40、 营业收入和营业成本
(1). 营业收入和营业成本情况
单位：元  币种：人民币
项目 本期发生额 上期发生额
收入 成本 收入 成本
主营业务 170,611,838,052.02 13,629,995,812.89 147,218,996,281.04 11,620,203,653.32
其他业务 287,314,224.32 159,486,555.09 474,608,713.10 247,070,198.46
合计 170,899,152,276.34 13,789,482,367.98 147,693,604,994.14 11,867,273,851.78
"""
    cells = extract_note_cost(text)
    assert len(cells) == 12
    main = {cell.column: cell.value for cell in cells if cell.row == "主营业务"}
    assert main["本期收入"] == "170611838052.02"
    assert main["本期成本"] == "13629995812.89"
    assert main["上期收入"] == "147218996281.04"
    assert main["上期成本"] == "11620203653.32"
    total = {cell.column: cell.value for cell in cells if cell.row == "合计"}
    assert total["本期收入"] == "170899152276.34"


def test_note_cost_header_order_variant() -> None:
    text = """61、营业收入和营业成本
(1)营业收入和营业成本情况
单位：元 币种：人民币
本期发生额 上期发生额
项目
收入 成本 收入 成本
主营业务 114,120,632,145.60 75,283,113,436.98 124,460,933,210.39 83,735,286,090.68
其他业务 1,272,678,831.09 1,015,719,596.95 1,297,235,706.12 1,053,305,986.66
合计 115,393,310,976.69 76,298,833,033.93 125,758,168,916.51 84,788,592,077.34
"""
    cells = extract_note_cost(text)
    assert len(cells) == 12
    main = {cell.column: cell.value for cell in cells if cell.row == "主营业务"}
    assert main["上期成本"] == "83735286090.68"


def test_segment_extracts_wrapped_labels() -> None:
    text = """(1)主营业务分行业、分产品、分地区、分销售模式情况
主营业务分行业情况
分行业 营业收入 营业成本 毛利率 营业收入比上年增减（%） 营业成本比上年增减（%） 毛利率比上年增减（%）
液体乳及乳制品
制造业 113,399,011,137.27 74,731,457,627.92 34.10 -8.42 -10.20 增加1.30 个百分点
其他 721,621,008.33 551,655,809.06 23.55 13.75 6.44 增加5.25 个百分点
主营业务分产品情况
分产品 营业收入 营业成本 毛利率 营业收入比上年增减（%） 营业成本比上年增减（%） 毛利率比上年增减（%）
液体乳 75,002,672,990.21 51,770,470,355.31 30.98 -12.32 -13.09 增加0.62 个百分点
"""
    cells = extract_segment(text)
    rows = {cell.row for cell in cells}
    assert rows == {"液体乳及乳制品制造业", "其他", "液体乳"}
    dairy = {cell.column: cell.value for cell in cells if cell.row == "液体乳及乳制品制造业"}
    assert dairy["营业收入"] == "113399011137.27"
    assert dairy["营业成本"] == "74731457627.92"
    assert dairy["毛利率"] == "34.10"


def test_annual_data_skips_yoy_and_handles_three_number_row() -> None:
    text = """七、 近三年主要会计数据和财务指标
(一) 主要会计数据
单位：元  币种：人民币
主要会计数据 2024年 2023年 本期比上年同期增减(%) 2022年
营业收入 170,899,152,276.34 147,693,604,994.14 15.71 124,099,843,771.99
归属于上市公司股东的净
利润 86,228,146,421.62 74,734,071,550.75 15.38 62,717,467,870.12
2024年末 2023年末 本期末比上年同期末增减（%） 2022年末
总资产 298,944,579,918.70 272,699,660,092.25 9.62 254,500,826,096.02
股本 1,256,197,800.00 1,256,197,800.00 1,256,197,800.00
"""
    cells = extract_annual_data(text)
    assert len(cells) == 12
    revenue = {cell.column: cell.value for cell in cells if cell.row == "营业收入"}
    assert revenue == {
        "2024年": "170899152276.34",
        "2023年": "147693604994.14",
        "2022年": "124099843771.99",
    }
    total_assets = {cell.column: cell.value for cell in cells if cell.row == "总资产"}
    assert total_assets["2024年"] == "298944579918.70"
    assert total_assets["2022年"] == "254500826096.02"
    equity = {cell.column: cell.value for cell in cells if cell.row == "股本"}
    assert equity["2022年"] == "1256197800.00"
