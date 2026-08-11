from findoc_rag.table_reconstruction import (
    Block,
    ExtractedCell,
    blocks_from_pymupdf_dict,
    detect_unit,
    extract_cells,
    merge_pages,
    normalize_label,
    normalize_value,
    reconstruct_cells,
)


def _keys(cells):
    return {(normalize_label(c.row), c.column, normalize_value(c.value)) for c in cells}


BLOCKS_A = [
    {"lines": [{"spans": [
        {"text": "归属于上市公司股东的扣除非经常性损益后的净利润",
         "bbox": (40.8, 200.0, 180.0, 209.0), "size": 9.0, "bold": False}]}]},
    {"lines": [{"spans": [
        {"text": "24,051,471,185.69", "bbox": (200.0, 200.0, 300.0, 209.0), "size": 9.0, "bold": False},
        {"text": "17,618,626,634.30", "bbox": (300.0, 200.0, 400.0, 209.0), "size": 9.0, "bold": False},
        {"text": "19,108,543,634.77", "bbox": (400.0, 200.0, 500.0, 209.0), "size": 9.0, "bold": False},
        {"text": "25,462,264,522.66", "bbox": (500.0, 200.0, 600.0, 209.0), "size": 9.0, "bold": False}]}]},
]

BLOCKS_B = [
    {"lines": [{"spans": [
        {"text": "液体乳及乳制品", "bbox": (40.8, 100.0, 150.0, 109.0), "size": 9.0, "bold": False}]}]},
    {"lines": [{"spans": [
        {"text": "制造业", "bbox": (40.8, 109.0, 90.0, 118.0), "size": 9.0, "bold": False},
        {"text": "113,399,011,137.27", "bbox": (200.0, 105.0, 300.0, 114.0), "size": 9.0, "bold": False},
        {"text": "74,731,457,627.92", "bbox": (300.0, 105.0, 400.0, 114.0), "size": 9.0, "bold": False}]}]},
]


def test_normalizers_and_unit_detection():
    assert normalize_value("+1,234.50") == "1234.50"
    assert normalize_value("−12.3") == "-12.3"
    assert normalize_label(" 归属于 股东\n的净利润 ") == "归属于股东的净利润"
    assert detect_unit("单位：万元 币种：人民币") == "万元"


def test_pymupdf_builder_accepts_realistic_dict_and_bold_flag():
    raw = {"blocks": [
        {"type": 1, "lines": [{"spans": [{"text": "image", "bbox": (0, 0, 1, 1)}]}]},
        {"type": 0, "lines": [{"spans": [
            {"text": "营业收入", "bbox": (10, 10, 60, 20), "font": "SimHei-Bold", "size": 9},
            {"text": "100.00", "bbox": (100, 10, 150, 20), "font": "Song", "size": 9},
        ]}]},
    ]}
    blocks = blocks_from_pymupdf_dict(raw)
    assert len(blocks) == 1
    assert isinstance(blocks[0], Block)
    assert blocks[0].lines[0].spans[0].bold is True


def test_fixture_a_quarterly_label_value_pairing():
    got = reconstruct_cells(BLOCKS_A, "quarterly")
    assert [(c.column, c.value) for c in got] == [
        ("第一季度", "24051471185.69"),
        ("第二季度", "17618626634.30"),
        ("第三季度", "19108543634.77"),
        ("第四季度", "25462264522.66"),
    ]
    assert {c.row for c in got} == {"归属于上市公司股东的扣除非经常性损益后的净利润"}


def test_geometry_is_independent_of_input_block_reading_order():
    got = reconstruct_cells(list(reversed(BLOCKS_A)), "quarterly")
    assert len(got) == 4
    assert got[0].row == "归属于上市公司股东的扣除非经常性损益后的净利润"
    assert got[0].value == "24051471185.69"


def test_fixture_b_wrapped_label_is_joined():
    got = reconstruct_cells(BLOCKS_B, "segment")
    assert [(c.row, c.column, c.value) for c in got] == [
        ("液体乳及乳制品制造业", "营业收入", "113399011137.27"),
        ("液体乳及乳制品制造业", "营业成本", "74731457627.92"),
    ]


def test_note_cost_four_default_value_columns():
    blocks = [{"lines": [{"spans": [
        {"text": "主营业务", "bbox": (40, 100, 120, 109)},
        {"text": "170,611,838,052.02", "bbox": (180, 100, 260, 109)},
        {"text": "13,629,995,812.89", "bbox": (280, 100, 360, 109)},
        {"text": "147,218,996,281.04", "bbox": (380, 100, 460, 109)},
        {"text": "11,620,203,653.32", "bbox": (480, 100, 560, 109)},
    ]}]}]
    got = reconstruct_cells(blocks, "note_cost")
    assert [(c.column, c.value) for c in got] == [
        ("本期收入", "170611838052.02"),
        ("本期成本", "13629995812.89"),
        ("上期收入", "147218996281.04"),
        ("上期成本", "11620203653.32"),
    ]


def test_annual_headers_use_geometry_and_skip_yoy_column():
    blocks = [
        {"lines": [{"spans": [
            {"text": "2024年", "bbox": (200, 50, 260, 59)},
            {"text": "2023年", "bbox": (300, 50, 360, 59)},
            {"text": "同比", "bbox": (400, 50, 450, 59)},
            {"text": "2022年", "bbox": (500, 50, 560, 59)},
        ]}]},
        {"lines": [{"spans": [
            {"text": "总资产", "bbox": (40, 100, 100, 109)},
            {"text": "298,944,579,918.70", "bbox": (200, 100, 260, 109)},
            {"text": "272,699,660,092.25", "bbox": (300, 100, 360, 109)},
            {"text": "9.62", "bbox": (400, 100, 450, 109)},
            {"text": "254,500,826,096.02", "bbox": (500, 100, 560, 109)},
        ]}]},
    ]
    got = reconstruct_cells(blocks, "annual_data")
    assert [(c.column, c.value) for c in got] == [
        ("2024年", "298944579918.70"),
        ("2023年", "272699660092.25"),
        ("2022年", "254500826096.02"),
    ]


def test_segment_keeps_first_three_business_metrics_only():
    blocks = [{"lines": [{"spans": [
        {"text": "酒类", "bbox": (40, 100, 80, 109)},
        {"text": "170,611,838,052.02", "bbox": (150, 100, 220, 109)},
        {"text": "13,629,995,812.89", "bbox": (240, 100, 310, 109)},
        {"text": "92.01", "bbox": (330, 100, 370, 109)},
        {"text": "15.89", "bbox": (390, 100, 430, 109)},
        {"text": "17.30", "bbox": (450, 100, 490, 109)},
        {"text": "0.10", "bbox": (510, 100, 550, 109)},
    ]}]}]
    got = reconstruct_cells(blocks, "segment")
    assert [(c.column, c.value) for c in got] == [
        ("营业收入", "170611838052.02"),
        ("营业成本", "13629995812.89"),
        ("毛利率", "92.01"),
    ]


def test_concentration_uses_text_fallback_from_span_geometry():
    blocks = [{"lines": [{"spans": [
        {"text": "前五名客户销售额711,963.51万元，占年度销售总额6.17%；",
         "bbox": (40, 100, 500, 110)}
    ]}]}]
    got = reconstruct_cells(blocks, "concentration")
    assert _keys(got) == {
        ("前五名客户", "销售额(万元)", "711963.51"),
        ("前五名客户", "销售占比(%)", "6.17"),
    }


def test_text_fallback_does_not_invent_missing_region_suffix():
    text = """主营业务分地区情况
分地区
营业收入
营业成本
毛利率
其他
14,779,601,866.72
10,680,973,204.28
27.73
-2.34
-3.08
增加0.55 个百分点
"""
    got = extract_cells(text, "segment")
    rows = {c.row for c in got}
    assert "其他" in rows
    assert "其他地区" not in rows


def test_merge_pages_is_order_preserving_and_deduplicates():
    a = ExtractedCell("营业收入", "2024年", "100.00")
    b = ExtractedCell("总资产", "2024年", "200.00")
    assert merge_pages([[a], [a, b]]) == [a, b]
