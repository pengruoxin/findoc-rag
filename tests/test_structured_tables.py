from findoc_rag.structured_tables import _restore_segment_sections, infer_table_types
from findoc_rag.table_reconstruction import ExtractedCell


def test_infer_table_types_covers_supported_families_without_gold() -> None:
    assert infer_table_types(
        "第一季度 第二季度 第三季度 第四季度 主营业务分产品情况 "
        "主要会计数据 2024年 2023年 2022年 前五名客户销售额 "
        "营业收入和营业成本 本期发生额 上期发生额 收入 成本"
    ) == ["quarterly", "note_cost", "segment", "annual_data", "concentration"]


def test_note_cost_inference_rejects_adjacent_investment_income_notes() -> None:
    text = (
        "57、公允价值变动收益 本期发生额 上期发生额 合计 664,223 46,270 "
        "58、投资收益 权益法核算的长期股权投资收益 3,743,040 股利收入 "
        "以摊余成本计量的金融资产终止确认收益"
    )

    assert "note_cost" not in infer_table_types(text)


def test_segment_section_is_restored_from_same_chunk_text() -> None:
    text = """主营业务分产品情况
分产品 营业收入 营业成本 毛利率 营业收入比上年增减（%） 营业成本比上年增减（%） 毛利率比上年增减（%）
奶粉及奶制品 100.00 60.00 40.00 1.00 2.00 增加3.00个百分点
"""
    cells = [
        ExtractedCell("奶粉及奶制品", "营业收入", "100.00"),
        ExtractedCell("奶粉及奶制品", "营业成本", "60.00"),
        ExtractedCell("奶粉及奶制品", "毛利率", "40.00"),
    ]

    restored = _restore_segment_sections(cells, text)

    assert {cell.section for cell in restored} == {"主营业务分产品情况"}


def test_existing_coordinate_segment_section_is_not_overwritten() -> None:
    cell = ExtractedCell("华北", "毛利率", "33.00", section="主营业务分地区情况")

    [restored] = _restore_segment_sections([cell], "主营业务分产品情况")

    assert restored.section == "主营业务分地区情况"
