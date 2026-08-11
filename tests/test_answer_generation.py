from types import SimpleNamespace

from findoc_rag.answer_generation import GeneratedAnswer, GroundedAnswerGenerator


def test_quarterly_cashflow_is_structured_into_four_pairs() -> None:
    text = """2024 年分季度主要财务数据
第一季度 第二季度 第三季度 第四季度
经营活动产生的现金流量净额
2,267,600,204.82
3,057,943,456.58
8,544,068,622.02
7,870,128,109.96"""
    answer = GroundedAnswerGenerator._extract_quarterly_metric(
        "伊利股份2024年各季度经营活动现金流量净额分别是多少", text
    )
    assert answer is not None
    assert "第1季度：2,267,600,204.82" in answer
    assert "第4季度：7,870,128,109.96" in answer


def test_quarterly_cashflow_tolerates_pdf_line_break_inside_metric() -> None:
    text = """经营活动产生的现金
流量净额
9,187,422,415.09
27,434,411,397.54
7,799,552,404.82
48,042,305,950.98"""
    answer = GroundedAnswerGenerator._extract_quarterly_metric(
        "贵州茅台2024年各季度经营活动现金流量净额", text
    )
    assert answer is not None
    assert "第1季度：9,187,422,415.09" in answer


def test_evidence_guard_rejects_wrong_company() -> None:
    chunk = SimpleNamespace(company_name="伊利股份", report_year=2024, text="伊利股份 2024 年营业收入")
    hit = SimpleNamespace(chunk=chunk)
    assert not GroundedAnswerGenerator._evidence_supports_query("贵州茅台2024年营业收入是多少", [hit])


def test_evidence_guard_accepts_matching_company_year_metric() -> None:
    chunk = SimpleNamespace(company_name="贵州茅台", report_year=2024, text="贵州茅台 2024 年营业收入")
    hit = SimpleNamespace(chunk=chunk)
    assert GroundedAnswerGenerator._evidence_supports_query("贵州茅台2024年营业收入是多少", [hit])


def test_generic_cost_question_requests_scope_clarification() -> None:
    prompt = GroundedAnswerGenerator._clarification_prompt("伊利股份2024年的成本是多少？")
    assert prompt is not None
    assert "明确成本口径" in prompt


def test_specific_operating_cost_question_does_not_request_clarification() -> None:
    prompt = GroundedAnswerGenerator._clarification_prompt("伊利股份2024年营业成本是多少？")
    assert prompt is None


def test_generic_profit_question_requests_year_and_scope() -> None:
    prompt = GroundedAnswerGenerator._clarification_prompt("茅台利润是多少？")
    assert prompt is not None
    assert "报告年份" in prompt


QUARTERLY_YILI_TEXT = """九、2024 年分季度主要财务数据
第一季度 （1-3 月份） 第二季度 （4-6 月份） 第三季度 （7-9 月份） 第四季度 （10-12 月份）
营业收入 32,462,896,350.54 27,232,860,611.05 29,037,177,483.84 26,660,376,531.26
归属于上市公司股东的净利润 5,922,814,507.71 1,608,319,620.91 3,337,346,216.72 -2,415,620,352.16
归属于上市公司股东的扣除非经常性损益后的净利润 3,727,609,925.90 1,596,850,118.63 3,184,409,980.19 -2,497,595,078.80
经营活动产生的现金流量净额 2,267,600,204.82 3,057,943,456.58 8,544,068,622.02 7,870,128,109.96
"""

ANNUAL_YILI_TEXT = """七、近三年主要会计数据和财务指标
(一)主要会计数据
单位：元 币种：人民币
主要会计数据 2024年 2023年 本期比上年同期增减(%) 2022年
营业收入 115,393,310,976.69 125,758,168,916.51 -8.24 122,698,004,080.99
归属于上市公司股东的净利润 8,452,859,993.18 10,428,540,457.94 -18.94 9,431,064,679.78
经营活动产生的现金流量净额 21,739,740,393.38 18,290,357,650.56 18.86 13,420,320,580.36
2024年末 2023年末 本期末比上年同期末增减（%） 2022年末
总资产 153,718,003,927.97 151,620,252,657.98 1.38 130,965,302,299.22
"""

ANNUAL_MOUTAI_TEXT = """七、 近三年主要会计数据和财务指标
(一) 主要会计数据
单位：元  币种：人民币
主要会计数据 2024年 2023年 本期比上年同期增减(%) 2022年
营业收入 170,899,152,276.34 147,693,604,994.14 15.71 124,099,843,771.99
归属于上市公司股东的净利润 86,228,146,421.62 74,734,071,550.75 15.38 62,717,467,870.12
"""

NOTE_COST_MOUTAI_TEXT = """40、 营业收入和营业成本
(1). 营业收入和营业成本情况
单位：元  币种：人民币
项目 本期发生额 上期发生额
收入 成本 收入 成本
主营业务 170,611,838,052.02 13,629,995,812.89 147,218,996,281.04 11,620,203,653.32
其他业务 287,314,224.32 159,486,555.09 474,608,713.10 247,070,198.46
合计 170,899,152,276.34 13,789,482,367.98 147,693,604,994.14 11,867,273,851.78
"""

NOTE_COST_PARENT_TEXT = """(1)营业收入和营业成本情况
单位：元 币种：人民币
项目 本期发生额 上期发生额
收入 成本 收入 成本
主营业务 101,014,629,405.71 73,439,646,767.04 111,024,883,534.90 82,114,172,067.76
其他业务 1,380,765,256.70 1,106,955,217.73 1,221,775,187.60 970,621,908.99
合计 102,395,394,662.41 74,546,601,984.77 112,246,658,722.50 83,084,793,976.75
"""

SEGMENT_YILI_TEXT = """(1)主营业务分行业、分产品、分地区、分销售模式情况
主营业务分产品情况
分产品 营业收入 营业成本 毛利率 营业收入比上年增减（%） 营业成本比上年增减（%） 毛利率比上年增减（%）
液体乳 75,002,672,990.21 51,770,470,355.31 30.98 -12.32 -13.09 增加0.62 个百分点
奶粉及奶制品 29,675,312,533.80 17,503,230,124.97 41.02 7.53 2.53 增加2.88 个百分点
冷饮产品 8,721,025,613.26 5,457,757,147.64 37.42 -18.41 -17.04 减少1.03 个百分点
其他产品 721,621,008.33 551,655,809.06 23.55 13.75 6.44 增加5.25 个百分点
主营业务分销售模式情况
销售模式 营业收入 营业成本 毛利率 营业收入比上年增减（%） 营业成本比上年增减（%） 毛利率比上年增减（%）
批发代理 95,768,511,021.23 10,136,042,973.30 89.42 19.73 18.28 增加0.13 个百分点
直销 74,843,327,030.79 3,493,952,839.59 95.33 11.32 14.52 减少0.13 个百分点
"""


def _hit(text: str, company: str = "伊利股份", year: int = 2024):
    chunk = SimpleNamespace(
        text=text,
        company_name=company,
        report_year=year,
        chunk_id="test-chunk",
        page_start=1,
        page_end=1,
        section_path=["测试章节"],
    )
    return SimpleNamespace(chunk=chunk)


def test_deterministic_quarterly_answer_with_citations() -> None:
    answer = GroundedAnswerGenerator._deterministic_table_answer(
        "伊利股份2024年各季度归母净利润分别是多少",
        [_hit(QUARTERLY_YILI_TEXT)],
    )
    assert answer is not None
    assert "第一至第四季度分别为5,922,814,507.71元" in answer
    assert "、-2,415,620,352.16元[1]" in answer


def test_deterministic_quarterly_reconcile_with_annual() -> None:
    answer = GroundedAnswerGenerator._deterministic_table_answer(
        "伊利股份2024年各季度归母净利润分别是多少，加总后是否等于全年披露值",
        [_hit(QUARTERLY_YILI_TEXT), _hit(ANNUAL_YILI_TEXT)],
    )
    assert answer is not None
    assert "合计8,452,859,993.18元" in answer
    assert "与全年披露值8,452,859,993.18元一致[2]" in answer


def test_deterministic_note_cost_scope() -> None:
    answer = GroundedAnswerGenerator._deterministic_table_answer(
        "伊利股份财务报表附注中2024年主营业务、其他业务和营业成本合计分别是多少",
        [_hit(NOTE_COST_MOUTAI_TEXT, company="贵州茅台")],
    )
    assert answer is not None
    assert "主营业务成本13,629,995,812.89元" in answer
    assert "其他业务成本159,486,555.09元" in answer
    assert "营业成本合计13,789,482,367.98元[1]" in answer


def test_deterministic_cost_reconciliation() -> None:
    answer = GroundedAnswerGenerator._deterministic_table_answer(
        "为什么贵州茅台2024年主营业务成本与营业成本合计不同，差额是多少",
        [_hit(NOTE_COST_MOUTAI_TEXT, company="贵州茅台")],
    )
    assert answer is not None
    assert "差额159,486,555.09元" in answer


def test_deterministic_segment_product_margin_highest() -> None:
    answer = GroundedAnswerGenerator._deterministic_table_answer(
        "伊利股份2024年各产品毛利率分别是多少，哪类最高",
        [_hit(SEGMENT_YILI_TEXT)],
    )
    assert answer is not None
    assert "液体乳30.98%" in answer
    assert "奶粉及奶制品41.02%" in answer
    assert "其中奶粉及奶制品最高" in answer


def test_deterministic_segment_channel_margin_delta() -> None:
    answer = GroundedAnswerGenerator._deterministic_table_answer(
        "贵州茅台2024年直销与批发代理哪种模式毛利率更高，高多少个百分点",
        [_hit(SEGMENT_YILI_TEXT, company="贵州茅台")],
    )
    assert answer is not None
    assert "直销毛利率95.33%" in answer
    assert "直销高5.91个百分点" in answer


def test_deterministic_annual_revenue_with_yoy() -> None:
    answer = GroundedAnswerGenerator._deterministic_table_answer(
        "贵州茅台2024年营业收入及同比增幅是多少",
        [_hit(ANNUAL_MOUTAI_TEXT, company="贵州茅台")],
    )
    assert answer is not None
    assert "营业收入为170,899,152,276.34元" in answer
    assert "同比增长15.71%[1]" in answer


def test_deterministic_cross_company_revenue_delta() -> None:
    answer = GroundedAnswerGenerator._deterministic_table_answer(
        "比较贵州茅台与伊利股份2024年营业收入，哪家更高，差额是多少",
        [
            _hit(ANNUAL_MOUTAI_TEXT, company="贵州茅台"),
            _hit(ANNUAL_YILI_TEXT, company="伊利股份"),
        ],
    )
    assert answer is not None
    assert "贵州茅台为170,899,152,276.34元[1]" in answer
    assert "伊利股份为115,393,310,976.69元[2]" in answer
    assert "贵州茅台高55,505,841,299.65元" in answer


def test_deterministic_consolidated_parent_revenue_delta() -> None:
    answer = GroundedAnswerGenerator._deterministic_table_answer(
        "伊利股份2024年合并口径与母公司口径营业收入分别是多少，差额多少",
        [
            _hit(NOTE_COST_MOUTAI_TEXT, company="伊利股份"),
            _hit(NOTE_COST_PARENT_TEXT, company="伊利股份"),
        ],
    )
    assert answer is not None
    assert "合并口径营业收入为170,899,152,276.34元" in answer
    assert "母公司口径为102,395,394,662.41元" in answer
    assert "合并口径高68,503,757,613.93元" in answer


def test_abstention_detection_marks_refusal() -> None:
    assert GroundedAnswerGenerator._is_abstention(
        "根据现有证据，无法确认伊利股份2024年前五名客户销售占比。因此，我无法给出准确回答。"
    )
    assert GroundedAnswerGenerator._is_abstention(
        "证据不足，无法回答贵州茅台2024年四个季度经营活动现金流量净额。"
    )


def test_abstention_detection_keeps_normal_answer() -> None:
    assert not GroundedAnswerGenerator._is_abstention(
        "贵州茅台2024年营业收入为170,899,152,276.34元，同比增幅为15.71%[1]。"
    )


def test_abstention_detection_ignores_uncertainty_noun() -> None:
    assert not GroundedAnswerGenerator._is_abstention(
        "年报披露了经营中存在的不确定性因素，收入确认遵循相关准则[1]。"
    )


def test_remote_mode_skips_deterministic_tables_by_default(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FINDOC_RAG_REMOTE_DETERMINISTIC_TABLES", raising=False)
    monkeypatch.delenv("FINDOC_RAG_DISABLE_DETERMINISTIC_TABLES", raising=False)
    generator = GroundedAnswerGenerator(enabled=True, model="deepseek-chat")
    generator.api_key = "test-key"
    sentinel = GeneratedAnswer(
        answer="remote answer",
        citations=[],
        provider="openai-compatible",
        grounded=True,
    )
    monkeypatch.setattr(
        generator, "_generate_remote", lambda *args, **kwargs: sentinel
    )
    answer = generator.generate(
        "伊利股份2024年各季度归母净利润分别是多少",
        [_hit(QUARTERLY_YILI_TEXT)],
    )
    assert answer.provider == "openai-compatible"


def test_remote_mode_deterministic_tables_priority(monkeypatch) -> None:
    monkeypatch.setenv("FINDOC_RAG_REMOTE_DETERMINISTIC_TABLES", "1")
    monkeypatch.delenv("FINDOC_RAG_DISABLE_DETERMINISTIC_TABLES", raising=False)
    generator = GroundedAnswerGenerator(enabled=True, model="deepseek-chat")
    generator.api_key = "test-key"
    answer = generator.generate(
        "伊利股份2024年各季度归母净利润分别是多少",
        [_hit(QUARTERLY_YILI_TEXT)],
    )
    assert answer.provider == "deterministic-table"
    assert "第一至第四季度分别为5,922,814,507.71元" in answer.answer
