from types import SimpleNamespace

from findoc_rag.answer_generation import GroundedAnswerGenerator


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
