"""Tests for deterministic financial synonym expansion."""

from __future__ import annotations

from findoc_rag.query_expansion import expand_query


def test_revenue_shorthand() -> None:
    assert "营业收入" in expand_query("贵州茅台2024年营收是多少？")
    assert "营收" not in expand_query("贵州茅台2024年营收是多少？")


def test_margin_paraphrase() -> None:
    assert expand_query("茅台酒毛利水平是多少？") == "茅台酒毛利率是多少？"


def test_roe_paraphrase() -> None:
    assert expand_query("净资产回报率是多少？") == "净资产收益率是多少？"


def test_risk_paraphrase() -> None:
    assert expand_query("年报提到的主要风险有哪些？") == "年报提到的可能面对的风险有哪些？"


def test_customer_paraphrase() -> None:
    expanded = expand_query("前五大客户销售占比是多少？")
    assert "前五名客户" in expanded
    assert "销售额占年度销售总额" in expanded


def test_plan_paraphrase_with_interleaved_year() -> None:
    assert expand_query("有没有承诺2025年一定实现1190亿元？") == "有没有承诺2025年计划实现1190亿元？"


def test_yoy_paraphrase() -> None:
    assert expand_query("营业收入及同比增幅是多少？") == "营业收入及比上年同期增减是多少？"


def test_unchanged_when_no_alias() -> None:
    query = "贵州茅台2024年经营活动产生的现金流量净额是多少？"
    assert expand_query(query) == query
