"""Deterministic financial synonym expansion for queries.

The synonym map is extracted from variant-regime failure cases (semantic
regime lexical misses), not hand-picked from intuition. Expansion happens
after relative-time resolution and before retrieval, so BM25 can match
professional terms that users paraphrase differently from the filings.
"""

from __future__ import annotations

FINANCIAL_SYNONYMS: tuple[tuple[str, str], ...] = (
    # Longest-first so longer aliases win over their substrings.
    ("净资产回报率", "净资产收益率"),
    ("同比增幅", "比上年同期增减"),
    ("一定实现", "计划实现"),
    ("前五大客户", "前五名客户"),
    ("毛利水平", "毛利率"),
    ("主要风险", "可能面对的风险"),
    ("销售占比", "销售额占年度销售总额"),
    ("营收", "营业收入"),
)


def expand_query(query: str) -> str:
    """Replace financial aliases with the canonical filing wording."""
    expanded = query
    for alias, canonical in FINANCIAL_SYNONYMS:
        expanded = expanded.replace(alias, canonical)
    return expanded
