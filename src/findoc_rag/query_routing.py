"""Shared, deterministic routing for production queries and evaluations.

The legacy routing benchmark measures whether company and year *signals* are
recognized.  Production metadata filtering needs a stricter distinction:
"2025 年经营目标" is usually disclosed in the 2024 annual report, while
"2025 年实际营业收入" belongs to a 2025 report.  This module keeps both
views explicit so evaluation code cannot silently substitute gold metadata.
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, Field

from findoc_rag.query_expansion import expand_query
from findoc_rag.query_rewriting import LLMQueryRewriter
from findoc_rag.time_utils import resolve_relative_time

YEAR_PATTERN = re.compile(r"20\d{2}")
REPORT_YEAR_PATTERN = re.compile(
    r"(?P<year>20\d{2})\s*年?\s*(?:年报|年度报告|报告期|报告年度)"
)

COMPANY_ALIASES = {
    "600519": "贵州茅台",
    "贵州茅台": "贵州茅台",
    "茅台": "贵州茅台",
    "kweichow moutai": "贵州茅台",
    "600887": "伊利股份",
    "伊利股份": "伊利股份",
    "伊利": "伊利股份",
}
COMPANY_ALIAS_KEYS = tuple(sorted(COMPANY_ALIASES, key=len, reverse=True))

# Forecast language can appear before or after a target year.  These cues are
# deliberately business-semantic rather than tied to either benchmark company.
FORECAST_CUES = (
    "经营计划",
    "发展计划",
    "未来规划",
    "业绩目标",
    "经营目标",
    "目标",
    "计划",
    "预计",
    "预期",
    "预测",
    "展望",
    "力争",
    "争取",
    "拟",
    "将实现",
    "能否实现",
    "是否实现",
    "保证",
    "承诺",
    "保证实现",
    "承诺实现",
)
ACTUAL_CUES = (
    "实际",
    "已实现",
    "实现了",
    "完成了",
    "年末",
    "期末",
    "全年",
    "同比",
    "营业收入",
    "营业成本",
    "净利润",
    "总资产",
    "现金流",
)


class FinanceQueryRoute(BaseModel):
    """Machine-readable filter signals derived only from the user query."""

    company_names: list[str] = Field(default_factory=list)
    mentioned_years: list[int] = Field(default_factory=list)
    report_years: list[int] = Field(default_factory=list)
    fact_periods: list[int] = Field(default_factory=list)
    forecast_target_years: list[int] = Field(default_factory=list)
    year_filter_policy: str = "none"


def _ordered_unique(values: list[int] | list[str]) -> list:
    return list(dict.fromkeys(values))


def _year_context(query: str, year: int, radius: int = 18) -> str:
    token = str(year)
    start = query.find(token)
    if start < 0:
        return query
    return query[max(0, start - radius) : start + len(token) + radius]


def route_finance_query(query: str) -> FinanceQueryRoute:
    """Classify query-derived metadata without consulting benchmark gold.

    Explicit annual-report wording always wins.  Otherwise forecast language
    keeps the target year in the lexical query but does not apply it as a
    ``report_year`` filter.  Factual/default years continue to use the current
    metadata filter behavior.
    """
    lowered = query.lower()
    companies = _ordered_unique(
        [COMPANY_ALIASES[alias] for alias in COMPANY_ALIAS_KEYS if alias in lowered]
    )
    years = _ordered_unique([int(value) for value in YEAR_PATTERN.findall(query)])
    explicit_report_years = _ordered_unique(
        [int(match.group("year")) for match in REPORT_YEAR_PATTERN.finditer(query)]
    )
    fact_periods: list[int] = []
    forecast_targets: list[int] = []

    for year in years:
        if year in explicit_report_years:
            continue
        context = _year_context(query, year)
        forecast = any(cue in context for cue in FORECAST_CUES)
        actual = any(cue in context for cue in ACTUAL_CUES)
        if forecast and not actual:
            forecast_targets.append(year)
        elif forecast and actual:
            # Explicitly completed/actual phrasing is factual.  Generic metric
            # names such as 营业收入 do not override clear plan/target wording.
            strong_actual = any(
                cue in context
                for cue in ("实际", "已实现", "实现了", "完成了", "年末", "期末")
            )
            if strong_actual:
                fact_periods.append(year)
            else:
                forecast_targets.append(year)
        else:
            fact_periods.append(year)

    report_years = _ordered_unique(explicit_report_years + fact_periods)
    if explicit_report_years and forecast_targets:
        policy = "explicit_report_and_forecast_target"
    elif explicit_report_years:
        policy = "explicit_report_year"
    elif forecast_targets and not report_years:
        policy = "forecast_target_no_report_filter"
    elif report_years:
        policy = "fact_period_as_report_year"
    else:
        policy = "none"
    return FinanceQueryRoute(
        company_names=companies,
        mentioned_years=years,
        report_years=report_years,
        fact_periods=_ordered_unique(fact_periods),
        forecast_target_years=_ordered_unique(forecast_targets),
        year_filter_policy=policy,
    )


def infer_finance_filters(query: str) -> tuple[list[str], list[int]]:
    """Return legacy company/year *recognition* signals.

    Kept for the existing query-routing-v1 18/18 regression.  Production and
    end-to-end evaluation must use :func:`route_finance_query` so target years
    are not confused with annual-report metadata.
    """
    route = route_finance_query(query)
    return route.company_names, route.mentioned_years


def prepare_finance_query(
    query: str,
    *,
    as_of_date: date | None,
    rewrite_mode: str = "deterministic",
    rewriter: LLMQueryRewriter | None = None,
) -> str:
    """Resolve relative time, then apply deterministic or LLM term rewriting."""
    resolved, _ = resolve_relative_time(query, as_of_date)
    if rewrite_mode == "llm" and rewriter is not None:
        return rewriter.rewrite(resolved)
    if rewrite_mode != "none":
        return expand_query(resolved)
    return resolved
