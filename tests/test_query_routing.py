"""Routing-signal regressions: the dataset and the inference must agree."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from findoc_rag.query_routing import (
    infer_finance_filters,
    prepare_finance_query,
    route_finance_query,
)
from findoc_rag.time_utils import resolve_relative_time

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/evaluation/query-routing-v1.json"


def test_all_routing_items_match_exactly() -> None:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    as_of = date.fromisoformat(dataset["default_as_of_date"])
    failures: list[str] = []
    for item in dataset["items"]:
        resolved_base, _ = resolve_relative_time(item["query"], as_of)
        companies, years = infer_finance_filters(resolved_base)
        prepare_finance_query(
            resolved_base, as_of_date=as_of, rewrite_mode="deterministic"
        )
        exact = (
            set(companies) == set(item["expected_companies"])
            and set(years) == set(item["expected_years"])
        )
        if not exact:
            failures.append(
                f"{item['query_id']}: got {companies}/{years}, "
                f"expected {item['expected_companies']}/{item['expected_years']}"
            )
    assert not failures, "\n".join(failures)


def test_forecast_target_year_is_not_used_as_report_year_filter() -> None:
    route = route_finance_query("伊利股份是否保证2025年实现1190亿元营业收入目标")

    assert route.company_names == ["伊利股份"]
    assert route.mentioned_years == [2025]
    assert route.report_years == []
    assert route.forecast_target_years == [2025]
    assert route.year_filter_policy == "forecast_target_no_report_filter"


def test_forecast_guarantee_with_year_between_cue_and_metric_is_not_report_filter() -> None:
    route = route_finance_query("伊利股份是否保证2025年一定实现1,190亿元营业总收入？")

    assert route.company_names == ["伊利股份"]
    assert route.report_years == []
    assert route.forecast_target_years == [2025]
    assert route.year_filter_policy == "forecast_target_no_report_filter"


def test_actual_period_remains_a_report_year_filter() -> None:
    route = route_finance_query("伊利股份2025年实际营业收入是多少")

    assert route.report_years == [2025]
    assert route.fact_periods == [2025]
    assert route.forecast_target_years == []


def test_explicit_report_year_and_forecast_target_are_kept_separate() -> None:
    route = route_finance_query("伊利股份2024年年报披露的2025年经营目标是多少")

    assert route.mentioned_years == [2024, 2025]
    assert route.report_years == [2024]
    assert route.forecast_target_years == [2025]
    assert route.year_filter_policy == "explicit_report_and_forecast_target"
