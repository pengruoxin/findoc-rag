"""Tests for deterministic relative-time resolution."""

from __future__ import annotations

from datetime import date

import pytest

from findoc_rag.time_utils import parse_as_of_date, resolve_relative_time


def test_resolves_qu_zhu_nian_to_previous_year() -> None:
    query = "贵州茅台去年营业收入是多少？"
    resolved, cues = resolve_relative_time(query, date(2025, 4, 30))
    assert resolved == "贵州茅台2024年营业收入是多少？"
    assert cues == [r"去年|上一年|上年"]


def test_resolves_this_year_to_anchor_year() -> None:
    resolved, cues = resolve_relative_time(
        "伊利股份今年净利润是多少？", date(2025, 4, 30)
    )
    assert resolved == "伊利股份2025年净利润是多少？"
    assert cues == [r"今年|本年|本年度"]


def test_resolves_qian_nian() -> None:
    resolved, _ = resolve_relative_time("前年营收是多少？", date(2026, 8, 7))
    assert resolved == "2024年营收是多少？"


def test_unchanged_query_without_relative_time() -> None:
    query = "贵州茅台2024年营业收入是多少？"
    resolved, cues = resolve_relative_time(query, date(2026, 8, 7))
    assert resolved == query
    assert cues == []


def test_relative_time_without_anchor_raises() -> None:
    with pytest.raises(ValueError, match="as_of_date anchor"):
        resolve_relative_time("贵州茅台去年营业收入是多少？", None)


def test_document_relative_last_year_uses_report_year_not_query_clock() -> None:
    resolved, cues = resolve_relative_time(
        "2022年年报里的去年营业收入是多少？", date(2026, 8, 20)
    )

    assert resolved == "2022年年报里的2021年营业收入是多少？"
    assert cues == [r"去年|上一年|上年"]


def test_document_relative_time_can_resolve_without_runtime_anchor() -> None:
    resolved, cues = resolve_relative_time("根据2022年度报告，本年净利润是多少？", None)

    assert resolved == "根据2022年度报告，2022年净利润是多少？"
    assert cues == [r"今年|本年|本年度"]


def test_relative_cue_before_document_reference_uses_query_clock() -> None:
    resolved, _ = resolve_relative_time(
        "去年发布的2022年年报披露了什么？", date(2026, 8, 20)
    )

    assert resolved == "2025年发布的2022年年报披露了什么？"


def test_previous_clause_does_not_leak_document_time_anchor() -> None:
    resolved, _ = resolve_relative_time(
        "先看2022年年报；去年行业收入是多少？", date(2026, 8, 20)
    )

    assert resolved == "先看2022年年报；2025年行业收入是多少？"


def test_parse_as_of_date() -> None:
    assert parse_as_of_date("2025-04-30") == date(2025, 4, 30)
    assert parse_as_of_date("") is None
    assert parse_as_of_date(None) is None
