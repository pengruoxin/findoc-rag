"""Tests for the dataset-level integrity gate."""

from __future__ import annotations

from findoc_rag.benchmark_integrity import (
    normalize_text,
    validate_benchmark,
)


def _chunk(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "text": text}


def _variant(variant_id: str, query: str, **extra) -> dict:
    payload = {
        "variant_id": variant_id,
        "query": query,
        "variant_types": ["paraphrase"],
        "query_regime": "semantic_or_relative_time",
    }
    payload.update(extra)
    return payload


def _item(query_id: str, **extra) -> dict:
    payload = {
        "query_id": query_id,
        "answerability": "answerable",
        "company_names": ["贵州茅台"],
        "report_years": [2024],
        "gold_chunk_ids": ["c1"],
        "gold_evidence": [
            {
                "evidence_id": f"{query_id}:e1",
                "chunk_id": "c1",
                "verbatim_quote": "营业收入\n170,899,152,276.34",
            }
        ],
        "query_variants": [],
    }
    payload.update(extra)
    return payload


def _benchmark(items: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "benchmark-v2",
        "corpus_index_id": "idx-1",
        "chunk_schema_version": 3,
        "item_count": len(items),
        "items": items,
    }


def test_valid_benchmark_passes() -> None:
    items = [
        _item(
            "q1",
            query_variants=[
                _variant("q1:v1", "600519 2024年营收是多少？", as_of_date=None),
                _variant(
                    "q1:v2",
                    "贵州茅台去年营收是多少？",
                    variant_types=["relative_time", "paraphrase"],
                    as_of_date="2025-04-30",
                ),
            ],
        )
    ]
    chunks = {"c1": _chunk("c1", "营业收入\n170,899,152,276.34")}
    result = validate_benchmark(
        _benchmark(items),
        corpus_index_id="idx-1",
        chunk_schema_version=3,
        chunks=chunks,
    )
    assert result.ok, result.errors


def test_missing_gold_chunk_fails() -> None:
    items = [_item("q1")]
    result = validate_benchmark(
        _benchmark(items),
        corpus_index_id="idx-1",
        chunk_schema_version=3,
        chunks={},
    )
    assert not result.ok
    assert any("gold chunk missing" in error for error in result.errors)


def test_quote_mismatch_fails() -> None:
    items = [_item("q1")]
    chunks = {"c1": _chunk("c1", "完全不同的文本")}
    result = validate_benchmark(
        _benchmark(items),
        corpus_index_id="idx-1",
        chunk_schema_version=3,
        chunks=chunks,
    )
    assert not result.ok
    assert any("quote not found" in error for error in result.errors)


def test_relative_variant_without_as_of_date_fails() -> None:
    items = [
        _item(
            "q1",
            query_variants=[
                _variant(
                    "q1:v1",
                    "贵州茅台去年营收是多少？",
                    variant_types=["relative_time", "paraphrase"],
                )
            ],
        )
    ]
    result = validate_benchmark(
        _benchmark(items),
        corpus_index_id="idx-1",
        chunk_schema_version=3,
        chunks={"c1": _chunk("c1", "营业收入 170,899,152,276.34")},
    )
    assert not result.ok
    assert any("requires as_of_date" in error for error in result.errors)


def test_year_outside_report_years_fails() -> None:
    items = [
        _item(
            "q1",
            query_variants=[
                _variant("q1:v1", "贵州茅台2026年营收是多少？")
            ],
        )
    ]
    result = validate_benchmark(
        _benchmark(items),
        corpus_index_id="idx-1",
        chunk_schema_version=3,
        chunks={"c1": _chunk("c1", "营业收入 170,899,152,276.34")},
    )
    assert not result.ok
    assert any("outside report_years" in error for error in result.errors)


def test_wrong_ticker_fails() -> None:
    items = [
        _item(
            "q1",
            query_variants=[_variant("q1:v1", "600887 2024年营收是多少？")],
        )
    ]
    result = validate_benchmark(
        _benchmark(items),
        corpus_index_id="idx-1",
        chunk_schema_version=3,
        chunks={"c1": _chunk("c1", "营业收入 170,899,152,276.34")},
    )
    assert not result.ok
    assert any("ticker 600887" in error for error in result.errors)


def test_chunk_schema_version_mismatch_fails() -> None:
    items = [_item("q1")]
    result = validate_benchmark(
        _benchmark(items),
        corpus_index_id="idx-1",
        chunk_schema_version=4,
        chunks={"c1": _chunk("c1", "营业收入 170,899,152,276.34")},
    )
    assert not result.ok
    assert any("chunk_schema_version" in error for error in result.errors)


def test_normalize_text_removes_whitespace() -> None:
    assert normalize_text("营业收入\n170,899,152,276.34") == "营业收入170,899,152,276.34"
