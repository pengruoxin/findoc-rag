"""Paired retrieval comparison tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts/compare_retrieval_runs.py"
    spec = importlib.util.spec_from_file_location("compare_retrieval_runs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = _load_module()


def _row(query_id: str, hit: int, mrr: float, regime: str = "canonical") -> dict:
    metrics = {
        "hit_at_5": hit,
        "mrr_at_5": mrr,
        "recall_at_5": float(hit),
        "ndcg_at_5": mrr,
        "candidate_recall": bool(hit),
    }
    return {
        "query_id": query_id,
        "canonical_id": query_id,
        "query": query_id,
        "regime": regime,
        "retrieval_judgment": "positive_gold",
        "results": {"query_parser": {"lexical": metrics}},
    }


def test_compare_rows_reports_fixed_and_regressed() -> None:
    old = {
        "fixed": _row("fixed", 0, 0.0),
        "regressed": _row("regressed", 1, 1.0, "ticker_or_finance_shorthand"),
        "same": _row("same", 1, 0.5, "semantic_or_relative_time"),
    }
    new = {
        "fixed": _row("fixed", 1, 0.5),
        "regressed": _row("regressed", 0, 0.0, "ticker_or_finance_shorthand"),
        "same": _row("same", 1, 0.5, "semantic_or_relative_time"),
    }

    report = compare.compare_rows(old, new, filter_name="query_parser", mode="lexical")

    assert report["fixed"] == ["fixed"]
    assert report["regressed"] == ["regressed"]
    assert report["mrr_improved"] == ["fixed"]
    assert report["mrr_regressed"] == ["regressed"]


def test_compare_rows_requires_identical_query_ids() -> None:
    with pytest.raises(ValueError, match="different query IDs"):
        compare.compare_rows(
            {"old": _row("old", 1, 1.0)},
            {"new": _row("new", 1, 1.0)},
            filter_name="query_parser",
            mode="lexical",
        )
