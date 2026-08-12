"""Routing-signal regressions: the dataset and the inference must agree."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from findoc_rag.api import infer_finance_filters, prepare_finance_query
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
