"""Stratified generation metrics with deterministic clustered bootstrap CIs."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from findoc_rag.generation_evaluation import (
    DeterministicCaseScore,
    GenerationEvaluationDataset,
    GenerationEvaluationItem,
    GenerationRunItem,
)


class BootstrapInterval(BaseModel):
    low: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)
    confidence_level: float = Field(gt=0, lt=1)
    samples: int = Field(ge=1)
    cluster_count: int = Field(ge=1)


class WilsonInterval(BaseModel):
    low: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)
    confidence_level: float = Field(gt=0, lt=1)
    eligible_count: int = Field(ge=1)
    success_count: int = Field(ge=0)


def wilson_interval(
    success_count: int,
    eligible_count: int,
    *,
    confidence_level: float = 0.95,
) -> WilsonInterval | None:
    """Wilson score interval for a binary proportion.

    The default 95% interval uses z=1.959963984540054. Other confidence
    levels are rejected until an inverse-normal implementation is needed.
    """
    if eligible_count <= 0:
        return None
    if not 0 <= success_count <= eligible_count:
        raise ValueError("success_count must be between zero and eligible_count")
    if confidence_level != 0.95:
        raise ValueError("only a 0.95 confidence level is currently supported")
    z = 1.959963984540054
    proportion = success_count / eligible_count
    denominator = 1 + z**2 / eligible_count
    center = (proportion + z**2 / (2 * eligible_count)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / eligible_count
            + z**2 / (4 * eligible_count**2)
        )
        / denominator
    )
    return WilsonInterval(
        low=max(0.0, center - margin),
        high=min(1.0, center + margin),
        confidence_level=confidence_level,
        eligible_count=eligible_count,
        success_count=success_count,
    )


def clustered_bootstrap_interval(
    observations: Sequence[tuple[str, float]],
    *,
    samples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval | None:
    """Return a percentile CI by resampling whole clusters with replacement."""
    if not observations:
        return None
    clusters: dict[str, list[float]] = defaultdict(list)
    for cluster_id, value in observations:
        clusters[cluster_id].append(float(value))
    cluster_ids = sorted(clusters)
    if len(cluster_ids) < 2:
        return None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(cluster_ids) for _ in cluster_ids]
        values = [value for cluster_id in selected for value in clusters[cluster_id]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    tail = (1 - confidence_level) / 2
    low_index = max(0, int(tail * samples))
    high_index = min(samples - 1, int((1 - tail) * samples) - 1)
    return BootstrapInterval(
        low=estimates[low_index],
        high=estimates[high_index],
        confidence_level=confidence_level,
        samples=samples,
        cluster_count=len(cluster_ids),
    )


def _document_cluster(item: GenerationEvaluationItem) -> str:
    documents = sorted({evidence.document_version_id for evidence in item.gold_evidence})
    if documents:
        return "documents:" + "+".join(documents)
    companies = "+".join(sorted(item.company_ids)) or "unspecified"
    years = "+".join(str(year) for year in sorted(item.report_years)) or "unspecified"
    return f"no-gold:{companies}:{years}"


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def _metric(
    rows: list[tuple[GenerationEvaluationItem, GenerationRunItem, DeterministicCaseScore]],
    selector: Callable[
        [GenerationEvaluationItem, GenerationRunItem, DeterministicCaseScore],
        bool | None,
    ],
    *,
    label: str,
) -> dict[str, Any]:
    selected = [
        (item, bool(value))
        for item, run, score in rows
        if (value := selector(item, run, score)) is not None
    ]
    if not selected:
        return {"eligible_count": 0, "success_count": 0, "rate": None}
    values = [float(value) for _, value in selected]
    success_count = int(sum(values))
    question_observations = [
        (item.query_id, float(value)) for item, value in selected
    ]
    document_observations = [
        (_document_cluster(item), float(value)) for item, value in selected
    ]
    question_interval = clustered_bootstrap_interval(
        question_observations, seed=_stable_seed(label + ":question")
    )
    document_interval = clustered_bootstrap_interval(
        document_observations, seed=_stable_seed(label + ":document")
    )
    return {
        "eligible_count": len(selected),
        "success_count": success_count,
        "rate": sum(values) / len(values),
        "wilson_ci": wilson_interval(success_count, len(selected)).model_dump(),
        "question_bootstrap_ci": (
            question_interval.model_dump() if question_interval else None
        ),
        "document_bootstrap_ci": (
            document_interval.model_dump() if document_interval else None
        ),
    }


def _summarize_rows(
    rows: list[tuple[GenerationEvaluationItem, GenerationRunItem, DeterministicCaseScore]],
    *,
    label: str,
) -> dict[str, Any]:
    strict_rows = [row for row in rows if row[2].strict_success_eligible]
    return {
        "item_count": len(rows),
        "provider_counts": dict(sorted(Counter(run.provider for _, run, _ in rows).items())),
        "strict_success": _metric(
            strict_rows,
            lambda _item, _run, score: score.strict_success,
            label=label + ":strict",
        ),
        "expected_behavior_accuracy": _metric(
            rows,
            lambda _item, _run, score: score.expected_behavior_correct,
            label=label + ":behavior",
        ),
        "run_error_rate": _metric(
            rows,
            lambda _item, run, _score: run.error is not None,
            label=label + ":error",
        ),
    }


def build_generation_quality_report(
    dataset: GenerationEvaluationDataset,
    run_items: list[GenerationRunItem],
    scores: list[DeterministicCaseScore],
) -> dict[str, Any]:
    """Build auditable overall and stratified metrics for canonical runs."""
    items = {item.query_id: item for item in dataset.items}
    runs = {item.query_id: item for item in run_items}
    score_by_id = {score.query_id: score for score in scores}
    query_ids = sorted(set(items) & set(runs) & set(score_by_id))
    rows = [(items[query_id], runs[query_id], score_by_id[query_id]) for query_id in query_ids]

    def strata(
        name: str,
        key: Callable[[GenerationEvaluationItem, GenerationRunItem], str],
    ) -> dict[str, Any]:
        grouped: dict[str, list[tuple[GenerationEvaluationItem, GenerationRunItem, DeterministicCaseScore]]] = defaultdict(list)
        for row in rows:
            grouped[key(row[0], row[1])].append(row)
        return {
            value: _summarize_rows(group, label=f"{name}:{value}")
            for value, group in sorted(grouped.items())
        }

    return {
        "report_version": "1.0",
        "dataset_id": dataset.dataset_id,
        "scored_item_count": len(rows),
        "unscored_run_item_count": len(run_items) - len(rows),
        "interval_notes": {
            "wilson_ci": "Binary-proportion uncertainty; use this for small or perfect samples.",
            "question_bootstrap_ci": "Percentile bootstrap resampled by query_id.",
            "document_bootstrap_ci": (
                "Percentile bootstrap resampled by the set of gold document IDs; "
                "no-gold behavior cases fall back to company/report-year scope. "
                "A perfect observed sample can still produce a degenerate [1, 1] interval."
            ),
        },
        "overall": _summarize_rows(rows, label="overall"),
        "strata": {
            "provider": strata("provider", lambda _, run: run.provider),
            "filter_source": strata(
                "filter_source", lambda _, run: run.filter_source or "not_recorded"
            ),
            "split": strata("split", lambda item, _: item.split),
            "category": strata("category", lambda item, _: item.category),
            "difficulty": strata("difficulty", lambda item, _: item.difficulty),
            "expected_behavior": strata(
                "expected_behavior", lambda item, _: item.answer_contract.expected_behavior
            ),
        },
    }
