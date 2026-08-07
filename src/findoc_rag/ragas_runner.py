"""Pure helpers for RAGAS generation-run validation (importable by tests)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from findoc_rag.generation_evaluation import (
    GenerationEvaluationDataset,
    GenerationLane,
    GenerationRunItem,
)


def load_run(path: Path) -> list[GenerationRunItem]:
    items = [
        GenerationRunItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        raise ValueError("run_jsonl contains no generation items")
    return items


def load_and_validate_run_manifest(
    run_jsonl: Path,
    dataset: GenerationEvaluationDataset,
    run_items: list[GenerationRunItem],
) -> tuple[dict[str, Any], GenerationLane]:
    """Load the sibling run summary and bind it to the dataset and JSONL."""

    summary_path = run_jsonl.parent / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"Generation run summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {"run_id", "dataset_id", "lane", "dataset_item_count", "item_count"}
    missing_fields = sorted(required - set(summary))
    if missing_fields:
        raise ValueError(f"Generation run summary is missing fields: {missing_fields}")
    if summary["run_id"] != run_jsonl.parent.name:
        raise ValueError("Generation run_id does not match its directory name")
    if summary["dataset_id"] != dataset.dataset_id:
        raise ValueError(
            "Generation run dataset_id does not match the selected dataset: "
            f"{summary['dataset_id']!r} != {dataset.dataset_id!r}"
        )
    if summary["dataset_item_count"] != dataset.item_count:
        raise ValueError("Generation run dataset_item_count does not match the dataset")
    if summary["item_count"] != len(run_items):
        raise ValueError("Generation run summary item_count does not match run_jsonl")
    lane = summary["lane"]
    if lane not in {"oracle_context", "retrieved_context", "robustness"}:
        raise ValueError(f"Unknown generation run lane: {lane!r}")
    return summary, cast(GenerationLane, lane)
