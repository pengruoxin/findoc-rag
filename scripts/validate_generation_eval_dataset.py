"""Validate the generation benchmark and export RAGAS oracle rows."""

import json
from pathlib import Path

from findoc_rag.benchmark_assets import benchmark_chunk_paths, validate_benchmark_lock
from findoc_rag.generation_evaluation import (
    GenerationEvaluationDataset,
    to_ragas_oracle_rows,
    validate_generation_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/evaluation/generation-eval-v1.json"
CHUNKS = benchmark_chunk_paths(ROOT)
REPORT = ROOT / "reports/generation/dataset-validation-v1.json"
RAGAS_EXPORT = ROOT / "data/evaluation/generation-eval-v1-ragas-oracle.jsonl"


def main() -> None:
    validate_benchmark_lock(ROOT)
    report = validate_generation_dataset(DATASET, CHUNKS)
    dataset = GenerationEvaluationDataset.model_validate_json(DATASET.read_text(encoding="utf-8"))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    rows = to_ragas_oracle_rows(dataset)
    RAGAS_EXPORT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Dataset valid: {report.item_count} items, {report.fact_count} facts")
    print(f"Splits: {report.split_counts}")
    print(f"RAGAS oracle rows: {len(rows)}")
    print(f"Warnings: {report.warning_count} pending PDF visual checks")


if __name__ == "__main__":
    main()
