"""Validate a generation benchmark and export RAGAS oracle rows."""

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--chunks", type=Path, nargs="+", default=CHUNKS)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--ragas-export", type=Path, default=RAGAS_EXPORT)
    parser.add_argument("--skip-benchmark-lock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_benchmark_lock:
        validate_benchmark_lock(ROOT)
    report = validate_generation_dataset(args.dataset, args.chunks)
    dataset = GenerationEvaluationDataset.model_validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    rows = to_ragas_oracle_rows(dataset)
    args.ragas_export.parent.mkdir(parents=True, exist_ok=True)
    args.ragas_export.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Dataset valid: {report.item_count} items, {report.fact_count} facts")
    print(f"Splits: {report.split_counts}")
    print(f"RAGAS oracle rows: {len(rows)}")
    print(f"Warnings: {report.warning_count} pending PDF visual checks")


if __name__ == "__main__":
    main()
