"""Build a stratified quality report from an existing immutable generation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.evaluation.reporting import build_generation_quality_report
from findoc_rag.generation_evaluation import (
    DeterministicCaseScore,
    GenerationEvaluationDataset,
    GenerationRunItem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/evaluation/benchmark-v2.json")
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_jsonl(path: Path, model_type):
    return [
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    dataset = GenerationEvaluationDataset.model_validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    run_items = _load_jsonl(args.run_dir / "items.jsonl", GenerationRunItem)
    scores = _load_jsonl(
        args.run_dir / "deterministic-scores.jsonl", DeterministicCaseScore
    )
    report = build_generation_quality_report(dataset, run_items, scores)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
