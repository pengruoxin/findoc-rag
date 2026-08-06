"""Recompute deterministic scores without rerunning retrieval or paid generation."""

import argparse
import json
from pathlib import Path

from findoc_rag.generation_evaluation import (
    GenerationEvaluationDataset,
    GenerationRunItem,
    score_generation_run_item,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/generation-eval-v1.json"))
    parser.add_argument("--revision", default="v2")
    args = parser.parse_args()
    dataset = GenerationEvaluationDataset.model_validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    run_items = {
        item.query_id: item
        for item in (
            GenerationRunItem.model_validate_json(line)
            for line in (args.run_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    scores = [score_generation_run_item(item, run_items[item.query_id]) for item in dataset.items]
    score_path = args.run_dir / f"deterministic-scores-{args.revision}.jsonl"
    score_path.write_text(
        "".join(score.model_dump_json() + "\n" for score in scores), encoding="utf-8"
    )
    strict = [score for score in scores if score.strict_success_eligible]
    summary = {
        "dataset_id": dataset.dataset_id,
        "scorer_revision": args.revision,
        "item_count": len(scores),
        "strict_success_eligible_count": len(strict),
        "strict_success_rate": sum(score.strict_success for score in strict) / len(strict),
        "semantic_review_required_count": sum(score.semantic_review_required for score in scores),
        "expected_behavior_accuracy": sum(score.expected_behavior_correct for score in scores)
        / len(scores),
    }
    (args.run_dir / f"summary-{args.revision}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
