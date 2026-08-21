"""Materialize sealed hard-v3 evaluation splits without changing case content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.agent_evaluation import AgentHardDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/agent-hard-v3.json"),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/agent-hard-v3-questions.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = AgentHardDataset.model_validate_json(args.dataset.read_bytes())
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    split_by_case = {
        question["case_id"]: question["split"] for question in questions["questions"]
    }
    if set(split_by_case) != {case.case_id for case in dataset.cases}:
        raise SystemExit("question and gold case IDs do not match")

    output_names = {
        "calibration": "agent-hard-v3-calibration.json",
        "dev": "agent-hard-v3-dev.json",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, output_name in output_names.items():
        cases = [
            case for case in dataset.cases if split_by_case[case.case_id] == split
        ]
        split_dataset = dataset.model_copy(
            update={
                "dataset_id": f"{dataset.dataset_id}-{split}",
                "description": f"{dataset.description} Split: {split}.",
                "index_scope": split,
                "cases": cases,
            }
        )
        output_path = args.output_dir / output_name
        output_path.write_text(
            split_dataset.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        print(f"{split}: cases={len(cases)} output={output_path.resolve()}")

        calculation_dataset = split_dataset.model_copy(
            update={
                "dataset_id": f"{split_dataset.dataset_id}-calculate",
                "description": f"{split_dataset.description} Task type: calculate.",
                "index_scope": f"{split}-calculate",
                "cases": [case for case in cases if case.task_type == "calculate"],
            }
        )
        calculation_path = args.output_dir / output_name.replace(
            ".json", "-calculate.json"
        )
        calculation_path.write_text(
            calculation_dataset.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{split}-calculate: cases={len(calculation_dataset.cases)} "
            f"output={calculation_path.resolve()}"
        )

        abstention_dataset = split_dataset.model_copy(
            update={
                "dataset_id": f"{split_dataset.dataset_id}-abstain",
                "description": f"{split_dataset.description} Expected behavior: abstain.",
                "index_scope": f"{split}-abstain",
                "cases": [
                    case for case in cases if case.expected_behavior == "abstain"
                ],
            }
        )
        abstention_path = args.output_dir / output_name.replace(
            ".json", "-abstain.json"
        )
        abstention_path.write_text(
            abstention_dataset.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{split}-abstain: cases={len(abstention_dataset.cases)} "
            f"output={abstention_path.resolve()}"
        )

        verification_dataset = split_dataset.model_copy(
            update={
                "dataset_id": f"{split_dataset.dataset_id}-verification",
                "description": (
                    f"{split_dataset.description} Challenge: claim verification."
                ),
                "index_scope": f"{split}-verification",
                "cases": [
                    case
                    for case in cases
                    if "claim_verification" in case.challenge_types
                ],
            }
        )
        verification_path = args.output_dir / output_name.replace(
            ".json", "-verification.json"
        )
        verification_path.write_text(
            verification_dataset.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{split}-verification: cases={len(verification_dataset.cases)} "
            f"output={verification_path.resolve()}"
        )

        multi_metric_compare_dataset = split_dataset.model_copy(
            update={
                "dataset_id": f"{split_dataset.dataset_id}-multi-metric-compare",
                "description": (
                    f"{split_dataset.description} Multi-year comparison tasks."
                ),
                "index_scope": f"{split}-multi-metric-compare",
                "cases": [
                    case
                    for case in cases
                    if case.task_type == "compare"
                    and "multi_year" in case.challenge_types
                ],
            }
        )
        multi_metric_compare_path = args.output_dir / output_name.replace(
            ".json", "-multi-metric-compare.json"
        )
        multi_metric_compare_path.write_text(
            multi_metric_compare_dataset.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{split}-multi-metric-compare: "
            f"cases={len(multi_metric_compare_dataset.cases)} "
            f"output={multi_metric_compare_path.resolve()}"
        )

        extraction_dataset = split_dataset.model_copy(
            update={
                "dataset_id": f"{split_dataset.dataset_id}-extract",
                "description": (
                    f"{split_dataset.description} Task type: extract."
                ),
                "index_scope": f"{split}-extract",
                "cases": [case for case in cases if case.task_type == "extract"],
            }
        )
        extraction_path = args.output_dir / output_name.replace(
            ".json", "-extract.json"
        )
        extraction_path.write_text(
            extraction_dataset.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{split}-extract: cases={len(extraction_dataset.cases)} "
            f"output={extraction_path.resolve()}"
        )


if __name__ == "__main__":
    main()
