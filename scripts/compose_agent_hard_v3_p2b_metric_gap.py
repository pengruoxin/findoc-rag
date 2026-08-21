"""Compose P2-B multi-metric comparison reruns with P2-A traces."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SPLITS = {
    "calibration": {
        "dataset": Path("data/evaluation/agent-hard-v3-calibration.json"),
        "baseline": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p2a-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p2b-metric-gap-retrieval-v3.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p2b-composed.json"
        ),
    },
    "dev": {
        "dataset": Path("data/evaluation/agent-hard-v3-dev.json"),
        "baseline": Path("reports/agent/agent-hard-v3-dev-deepseek-p2a-composed.json"),
        "targeted": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p2b-metric-gap-retrieval.json"
        ),
        "output": Path("reports/agent/agent-hard-v3-dev-deepseek-p2b-composed.json"),
    },
}

EXPERIMENTS = {
    "calibration": [
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p2b-metric-gap-retrieval.json"
        ),
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p2b-metric-gap-retrieval-v2.json"
        ),
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p2b-metric-gap-retrieval-v3.json"
        ),
    ],
    "dev": [
        Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p2b-metric-gap-retrieval.json"
        )
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metrics(rows: list[dict[str, Any]], dataset: dict[str, Any]) -> dict[str, Any]:
    executed = [row for row in rows if row["status"] == "executed"]
    fact_count = sum(row["expected_fact_count"] for row in executed)
    fact_matches = sum(
        sum(fact["matched"] for fact in row["score"]["fact_scores"])
        for row in executed
    )
    all_fact_count = sum(row["expected_fact_count"] for row in rows)
    behaviors = {case["case_id"]: case["expected_behavior"] for case in dataset["cases"]}
    abstentions = [
        row for row in executed if behaviors[row["case_id"]] == "abstain"
    ]
    clarifications = [
        row for row in executed if behaviors[row["case_id"]] == "clarify"
    ]
    return {
        "case_count": len(rows),
        "executed_case_count": len(executed),
        "unsupported_case_count": len(rows) - len(executed),
        "task_coverage_rate": len(executed) / len(rows),
        "plan_target_exact_rate": _rate(
            [row["score"]["plan_target_exact"] for row in executed]
        ),
        "behavior_accuracy": _rate(
            [row["score"]["behavior_correct"] for row in executed]
        ),
        "supported_fact_accuracy": fact_matches / fact_count if fact_count else 0.0,
        "end_to_end_fact_accuracy": (
            fact_matches / all_fact_count if all_fact_count else 0.0
        ),
        "supported_case_pass_rate": _rate(
            [row["score"]["case_pass"] for row in executed]
        ),
        "end_to_end_case_pass_rate": (
            sum(row["score"]["case_pass"] for row in executed) / len(rows)
        ),
        "safe_abstention_accuracy": _rate(
            [row["score"]["behavior_correct"] for row in abstentions]
        ),
        "clarification_accuracy": _rate(
            [row["score"]["behavior_correct"] for row in clarifications]
        ),
    }


def _runtime_cost(report: dict[str, Any]) -> Counter:
    executed = [item for item in report["items"] if item["status"] == "executed"]
    traces = [item["trace"].get("model_trace") for item in executed]
    traces = [trace for trace in traces if trace]
    calls = [call for item in executed for call in item["trace"]["tool_calls"]]
    input_tokens = sum(trace.get("input_tokens") or 0 for trace in traces)
    output_tokens = sum(trace.get("output_tokens") or 0 for trace in traces)
    return Counter(
        model_requests=sum(trace["request_count"] for trace in traces),
        model_input_tokens=input_tokens,
        model_output_tokens=output_tokens,
        model_total_tokens=input_tokens + output_tokens,
        tool_calls=len(calls),
    )


def main() -> None:
    summaries: dict[str, Any] = {}
    combined_before = 0
    combined_after = 0
    combined_cases = 0
    final_cost = Counter()

    for split, paths in SPLITS.items():
        dataset_path = paths["dataset"]
        baseline_path = paths["baseline"]
        targeted_path = paths["targeted"]
        output_path = paths["output"]
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        targeted = json.loads(targeted_path.read_text(encoding="utf-8"))
        if baseline["index_id"] != targeted["index_id"]:
            raise SystemExit(f"{split}: baseline and targeted index IDs differ")
        if baseline["provider"] != targeted["provider"]:
            raise SystemExit(f"{split}: baseline and targeted providers differ")

        expected_ids = {
            case["case_id"]
            for case in dataset["cases"]
            if case["task_type"] == "compare"
            and "multi_year" in case["challenge_types"]
        }
        replacements = {item["case_id"]: item for item in targeted["items"]}
        if set(replacements) != expected_ids:
            raise SystemExit(f"{split}: targeted report does not match multi-year cases")
        rows = [replacements.get(item["case_id"], item) for item in baseline["items"]]
        metrics = _metrics(rows, dataset)
        composed = {
            **baseline,
            "status": "complete",
            "evaluated_at": datetime.now(UTC).isoformat(),
            "evaluation_mode": "composed_single_variable_targeted_rerun",
            "composition": {
                "baseline_report": baseline_path.as_posix(),
                "baseline_report_sha256": _sha256(baseline_path),
                "targeted_report": targeted_path.as_posix(),
                "targeted_report_sha256": _sha256(targeted_path),
                "replaced_case_ids": sorted(replacements),
                "unchanged_case_count": len(rows) - len(replacements),
            },
            "metrics": metrics,
            "items": rows,
        }
        output_path.write_text(
            json.dumps(composed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        before = sum(item["score"]["case_pass"] for item in baseline["items"])
        after = sum(item["score"]["case_pass"] for item in rows)
        combined_before += before
        combined_after += after
        combined_cases += len(rows)
        cost = _runtime_cost(targeted)
        final_cost.update(cost)
        summaries[split] = {
            "dataset_sha256": _sha256(dataset_path),
            "index_id": baseline["index_id"],
            "before_metrics": baseline["metrics"],
            "after_metrics": metrics,
            "strict_case_pass_count_before": before,
            "strict_case_pass_count_after": after,
            "strict_case_pass_count_delta": after - before,
            "final_targeted_runtime_cost": dict(cost),
            "composed_report": output_path.as_posix(),
            "composed_report_sha256": _sha256(output_path),
        }

    iterations = []
    experiment_cost = Counter()
    for split, paths in EXPERIMENTS.items():
        for iteration, path in enumerate(paths, start=1):
            report = json.loads(path.read_text(encoding="utf-8"))
            cost = _runtime_cost(report)
            experiment_cost.update(cost)
            iterations.append(
                {
                    "split": split,
                    "iteration": iteration,
                    "report": path.as_posix(),
                    "report_sha256": _sha256(path),
                    "fact_accuracy": report["metrics"]["supported_fact_accuracy"],
                    "strict_case_pass_rate": report["metrics"][
                        "end_to_end_case_pass_rate"
                    ],
                    "runtime_cost": dict(cost),
                }
            )

    summary = {
        "schema_version": "1",
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "change_id": "p2b-multi-metric-gap-driven-retrieval",
        "change_scope": (
            "Comparison planning extracts non-overlapping metric requirements, includes all "
            "metrics in retrieval hints, judges evidence sufficiency per metric, exposes "
            "remaining metric gaps to DeepSeek, and safely repairs explicit-year claims assigned "
            "to the wrong same-company target."
        ),
        "evaluation_scope": "calibration + dev multi-year comparisons; frozen_test sealed",
        "composition_policy": (
            "Eight multi-year comparison traces were replaced; forty P2-A traces were reused."
        ),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "splits": summaries,
        "combined": {
            "case_count": combined_cases,
            "strict_case_pass_count_before": combined_before,
            "strict_case_pass_rate_before": combined_before / combined_cases,
            "strict_case_pass_count_after": combined_after,
            "strict_case_pass_rate_after": combined_after / combined_cases,
            "strict_case_pass_count_delta": combined_after - combined_before,
            "strict_case_pass_rate_delta": (
                combined_after - combined_before
            )
            / combined_cases,
            "final_targeted_runtime_cost": dict(final_cost),
            "all_experiment_runtime_cost": dict(experiment_cost),
        },
        "iterations": iterations,
        "frozen_test_opened": False,
    }
    output = Path("reports/agent/agent-hard-v3-p2b-metric-gap-improvement.json")
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("status=complete")
    print(
        "strict_case_pass_rate="
        f"{summary['combined']['strict_case_pass_rate_before']:.4f}->"
        f"{summary['combined']['strict_case_pass_rate_after']:.4f}"
    )
    print(f"report={output.resolve()}")


if __name__ == "__main__":
    main()
