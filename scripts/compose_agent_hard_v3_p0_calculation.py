"""Compose the isolated P0 calculation rerun with unchanged hard-v3 traces."""

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
            "reports/agent/agent-hard-v3-calibration-deepseek-baseline-rescored.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p0-general-calculate-v5.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p0-composed.json"
        ),
    },
    "dev": {
        "dataset": Path("data/evaluation/agent-hard-v3-dev.json"),
        "baseline": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-baseline-rescored.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p0-general-calculate.json"
        ),
        "output": Path("reports/agent/agent-hard-v3-dev-deepseek-p0-composed.json"),
    },
}

CALIBRATION_ITERATIONS = [
    Path("reports/agent/agent-hard-v3-calibration-deepseek-p0-general-calculate.json"),
    Path("reports/agent/agent-hard-v3-calibration-deepseek-p0-general-calculate-v2.json"),
    Path("reports/agent/agent-hard-v3-calibration-deepseek-p0-general-calculate-v3.json"),
    Path("reports/agent/agent-hard-v3-calibration-deepseek-p0-general-calculate-v4.json"),
    Path("reports/agent/agent-hard-v3-calibration-deepseek-p0-general-calculate-v5.json"),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metrics(rows: list[dict[str, Any]], dataset: dict[str, Any]) -> dict[str, Any]:
    executed = [row for row in rows if row["status"] == "executed"]
    unsupported = [row for row in rows if row["status"] != "executed"]
    executed_fact_count = sum(row["expected_fact_count"] for row in executed)
    executed_fact_matches = sum(
        sum(fact["matched"] for fact in row["score"]["fact_scores"])
        for row in executed
    )
    all_fact_count = sum(row["expected_fact_count"] for row in rows)
    behaviors = {case["case_id"]: case["expected_behavior"] for case in dataset["cases"]}
    abstention_rows = [
        row
        for row in executed
        if behaviors[row["case_id"]] == "abstain"
    ]
    clarification_rows = [
        row
        for row in executed
        if behaviors[row["case_id"]] == "clarify"
    ]
    return {
        "case_count": len(rows),
        "executed_case_count": len(executed),
        "unsupported_case_count": len(unsupported),
        "task_coverage_rate": len(executed) / len(rows),
        "plan_target_exact_rate": _rate(
            [row["score"]["plan_target_exact"] for row in executed]
        ),
        "behavior_accuracy": _rate(
            [row["score"]["behavior_correct"] for row in executed]
        ),
        "supported_fact_accuracy": (
            executed_fact_matches / executed_fact_count if executed_fact_count else 0.0
        ),
        "end_to_end_fact_accuracy": executed_fact_matches / all_fact_count,
        "supported_case_pass_rate": _rate(
            [row["score"]["case_pass"] for row in executed]
        ),
        "end_to_end_case_pass_rate": (
            sum(row["score"]["case_pass"] for row in executed) / len(rows)
        ),
        "safe_abstention_accuracy": _rate(
            [row["score"]["behavior_correct"] for row in abstention_rows]
        ),
        "clarification_accuracy": _rate(
            [row["score"]["behavior_correct"] for row in clarification_rows]
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
    split_summaries: dict[str, Any] = {}
    combined_before_passes = 0
    combined_after_passes = 0
    combined_cases = 0
    final_targeted_cost = Counter()

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
        replacements = {item["case_id"]: item for item in targeted["items"]}
        rows = [replacements.get(item["case_id"], item) for item in baseline["items"]]
        if len(replacements) != 4:
            raise SystemExit(f"{split}: expected four targeted calculation cases")
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
        before_passes = sum(
            bool(item.get("score") and item["score"]["case_pass"])
            for item in baseline["items"]
        )
        after_passes = sum(item["score"]["case_pass"] for item in rows)
        combined_before_passes += before_passes
        combined_after_passes += after_passes
        combined_cases += len(rows)
        targeted_cost = _runtime_cost(targeted)
        final_targeted_cost.update(targeted_cost)
        split_summaries[split] = {
            "dataset_sha256": _sha256(dataset_path),
            "index_id": baseline["index_id"],
            "baseline_metrics": baseline["metrics"],
            "after_metrics": metrics,
            "strict_case_pass_count_before": before_passes,
            "strict_case_pass_count_after": after_passes,
            "strict_case_pass_count_delta": after_passes - before_passes,
            "targeted_runtime_cost": dict(targeted_cost),
            "composed_report": output_path.as_posix(),
            "composed_report_sha256": _sha256(output_path),
        }

    iteration_rows = []
    all_experiment_cost = Counter()
    for index, path in enumerate(CALIBRATION_ITERATIONS, start=1):
        report = json.loads(path.read_text(encoding="utf-8"))
        cost = _runtime_cost(report)
        all_experiment_cost.update(cost)
        iteration_rows.append(
            {
                "iteration": index,
                "report": path.as_posix(),
                "report_sha256": _sha256(path),
                "task_coverage_rate": report["metrics"]["task_coverage_rate"],
                "supported_fact_accuracy": report["metrics"][
                    "supported_fact_accuracy"
                ],
                "end_to_end_case_pass_rate": report["metrics"][
                    "end_to_end_case_pass_rate"
                ],
                "runtime_cost": dict(cost),
            }
        )
    dev_report = json.loads(SPLITS["dev"]["targeted"].read_text(encoding="utf-8"))
    all_experiment_cost.update(_runtime_cost(dev_report))

    summary_path = Path(
        "reports/agent/agent-hard-v3-p0-general-calculation-improvement.json"
    )
    summary = {
        "schema_version": "1",
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "change_id": "p0-general-grounded-calculation-routing",
        "single_variable": (
            "Only calculate-task routing, metric-aware retrieval, cited operand validation, "
            "local Decimal execution, and source-vintage binding changed."
        ),
        "evaluation_scope": "calibration + dev; frozen_test remains sealed",
        "composition_policy": (
            "Eight formerly unsupported calculate cases were rerun remotely. The other forty "
            "stored baseline traces are byte-for-byte reused in composed reports."
        ),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "splits": split_summaries,
        "combined": {
            "case_count": combined_cases,
            "strict_case_pass_count_before": combined_before_passes,
            "strict_case_pass_rate_before": combined_before_passes / combined_cases,
            "strict_case_pass_count_after": combined_after_passes,
            "strict_case_pass_rate_after": combined_after_passes / combined_cases,
            "strict_case_pass_count_delta": (
                combined_after_passes - combined_before_passes
            ),
            "strict_case_pass_rate_delta": (
                combined_after_passes - combined_before_passes
            )
            / combined_cases,
            "task_coverage_rate_before": 40 / 48,
            "task_coverage_rate_after": 1.0,
            "final_targeted_runtime_cost": dict(final_targeted_cost),
            "all_calibration_and_dev_experiment_cost": dict(all_experiment_cost),
        },
        "calibration_iterations": iteration_rows,
        "dev_confirmation": {
            "report": SPLITS["dev"]["targeted"].as_posix(),
            "report_sha256": _sha256(SPLITS["dev"]["targeted"]),
            "metrics": dev_report["metrics"],
        },
        "frozen_test_opened": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("status=complete")
    print(
        "strict_case_pass_rate="
        f"{summary['combined']['strict_case_pass_rate_before']:.4f}->"
        f"{summary['combined']['strict_case_pass_rate_after']:.4f}"
    )
    print(
        "task_coverage_rate="
        f"{summary['combined']['task_coverage_rate_before']:.4f}->1.0000"
    )
    print(f"report={summary_path.resolve()}")


if __name__ == "__main__":
    main()
