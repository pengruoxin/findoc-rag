"""Compose P2-A verification answer-contract reruns with P1 traces."""

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
            "reports/agent/agent-hard-v3-calibration-deepseek-p1-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p2a-verdict-contract-v3.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p2a-composed.json"
        ),
    },
    "dev": {
        "dataset": Path("data/evaluation/agent-hard-v3-dev.json"),
        "baseline": Path("reports/agent/agent-hard-v3-dev-deepseek-p1-composed.json"),
        "targeted": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p2a-verdict-contract-v3.json"
        ),
        "output": Path("reports/agent/agent-hard-v3-dev-deepseek-p2a-composed.json"),
    },
}

EXPERIMENTS = {
    "calibration": [
        Path("reports/agent/agent-hard-v3-calibration-deepseek-p2a-verdict-contract.json"),
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p2a-verdict-contract-v2.json"
        ),
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p2a-verdict-contract-v3.json"
        ),
    ],
    "dev": [
        Path("reports/agent/agent-hard-v3-dev-deepseek-p2a-verdict-contract.json"),
        Path("reports/agent/agent-hard-v3-dev-deepseek-p2a-verdict-contract-v2.json"),
        Path("reports/agent/agent-hard-v3-dev-deepseek-p2a-verdict-contract-v3.json"),
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metrics(rows: list[dict[str, Any]], dataset: dict[str, Any]) -> dict[str, Any]:
    executed = [row for row in rows if row["status"] == "executed"]
    unsupported = [row for row in rows if row["status"] != "executed"]
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
        "unsupported_case_count": len(unsupported),
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
    split_summaries: dict[str, Any] = {}
    before_passes = 0
    after_passes = 0
    case_count = 0
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
            if "claim_verification" in case["challenge_types"]
        }
        replacements = {item["case_id"]: item for item in targeted["items"]}
        if set(replacements) != expected_ids:
            raise SystemExit(f"{split}: targeted report does not match verification cases")
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

        split_before = sum(item["score"]["case_pass"] for item in baseline["items"])
        split_after = sum(item["score"]["case_pass"] for item in rows)
        before_passes += split_before
        after_passes += split_after
        case_count += len(rows)
        cost = _runtime_cost(targeted)
        final_cost.update(cost)
        split_summaries[split] = {
            "dataset_sha256": _sha256(dataset_path),
            "index_id": baseline["index_id"],
            "before_metrics": baseline["metrics"],
            "after_metrics": metrics,
            "strict_case_pass_count_before": split_before,
            "strict_case_pass_count_after": split_after,
            "strict_case_pass_count_delta": split_after - split_before,
            "final_targeted_runtime_cost": dict(cost),
            "composed_report": output_path.as_posix(),
            "composed_report_sha256": _sha256(output_path),
        }

    experiment_rows = []
    all_experiment_cost = Counter()
    for split, paths in EXPERIMENTS.items():
        for iteration, path in enumerate(paths, start=1):
            report = json.loads(path.read_text(encoding="utf-8"))
            cost = _runtime_cost(report)
            all_experiment_cost.update(cost)
            experiment_rows.append(
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
        "change_id": "p2a-verification-answer-contract-and-cross-period-provenance",
        "change_scope": (
            "Verification tasks receive an explicit completeness contract. Submissions must "
            "state a verdict or reproduce every directional predicate. Same-company cross-year "
            "comparison claims may cite another planned year's evidence only when that evidence "
            "year is explicit in the claim."
        ),
        "evaluation_scope": "calibration + dev verification cases; frozen_test remains sealed",
        "composition_policy": (
            "Four claim-verification cases were rerun. The other forty-four P1 traces are reused."
        ),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "splits": split_summaries,
        "combined": {
            "case_count": case_count,
            "strict_case_pass_count_before": before_passes,
            "strict_case_pass_rate_before": before_passes / case_count,
            "strict_case_pass_count_after": after_passes,
            "strict_case_pass_rate_after": after_passes / case_count,
            "strict_case_pass_count_delta": after_passes - before_passes,
            "strict_case_pass_rate_delta": (after_passes - before_passes) / case_count,
            "final_targeted_runtime_cost": dict(final_cost),
            "all_experiment_runtime_cost": dict(all_experiment_cost),
        },
        "iterations": experiment_rows,
        "known_remaining_failure": (
            "v3_601318_verify_nbv now matches 5/5 facts but still fails strict citation "
            "source coverage because cited pages do not overlap the Gold source page."
        ),
        "frozen_test_opened": False,
    }
    summary_path = Path(
        "reports/agent/agent-hard-v3-p2a-answer-contract-improvement.json"
    )
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
    print(f"report={summary_path.resolve()}")


if __name__ == "__main__":
    main()
