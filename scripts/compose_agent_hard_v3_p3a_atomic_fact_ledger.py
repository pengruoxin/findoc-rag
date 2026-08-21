"""Compose P3-A extraction reruns with the frozen P2-B non-extraction traces."""

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
            "reports/agent/agent-hard-v3-calibration-deepseek-p2b-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p3a-atomic-fact-ledger-v4.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p3a-composed.json"
        ),
    },
    "dev": {
        "dataset": Path("data/evaluation/agent-hard-v3-dev.json"),
        "baseline": Path("reports/agent/agent-hard-v3-dev-deepseek-p2b-composed.json"),
        "targeted": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p3a-atomic-fact-ledger-posthoc-v3.json"
        ),
        "output": Path("reports/agent/agent-hard-v3-dev-deepseek-p3a-composed.json"),
    },
}

EXPERIMENTS = [
    (
        "calibration-invalid-index",
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p3a-atomic-fact-ledger-v1.json"
        ),
        False,
    ),
    (
        "calibration-v2",
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p3a-atomic-fact-ledger-v2.json"
        ),
        True,
    ),
    (
        "calibration-v3",
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p3a-atomic-fact-ledger-v3.json"
        ),
        True,
    ),
    (
        "calibration-v4-final",
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p3a-atomic-fact-ledger-v4.json"
        ),
        True,
    ),
    (
        "dev-first-confirmation",
        Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p3a-atomic-fact-ledger.json"
        ),
        True,
    ),
    (
        "dev-posthoc-v2",
        Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p3a-atomic-fact-ledger-posthoc-v2.json"
        ),
        True,
    ),
    (
        "dev-posthoc-v3-final",
        Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p3a-atomic-fact-ledger-posthoc-v3.json"
        ),
        True,
    ),
]


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
    behaviors = {case["case_id"]: case["expected_behavior"] for case in dataset["cases"]}
    abstentions = [row for row in executed if behaviors[row["case_id"]] == "abstain"]
    clarifications = [row for row in executed if behaviors[row["case_id"]] == "clarify"]
    requirement_rows = [
        row["requirement_diagnostics"]
        for row in executed
        if row.get("requirement_diagnostics")
        and row["requirement_diagnostics"]["applicable"]
    ]
    planned = sum(row["planned_requirement_count"] for row in requirement_rows)
    scoped = sum(row["scoped_requirement_count"] for row in requirement_rows)
    return {
        "case_count": len(rows),
        "executed_case_count": len(executed),
        "task_coverage_rate": len(executed) / len(rows),
        "plan_target_exact_rate": _rate(
            [row["score"]["plan_target_exact"] for row in executed]
        ),
        "behavior_accuracy": _rate(
            [row["score"]["behavior_correct"] for row in executed]
        ),
        "supported_fact_accuracy": fact_matches / fact_count if fact_count else 0.0,
        "supported_case_pass_rate": _rate(
            [row["score"]["case_pass"] for row in executed]
        ),
        "safe_abstention_accuracy": _rate(
            [row["score"]["behavior_correct"] for row in abstentions]
        ),
        "clarification_accuracy": _rate(
            [row["score"]["behavior_correct"] for row in clarifications]
        ),
        "atomic_requirement_case_count": len(requirement_rows),
        "task_requirement_coverage": (
            sum(row["covered_requirement_count"] for row in requirement_rows) / planned
            if planned
            else None
        ),
        "requirement_evidence_coverage": (
            sum(row["evidence_bound_requirement_count"] for row in requirement_rows)
            / planned
            if planned
            else None
        ),
        "scope_validation_rate": (
            sum(
                row["scope_validated_requirement_count"]
                for row in requirement_rows
            )
            / scoped
            if scoped
            else None
        ),
        "claim_citation_completeness": (
            sum(row["claim_citation_completeness"] for row in requirement_rows)
            / len(requirement_rows)
            if requirement_rows
            else None
        ),
    }


def _runtime_cost(report: dict[str, Any]) -> Counter:
    traces = [
        item["trace"].get("model_trace")
        for item in report["items"]
        if item["status"] == "executed" and item.get("trace")
    ]
    traces = [trace for trace in traces if trace]
    calls = [
        call
        for item in report["items"]
        if item["status"] == "executed" and item.get("trace")
        for call in item["trace"]["tool_calls"]
    ]
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
        dataset = json.loads(paths["dataset"].read_text(encoding="utf-8"))
        baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
        targeted = json.loads(paths["targeted"].read_text(encoding="utf-8"))
        if baseline["index_id"] != targeted["index_id"]:
            raise SystemExit(f"{split}: baseline and targeted index IDs differ")
        expected_ids = {
            case["case_id"]
            for case in dataset["cases"]
            if case["task_type"] == "extract"
        }
        replacements = {item["case_id"]: item for item in targeted["items"]}
        if set(replacements) != expected_ids:
            raise SystemExit(f"{split}: targeted report does not match extract cases")
        rows = [replacements.get(item["case_id"], item) for item in baseline["items"]]
        metrics = _metrics(rows, dataset)
        composed = {
            **baseline,
            "status": "complete",
            "evaluated_at": datetime.now(UTC).isoformat(),
            "evaluation_mode": "composed_single_variable_extract_rerun",
            "composition": {
                "baseline_report": paths["baseline"].as_posix(),
                "baseline_report_sha256": _sha256(paths["baseline"]),
                "targeted_report": paths["targeted"].as_posix(),
                "targeted_report_sha256": _sha256(paths["targeted"]),
                "replaced_case_ids": sorted(replacements),
                "unchanged_case_count": len(rows) - len(replacements),
            },
            "metrics": metrics,
            "items": rows,
        }
        paths["output"].write_text(
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
            "dataset_sha256": _sha256(paths["dataset"]),
            "index_id": baseline["index_id"],
            "before_metrics": baseline["metrics"],
            "after_metrics": metrics,
            "strict_case_pass_count_before": before,
            "strict_case_pass_count_after": after,
            "strict_case_pass_count_delta": after - before,
            "final_targeted_runtime_cost": dict(cost),
            "composed_report": paths["output"].as_posix(),
            "composed_report_sha256": _sha256(paths["output"]),
        }

    iterations = []
    experiment_cost = Counter()
    for label, path, valid in EXPERIMENTS:
        report = json.loads(path.read_text(encoding="utf-8"))
        cost = _runtime_cost(report)
        experiment_cost.update(cost)
        iterations.append(
            {
                "label": label,
                "valid_for_effect_comparison": valid,
                "report": path.as_posix(),
                "report_sha256": _sha256(path),
                "fact_accuracy": report["metrics"]["supported_fact_accuracy"],
                "strict_case_pass_rate": report["metrics"][
                    "end_to_end_case_pass_rate"
                ],
                "behavior_accuracy": report["metrics"]["behavior_accuracy"],
                "runtime_cost": dict(cost),
            }
        )

    summary = {
        "schema_version": "1",
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "change_id": "p3a-evidence-aware-atomic-fact-ledger",
        "change_scope": (
            "DeepSeek converts retrieved extraction evidence into an atomic fact ledger; "
            "the controller fills deterministic metric omissions, searches only uncovered "
            "requirements, rejects group/segment scope conflicts, and requires every final "
            "fact to bind requirement IDs and evidence chunks."
        ),
        "evaluation_scope": "calibration + dev extraction tasks; frozen_test sealed",
        "composition_policy": (
            "Twenty-eight extract traces were replaced; twenty P2-B compare/calculate "
            "traces were reused unchanged."
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
        "dev_governance": {
            "first_dev_run_exposed_validator_bugs": True,
            "final_dev_run_is_posthoc": True,
            "unbiased_dev_confirmation": False,
            "frozen_test_opened": False,
        },
        "iterations": iterations,
    }
    output = Path("reports/agent/agent-hard-v3-p3a-atomic-fact-ledger-improvement.json")
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
