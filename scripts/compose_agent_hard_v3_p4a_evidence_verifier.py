"""Compose P4-A verifier traces with the P3-B full reports."""

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
            "reports/agent/agent-hard-v3-calibration-deepseek-p3b-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4a-evidence-verifier-v3.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p4a-composed.json"
        ),
    },
    "dev": {
        "dataset": Path("data/evaluation/agent-hard-v3-dev.json"),
        "baseline": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p3b-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-"
            "p4a-evidence-verifier-posthoc-v2.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p4a-composed.json"
        ),
    },
}

NORMAL_EXPERIMENTS = [
    (
        "calibration-v1-requirement-only-negative",
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4a-evidence-verifier-v1.json"
        ),
    ),
    (
        "calibration-v2-subject-field-diff",
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4a-evidence-verifier-v2.json"
        ),
    ),
    (
        "calibration-v3-final",
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4a-evidence-verifier-v3.json"
        ),
    ),
    (
        "dev-posthoc-v1-supported-field-regression",
        Path(
            "reports/agent/agent-hard-v3-dev-deepseek-"
            "p4a-evidence-verifier-posthoc-v1.json"
        ),
    ),
    (
        "dev-posthoc-v2-final",
        Path(
            "reports/agent/agent-hard-v3-dev-deepseek-"
            "p4a-evidence-verifier-posthoc-v2.json"
        ),
    ),
]

FAULT_EXPERIMENTS = [
    (
        f"faults-v{version}",
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            f"p4a-evidence-verifier-faults-v{version}.json"
        ),
    )
    for version in range(1, 5)
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
    behavior = {
        case["case_id"]: case["expected_behavior"] for case in dataset["cases"]
    }
    abstentions = [
        row for row in executed if behavior[row["case_id"]] == "abstain"
    ]
    clarifications = [
        row for row in executed if behavior[row["case_id"]] == "clarify"
    ]
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
        "supported_fact_accuracy": (
            fact_matches / fact_count if fact_count else 0.0
        ),
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
            sum(row["covered_requirement_count"] for row in requirement_rows)
            / planned
            if planned
            else None
        ),
        "requirement_evidence_coverage": (
            sum(
                row["evidence_bound_requirement_count"]
                for row in requirement_rows
            )
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


def _verifier_cost(report: dict[str, Any]) -> Counter:
    verifications = [
        item["trace"].get("evidence_verification")
        for item in report["items"]
        if item.get("trace")
    ]
    verifications = [item for item in verifications if item]
    input_tokens = sum(item.get("input_tokens") or 0 for item in verifications)
    output_tokens = sum(item.get("output_tokens") or 0 for item in verifications)
    return Counter(
        model_requests=sum(item.get("request_count", 0) for item in verifications),
        model_input_tokens=input_tokens,
        model_output_tokens=output_tokens,
        model_total_tokens=input_tokens + output_tokens,
    )


def _decision_counts(report: dict[str, Any]) -> Counter:
    return Counter(
        item["trace"]["evidence_verification"]["final_decision"]
        for item in report["items"]
        if item.get("trace") and item["trace"].get("evidence_verification")
    )


def main() -> None:
    summaries: dict[str, Any] = {}
    combined_before = 0
    combined_after = 0
    combined_cases = 0
    final_normal_cost = Counter()
    final_decisions = Counter()
    changed_cases: list[dict[str, Any]] = []

    for split, paths in SPLITS.items():
        dataset = json.loads(paths["dataset"].read_text(encoding="utf-8"))
        baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
        targeted = json.loads(paths["targeted"].read_text(encoding="utf-8"))
        if targeted["index_id"] != baseline["index_id"]:
            raise SystemExit(f"{split}: index IDs differ")
        replacements = {item["case_id"]: item for item in targeted["items"]}
        expected_ids = {
            case["case_id"]
            for case in dataset["cases"]
            if case["task_type"] == "extract"
        }
        if set(replacements) != expected_ids:
            raise SystemExit(f"{split}: extract replacement set mismatch")
        baseline_by_id = {item["case_id"]: item for item in baseline["items"]}
        rows = [replacements.get(item["case_id"], item) for item in baseline["items"]]
        for case_id, replacement in replacements.items():
            before_pass = baseline_by_id[case_id]["score"]["case_pass"]
            after_pass = replacement["score"]["case_pass"]
            if before_pass != after_pass:
                changed_cases.append(
                    {
                        "split": split,
                        "case_id": case_id,
                        "before_pass": before_pass,
                        "after_pass": after_pass,
                    }
                )
        metrics = _metrics(rows, dataset)
        composed = {
            **baseline,
            "status": "complete",
            "evaluated_at": datetime.now(UTC).isoformat(),
            "evaluation_mode": "composed_p4a_evidence_verifier",
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
        split_cost = _verifier_cost(targeted)
        split_decisions = _decision_counts(targeted)
        final_normal_cost.update(split_cost)
        final_decisions.update(split_decisions)
        summaries[split] = {
            "dataset_sha256": _sha256(paths["dataset"]),
            "index_id": baseline["index_id"],
            "before_metrics": baseline["metrics"],
            "after_metrics": metrics,
            "strict_case_pass_count_before": before,
            "strict_case_pass_count_after": after,
            "strict_case_pass_count_delta": after - before,
            "verifier_decisions": dict(split_decisions),
            "incremental_verifier_cost": dict(split_cost),
            "composed_report": paths["output"].as_posix(),
            "composed_report_sha256": _sha256(paths["output"]),
        }

    iterations: list[dict[str, Any]] = []
    recorded_experiment_cost = Counter()
    for label, path in NORMAL_EXPERIMENTS:
        report = json.loads(path.read_text(encoding="utf-8"))
        cost = _verifier_cost(report)
        recorded_experiment_cost.update(cost)
        revisions = sorted(
            {
                item["trace"]["evidence_verification"]["prompt_revision"]
                for item in report["items"]
                if item.get("trace")
                and item["trace"].get("evidence_verification")
            }
        )
        iterations.append(
            {
                "label": label,
                "kind": "normal_answers",
                "report": path.as_posix(),
                "report_sha256": _sha256(path),
                "prompt_revisions": revisions,
                "fact_accuracy": report["metrics"]["supported_fact_accuracy"],
                "strict_case_pass_rate": report["metrics"][
                    "end_to_end_case_pass_rate"
                ],
                "verifier_decisions": dict(_decision_counts(report)),
                "incremental_verifier_cost": dict(cost),
            }
        )
    for label, path in FAULT_EXPERIMENTS:
        report = json.loads(path.read_text(encoding="utf-8"))
        cost = Counter(
            model_requests=report["runtime_cost"]["model_requests"],
            model_input_tokens=report["runtime_cost"]["input_tokens"],
            model_output_tokens=report["runtime_cost"]["output_tokens"],
            model_total_tokens=report["runtime_cost"]["total_tokens"],
        )
        recorded_experiment_cost.update(cost)
        iterations.append(
            {
                "label": label,
                "kind": "stored_trace_fault_injection",
                "report": path.as_posix(),
                "report_sha256": _sha256(path),
                "metrics": report["metrics"],
                "runtime_cost": dict(cost),
            }
        )

    final_fault_path = FAULT_EXPERIMENTS[-1][1]
    final_fault = json.loads(final_fault_path.read_text(encoding="utf-8"))
    summary = {
        "schema_version": "1",
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "change_id": "p4a-independent-evidence-verifier",
        "change_scope": (
            "Complex extraction answers are reviewed in a separate model context. "
            "The verifier sees only query, atomic requirements, claims, and bounded evidence; "
            "it may accept, reject, or request one evidence-bound repair followed by a second "
            "verification. Simple and non-answer cases stay on the single-agent path."
        ),
        "evaluation_scope": (
            "calibration + post-hoc dev stored-trace paired verification and calibration "
            "fault injection; frozen_test sealed"
        ),
        "composition_policy": (
            "Only extract traces were replaced in the P3-B composed reports. Retrieval, base "
            "Agent outputs, Gold, scorer, indexes, and frozen test were unchanged."
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
            "normal_answer_verifier_decisions": dict(final_decisions),
            "final_incremental_verifier_cost": dict(final_normal_cost),
        },
        "fault_injection": {
            "report": final_fault_path.as_posix(),
            "report_sha256": _sha256(final_fault_path),
            "metrics": final_fault["metrics"],
            "runtime_cost": final_fault["runtime_cost"],
        },
        "paired_changed_cases": changed_cases,
        "recorded_all_experiment_incremental_cost": dict(
            recorded_experiment_cost
        ),
        "unmetered_aborted_attempt": {
            "status": "invalid_harness_run",
            "reason": (
                "The first fault-injection harness aborted before report persistence because "
                "one simple claim had no delimiter suitable for truncation."
            ),
            "minimum_model_requests": 2,
            "token_cost": "unknown",
            "included_in_recorded_cost": False,
        },
        "decision": {
            "default_multi_agent_recommended": False,
            "conditional_high_risk_verifier_recommended": True,
            "reason": (
                "The verifier added no strict accuracy on normal answers and cost 80,817 "
                "tokens, but detected or repaired 14/15 injected claim faults. Keep it opt-in "
                "for audit and high-risk multi-fact extraction rather than the default path."
            ),
        },
        "dev_governance": {
            "final_dev_run_is_posthoc": True,
            "unbiased_dev_confirmation": False,
            "frozen_test_opened": False,
        },
        "iterations": iterations,
    }
    output = Path(
        "reports/agent/agent-hard-v3-p4a-evidence-verifier-improvement.json"
    )
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
    print(
        "fault_safe_detection_rate="
        f"{summary['fault_injection']['metrics']['safe_detection_rate']:.4f}"
    )
    print(f"report={output.resolve()}")


if __name__ == "__main__":
    main()
