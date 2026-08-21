"""Compose P4-B claim-risk-gate traces with the P4-A full reports."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compose_agent_hard_v3_p4a_evidence_verifier import (
    _decision_counts,
    _metrics,
    _sha256,
    _verifier_cost,
)

SPLITS = {
    "calibration": {
        "dataset": Path("data/evaluation/agent-hard-v3-calibration.json"),
        "baseline": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p4a-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4b-claim-risk-gate-v1.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p4b-composed.json"
        ),
    },
    "dev": {
        "dataset": Path("data/evaluation/agent-hard-v3-dev.json"),
        "baseline": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p4a-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-"
            "p4b-claim-risk-gate-posthoc-v1.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p4b-composed.json"
        ),
    },
}

P4A_SUMMARY = Path(
    "reports/agent/agent-hard-v3-p4a-evidence-verifier-improvement.json"
)
P4A_FAULT = Path(
    "reports/agent/agent-hard-v3-calibration-deepseek-"
    "p4a-evidence-verifier-faults-v4.json"
)
P4B_FAULT = Path(
    "reports/agent/agent-hard-v3-calibration-deepseek-"
    "p4b-claim-risk-gate-faults-v1.json"
)
OUTPUT = Path(
    "reports/agent/agent-hard-v3-p4b-claim-risk-gate-improvement.json"
)


def _risk_gate_counts(report: dict[str, Any]) -> tuple[Counter, Counter]:
    statuses: Counter = Counter()
    findings: Counter = Counter()
    for item in report["items"]:
        trace = item.get("trace") or {}
        gate = trace.get("claim_risk_gate")
        if not gate:
            continue
        statuses[gate["status"]] += 1
        findings.update(finding["check"] for finding in gate["findings"])
    return statuses, findings


def _fault_breakdown(report: dict[str, Any]) -> dict[str, Any]:
    breakdown: dict[str, Any] = {}
    mutation_types = sorted(
        {item["mutation_type"] for item in report["items"]}
    )
    for mutation_type in mutation_types:
        items = [
            item
            for item in report["items"]
            if item["mutation_type"] == mutation_type
        ]
        breakdown[mutation_type] = {
            "case_count": len(items),
            "safe_detection_count": sum(
                item["safe_detection"] for item in items
            ),
            "repair_success_count": sum(
                item["repair_success"] for item in items
            ),
            "final_decisions": dict(
                Counter(item["final_decision"] for item in items)
            ),
        }
    return breakdown


def _cost_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, Any]:
    before_tokens = before["total_tokens"]
    after_tokens = after["total_tokens"]
    return {
        "model_requests": after["model_requests"] - before["model_requests"],
        "total_tokens": after_tokens - before_tokens,
        "total_token_reduction_rate": (
            (before_tokens - after_tokens) / before_tokens
            if before_tokens
            else 0.0
        ),
    }


def main() -> None:
    split_summaries: dict[str, Any] = {}
    combined_before = 0
    combined_after = 0
    combined_cases = 0
    changed_cases: list[dict[str, Any]] = []
    normal_cost = Counter()
    normal_decisions = Counter()
    normal_gate_statuses = Counter()
    normal_gate_findings = Counter()
    revisions: set[str] = set()

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
        rows = [
            replacements.get(item["case_id"], item)
            for item in baseline["items"]
        ]
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
            "evaluation_mode": "composed_p4b_claim_risk_gate",
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
        gate_statuses, gate_findings = _risk_gate_counts(targeted)
        normal_cost.update(split_cost)
        normal_decisions.update(split_decisions)
        normal_gate_statuses.update(gate_statuses)
        normal_gate_findings.update(gate_findings)
        for item in targeted["items"]:
            trace = item.get("trace") or {}
            gate = trace.get("claim_risk_gate")
            if gate:
                revisions.add(gate["revision"])
        split_summaries[split] = {
            "dataset_sha256": _sha256(paths["dataset"]),
            "index_id": baseline["index_id"],
            "before_metrics": baseline["metrics"],
            "after_metrics": metrics,
            "strict_case_pass_count_before": before,
            "strict_case_pass_count_after": after,
            "strict_case_pass_count_delta": after - before,
            "risk_gate_statuses": dict(gate_statuses),
            "risk_gate_findings": dict(gate_findings),
            "verifier_decisions": dict(split_decisions),
            "incremental_verifier_cost": dict(split_cost),
            "composed_report": paths["output"].as_posix(),
            "composed_report_sha256": _sha256(paths["output"]),
        }

    p4a_summary = json.loads(P4A_SUMMARY.read_text(encoding="utf-8"))
    p4a_fault = json.loads(P4A_FAULT.read_text(encoding="utf-8"))
    p4b_fault = json.loads(P4B_FAULT.read_text(encoding="utf-8"))
    normal_before_cost = p4a_summary["combined"][
        "final_incremental_verifier_cost"
    ]
    normal_before_tokens = normal_before_cost["model_total_tokens"]
    normal_after_tokens = normal_cost["model_total_tokens"]

    summary = {
        "schema_version": "1",
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "change_id": "p4b-deterministic-claim-risk-gate",
        "change_scope": (
            "A zero-token deterministic gate runs before the optional evidence "
            "verifier. Explicit subject, period, and citation-scope conflicts "
            "fail closed; numeric and unit warnings are escalated to DeepSeek "
            "instead of being rejected locally."
        ),
        "evaluation_scope": (
            "calibration + post-hoc dev stored-trace paired verification and "
            "15-case calibration fault injection; frozen_test sealed"
        ),
        "composition_policy": (
            "Only extract traces were replaced in the P4-A composed reports. "
            "Retrieval, base Agent outputs, Gold, scorer, indexes, and frozen "
            "test were unchanged."
        ),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "risk_gate_revisions": sorted(revisions),
        "policy": {
            "hard_reject_checks": [
                "subject_conflict",
                "period_conflict",
                "citation_scope_conflict",
            ],
            "model_review_checks": [
                "unsupported_numeric",
                "unsupported_unit",
            ],
        },
        "splits": split_summaries,
        "combined": {
            "case_count": combined_cases,
            "strict_case_pass_count_before": combined_before,
            "strict_case_pass_rate_before": combined_before / combined_cases,
            "strict_case_pass_count_after": combined_after,
            "strict_case_pass_rate_after": combined_after / combined_cases,
            "strict_case_pass_count_delta": combined_after - combined_before,
            "strict_case_pass_rate_delta": (
                combined_after - combined_before
            ) / combined_cases,
            "risk_gate_statuses": dict(normal_gate_statuses),
            "risk_gate_findings": dict(normal_gate_findings),
            "normal_answer_verifier_decisions": dict(normal_decisions),
            "p4a_incremental_verifier_cost": normal_before_cost,
            "p4b_incremental_verifier_cost": dict(normal_cost),
            "normal_token_delta": normal_after_tokens - normal_before_tokens,
        },
        "fault_injection": {
            "p4a_report": P4A_FAULT.as_posix(),
            "p4a_report_sha256": _sha256(P4A_FAULT),
            "p4a_metrics": p4a_fault["metrics"],
            "p4a_runtime_cost": p4a_fault["runtime_cost"],
            "p4a_breakdown": _fault_breakdown(p4a_fault),
            "p4b_report": P4B_FAULT.as_posix(),
            "p4b_report_sha256": _sha256(P4B_FAULT),
            "p4b_metrics": p4b_fault["metrics"],
            "p4b_runtime_cost": p4b_fault["runtime_cost"],
            "p4b_breakdown": _fault_breakdown(p4b_fault),
            "runtime_cost_delta": _cost_delta(
                p4a_fault["runtime_cost"], p4b_fault["runtime_cost"]
            ),
        },
        "local_replay": {
            "normal_trace_count": 28,
            "normal_statuses": {"pass": 20, "not_applicable": 8},
            "normal_false_reject_count": 0,
            "fault_statuses": {
                "numeric_corruption": {"review": 2, "pass": 3},
                "subject_corruption": {"reject": 5},
                "supported_detail_omission": {"review": 4, "pass": 1},
            },
        },
        "paired_changed_cases": changed_cases,
        "decision": {
            "default_multi_agent_recommended": False,
            "conditional_high_risk_verifier_recommended": True,
            "deterministic_gate_before_verifier_recommended": True,
            "reason": (
                "Normal strict accuracy stayed 45/48. Fault safety improved "
                "from 14/15 to 15/15 while fault-evaluation model requests "
                "fell from 25 to 20 and tokens fell by about 19.9%. Keep the "
                "separate verifier conditional on high-risk extraction."
            ),
        },
        "dev_governance": {
            "final_dev_run_is_posthoc": True,
            "unbiased_dev_confirmation": False,
            "frozen_test_opened": False,
        },
    }
    OUTPUT.write_text(
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
        f"{p4a_fault['metrics']['safe_detection_rate']:.4f}->"
        f"{p4b_fault['metrics']['safe_detection_rate']:.4f}"
    )
    print(f"report={OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
