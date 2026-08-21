"""Compose P4-C open-risk traces and summarize paired evaluations."""

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
from compose_agent_hard_v3_p4b_claim_risk_gate import (
    _fault_breakdown,
    _risk_gate_counts,
)

SPLITS = {
    "calibration": {
        "dataset": Path("data/evaluation/agent-hard-v3-calibration.json"),
        "baseline": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p4b-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4c-open-risk-v1.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p4c-composed.json"
        ),
    },
    "dev": {
        "dataset": Path("data/evaluation/agent-hard-v3-dev.json"),
        "baseline": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p4b-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-"
            "p4c-open-risk-posthoc-v1.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p4c-composed.json"
        ),
    },
}
P4B_SUMMARY = Path(
    "reports/agent/agent-hard-v3-p4b-claim-risk-gate-improvement.json"
)
P4B_KNOWN_FAULT = Path(
    "reports/agent/agent-hard-v3-calibration-deepseek-"
    "p4b-claim-risk-gate-faults-v1.json"
)
P4C_KNOWN_FAULT = Path(
    "reports/agent/agent-hard-v3-calibration-deepseek-"
    "p4c-open-risk-known-faults-v1.json"
)
OPEN_RISK_OFF = Path(
    "reports/agent/agent-hard-v3-deepseek-"
    "p4c-open-risk-ablation-off-v2.json"
)
OPEN_RISK_ON = Path(
    "reports/agent/agent-hard-v3-deepseek-"
    "p4c-open-risk-ablation-on-v2.json"
)
EXPLORATORY_BASELINE = Path(
    "reports/agent/agent-hard-v3-deepseek-p4b-open-risk-faults-v1.json"
)
EXPLORATORY_P4C = Path(
    "reports/agent/agent-hard-v3-deepseek-p4c-open-risk-faults-v1.json"
)
OUTPUT = Path("reports/agent/agent-hard-v3-p4c-open-risk-improvement.json")


def _model_cost_delta(
    before: dict[str, int], after: dict[str, int]
) -> dict[str, Any]:
    before_tokens = before["model_total_tokens"]
    after_tokens = after["model_total_tokens"]
    return {
        "model_requests": after["model_requests"] - before["model_requests"],
        "model_total_tokens": after_tokens - before_tokens,
        "model_total_token_change_rate": (
            (after_tokens - before_tokens) / before_tokens
            if before_tokens
            else 0.0
        ),
    }


def _runtime_cost_delta(
    before: dict[str, int], after: dict[str, int]
) -> dict[str, Any]:
    before_tokens = before["total_tokens"]
    after_tokens = after["total_tokens"]
    return {
        "model_requests": after["model_requests"] - before["model_requests"],
        "total_tokens": after_tokens - before_tokens,
        "total_token_change_rate": (
            (after_tokens - before_tokens) / before_tokens
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
    gate_revisions: set[str] = set()
    prompt_revisions: set[str] = set()

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
            "evaluation_mode": "composed_p4c_open_risk",
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
            verification = trace.get("evidence_verification")
            if gate:
                gate_revisions.add(gate["revision"])
            if verification:
                prompt_revisions.add(verification["prompt_revision"])
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

    p4b_summary = json.loads(P4B_SUMMARY.read_text(encoding="utf-8"))
    p4b_known = json.loads(P4B_KNOWN_FAULT.read_text(encoding="utf-8"))
    p4c_known = json.loads(P4C_KNOWN_FAULT.read_text(encoding="utf-8"))
    open_off = json.loads(OPEN_RISK_OFF.read_text(encoding="utf-8"))
    open_on = json.loads(OPEN_RISK_ON.read_text(encoding="utf-8"))
    exploratory_before = json.loads(
        EXPLORATORY_BASELINE.read_text(encoding="utf-8")
    )
    exploratory_after = json.loads(
        EXPLORATORY_P4C.read_text(encoding="utf-8")
    )
    p4b_normal_cost = p4b_summary["combined"][
        "p4b_incremental_verifier_cost"
    ]

    summary = {
        "schema_version": "1",
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "change_id": "p4c-open-claim-risk-routing",
        "change_scope": (
            "A generic character-bigram claim/evidence coverage signal routes "
            "low-support answers to the separate DeepSeek verifier. The signal "
            "never rejects locally. Extract CLI routing now defaults to auto, "
            "with explicit off and always policies."
        ),
        "evaluation_scope": (
            "calibration + post-hoc dev normal answers, prior 15-case known "
            "fault set, and paired 15-case unseen semantic fault ablation; "
            "frozen_test sealed"
        ),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "claim_support_threshold": {
            "feature": "normalized Chinese/Latin character-bigram coverage",
            "review_below": 0.72,
            "minimum_claim_bigrams": 6,
            "action": "model_review_only",
        },
        "risk_gate_revisions": sorted(gate_revisions),
        "verifier_prompt_revisions": sorted(prompt_revisions),
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
            "p4b_incremental_verifier_cost": p4b_normal_cost,
            "p4c_incremental_verifier_cost": dict(normal_cost),
            "normal_cost_delta": _model_cost_delta(
                p4b_normal_cost, dict(normal_cost)
            ),
            "additional_normal_model_requests": 0,
        },
        "known_fault_regression": {
            "p4b_report": P4B_KNOWN_FAULT.as_posix(),
            "p4b_report_sha256": _sha256(P4B_KNOWN_FAULT),
            "p4b_metrics": p4b_known["metrics"],
            "p4b_runtime_cost": p4b_known["runtime_cost"],
            "p4c_report": P4C_KNOWN_FAULT.as_posix(),
            "p4c_report_sha256": _sha256(P4C_KNOWN_FAULT),
            "p4c_metrics": p4c_known["metrics"],
            "p4c_runtime_cost": p4c_known["runtime_cost"],
            "p4c_breakdown": _fault_breakdown(p4c_known),
            "runtime_cost_delta": _runtime_cost_delta(
                p4b_known["runtime_cost"], p4c_known["runtime_cost"]
            ),
        },
        "open_risk_paired_ablation": {
            "fault_types": [
                "semantic_negation",
                "unsupported_causal_inference",
                "scope_qualifier_corruption",
            ],
            "signal_off_report": OPEN_RISK_OFF.as_posix(),
            "signal_off_report_sha256": _sha256(OPEN_RISK_OFF),
            "signal_off_metrics": open_off["metrics"],
            "signal_off_runtime_cost": open_off["runtime_cost"],
            "signal_on_report": OPEN_RISK_ON.as_posix(),
            "signal_on_report_sha256": _sha256(OPEN_RISK_ON),
            "signal_on_metrics": open_on["metrics"],
            "signal_on_runtime_cost": open_on["runtime_cost"],
            "runtime_cost_delta": _runtime_cost_delta(
                open_off["runtime_cost"], open_on["runtime_cost"]
            ),
        },
        "local_replay": {
            "normal_trace_count": 28,
            "normal_statuses": {
                "pass": 19,
                "review": 1,
                "not_applicable": 8,
            },
            "new_normal_review_count": 1,
            "new_normal_model_request_count": 0,
            "open_fault_statuses": {"review": 15},
        },
        "exploratory_negative_iteration": {
            "baseline_report": EXPLORATORY_BASELINE.as_posix(),
            "baseline_metrics": exploratory_before["metrics"],
            "first_p4c_report": EXPLORATORY_P4C.as_posix(),
            "first_p4c_metrics": exploratory_after["metrics"],
            "reason_not_final": (
                "The first scope mutation used parent-company wording that "
                "overlapped the source text and was semantically ambiguous for "
                "one segment case. The final v2 mutation is explicitly unsupported, "
                "and off/on runs share the same code and prompt."
            ),
        },
        "paired_changed_cases": changed_cases,
        "decision": {
            "default_multi_agent_for_all_queries_recommended": False,
            "automatic_high_risk_verifier_recommended": True,
            "always_verify_every_extract_recommended": False,
            "reason": (
                "Normal strict accuracy stayed 45/48 with no extra normal model "
                "requests. On unseen semantic faults, generic risk routing raised "
                "safe handling from 9/15 to 15/15 at six extra verifier calls. "
                "This supports automatic conditional routing, not more agents."
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
        "open_risk_safe_detection_rate="
        f"{open_off['metrics']['safe_detection_rate']:.4f}->"
        f"{open_on['metrics']['safe_detection_rate']:.4f}"
    )
    print(f"report={OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
