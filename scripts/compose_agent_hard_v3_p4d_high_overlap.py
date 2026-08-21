"""Compose P4-D reports and summarize high-overlap fault experiments."""

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
from compose_agent_hard_v3_p4b_claim_risk_gate import _risk_gate_counts

SPLITS = {
    "calibration": {
        "dataset": Path("data/evaluation/agent-hard-v3-calibration.json"),
        "baseline": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p4c-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4d-contract-completeness-v3.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p4d-composed.json"
        ),
    },
    "dev": {
        "dataset": Path("data/evaluation/agent-hard-v3-dev.json"),
        "baseline": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p4c-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-"
            "p4d-contract-completeness-posthoc-v3.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p4d-composed.json"
        ),
    },
}
P4C_SUMMARY = Path("reports/agent/agent-hard-v3-p4c-open-risk-improvement.json")
P4C_HIGH_BASELINE = Path(
    "reports/agent/agent-hard-v3-deepseek-p4c-high-overlap-baseline-v1.json"
)
CONTRACT_OFF = Path(
    "reports/agent/agent-hard-v3-deepseek-p4d-contract-ablation-off-v1.json"
)
CONTRACT_ON = Path(
    "reports/agent/agent-hard-v3-deepseek-p4d-contract-ablation-on-v1.json"
)
HIGH_COST_FINAL = Path(
    "reports/agent/agent-hard-v3-deepseek-p4d-high-overlap-final-v1.json"
)
LIGHTWEIGHT_V2 = Path(
    "reports/agent/agent-hard-v3-deepseek-p4d-high-overlap-final-v2.json"
)
FINAL_HIGH = Path(
    "reports/agent/agent-hard-v3-deepseek-p4d-high-overlap-final-v3.json"
)
KNOWN_OLD = Path(
    "reports/agent/agent-hard-v3-calibration-deepseek-"
    "p4d-known-faults-v1.json"
)
KNOWN_FIXED_V2 = Path(
    "reports/agent/agent-hard-v3-calibration-deepseek-"
    "p4d-known-faults-fixed-v2.json"
)
FINAL_KNOWN = Path(
    "reports/agent/agent-hard-v3-calibration-deepseek-"
    "p4d-known-faults-fixed-v3.json"
)
OPEN_REGRESSION_PRIOR = Path(
    "reports/agent/agent-hard-v3-deepseek-p4d-open-risk-regression-v1.json"
)
OPEN_REGRESSION = Path(
    "reports/agent/agent-hard-v3-deepseek-p4d-open-risk-regression-v2.json"
)
OUTPUT = Path(
    "reports/agent/agent-hard-v3-p4d-high-overlap-improvement.json"
)


def _cost_delta(
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


def _report_ref(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": _sha256(path)}


def _normal_cost(paths: list[Path]) -> Counter:
    cost = Counter()
    for path in paths:
        cost.update(_verifier_cost(json.loads(path.read_text(encoding="utf-8"))))
    return cost


def main() -> None:
    split_summaries: dict[str, Any] = {}
    combined_before = 0
    combined_after = 0
    combined_cases = 0
    changed_cases: list[dict[str, Any]] = []
    normal_cost = Counter()
    decisions = Counter()
    gate_statuses = Counter()
    gate_findings = Counter()
    gate_revisions: set[str] = set()
    prompt_revisions: set[str] = set()

    for split, paths in SPLITS.items():
        dataset = json.loads(paths["dataset"].read_text(encoding="utf-8"))
        baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
        targeted = json.loads(paths["targeted"].read_text(encoding="utf-8"))
        if baseline["index_id"] != targeted["index_id"]:
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
            "evaluation_mode": "composed_p4d_high_overlap_contract",
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
        split_gate_statuses, split_gate_findings = _risk_gate_counts(targeted)
        normal_cost.update(split_cost)
        decisions.update(split_decisions)
        gate_statuses.update(split_gate_statuses)
        gate_findings.update(split_gate_findings)
        for item in targeted["items"]:
            trace = item.get("trace") or {}
            if trace.get("claim_risk_gate"):
                gate_revisions.add(trace["claim_risk_gate"]["revision"])
            if trace.get("evidence_verification"):
                prompt_revisions.add(
                    trace["evidence_verification"]["prompt_revision"]
                )
        split_summaries[split] = {
            "dataset_sha256": _sha256(paths["dataset"]),
            "index_id": baseline["index_id"],
            "before_metrics": baseline["metrics"],
            "after_metrics": metrics,
            "strict_case_pass_count_before": before,
            "strict_case_pass_count_after": after,
            "strict_case_pass_count_delta": after - before,
            "risk_gate_statuses": dict(split_gate_statuses),
            "risk_gate_findings": dict(split_gate_findings),
            "verifier_decisions": dict(split_decisions),
            "incremental_verifier_cost": dict(split_cost),
            "composed_report": paths["output"].as_posix(),
            "composed_report_sha256": _sha256(paths["output"]),
        }

    p4c = json.loads(P4C_SUMMARY.read_text(encoding="utf-8"))
    p4c_cost = p4c["combined"]["p4c_incremental_verifier_cost"]
    p4c_high = json.loads(P4C_HIGH_BASELINE.read_text(encoding="utf-8"))
    contract_off = json.loads(CONTRACT_OFF.read_text(encoding="utf-8"))
    contract_on = json.loads(CONTRACT_ON.read_text(encoding="utf-8"))
    high_cost = json.loads(HIGH_COST_FINAL.read_text(encoding="utf-8"))
    lightweight_v2 = json.loads(LIGHTWEIGHT_V2.read_text(encoding="utf-8"))
    final_high = json.loads(FINAL_HIGH.read_text(encoding="utf-8"))
    known_old = json.loads(KNOWN_OLD.read_text(encoding="utf-8"))
    known_fixed_v2 = json.loads(KNOWN_FIXED_V2.read_text(encoding="utf-8"))
    final_known = json.loads(FINAL_KNOWN.read_text(encoding="utf-8"))
    open_regression_prior = json.loads(
        OPEN_REGRESSION_PRIOR.read_text(encoding="utf-8")
    )
    open_regression = json.loads(OPEN_REGRESSION.read_text(encoding="utf-8"))
    focused_normal_cost = _normal_cost(
        [
            Path(
                "reports/agent/agent-hard-v3-calibration-deepseek-"
                "p4d-focused-contract-v1.json"
            ),
            Path(
                "reports/agent/agent-hard-v3-dev-deepseek-"
                "p4d-focused-contract-posthoc-v1.json"
            ),
        ]
    )

    summary = {
        "schema_version": "1",
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "change_id": "p4d-high-overlap-contract-and-accounting-sign",
        "change_scope": (
            "Low-complexity extraction claims that diverge from their atomic "
            "requirement contract are routed to the verifier. Explicit negative "
            "or accounting-parenthesis signs absent from the contract fail "
            "closed locally. Required numeric omissions are routed for repair."
        ),
        "evaluation_scope": (
            "normal calibration + post-hoc dev, corrected known-fault regression, "
            "P4-C open-risk regression, and 15 high-overlap label/relation/sign "
            "faults; frozen_test sealed"
        ),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "requirement_contract_similarity_review_below": 0.99,
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
            "risk_gate_statuses": dict(gate_statuses),
            "risk_gate_findings": dict(gate_findings),
            "normal_answer_verifier_decisions": dict(decisions),
            "p4c_incremental_verifier_cost": p4c_cost,
            "p4d_incremental_verifier_cost": dict(normal_cost),
            "normal_cost_delta": _cost_delta(p4c_cost, dict(normal_cost)),
            "additional_normal_model_requests": 0,
        },
        "high_overlap_faults": {
            "fault_types": [
                "label_swap",
                "relation_swap",
                "accounting_sign_flip",
            ],
            "p4c_baseline_report": _report_ref(P4C_HIGH_BASELINE),
            "p4c_baseline_metrics": p4c_high["metrics"],
            "p4c_baseline_runtime_cost": p4c_high["runtime_cost"],
            "final_report": _report_ref(FINAL_HIGH),
            "final_metrics": final_high["metrics"],
            "final_runtime_cost": final_high["runtime_cost"],
            "remaining_unsafe_case_ids": [
                item["mutation_id"]
                for item in final_high["items"]
                if item["unsafe_accept"]
            ],
        },
        "known_fault_regression": {
            "report": _report_ref(FINAL_KNOWN),
            "metrics": final_known["metrics"],
            "runtime_cost": final_known["runtime_cost"],
            "harness_fix": (
                "ASCII commas between digits are no longer treated as clause "
                "separators. Two prior omission mutations were actually truncated "
                "numbers and are excluded from omission-repair comparison."
            ),
            "iterations": [
                {
                    "label": "legacy-comma-splitting-harness",
                    "report": _report_ref(KNOWN_OLD),
                    "metrics": known_old["metrics"],
                    "runtime_cost": known_old["runtime_cost"],
                    "comparable": False,
                },
                {
                    "label": "corrected-harness-before-completeness-gate",
                    "report": _report_ref(KNOWN_FIXED_V2),
                    "metrics": known_fixed_v2["metrics"],
                    "runtime_cost": known_fixed_v2["runtime_cost"],
                    "comparable": True,
                },
                {
                    "label": "corrected-harness-with-completeness-gate",
                    "report": _report_ref(FINAL_KNOWN),
                    "metrics": final_known["metrics"],
                    "runtime_cost": final_known["runtime_cost"],
                    "comparable": True,
                },
            ],
        },
        "open_risk_regression": {
            "report": _report_ref(OPEN_REGRESSION),
            "metrics": open_regression["metrics"],
            "runtime_cost": open_regression["runtime_cost"],
            "prior_report": _report_ref(OPEN_REGRESSION_PRIOR),
            "prior_metrics": open_regression_prior["metrics"],
            "observed_safe_detection_count_range": [14, 15],
            "note": (
                "Routing stayed 15/15 in both runs, but the same-provider "
                "verifier accepted one semantic-negation fault in the current "
                "code replication. Do not report the prior 15/15 as stable."
            ),
        },
        "iterations": [
            {
                "label": "p4c-high-overlap-baseline",
                "report": _report_ref(P4C_HIGH_BASELINE),
                "metrics": p4c_high["metrics"],
                "runtime_cost": p4c_high["runtime_cost"],
            },
            {
                "label": "contract-signal-off",
                "report": _report_ref(CONTRACT_OFF),
                "metrics": contract_off["metrics"],
                "runtime_cost": contract_off["runtime_cost"],
            },
            {
                "label": "contract-signal-on",
                "report": _report_ref(CONTRACT_ON),
                "metrics": contract_on["metrics"],
                "runtime_cost": contract_on["runtime_cost"],
            },
            {
                "label": "focused-evidence-negative-cost-experiment",
                "report": _report_ref(HIGH_COST_FINAL),
                "metrics": high_cost["metrics"],
                "runtime_cost": high_cost["runtime_cost"],
                "normal_cost": dict(focused_normal_cost),
                "adopted": False,
            },
            {
                "label": "lightweight-contract-and-sign-gate",
                "report": _report_ref(LIGHTWEIGHT_V2),
                "metrics": lightweight_v2["metrics"],
                "runtime_cost": lightweight_v2["runtime_cost"],
                "adopted": True,
            },
            {
                "label": "final-lightweight-contract-sign-completeness",
                "report": _report_ref(FINAL_HIGH),
                "metrics": final_high["metrics"],
                "runtime_cost": final_high["runtime_cost"],
                "adopted": True,
            },
        ],
        "paired_changed_cases": changed_cases,
        "decision": {
            "adopt_lightweight_contract_and_sign_gate": True,
            "adopt_focused_evidence_duplication": False,
            "add_more_relation_keyword_rules": False,
            "automatic_high_risk_verifier_recommended": True,
            "reason": (
                "High-overlap safety improved from 5/15 to 14/15 while normal "
                "strict accuracy stayed 45/48 and normal requests stayed at 18. "
                "One direction swap still passed; adding a benchmark-specific "
                "direction rule is not justified."
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
        "high_overlap_safe_detection_rate="
        f"{p4c_high['metrics']['safe_detection_rate']:.4f}->"
        f"{final_high['metrics']['safe_detection_rate']:.4f}"
    )
    print(f"report={OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
