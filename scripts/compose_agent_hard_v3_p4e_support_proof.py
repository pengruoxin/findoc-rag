"""Compose P4-E reports and summarize bounded support-proof experiments."""

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
            "reports/agent/agent-hard-v3-calibration-deepseek-p4d-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4e-bounded-proof-context-v7.json"
        ),
        "off": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4e-support-proof-ablation-off-v1.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-p4e-composed.json"
        ),
    },
    "dev": {
        "dataset": Path("data/evaluation/agent-hard-v3-dev.json"),
        "baseline": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p4d-composed.json"
        ),
        "targeted": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-"
            "p4e-bounded-proof-context-posthoc-v7.json"
        ),
        "off": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-"
            "p4e-support-proof-ablation-off-posthoc-v1.json"
        ),
        "output": Path(
            "reports/agent/agent-hard-v3-dev-deepseek-p4e-composed.json"
        ),
    },
}
HIGH_OFF = Path(
    "reports/agent/agent-hard-v3-deepseek-"
    "p4e-support-proof-ablation-off-high-overlap-v1.json"
)
HIGH_ON = Path(
    "reports/agent/agent-hard-v3-deepseek-"
    "p4e-bounded-proof-context-high-overlap-v7.json"
)
KNOWN = Path(
    "reports/agent/agent-hard-v3-calibration-deepseek-p4e-known-faults-v2.json"
)
OPEN = Path(
    "reports/agent/agent-hard-v3-deepseek-p4e-open-risk-regression-v1.json"
)
ITERATIONS = [
    (
        "all-requirements-proof-v1",
        Path(
            "reports/agent/agent-hard-v3-deepseek-"
            "p4e-support-proof-high-overlap-v1.json"
        ),
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4e-support-proof-v1.json"
        ),
        False,
    ),
    (
        "normalized-quotes-with-challenge-v2",
        Path(
            "reports/agent/agent-hard-v3-deepseek-"
            "p4e-support-proof-challenge-high-overlap-v2.json"
        ),
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4e-support-proof-challenge-v2.json"
        ),
        False,
    ),
    (
        "bounded-weakest-contract-v3",
        Path(
            "reports/agent/agent-hard-v3-deepseek-"
            "p4e-bounded-support-proof-high-overlap-v3.json"
        ),
        Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4e-bounded-proof-v3.json"
        ),
        False,
    ),
    (
        "finding-wins-normalization-v4",
        Path(
            "reports/agent/agent-hard-v3-deepseek-"
            "p4e-bounded-proof-normalized-high-overlap-v4.json"
        ),
        None,
        False,
    ),
    (
        "manual-fallback-v5",
        Path(
            "reports/agent/agent-hard-v3-deepseek-"
            "p4e-bounded-proof-manual-fallback-high-overlap-v5.json"
        ),
        Path(
            "reports/agent/agent-hard-v3-dev-deepseek-"
            "p4e-bounded-proof-posthoc-v4.json"
        ),
        False,
    ),
    (
        "bounded-proof-context-retry-v7",
        HIGH_ON,
        SPLITS["calibration"]["targeted"],
        True,
    ),
]
OUTPUT = Path("reports/agent/agent-hard-v3-p4e-support-proof-improvement.json")


def _report_ref(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": _sha256(path)}


def _normal_cost(paths: list[Path]) -> Counter:
    cost = Counter()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        cost.update(_verifier_cost(report))
    return cost


def _cost_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, Any]:
    token_delta = after["model_total_tokens"] - before["model_total_tokens"]
    return {
        "model_requests": after["model_requests"] - before["model_requests"],
        "model_total_tokens": token_delta,
        "model_total_token_change_rate": (
            token_delta / before["model_total_tokens"]
            if before["model_total_tokens"]
            else 0.0
        ),
    }


def _iteration_summary(
    label: str,
    fault_path: Path,
    normal_path: Path | None,
    adopted: bool,
) -> dict[str, Any]:
    fault = json.loads(fault_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "label": label,
        "fault_report": _report_ref(fault_path),
        "fault_metrics": fault["metrics"],
        "fault_runtime_cost": fault["runtime_cost"],
        "adopted": adopted,
    }
    if normal_path is not None:
        normal = json.loads(normal_path.read_text(encoding="utf-8"))
        result.update(
            {
                "normal_report": _report_ref(normal_path),
                "normal_metrics": normal["metrics"],
                "normal_verifier_cost": _verifier_cost(normal),
            }
        )
    return result


def main() -> None:
    split_summaries: dict[str, Any] = {}
    combined_before = 0
    combined_after = 0
    combined_cases = 0
    changed_cases: list[dict[str, Any]] = []
    on_cost = Counter()
    decisions = Counter()
    gate_statuses = Counter()
    gate_findings = Counter()
    prompt_revisions: set[str] = set()

    for split, paths in SPLITS.items():
        dataset = json.loads(paths["dataset"].read_text(encoding="utf-8"))
        baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))
        targeted = json.loads(paths["targeted"].read_text(encoding="utf-8"))
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
            replacements.get(item["case_id"], item) for item in baseline["items"]
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
            "evaluation_mode": "composed_p4e_bounded_support_proof",
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
        on_cost.update(split_cost)
        decisions.update(split_decisions)
        gate_statuses.update(split_gate_statuses)
        gate_findings.update(split_gate_findings)
        for item in targeted["items"]:
            verification = (item.get("trace") or {}).get("evidence_verification")
            if verification:
                prompt_revisions.add(verification["prompt_revision"])
        split_summaries[split] = {
            "dataset_sha256": _sha256(paths["dataset"]),
            "strict_case_pass_count_before": before,
            "strict_case_pass_count_after": after,
            "strict_case_pass_count_delta": after - before,
            "before_metrics": baseline["metrics"],
            "after_metrics": metrics,
            "verifier_decisions": dict(split_decisions),
            "risk_gate_statuses": dict(split_gate_statuses),
            "risk_gate_findings": dict(split_gate_findings),
            "incremental_verifier_cost": dict(split_cost),
            "composed_report": paths["output"].as_posix(),
            "composed_report_sha256": _sha256(paths["output"]),
        }

    off_cost = _normal_cost([paths["off"] for paths in SPLITS.values()])
    high_off = json.loads(HIGH_OFF.read_text(encoding="utf-8"))
    high_on = json.loads(HIGH_ON.read_text(encoding="utf-8"))
    known = json.loads(KNOWN.read_text(encoding="utf-8"))
    open_risk = json.loads(OPEN.read_text(encoding="utf-8"))
    summary = {
        "schema_version": "1",
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "change_id": "p4e-bounded-atomic-support-proof",
        "change_scope": (
            "For non-audit verified extracts, require a verbatim support proof "
            "for the atomic requirement with the weakest claim/contract match. "
            "Low proof-language alignment gets one contrastive challenge; invalid "
            "proofs escalate to manual review and partition errors get one retry."
        ),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "same_provider_independent_verifier": False,
        "proof_selection": "one weakest claim/requirement contract per non-audit trace",
        "proof_challenge_language_coverage_below": 0.90,
        "prompt_revisions": sorted(prompt_revisions),
        "splits": split_summaries,
        "combined": {
            "case_count": combined_cases,
            "strict_case_pass_count_before": combined_before,
            "strict_case_pass_rate_before": combined_before / combined_cases,
            "strict_case_pass_count_after": combined_after,
            "strict_case_pass_rate_after": combined_after / combined_cases,
            "strict_case_pass_count_delta": combined_after - combined_before,
            "proof_off_normal_verifier_cost": dict(off_cost),
            "proof_on_normal_verifier_cost": dict(on_cost),
            "normal_cost_delta": _cost_delta(dict(off_cost), dict(on_cost)),
            "normal_answer_verifier_decisions": dict(decisions),
            "risk_gate_statuses": dict(gate_statuses),
            "risk_gate_findings": dict(gate_findings),
        },
        "high_overlap_ablation": {
            "proof_off_report": _report_ref(HIGH_OFF),
            "proof_off_metrics": high_off["metrics"],
            "proof_off_runtime_cost": high_off["runtime_cost"],
            "proof_on_report": _report_ref(HIGH_ON),
            "proof_on_metrics": high_on["metrics"],
            "proof_on_runtime_cost": high_on["runtime_cost"],
            "fixed_unsafe_case_ids": [
                item["mutation_id"]
                for item in high_off["items"]
                if item["unsafe_accept"]
                and any(
                    candidate["mutation_id"] == item["mutation_id"]
                    and candidate["safe_detection"]
                    for candidate in high_on["items"]
                )
            ],
        },
        "known_fault_regression": {
            "report": _report_ref(KNOWN),
            "metrics": known["metrics"],
            "runtime_cost": known["runtime_cost"],
        },
        "open_risk_regression": {
            "report": _report_ref(OPEN),
            "metrics": open_risk["metrics"],
            "runtime_cost": open_risk["runtime_cost"],
        },
        "iterations": [
            _iteration_summary(label, fault, normal, adopted)
            for label, fault, normal, adopted in ITERATIONS
        ],
        "paired_changed_cases": changed_cases,
        "decision": {
            "adopt_bounded_support_proof_for_non_audit_verifier_routes": True,
            "require_proof_for_every_requirement": False,
            "treat_manual_review_as_automatic_success": False,
            "reason": (
                "The same-code ablation fixed the remaining high-overlap relation "
                "swap (14/15 to 15/15) with no normal strict regression. Normal "
                "verifier cost rose by about 19%, so proof is bounded to one weak "
                "atomic contract and existing audit reviews keep the compact path."
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
        f"{high_off['metrics']['safe_detection_rate']:.4f}->"
        f"{high_on['metrics']['safe_detection_rate']:.4f}"
    )
    print(f"report={OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
