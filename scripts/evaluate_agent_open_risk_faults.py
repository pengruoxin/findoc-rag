"""Measure verifier handling of semantic faults absent from hard local rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from findoc_rag.agent_tasks import AgentTaskTrace
from findoc_rag.deepseek_agent import DeepSeekToolCallingModel
from findoc_rag.evidence_verifier import EvidenceVerifierAgent

SOURCE_REPORTS = (
    Path(
        "reports/agent/agent-hard-v3-calibration-deepseek-"
        "p3b2-authority-ranking-extract-v4.json"
    ),
    Path(
        "reports/agent/agent-hard-v3-dev-deepseek-"
        "p3b2-authority-ranking-extract-posthoc-v3.json"
    ),
)
CASE_IDS = (
    "v3_601318_y23_segments",
    "v3_601318_y24_customer",
    "v3_601398_y23_asset_quality",
    "v3_601398_y24_interest",
    "v3_002594_y23_products",
)
MUTATIONS = {
    "semantic_negation": "，但上述事实实际并不存在。",
    "unsupported_causal_inference": "，因此必然导致下一年度利润下降。",
    "scope_qualifier_corruption": (
        "，且该数据仅为境外零售业务口径，不包含任何境内业务。"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/agent/agent-hard-v3-deepseek-p4c-open-risk-faults-v1.json"
        ),
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--disable-open-risk-signal", action="store_true")
    parser.add_argument("--disable-support-proof", action="store_true")
    parser.add_argument("--require-remote", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_requirement_with_claim(trace: AgentTaskTrace) -> tuple[str, str]:
    for requirement in trace.plan.fact_requirements:
        claims = trace.result.requirement_claims.get(requirement.requirement_id, [])
        if claims:
            return requirement.requirement_id, claims[0]
    raise ValueError("trace has no requirement-bound claim")


def _mutate(
    trace: AgentTaskTrace,
    suffix: str,
) -> tuple[AgentTaskTrace, str, str, str]:
    requirement_id, original_claim = _first_requirement_with_claim(trace)
    mutated_claim = original_claim.rstrip("。") + suffix
    updated_claims = {
        key: list(values) for key, values in trace.result.requirement_claims.items()
    }
    updated_claims[requirement_id][0] = mutated_claim
    result = trace.result.model_copy(
        update={"requirement_claims": updated_claims}
    )
    return (
        trace.model_copy(update={"result": result}),
        requirement_id,
        original_claim,
        mutated_claim,
    )


def _breakdown(items: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for mutation_type in MUTATIONS:
        selected = [
            item
            for item in items
            if item["mutation_type"] == mutation_type
        ]
        result[mutation_type] = {
            "case_count": len(selected),
            "safe_detection_count": sum(
                item["safe_detection"] for item in selected
            ),
            "routed_count": sum(item["routed"] for item in selected),
            "final_decisions": dict(
                Counter(item["final_decision"] for item in selected)
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    sources = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in SOURCE_REPORTS
    ]
    source_by_case = {
        item["case_id"]: item
        for source in sources
        for item in source["items"]
    }
    verifier_model = DeepSeekToolCallingModel(
        model=args.model,
        endpoint=args.endpoint,
    )
    optimizer_model = DeepSeekToolCallingModel(
        model=args.model,
        endpoint=args.endpoint,
    )
    if not verifier_model.available or not optimizer_model.available:
        report = {
            "schema_version": "1",
            "status": "not_run",
            "reason": "missing_provider_api_key",
            "source_reports": [path.as_posix() for path in SOURCE_REPORTS],
            "items": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("status=not_run")
        print(f"report={args.output.resolve()}")
        if args.require_remote:
            raise SystemExit(2)
        return
    known_companies = sorted(
        {
            evidence["company_name"]
            for source in sources
            for source_item in source["items"]
            if source_item.get("trace")
            for evidence in source_item["trace"]["evidence_memory"]["items"]
            if evidence.get("company_name")
        }
    )
    reviewer = EvidenceVerifierAgent(
        verifier_model,
        optimizer_model=optimizer_model,
        known_companies=known_companies,
        enable_open_language_risk=not args.disable_open_risk_signal,
        require_support_proof=not args.disable_support_proof,
    )
    items: list[dict] = []
    for case_id in CASE_IDS:
        source_item = source_by_case.get(case_id)
        if source_item is None or source_item["status"] != "executed":
            raise SystemExit(f"missing executed source trace: {case_id}")
        original = AgentTaskTrace.model_validate(source_item["trace"])
        for mutation_type, suffix in MUTATIONS.items():
            mutated, requirement_id, original_claim, mutated_claim = _mutate(
                original, suffix
            )
            reviewed = reviewer.review(mutated)
            verification = reviewed.evidence_verification
            assert verification is not None
            risk_gate = reviewed.claim_risk_gate
            local_reject = bool(risk_gate and risk_gate.status == "reject")
            final_decision = (
                "local_gate_rejected"
                if local_reject
                else verification.final_decision
            )
            safe_detection = (
                local_reject
                or final_decision
                in {"accept_repaired", "abstain", "manual_review"}
            )
            items.append(
                {
                    "mutation_id": f"{case_id}:{mutation_type}",
                    "case_id": case_id,
                    "mutation_type": mutation_type,
                    "requirement_id": requirement_id,
                    "original_claim": original_claim,
                    "mutated_claim": mutated_claim,
                    "source_requirement_count": len(
                        original.plan.fact_requirements
                    ),
                    "routed": verification.routed,
                    "route_reason": verification.route_reason,
                    "final_decision": final_decision,
                    "safe_detection": safe_detection,
                    "unsafe_accept": final_decision
                    in {"accept_original", "not_routed"},
                    "claim_risk_gate": (
                        risk_gate.model_dump(mode="json")
                        if risk_gate is not None
                        else None
                    ),
                    "verification": verification.model_dump(mode="json"),
                }
            )
    cost = Counter()
    for item in items:
        verification = item["verification"]
        cost["model_requests"] += verification["request_count"]
        cost["input_tokens"] += verification.get("input_tokens") or 0
        cost["output_tokens"] += verification.get("output_tokens") or 0
    cost["total_tokens"] = cost["input_tokens"] + cost["output_tokens"]
    metrics = {
        "mutation_count": len(items),
        "routed_rate": sum(item["routed"] for item in items) / len(items),
        "safe_detection_rate": sum(item["safe_detection"] for item in items)
        / len(items),
        "unsafe_accept_rate": sum(item["unsafe_accept"] for item in items)
        / len(items),
        "fail_closed_rate": sum(
            item["safe_detection"] or item["final_decision"] == "error"
            for item in items
        )
        / len(items),
        "verifier_error_rate": sum(
            item["final_decision"] == "error" for item in items
        )
        / len(items),
        "manual_review_rate": sum(
            item["final_decision"] == "manual_review" for item in items
        )
        / len(items),
        "breakdown": _breakdown(items),
    }
    report = {
        "schema_version": "1",
        "status": "complete",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "evaluation": "stored-trace open-set semantic fault injection",
        "fault_design": (
            "Semantic negation, unsupported causal inference, and scope "
            "qualifier corruption do not change company, explicit year, "
            "number, unit, or citation document scope."
        ),
        "open_risk_signal_enabled": not args.disable_open_risk_signal,
        "support_proof_required": not args.disable_support_proof,
        "source_reports": [
            {
                "path": path.as_posix(),
                "sha256": _sha256(path),
            }
            for path in SOURCE_REPORTS
        ],
        "provider": {
            "name": verifier_model.provider,
            "model": verifier_model.model,
            "endpoint": verifier_model.endpoint,
        },
        "metrics": metrics,
        "runtime_cost": dict(cost),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("status=complete")
    print(f"routed_rate={metrics['routed_rate']:.4f}")
    print(f"safe_detection_rate={metrics['safe_detection_rate']:.4f}")
    print(f"unsafe_accept_rate={metrics['unsafe_accept_rate']:.4f}")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
