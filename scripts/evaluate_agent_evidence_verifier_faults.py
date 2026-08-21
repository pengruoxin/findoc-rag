"""Inject claim-level faults into stored traces and measure verifier behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from findoc_rag.agent_tasks import AgentTaskTrace
from findoc_rag.deepseek_agent import DeepSeekToolCallingModel
from findoc_rag.evidence_verifier import EvidenceVerifierAgent

DEFAULT_CASE_IDS = (
    "v3_601398_y23_core",
    "v3_601398_y23_asset_quality",
    "v3_601398_y24_core",
    "v3_002594_y23_core",
    "v3_002594_y23_products",
)
NUMBER_PATTERN = re.compile(r"(?<![\d])\d[\d,.]*(?![\d])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p3b2-authority-ranking-extract-v4.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/agent/agent-hard-v3-calibration-deepseek-"
            "p4a-evidence-verifier-faults-v1.json"
        ),
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--disable-support-proof", action="store_true")
    parser.add_argument("--require-remote", action="store_true")
    return parser.parse_args()


def _different_number(value: str) -> str:
    characters = list(value)
    for index in range(len(characters) - 1, -1, -1):
        if characters[index].isdigit():
            characters[index] = "1" if characters[index] != "1" else "2"
            return "".join(characters)
    raise ValueError("value contains no digit")


def _first_requirement_with_claim(trace: AgentTaskTrace) -> tuple[str, str]:
    for requirement in trace.plan.fact_requirements:
        claims = trace.result.requirement_claims.get(requirement.requirement_id, [])
        if claims:
            return requirement.requirement_id, claims[0]
    raise ValueError("trace has no requirement-bound claim")


def _mutate_numeric(trace: AgentTaskTrace) -> tuple[AgentTaskTrace, dict]:
    for requirement in trace.plan.fact_requirements:
        claims = trace.result.requirement_claims.get(requirement.requirement_id, [])
        for claim_index, claim in enumerate(claims):
            for match in NUMBER_PATTERN.finditer(claim):
                value = match.group()
                if value.rstrip(".,") in {"2023", "2024"}:
                    continue
                replacement = _different_number(value)
                mutated_claim = claim[: match.start()] + replacement + claim[match.end() :]
                updated_claims = {
                    key: list(values)
                    for key, values in trace.result.requirement_claims.items()
                }
                updated_claims[requirement.requirement_id][claim_index] = mutated_claim
                result = trace.result.model_copy(
                    update={"requirement_claims": updated_claims}
                )
                return (
                    trace.model_copy(update={"result": result}),
                    {
                        "requirement_id": requirement.requirement_id,
                        "original_claim": claim,
                        "mutated_claim": mutated_claim,
                        "mutation": f"numeric:{value}->{replacement}",
                    },
                )
    raise ValueError("trace has no mutable non-year number")


def _mutate_subject(trace: AgentTaskTrace) -> tuple[AgentTaskTrace, dict]:
    requirement_id, claim = _first_requirement_with_claim(trace)
    company = (
        trace.plan.document_scope.company_names[0]
        if trace.plan.document_scope
        and trace.plan.document_scope.company_names
        else ""
    )
    wrong_company = "工商银行" if company == "比亚迪" else "比亚迪"
    mutated_claim = (
        claim.replace(company, wrong_company, 1)
        if company and company in claim
        else f"{wrong_company}：{claim}"
    )
    updated_claims = {
        key: list(values) for key, values in trace.result.requirement_claims.items()
    }
    updated_claims[requirement_id][0] = mutated_claim
    return (
        trace.model_copy(
            update={
                "result": trace.result.model_copy(
                    update={"requirement_claims": updated_claims}
                )
            }
        ),
        {
            "requirement_id": requirement_id,
            "original_claim": claim,
            "mutated_claim": mutated_claim,
            "mutation": "subject:wrong_company",
        },
    )


def _mutate_omission(trace: AgentTaskTrace) -> tuple[AgentTaskTrace, dict]:
    delimiters = ("；", "，", ";", ",")
    candidates: list[tuple[int, str, int, str, str]] = []
    for requirement in trace.plan.fact_requirements:
        for claim_index, claim in enumerate(
            trace.result.requirement_claims.get(requirement.requirement_id, [])
        ):
            for delimiter in delimiters:
                for match in re.finditer(re.escape(delimiter), claim):
                    if (
                        delimiter == ","
                        and match.start() > 0
                        and match.end() < len(claim)
                        and claim[match.start() - 1].isdigit()
                        and claim[match.end()].isdigit()
                    ):
                        continue
                    shortened = (
                        claim[: match.start()].rstrip("。；;,，") + "。"
                    )
                    removed = len(claim) - len(shortened)
                    if removed >= 8:
                        candidates.append(
                            (
                                removed,
                                requirement.requirement_id,
                                claim_index,
                                claim,
                                shortened,
                            )
                        )
    if not candidates:
        for requirement in trace.plan.fact_requirements:
            for claim_index, claim in enumerate(
                trace.result.requirement_claims.get(
                    requirement.requirement_id, []
                )
            ):
                for match in NUMBER_PATTERN.finditer(claim):
                    value = match.group()
                    if value.rstrip(".,") in {"2023", "2024"}:
                        continue
                    shortened = claim.replace(value, "").strip()
                    candidates.append(
                        (
                            len(value),
                            requirement.requirement_id,
                            claim_index,
                            claim,
                            shortened,
                        )
                    )
                    break
            if candidates:
                break
    if not candidates:
        raise ValueError("trace has no fact detail suitable for omission")
    _, requirement_id, claim_index, original_claim, mutated_claim = max(candidates)
    updated_claims = {
        key: list(values) for key, values in trace.result.requirement_claims.items()
    }
    updated_claims[requirement_id][claim_index] = mutated_claim
    return (
        trace.model_copy(
            update={
                "result": trace.result.model_copy(
                    update={"requirement_claims": updated_claims}
                )
            }
        ),
        {
            "requirement_id": requirement_id,
            "original_claim": original_claim,
            "mutated_claim": mutated_claim,
            "mutation": "omission:truncate_after_first_clause",
        },
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    source_by_case = {item["case_id"]: item for item in source["items"]}
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
            "source_report": args.source_report.as_posix(),
            "source_report_sha256": _sha256(args.source_report),
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
    reviewer = EvidenceVerifierAgent(
        verifier_model,
        optimizer_model=optimizer_model,
        known_companies=sorted(
            {
                item["company_name"]
                for source_item in source.get("items", [])
                if source_item.get("trace")
                for item in source_item["trace"]["evidence_memory"]["items"]
                if item.get("company_name")
            }
        ),
        require_support_proof=not args.disable_support_proof,
    )
    mutation_functions = {
        "numeric_corruption": _mutate_numeric,
        "subject_corruption": _mutate_subject,
        "supported_detail_omission": _mutate_omission,
    }
    items: list[dict] = []
    for case_id in DEFAULT_CASE_IDS:
        source_item = source_by_case.get(case_id)
        if source_item is None or source_item["status"] != "executed":
            raise SystemExit(f"missing executed source trace: {case_id}")
        original = AgentTaskTrace.model_validate(source_item["trace"])
        for mutation_type, mutate in mutation_functions.items():
            mutated, mutation = mutate(original)
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
                    **mutation,
                    "final_decision": final_decision,
                    "safe_detection": safe_detection,
                    "repair_success": final_decision == "accept_repaired",
                    "unsafe_accept": final_decision == "accept_original",
                    "verifier_error": final_decision == "error",
                    "claim_risk_gate": (
                        risk_gate.model_dump(mode="json")
                        if risk_gate is not None
                        else None
                    ),
                    "verification": verification.model_dump(mode="json"),
                }
            )
    metrics = {
        "mutation_count": len(items),
        "safe_detection_rate": sum(item["safe_detection"] for item in items)
        / len(items),
        "unsafe_accept_rate": sum(item["unsafe_accept"] for item in items)
        / len(items),
        "verifier_error_rate": sum(item["verifier_error"] for item in items)
        / len(items),
        "manual_review_rate": sum(
            item["final_decision"] == "manual_review" for item in items
        )
        / len(items),
        "omission_repair_rate": (
            sum(
                item["repair_success"]
                for item in items
                if item["mutation_type"] == "supported_detail_omission"
            )
            / sum(
                item["mutation_type"] == "supported_detail_omission"
                for item in items
            )
        ),
    }
    cost = Counter()
    for item in items:
        verification = item["verification"]
        cost["model_requests"] += verification["request_count"]
        cost["input_tokens"] += verification.get("input_tokens") or 0
        cost["output_tokens"] += verification.get("output_tokens") or 0
    cost["total_tokens"] = cost["input_tokens"] + cost["output_tokens"]
    report = {
        "schema_version": "1",
        "status": "complete",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "evaluation": "stored-trace claim fault injection",
        "support_proof_required": not args.disable_support_proof,
        "source_report": args.source_report.as_posix(),
        "source_report_sha256": _sha256(args.source_report),
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
    print(f"safe_detection_rate={metrics['safe_detection_rate']:.4f}")
    print(f"unsafe_accept_rate={metrics['unsafe_accept_rate']:.4f}")
    print(f"omission_repair_rate={metrics['omission_repair_rate']:.4f}")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
