"""Create a reproducible assistant audit of the hard-v3 DeepSeek baseline."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _audit(
    verdict: str,
    category: str,
    reason: str,
    *,
    facts: tuple[str, ...] = (),
    citation_coverage: float | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "failure_category": category,
        "reason": reason,
        "matched_fact_overrides": list(facts),
        "citation_coverage_override": citation_coverage,
    }


AUDITS: dict[str, dict[str, Any]] = {
    "v3_601398_y23_ecl_audit": _audit(
        "genuine_agent_failure",
        "multi_fact_omission",
        "同义表述覆盖多个被机器漏判的事实，但回答仍遗漏经济情景及其权重。",
        facts=(
            "risk_ead",
            "audit_controls",
            "audit_models",
            "audit_it",
            "audit_collateral",
        ),
    ),
    "v3_601398_y24_interest": _audit(
        "evaluator_false_negative",
        "gold_matcher_false_negative",
        "回答明确包含LPR下调和存款期限结构变动，机器被括号和同义词阻断。",
        facts=("reason_lpr", "reason_deposit"),
    ),
    "v3_601398_trend_core": _audit(
        "genuine_agent_failure",
        "multi_document_retrieval_failure",
        "比较代理没有取得两期四项核心数据并错误拒答。",
    ),
    "v3_601398_trend_asset_quality": _audit(
        "genuine_agent_failure",
        "multi_document_retrieval_failure",
        "比较代理在模型预算内未形成通过门禁的答案。",
    ),
    "v3_601398_calc_profit_growth": _audit(
        "genuine_agent_failure",
        "unsupported_general_calculation",
        "通用带来源计算未被路由到可执行控制器。",
    ),
    "v3_601398_calc_loan_deposit_gap": _audit(
        "genuine_agent_failure",
        "unsupported_general_calculation",
        "通用带来源计算未被路由到可执行控制器。",
    ),
    "v3_601398_verify_quality": _audit(
        "evaluator_false_negative",
        "gold_matcher_false_negative",
        "回答给出两期数值并明确判断不良率下降、资本充足率上升，语义结论成立。",
        facts=("verdict",),
    ),
    "v3_601398_future_unavailable": _audit(
        "genuine_agent_failure",
        "future_fact_behavior_confusion",
        "已明确公司且事实超出语料年份，应安全拒答而不是要求补充公司和报告年度。",
    ),
    "v3_002594_calc_profit_margin": _audit(
        "genuine_agent_failure",
        "unsupported_general_calculation",
        "通用带来源计算未被路由到可执行控制器。",
    ),
    "v3_002594_calc_product_share": _audit(
        "genuine_agent_failure",
        "unsupported_general_calculation",
        "通用带来源计算未被路由到可执行控制器。",
    ),
    "v3_002594_verify_growth": _audit(
        "evaluator_false_negative",
        "gold_matcher_false_negative",
        "回答明确说明两项均增长且净利润增速快于收入增速。",
        facts=("verdict",),
    ),
    "v3_002594_future_unavailable": _audit(
        "genuine_agent_failure",
        "future_fact_behavior_confusion",
        "已明确公司且事实超出语料年份，应安全拒答而不是要求澄清。",
    ),
    "v3_601318_y23_core": _audit(
        "evaluator_false_negative",
        "citation_whitelist_false_negative",
        "引用的同一份年报第16、17、49页直接支持答案，只是不在原金标页白名单。",
        citation_coverage=1.0,
    ),
    "v3_601318_y23_audit": _audit(
        "evaluator_false_negative",
        "gold_matcher_false_negative",
        "责任单元、评估主要假设和独立计算代表性合同组均与金标语义一致。",
        facts=("risk_unit", "audit_assumptions", "audit_recalculation"),
    ),
    "v3_601318_y24_core": _audit(
        "genuine_agent_failure",
        "metric_scope_confusion",
        "回答把寿险分部营运利润和净利润当成集团归母口径。",
    ),
    "v3_601318_trend_core": _audit(
        "genuine_agent_failure",
        "multi_document_retrieval_failure",
        "回答只覆盖部分比较项并错误拒答。",
    ),
    "v3_601318_trend_customer": _audit(
        "evaluator_false_negative",
        "gold_matcher_false_negative",
        "回答明确写出客均合同数由2.95降至2.92，方向事实完整。",
        facts=("direction",),
    ),
    "v3_601318_calc_nbv_growth": _audit(
        "genuine_agent_failure",
        "unsupported_general_calculation",
        "通用带来源计算未被路由到可执行控制器。",
    ),
    "v3_601318_calc_segment_share": _audit(
        "genuine_agent_failure",
        "unsupported_general_calculation",
        "通用带来源计算未被路由到可执行控制器。",
    ),
    "v3_601318_verify_nbv": _audit(
        "evaluator_false_negative",
        "citation_whitelist_and_matcher_false_negative",
        "同一份年报第65、66页直接支持全部数值，回答也明确指出原说法不成立。",
        facts=("verdict",),
        citation_coverage=1.0,
    ),
    "v3_601318_future_unavailable": _audit(
        "genuine_agent_failure",
        "future_fact_behavior_confusion",
        "已明确公司且事实超出语料年份，应安全拒答而不是要求澄清。",
    ),
    "v3_300750_y24_products": _audit(
        "evaluator_false_negative",
        "gold_matcher_false_negative",
        "回答用“减少”表达金标中的“下降”，三项同比数值均正确。",
        facts=("power_yoy", "storage_yoy", "material_yoy"),
    ),
    "v3_300750_trend_core": _audit(
        "evaluator_false_negative",
        "gold_matcher_false_negative",
        "回答逐项明确营业收入下降、净利润和经营现金流上升。",
        facts=("direction",),
    ),
    "v3_300750_calc_storage_share": _audit(
        "genuine_agent_failure",
        "unsupported_general_calculation",
        "通用带来源计算未被路由到可执行控制器。",
    ),
    "v3_300750_calc_net_margin": _audit(
        "genuine_agent_failure",
        "unsupported_general_calculation",
        "通用带来源计算未被路由到可执行控制器。",
    ),
    "v3_300750_future_unavailable": _audit(
        "genuine_agent_failure",
        "future_fact_behavior_confusion",
        "已明确公司且事实超出语料年份，应安全拒答而不是要求澄清。",
    ),
}


SPLITS = {
    "calibration": (
        Path("data/evaluation/agent-hard-v3-calibration.json"),
        Path("reports/agent/agent-hard-v3-calibration-deepseek-baseline-rescored.json"),
    ),
    "dev": (
        Path("data/evaluation/agent-hard-v3-dev.json"),
        Path("reports/agent/agent-hard-v3-dev-deepseek-baseline-rescored.json"),
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    output = Path("reports/agent/agent-hard-v3-deepseek-baseline-analysis.json")
    split_payloads: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    all_failures: set[str] = set()
    total_cost = Counter()

    for split, (dataset_path, report_path) in SPLITS.items():
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cases = {case["case_id"]: case for case in dataset["cases"]}
        rows: list[dict[str, Any]] = []
        for item in report["items"]:
            case = cases[item["case_id"]]
            score = item.get("score") or {}
            machine_pass = bool(score.get("case_pass", False))
            if not machine_pass:
                all_failures.add(item["case_id"])
            audit = AUDITS.get(item["case_id"])
            matched = {
                fact["fact_id"]: bool(fact["matched"])
                for fact in score.get("fact_scores", [])
            }
            if audit:
                for fact_id in audit["matched_fact_overrides"]:
                    matched[fact_id] = True
            expected_fact_ids = [fact["fact_id"] for fact in case["expected_facts"]]
            fact_complete = all(matched.get(fact_id, False) for fact_id in expected_fact_ids)
            citation_coverage = score.get("citation_source_coverage", 0.0)
            if audit and audit["citation_coverage_override"] is not None:
                citation_coverage = audit["citation_coverage_override"]
            citation_gate = (
                bool(score.get("citation_integrity")) and citation_coverage == 1.0
                if case["expected_behavior"] == "answer"
                else True
            )
            audited_pass = bool(
                item["status"] == "executed"
                and score.get("behavior_correct")
                and score.get("plan_target_exact")
                and fact_complete
                and citation_gate
            )
            row = {
                "split": split,
                "case_id": item["case_id"],
                "task_type": case["task_type"],
                "challenge_types": case["challenge_types"],
                "runtime_status": item["status"],
                "machine_case_pass": machine_pass,
                "audited_case_pass": audited_pass,
                "machine_fact_accuracy": score.get("fact_accuracy"),
                "audited_missing_fact_ids": [
                    fact_id
                    for fact_id in expected_fact_ids
                    if not matched.get(fact_id, False)
                ],
                "machine_citation_source_coverage": score.get(
                    "citation_source_coverage"
                ),
                "audited_citation_source_coverage": citation_coverage,
                "manual_audit": audit,
            }
            rows.append(row)
            all_rows.append(row)

        executed = [item for item in report["items"] if item["status"] == "executed"]
        traces = [item["trace"].get("model_trace") for item in executed]
        traces = [trace for trace in traces if trace]
        calls = [call for item in executed for call in item["trace"]["tool_calls"]]
        cost = {
            "model_requests": sum(trace["request_count"] for trace in traces),
            "model_input_tokens": sum(trace.get("input_tokens") or 0 for trace in traces),
            "model_output_tokens": sum(trace.get("output_tokens") or 0 for trace in traces),
            "tool_calls": len(calls),
            "tool_duration_ms": sum(call.get("duration_ms") or 0 for call in calls),
        }
        cost["model_total_tokens"] = cost["model_input_tokens"] + cost["model_output_tokens"]
        total_cost.update(cost)
        machine_passes = sum(row["machine_case_pass"] for row in rows)
        audited_passes = sum(row["audited_case_pass"] for row in rows)
        split_payloads[split] = {
            "dataset": dataset_path.as_posix(),
            "dataset_sha256": _sha256(dataset_path),
            "report": report_path.as_posix(),
            "report_sha256": _sha256(report_path),
            "index_id": report["index_id"],
            "raw_machine_metrics": report["metrics"],
            "assistant_audited_case_pass_count": audited_passes,
            "assistant_audited_case_pass_rate": _rate(audited_passes, len(rows)),
            "machine_case_pass_count": machine_passes,
            "runtime_cost": cost,
        }

    errors = []
    if all_failures != set(AUDITS):
        errors.append(
            "audit coverage mismatch: "
            f"missing={sorted(all_failures - set(AUDITS))}; "
            f"extra={sorted(set(AUDITS) - all_failures)}"
        )
    machine_passes = sum(row["machine_case_pass"] for row in all_rows)
    audited_passes = sum(row["audited_case_pass"] for row in all_rows)
    verdict_counts = Counter(
        audit["verdict"] for audit in AUDITS.values()
    )
    genuine_categories = Counter(
        audit["failure_category"]
        for audit in AUDITS.values()
        if audit["verdict"] == "genuine_agent_failure"
    )
    payload = {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "complete" if not errors else "invalid",
        "evaluation_scope": "calibration + dev only; frozen_test remains sealed",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "scoring_note": (
            "Machine metrics use deterministic local scoring after the generic "
            "percentage-boundary correction. Assistant audit is separate and provisional."
        ),
        "external_claims_ready": False,
        "external_claim_blockers": [
            "gold remains assistant-source-verified provisional",
            "independent human double review is incomplete",
            "frozen_test was not opened",
        ],
        "splits": split_payloads,
        "combined": {
            "case_count": len(all_rows),
            "machine_case_pass_count": machine_passes,
            "machine_case_pass_rate": _rate(machine_passes, len(all_rows)),
            "assistant_audited_case_pass_count": audited_passes,
            "assistant_audited_case_pass_rate": _rate(audited_passes, len(all_rows)),
            "machine_failure_count": len(all_failures),
            "manual_audit_verdict_counts": dict(sorted(verdict_counts.items())),
            "genuine_failure_category_counts": dict(sorted(genuine_categories.items())),
            "runtime_cost": dict(total_cost),
        },
        "next_single_variable_recommendation": {
            "id": "p0-general-grounded-calculation-routing",
            "change": "让现有通用计算代理支持带来源的跨文档派生计算，不改检索和答案评分。",
            "direct_target_case_count": 8,
            "expected_primary_metric": "task_coverage_rate",
            "current_combined_task_coverage_rate": 40 / 48,
        },
        "cases": all_rows,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"status={payload['status']}")
    print(f"machine_case_pass={machine_passes}/{len(all_rows)}")
    print(f"assistant_audited_case_pass={audited_passes}/{len(all_rows)}")
    print(f"output={output.resolve()}")


if __name__ == "__main__":
    main()
