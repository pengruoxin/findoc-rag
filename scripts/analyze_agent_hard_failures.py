"""Analyze the frozen Agent hard-v2 baseline without changing its raw scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# This audit is intentionally coupled to the first frozen hard-v2 baseline.  It records
# source-backed semantic corrections separately from the immutable machine score.
MANUAL_AUDITS: dict[str, dict[str, Any]] = {
    "blind_moutai_cashflow_change": {
        "verdict": "evaluator_false_negative",
        "failure_category": "gold_matcher_false_negative",
        "matched_fact_overrides": ["cashflow_reason"],
        "audited_case_pass": True,
        "reason": (
            "回答中的两项增长原因与年报第8-9页原文语义一致；自动评分只接受压缩后的"
            "金标短语，未接受完整同义表述。"
        ),
    },
    "blind_yili_inventory_policy": {
        "verdict": "genuine_agent_failure",
        "failure_category": "planner_missing_document_scope_inference",
        "audited_case_pass": False,
        "reason": "公司唯一且问题不依赖事实年度，规划器仍因缺少显式年份而错误要求澄清。",
    },
    "blind_yili_cashflow_change": {
        "verdict": "evaluator_false_negative",
        "failure_category": "gold_matcher_false_negative",
        "matched_fact_overrides": ["reason"],
        "audited_case_pass": True,
        "reason": (
            "回答准确复述2025年春节备货和预收经销商货款增加；自动评分未接受完整句式。"
        ),
    },
    "blind_yili_annual_deducted_profit": {
        "verdict": "genuine_agent_failure",
        "failure_category": "unsupported_general_calculation",
        "audited_case_pass": False,
        "reason": "任务需要通用带来源算术，当前 calculate 仅实现P1的两种固定操作。",
    },
    "blind_midea_2023_revenue_yoy": {
        "verdict": "evaluator_false_negative",
        "failure_category": "citation_whitelist_false_negative",
        "audited_citation_source_coverage": 1.0,
        "audited_case_pass": True,
        "reason": (
            "引用第51-52页的营业收入构成表同时包含372,037,280千元和8.18%，"
            "但金标页集合只列第10-11页。"
        ),
    },
    "blind_midea_2023_inventory_increase": {
        "verdict": "genuine_agent_failure",
        "failure_category": "unsupported_general_calculation",
        "audited_case_pass": False,
        "reason": "任务需要通用带来源算术，当前 calculate 仅实现P1的两种固定操作。",
    },
    "blind_midea_2023_key_audit_matter": {
        "verdict": "evaluator_false_negative",
        "failure_category": "gold_matcher_false_negative",
        "matched_fact_overrides": ["kam_reason"],
        "audited_case_pass": True,
        "reason": "回答完整覆盖销售渠道、客户/销量、金额重大和审计资源投入，符合第154页原文。",
    },
    "blind_midea_2023_future_actual": {
        "verdict": "genuine_agent_failure",
        "failure_category": "planner_document_fact_year_confusion",
        "audited_case_pass": False,
        "reason": "问题中的未来事实年份与2023年文档年份并存，规划器错误澄清而未安全拒答。",
    },
    "blind_midea_2024_revenue_yoy": {
        "verdict": "evaluator_false_negative",
        "failure_category": "citation_whitelist_false_negative",
        "audited_citation_source_coverage": 1.0,
        "audited_case_pass": True,
        "reason": (
            "引用第49-50页的营业收入构成表同时包含407,149,600千元和9.44%，"
            "但金标页集合只列第9-10页。"
        ),
    },
    "blind_midea_2024_inventory_increase": {
        "verdict": "genuine_agent_failure",
        "failure_category": "unsupported_general_calculation",
        "audited_case_pass": False,
        "reason": "任务需要通用带来源算术，当前 calculate 仅实现P1的两种固定操作。",
    },
    "blind_midea_2024_key_audit_matter": {
        "verdict": "evaluator_false_negative",
        "failure_category": "gold_matcher_false_negative",
        "matched_fact_overrides": ["kam_reason"],
        "audited_case_pass": True,
        "reason": "回答完整覆盖销售渠道、客户/销量、金额重大和审计资源投入，符合第151页原文。",
    },
    "blind_midea_2024_future_actual": {
        "verdict": "genuine_agent_failure",
        "failure_category": "planner_document_fact_year_confusion",
        "audited_case_pass": False,
        "reason": "问题中的未来事实年份与2024年文档年份并存，规划器错误澄清而未安全拒答。",
    },
    "blind_shenhua_2024_coal_sales_production_gap": {
        "verdict": "genuine_agent_failure",
        "failure_category": "unsupported_general_calculation",
        "audited_case_pass": False,
        "reason": "任务需要通用带来源算术，当前 calculate 仅实现P1的两种固定操作。",
    },
    "blind_shenhua_2024_coal_revenue_recognition": {
        "verdict": "evaluator_false_negative",
        "failure_category": "gold_matcher_false_negative",
        "matched_fact_overrides": ["recognition"],
        "audited_case_pass": True,
        "reason": "回答明确写出客户取得煤炭商品控制权时确认收入，符合第132页原文。",
    },
    "blind_shenhua_2024_future_actual": {
        "verdict": "genuine_agent_failure",
        "failure_category": "planner_document_fact_year_confusion",
        "audited_case_pass": False,
        "reason": "问题中的未来事实年份与2024年文档年份并存，规划器错误澄清而未安全拒答。",
    },
    "blind_midea_2023_2024_revenue_compare": {
        "verdict": "genuine_agent_failure",
        "failure_category": "multi_fact_omission",
        "audited_case_pass": False,
        "reason": "回答遗漏2023年营业收入同比增幅8.18%，只给出2024年同比。",
    },
    "blind_moutai_yili_cashflow_compare": {
        "verdict": "genuine_agent_failure",
        "failure_category": "multi_fact_omission",
        "matched_fact_overrides": ["yili_cashflow_change:reason"],
        "audited_citation_source_coverage": 1.0,
        "audited_case_pass": False,
        "reason": (
            "伊利原因及全部已回答数值都有有效替代页引用，但回答遗漏贵州茅台现金流增长原因。"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/agent-hard-v2.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "reports/agent/agent-hard-v2-deepseek-p2a-baseline.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/agent/agent-hard-v2-p2a-failure-analysis.json"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _answer_payload(item: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    trace = item.get("trace") or {}
    result = trace.get("result") or {}
    answer = result.get("answer") or {}
    if not isinstance(answer, dict):
        return None, []
    return answer.get("answer"), answer.get("citations") or []


def _machine_tags(item: dict[str, Any], expected_behavior: str) -> list[str]:
    tags: list[str] = []
    if item["status"] == "unsupported_task_type":
        tags.append("unsupported_task_type")
    score = item.get("score") or {}
    if score.get("behavior_correct") is False:
        tags.append("behavior_mismatch")
    outcome = ((item.get("trace") or {}).get("result") or {}).get("outcome")
    if outcome == "clarify" and expected_behavior != "clarify":
        tags.append("unexpected_clarification")
    if score.get("fact_accuracy") is not None and score.get("fact_accuracy") < 1:
        tags.append("fact_match_failure")
    if score.get("citation_source_coverage") is not None and (
        score.get("citation_source_coverage") < 1
    ):
        tags.append("citation_gold_page_miss")
    return tags


def _group_rates(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, float | int]]:
    names = sorted(
        {
            name
            for row in rows
            for name in (
                row[key] if isinstance(row[key], list) else [row[key]]
            )
        }
    )
    output: dict[str, dict[str, float | int]] = {}
    for name in names:
        selected = [
            row
            for row in rows
            if name in (row[key] if isinstance(row[key], list) else [row[key]])
        ]
        machine_passes = sum(row["machine_case_pass"] for row in selected)
        audited_passes = sum(row["audited_case_pass"] for row in selected)
        output[name] = {
            "case_count": len(selected),
            "machine_pass_count": machine_passes,
            "machine_pass_rate": _rate(machine_passes, len(selected)) or 0.0,
            "audited_pass_count": audited_passes,
            "audited_pass_rate": _rate(audited_passes, len(selected)) or 0.0,
        }
    return output


def main() -> int:
    args = parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in dataset["cases"]}
    items = {item["case_id"]: item for item in report["items"]}
    errors: list[str] = []
    if set(cases) != set(items):
        errors.append("dataset/report case IDs differ")

    machine_failures = {
        case_id
        for case_id, item in items.items()
        if not (item.get("score") or {}).get("case_pass", False)
    }
    if machine_failures != set(MANUAL_AUDITS):
        errors.append(
            "manual audit does not exactly cover frozen machine failures: "
            f"missing={sorted(machine_failures - set(MANUAL_AUDITS))}; "
            f"extra={sorted(set(MANUAL_AUDITS) - machine_failures)}"
        )

    rows: list[dict[str, Any]] = []
    supported_fact_total = 0
    supported_fact_matches = 0
    audited_supported_fact_matches = 0
    end_to_end_fact_total = 0
    audited_end_to_end_fact_matches = 0

    for case_id, case in cases.items():
        item = items[case_id]
        score = item.get("score") or {}
        audit = MANUAL_AUDITS.get(case_id)
        machine_fact_matches = {
            fact["fact_id"]: bool(fact.get("matched"))
            for fact in score.get("fact_scores") or []
        }
        audited_fact_matches = dict(machine_fact_matches)
        if audit:
            for fact_id in audit.get("matched_fact_overrides", []):
                if fact_id not in {fact["fact_id"] for fact in case["expected_facts"]}:
                    errors.append(f"unknown manual fact override: {case_id}/{fact_id}")
                audited_fact_matches[fact_id] = True

        expected_fact_ids = [fact["fact_id"] for fact in case["expected_facts"]]
        end_to_end_fact_total += len(expected_fact_ids)
        audited_end_to_end_fact_matches += sum(
            audited_fact_matches.get(fact_id, False) for fact_id in expected_fact_ids
        )
        if item["status"] == "executed":
            supported_fact_total += len(expected_fact_ids)
            supported_fact_matches += sum(
                machine_fact_matches.get(fact_id, False)
                for fact_id in expected_fact_ids
            )
            audited_supported_fact_matches += sum(
                audited_fact_matches.get(fact_id, False)
                for fact_id in expected_fact_ids
            )

        machine_case_pass = bool(score.get("case_pass", False))
        audited_case_pass = (
            bool(audit["audited_case_pass"]) if audit else machine_case_pass
        )
        machine_citation_coverage = score.get("citation_source_coverage")
        audited_citation_coverage = (
            audit.get("audited_citation_source_coverage", machine_citation_coverage)
            if audit
            else machine_citation_coverage
        )
        answer, citations = _answer_payload(item)
        rows.append(
            {
                "case_id": case_id,
                "task_type": case["task_type"],
                "challenge_types": case["challenge_types"],
                "expected_behavior": case["expected_behavior"],
                "runtime_status": item["status"],
                "machine_case_pass": machine_case_pass,
                "machine_failure_tags": _machine_tags(
                    item, case["expected_behavior"]
                ),
                "machine_fact_accuracy": score.get("fact_accuracy"),
                "machine_citation_source_coverage": machine_citation_coverage,
                "audited_case_pass": audited_case_pass,
                "audited_matched_fact_ids": [
                    fact_id
                    for fact_id in expected_fact_ids
                    if audited_fact_matches.get(fact_id, False)
                ],
                "audited_missing_fact_ids": [
                    fact_id
                    for fact_id in expected_fact_ids
                    if not audited_fact_matches.get(fact_id, False)
                ],
                "audited_citation_source_coverage": audited_citation_coverage,
                "manual_audit": audit,
                "answer": answer,
                "citation_pages": [
                    [citation.get("page_start"), citation.get("page_end")]
                    for citation in citations
                ],
                "tool_sequence": [
                    call["tool"] for call in (item.get("trace") or {}).get("tool_calls", [])
                ],
            }
        )

    executed = [item for item in items.values() if item["status"] == "executed"]
    model_traces = [
        item["trace"]["model_trace"]
        for item in executed
        if item.get("trace") and item["trace"].get("model_trace")
    ]
    tool_calls = [call for item in executed for call in item["trace"]["tool_calls"]]
    tool_counts = Counter(call["tool"] for call in tool_calls)
    model_requests = sum(trace["request_count"] for trace in model_traces)
    input_tokens = sum(trace.get("input_tokens") or 0 for trace in model_traces)
    output_tokens = sum(trace.get("output_tokens") or 0 for trace in model_traces)
    audited_passes = sum(row["audited_case_pass"] for row in rows)
    audited_supported_passes = sum(
        row["audited_case_pass"]
        for row in rows
        if row["runtime_status"] == "executed"
    )
    verdict_counts = Counter(
        row["manual_audit"]["verdict"]
        for row in rows
        if row["manual_audit"]
    )
    failure_category_counts = Counter(
        row["manual_audit"]["failure_category"]
        for row in rows
        if row["manual_audit"]
        and row["manual_audit"]["verdict"] == "genuine_agent_failure"
    )

    payload = {
        "schema_version": "1",
        "dataset_id": dataset["dataset_id"],
        "analysis_scope": "frozen P2-A baseline; no capability or raw score changes",
        "generated_at": datetime.now(UTC).isoformat(),
        "valid": not errors,
        "ready_for_external_claims": False,
        "external_claim_blockers": [
            "gold and semantic audit are assistant-verified provisional",
            "independent human double review is incomplete",
            "the document-blind set contains only five annual reports",
            "the set contains no true scanned annual report",
        ],
        "inputs": {
            "dataset": args.dataset.as_posix(),
            "dataset_sha256": _sha256(args.dataset),
            "report": args.report.as_posix(),
            "report_sha256": _sha256(args.report),
        },
        "raw_machine_metrics": report["metrics"],
        "assistant_source_audited_metrics": {
            "case_count": len(rows),
            "case_pass_count": audited_passes,
            "case_pass_rate": _rate(audited_passes, len(rows)),
            "executed_case_count": len(executed),
            "executed_case_pass_count": audited_supported_passes,
            "executed_case_pass_rate": _rate(audited_supported_passes, len(executed)),
            "supported_fact_match_count": audited_supported_fact_matches,
            "supported_fact_count": supported_fact_total,
            "supported_fact_accuracy": _rate(
                audited_supported_fact_matches, supported_fact_total
            ),
            "end_to_end_fact_match_count": audited_end_to_end_fact_matches,
            "end_to_end_fact_count": end_to_end_fact_total,
            "end_to_end_fact_accuracy": _rate(
                audited_end_to_end_fact_matches, end_to_end_fact_total
            ),
        },
        "machine_fact_counts": {
            "supported_fact_match_count": supported_fact_matches,
            "supported_fact_count": supported_fact_total,
        },
        "manual_audit_summary": {
            "machine_failure_count": len(machine_failures),
            "reviewed_failure_count": sum(verdict_counts.values()),
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "genuine_failure_category_counts": dict(
                sorted(failure_category_counts.items())
            ),
            "genuine_failure_case_ids": [
                row["case_id"]
                for row in rows
                if row["manual_audit"]
                and row["manual_audit"]["verdict"] == "genuine_agent_failure"
            ],
            "evaluator_false_negative_case_ids": [
                row["case_id"]
                for row in rows
                if row["manual_audit"]
                and row["manual_audit"]["verdict"] == "evaluator_false_negative"
            ],
        },
        "rates_by_task_type": _group_rates(rows, "task_type"),
        "rates_by_challenge_type": _group_rates(rows, "challenge_types"),
        "runtime_cost": {
            "model_requests": model_requests,
            "model_input_tokens": input_tokens,
            "model_output_tokens": output_tokens,
            "model_total_tokens": input_tokens + output_tokens,
            "average_total_tokens_per_executed_case": _rate(
                input_tokens + output_tokens, len(executed)
            ),
            "tool_calls": len(tool_calls),
            "tool_counts": dict(sorted(tool_counts.items())),
            "tool_duration_ms": sum(call.get("duration_ms") or 0 for call in tool_calls),
        },
        "next_single_variable_recommendation": {
            "id": "p2-b1-document-scope-vs-fact-period",
            "change": (
                "只修改extract规划器：把报告文档年份与问题事实年份分开，并在命名公司"
                "只有唯一候选年报时允许从索引元数据推断文档范围。"
            ),
            "direct_target_cases": [
                "blind_yili_inventory_policy",
                "blind_midea_2023_future_actual",
                "blind_midea_2024_future_actual",
                "blind_shenhua_2024_future_actual",
            ],
            "expected_direction": (
                "修复1个错误澄清和3个未来事实未安全拒答；不同时加入计算器或完整性检查。"
            ),
        },
        "cases": rows,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"valid={str(payload['valid']).lower()}")
    print(f"machine_passes={report['metrics']['end_to_end_case_pass_rate']:.4f}")
    print(
        "audited_passes="
        f"{payload['assistant_source_audited_metrics']['case_pass_count']}/"
        f"{payload['assistant_source_audited_metrics']['case_count']}"
    )
    print(f"output={args.output.resolve()}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
