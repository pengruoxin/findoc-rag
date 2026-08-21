"""Summarize frozen P2-A/P2-B1 Agent hard-v2 runs and paired changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analyze_agent_hard_failures import MANUAL_AUDITS

RUNS = [
    (
        "p2a_baseline",
        "agent-hard-v2-deepseek-p2a-baseline.json",
        "冻结P1代码的五文档盲测基线",
        "baseline",
    ),
    (
        "p2b1a_document_fact_period",
        "agent-hard-v2-deepseek-p2b1-document-fact-period.json",
        "分离报告年份/事实期间并增加唯一文档范围推断；尚未识别“2023年年报”",
        "improved_partial",
    ),
    (
        "p2b1b_document_year_syntax",
        "agent-hard-v2-deepseek-p2b1b-document-year-pattern.json",
        "同一规划变量内补齐“YYYY年年报”文档年份语法",
        "improved",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/agent-hard-v2.json"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports/agent"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/agent/agent-hard-v2-p2b1-summary.json"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _runtime(items: list[dict[str, Any]]) -> dict[str, Any]:
    executed = [item for item in items if item["status"] == "executed"]
    model_traces = [
        item["trace"].get("model_trace")
        for item in executed
        if item.get("trace") and item["trace"].get("model_trace")
    ]
    calls = [
        call
        for item in executed
        for call in (item.get("trace") or {}).get("tool_calls", [])
    ]
    input_tokens = sum(trace.get("input_tokens") or 0 for trace in model_traces)
    output_tokens = sum(trace.get("output_tokens") or 0 for trace in model_traces)
    return {
        "model_requests": sum(trace["request_count"] for trace in model_traces),
        "model_input_tokens": input_tokens,
        "model_output_tokens": output_tokens,
        "model_total_tokens": input_tokens + output_tokens,
        "tool_calls": len(calls),
        "tool_counts": dict(sorted(Counter(call["tool"] for call in calls).items())),
    }


def _audited_metrics(
    report: dict[str, Any], cases: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    case_passes = 0
    supported_passes = 0
    supported_fact_total = 0
    supported_fact_matches = 0
    end_to_end_fact_total = 0
    end_to_end_fact_matches = 0
    genuine_failures: list[str] = []
    evaluator_false_negatives: list[str] = []

    for item in report["items"]:
        case_id = item["case_id"]
        expected_fact_ids = [fact["fact_id"] for fact in cases[case_id]["expected_facts"]]
        machine_matches = {
            fact["fact_id"]: bool(fact.get("matched"))
            for fact in (item.get("score") or {}).get("fact_scores") or []
        }
        audit = MANUAL_AUDITS.get(case_id)
        if audit:
            for fact_id in audit.get("matched_fact_overrides", []):
                machine_matches[fact_id] = True

        fact_matches = sum(machine_matches.get(fact_id, False) for fact_id in expected_fact_ids)
        end_to_end_fact_total += len(expected_fact_ids)
        end_to_end_fact_matches += fact_matches
        if item["status"] == "executed":
            supported_fact_total += len(expected_fact_ids)
            supported_fact_matches += fact_matches

        machine_pass = bool((item.get("score") or {}).get("case_pass", False))
        audited_pass = machine_pass or bool(audit and audit["audited_case_pass"])
        case_passes += audited_pass
        if item["status"] == "executed":
            supported_passes += audited_pass
        if not audited_pass:
            genuine_failures.append(case_id)
        elif not machine_pass:
            evaluator_false_negatives.append(case_id)

    executed_count = report["metrics"]["executed_case_count"]
    return {
        "case_pass_count": case_passes,
        "case_count": len(report["items"]),
        "case_pass_rate": _rate(case_passes, len(report["items"])),
        "executed_case_pass_count": supported_passes,
        "executed_case_count": executed_count,
        "executed_case_pass_rate": _rate(supported_passes, executed_count),
        "supported_fact_match_count": supported_fact_matches,
        "supported_fact_count": supported_fact_total,
        "supported_fact_accuracy": _rate(supported_fact_matches, supported_fact_total),
        "end_to_end_fact_match_count": end_to_end_fact_matches,
        "end_to_end_fact_count": end_to_end_fact_total,
        "end_to_end_fact_accuracy": _rate(end_to_end_fact_matches, end_to_end_fact_total),
        "genuine_failure_case_ids": genuine_failures,
        "evaluator_false_negative_case_ids": evaluator_false_negatives,
    }


def _paired(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_items = {item["case_id"]: item for item in before["items"]}
    after_items = {item["case_id"]: item for item in after["items"]}
    fixed = [
        case_id
        for case_id, item in after_items.items()
        if not bool((before_items[case_id].get("score") or {}).get("case_pass", False))
        and bool((item.get("score") or {}).get("case_pass", False))
    ]
    regressed = [
        case_id
        for case_id, item in after_items.items()
        if bool((before_items[case_id].get("score") or {}).get("case_pass", False))
        and not bool((item.get("score") or {}).get("case_pass", False))
    ]
    return {
        "fixed_case_ids": fixed,
        "regressed_case_ids": regressed,
        "net_case_delta": len(fixed) - len(regressed),
    }


def main() -> int:
    args = parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in dataset["cases"]}
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    errors: list[str] = []

    for run_id, filename, change, verdict in RUNS:
        path = args.report_dir / filename
        if not path.is_file():
            errors.append(f"missing report: {path.as_posix()}")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append(report)
        rows.append(
            {
                "run_id": run_id,
                "report": path.as_posix(),
                "report_sha256": _sha256(path),
                "controlled_change": change,
                "verdict": verdict,
                "planner_revisions": sorted(
                    {
                        item["trace"]["plan"]["planner_revision"]
                        for item in report["items"]
                        if item.get("trace")
                    }
                ),
                "machine_metrics": report["metrics"],
                "assistant_source_audited_metrics": _audited_metrics(report, cases),
                "runtime": _runtime(report["items"]),
            }
        )

    paired = []
    for index in range(1, len(reports)):
        paired.append(
            {
                "before_run_id": rows[index - 1]["run_id"],
                "after_run_id": rows[index]["run_id"],
                **_paired(reports[index - 1], reports[index]),
            }
        )
    if len(reports) == len(RUNS):
        paired.append(
            {
                "before_run_id": rows[0]["run_id"],
                "after_run_id": rows[-1]["run_id"],
                **_paired(reports[0], reports[-1]),
            }
        )

    payload = {
        "schema_version": "1",
        "dataset_id": dataset["dataset_id"],
        "generated_at": datetime.now(UTC).isoformat(),
        "valid": not errors,
        "ready_for_external_claims": False,
        "audit_boundary": (
            "machine metrics are immutable; source-audited metrics apply the frozen P2-A "
            "assistant review and are not independent human scores"
        ),
        "dataset": args.dataset.as_posix(),
        "dataset_sha256": _sha256(args.dataset),
        "runs": rows,
        "paired_comparisons": paired,
        "conclusion": {
            "p2b1_direct_targets": [
                "blind_yili_inventory_policy",
                "blind_midea_2023_future_actual",
                "blind_midea_2024_future_actual",
                "blind_shenhua_2024_future_actual",
            ],
            "final_machine_delta": "+4 cases, 0 regressions",
            "final_remaining_genuine_failures": (
                "4 unsupported general calculations and 2 multi-fact omissions"
            ),
            "next_single_variable": (
                "P2-B2: add an explicit required-fact checklist to extraction/comparison "
                "submission; do not add general calculation in the same run"
            ),
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"valid={str(payload['valid']).lower()}")
    print(f"runs={len(rows)}")
    if rows:
        final = rows[-1]
        print(
            "final_machine_pass_rate="
            f"{final['machine_metrics']['end_to_end_case_pass_rate']:.4f}"
        )
        print(
            "final_audited_passes="
            f"{final['assistant_source_audited_metrics']['case_pass_count']}/"
            f"{final['assistant_source_audited_metrics']['case_count']}"
        )
    print(f"output={args.output.resolve()}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
