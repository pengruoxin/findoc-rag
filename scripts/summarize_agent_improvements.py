"""Summarize frozen DeepSeek Agent P0/P1 experiment sequences."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

RUNS = [
    (
        "baseline",
        "agent-hard-v1-deepseek.json",
        "首次真实 DeepSeek 困难集基线",
        "baseline",
    ),
    (
        "p0a1_loop_control",
        "agent-hard-v1-deepseek-p0a-loop-control.json",
        "缺口重试、重复检索限制与提交预算保留",
        "improved",
    ),
    (
        "p0a2_forced_submit",
        "agent-hard-v1-deepseek-p0a2-forced-submit.json",
        "仅用 named tool_choice 强制提交",
        "no_gain",
    ),
    (
        "p0a3_fresh_finalizer",
        "agent-hard-v1-deepseek-p0a3-fresh-finalizer.json",
        "丢弃陈旧检索历史的独立最终提交上下文",
        "improved",
    ),
    (
        "p0b1_provenance_targets",
        "agent-hard-v1-deepseek-p0b-provenance-targets.json",
        "拆分事实年度、报告年份、报表口径和数值版本",
        "mixed",
    ),
    (
        "p0b2_table_hints",
        "agent-hard-v1-deepseek-p0b2-table-hints.json",
        "财务表定位提示、同表多版本证据绑定与跨报告过滤修正",
        "improved",
    ),
    (
        "p0b3_optional_submit_schema",
        "agent-hard-v1-deepseek-p0b3-cross-target-claims.json",
        "放宽提交工具必填字段",
        "regressed",
    ),
    (
        "p0b4_strict_submit_schema",
        "agent-hard-v1-deepseek-p0b4-strict-schema-lenient-abstain.json",
        "恢复模型侧严格 schema，保留本地非答案默认值",
        "improved",
    ),
    (
        "p0b5_nonanswer_parser",
        "agent-hard-v1-deepseek-p0b5-safe-nonanswer-parser.json",
        "仅对畸形非答案载荷做安全解析",
        "regressed",
    ),
    (
        "p0b6_no_evidence_gate",
        "agent-hard-v1-deepseek-p0b6-local-no-evidence-gate.json",
        "零证据缺口由本地门禁确定性拒答，并记录 validation_errors",
        "improved",
    ),
    (
        "p0b7_cross_year_groups",
        "agent-hard-v1-deepseek-p0b7-cross-year-evidence-groups.json",
        "同公司同比或重述任务建立跨年度证据组",
        "mixed",
    ),
    (
        "p0b8_provenance_repair",
        "agent-hard-v1-deepseek-p0b8-provenance-citation-repair.json",
        "补齐已检索的指定报告年份引用，受控支持跨公司比较 claim",
        "improved",
    ),
    (
        "p0b8_replication",
        "agent-hard-v1-deepseek-p0b8-replication.json",
        "同代码、同数据、同索引复现实验",
        "replicated",
    ),
]

P1_RUNS = [
    (
        "p0b8_start",
        "agent-hard-v1-deepseek-p0b8-provenance-citation-repair.json",
        "P1 起点：仅支持 5 个 compare 难例",
        "baseline",
    ),
    (
        "p1a0_page_window",
        "agent-hard-v1-deepseek-p1a-extract-page-window.json",
        "跨页 extract 与 page window；旧工具上下文干扰最终提交",
        "no_gain",
    ),
    (
        "p1a1_phase_isolated_extract",
        "agent-hard-v1-deepseek-p1a1-single-window-finalizer.json",
        "单窗口展开与独立抽取 finalizer",
        "improved",
    ),
    (
        "p1b0_cited_calculator",
        "agent-hard-v1-deepseek-p1b-cited-decimal-calculator.json",
        "带来源 Decimal 计算器；并行工具调用导致阶段协议失败",
        "no_gain",
    ),
    (
        "p1b1_phase_isolated_calculator",
        "agent-hard-v1-deepseek-p1b1-phase-isolated-calculator.json",
        "检索、页面窗口、计算器三阶段隔离",
        "improved",
    ),
    (
        "p1c_pdf_geometry_relationships",
        "agent-hard-v1-deepseek-p1c-pdf-geometry-relationships.json",
        "PDF 文字坐标与连接线关系重建，DeepSeek 关系确认，本地百分比求和",
        "improved",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=Path("reports/agent"))
    parser.add_argument("--phase", choices=("p0", "p1", "all"), default="p0")
    parser.add_argument(
        "--output",
        type=Path,
    )
    return parser.parse_args()


def _sum_runtime(items: list[dict]) -> dict[str, int]:
    executed = [item for item in items if item.get("status") == "executed"]
    model_traces = [
        item["trace"]["model_trace"]
        for item in executed
        if item.get("trace") and item["trace"].get("model_trace")
    ]
    return {
        "model_requests": sum(trace["request_count"] for trace in model_traces),
        "model_input_tokens": sum(trace["input_tokens"] or 0 for trace in model_traces),
        "model_output_tokens": sum(trace["output_tokens"] or 0 for trace in model_traces),
        "tool_calls": sum(len(item["trace"]["tool_calls"]) for item in executed),
        "actual_retrieval_calls": sum(
            call.get("tool", "search_evidence") == "search_evidence"
            for item in executed
            for call in item["trace"]["tool_calls"]
        ),
    }


def main() -> int:
    args = parse_args()
    selected_runs = (
        RUNS
        if args.phase == "p0"
        else P1_RUNS
        if args.phase == "p1"
        else RUNS + P1_RUNS[1:]
    )
    output = args.output or Path(
        f"reports/agent/agent-hard-v1-{args.phase}-summary.json"
    )
    rows: list[dict] = []
    missing: list[str] = []
    for run_id, filename, change, verdict in selected_runs:
        path = args.report_dir / filename
        if not path.is_file():
            missing.append(path.as_posix())
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report["metrics"]
        rows.append(
            {
                "run_id": run_id,
                "report": path.as_posix(),
                "change": change,
                "verdict": verdict,
                "metrics": metrics,
                "runtime": _sum_runtime(report["items"]),
            }
        )

    payload = {
        "schema_version": "1",
        "dataset_id": "agent-hard-v1",
        "phase": args.phase,
        "generated_at": datetime.now(UTC).isoformat(),
        "missing_reports": missing,
        "runs": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"runs={len(rows)}")
    print(f"missing={len(missing)}")
    print(f"output={output.resolve()}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
