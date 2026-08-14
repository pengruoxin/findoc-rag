"""Compare two generation runs on the same frozen dataset and lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--change",
        help=(
            "required when the two runs come from different code revisions: "
            "describe the single controlled variable, e.g. 'abstention detection'"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {row["query_id"]: row for row in rows}


def compare_runs(baseline_dir: Path, candidate_dir: Path, change: str | None = None) -> dict:
    baseline_summary = read_json(baseline_dir / "summary.json")
    candidate_summary = read_json(candidate_dir / "summary.json")
    for field in ("dataset_id", "lane"):
        if baseline_summary[field] != candidate_summary[field]:
            raise ValueError(f"Cannot compare runs with different {field}")
    if (
        baseline_summary.get("remote_generation")
        != candidate_summary.get("remote_generation")
    ):
        raise ValueError(
            "Cannot compare runs with different remote_generation flags "
            "(LLM vs no-LLM is more than one variable)"
        )

    baseline_revision = baseline_summary.get("code_revision")
    candidate_revision = candidate_summary.get("code_revision")
    baseline_dirty = bool(baseline_summary.get("code_dirty"))
    candidate_dirty = bool(candidate_summary.get("code_dirty"))
    baseline_fingerprint = baseline_summary.get("code_fingerprint")
    candidate_fingerprint = candidate_summary.get("code_fingerprint")
    revision_match = (
        baseline_revision == candidate_revision
        and baseline_revision is not None
        and baseline_revision != "unknown"
    )
    fingerprint_match = bool(
        baseline_fingerprint
        and baseline_fingerprint == candidate_fingerprint
    )
    code_state_match = revision_match and (
        (not baseline_dirty and not candidate_dirty) or fingerprint_match
    )
    if not code_state_match and not change:
        raise ValueError(
            "Runs were produced by different, dirty, or unrecorded code states; "
            "this is not a controlled comparison. Pass --change "
            "'<single variable description>' to declare the controlled change, "
            "or compare two runs from the same revision."
        )

    baseline_scores = read_jsonl(baseline_dir / "deterministic-scores.jsonl")
    candidate_scores = read_jsonl(candidate_dir / "deterministic-scores.jsonl")
    if set(baseline_scores) != set(candidate_scores):
        raise ValueError("Compared runs do not contain the same query IDs")

    behavior_fixed: list[str] = []
    behavior_regressed: list[str] = []
    strict_fixed: list[str] = []
    strict_regressed: list[str] = []
    for query_id in sorted(baseline_scores):
        before = baseline_scores[query_id]
        after = candidate_scores[query_id]
        if not before["expected_behavior_correct"] and after["expected_behavior_correct"]:
            behavior_fixed.append(query_id)
        elif before["expected_behavior_correct"] and not after["expected_behavior_correct"]:
            behavior_regressed.append(query_id)
        if before["strict_success_eligible"] and after["strict_success_eligible"]:
            if not before["strict_success"] and after["strict_success"]:
                strict_fixed.append(query_id)
            elif before["strict_success"] and not after["strict_success"]:
                strict_regressed.append(query_id)

    metric_names = (
        "strict_success_rate",
        "expected_behavior_accuracy",
        "run_error_rate",
    )
    metrics = {
        name: {
            "baseline": baseline_summary[name],
            "candidate": candidate_summary[name],
            "delta": candidate_summary[name] - baseline_summary[name],
        }
        for name in metric_names
    }
    return {
        "dataset_id": baseline_summary["dataset_id"],
        "lane": baseline_summary["lane"],
        "baseline_run_id": baseline_summary["run_id"],
        "candidate_run_id": candidate_summary["run_id"],
        "baseline_code_revision": baseline_revision,
        "candidate_code_revision": candidate_revision,
        "code_revision_match": revision_match,
        "baseline_code_dirty": baseline_dirty,
        "candidate_code_dirty": candidate_dirty,
        "baseline_code_fingerprint": baseline_fingerprint,
        "candidate_code_fingerprint": candidate_fingerprint,
        "code_fingerprint_match": fingerprint_match,
        "code_state_match": code_state_match,
        "controlled_change": change if not code_state_match else None,
        "paired_item_count": len(baseline_scores),
        "metrics": metrics,
        "behavior_fixed": behavior_fixed,
        "behavior_regressed": behavior_regressed,
        "strict_fixed": strict_fixed,
        "strict_regressed": strict_regressed,
    }


def render_markdown(report: dict) -> str:
    metrics = report["metrics"]
    lines = [
        f"# 生成评测配对对比：{report['lane']}",
        "",
        f"- Dataset：`{report['dataset_id']}`",
        f"- Baseline：`{report['baseline_run_id']}`",
        f"- Candidate：`{report['candidate_run_id']}`",
        f"- 代码版本：baseline `{report['baseline_code_revision']}` / "
        f"candidate `{report['candidate_code_revision']}`"
        + ("（一致）" if report["code_revision_match"] else "（不同）"),
        (
            f"- 代码状态：{'一致' if report['code_state_match'] else '不同或不可证明'}；"
            f"fingerprint {'一致' if report['code_fingerprint_match'] else '不一致/缺失'}"
        ),
        f"- 受控变量：{report['controlled_change'] or '无（同代码复现/稳定性对照）'}",
        f"- 配对样本：{report['paired_item_count']}",
        "",
        "| 指标 | Baseline | Candidate | 变化 |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "strict_success_rate": "Strict success",
        "expected_behavior_accuracy": "Expected behavior accuracy",
        "run_error_rate": "Run error rate",
    }
    for name, label in labels.items():
        value = metrics[name]
        lines.append(
            f"| {label} | {value['baseline']:.4f} | "
            f"{value['candidate']:.4f} | {value['delta']:+.4f} |"
        )
    for key, title in (
        ("behavior_fixed", "行为修复"),
        ("behavior_regressed", "行为回归"),
        ("strict_fixed", "Strict 修复"),
        ("strict_regressed", "Strict 回归"),
    ):
        cases = report[key]
        lines.extend(["", f"## {title}（{len(cases)}）", ""])
        lines.extend(f"- `{case}`" for case in cases)
        if not cases:
            lines.append("- 无")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = compare_runs(args.baseline, args.candidate, change=args.change)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output_prefix.with_suffix(".md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
