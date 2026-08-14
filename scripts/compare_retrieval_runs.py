#!/usr/bin/env python3
"""Create a paired fixed/regressed report for two retrieval per-query runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

REGIMES = ("canonical", "ticker_or_finance_shorthand", "semantic_or_relative_time")
METRICS = ("hit_at_5", "mrr_at_5", "recall_at_5", "ndcg_at_5", "candidate_recall")


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = row["query_id"]
        if query_id in rows:
            raise ValueError(f"Duplicate query_id: {query_id}")
        rows[query_id] = row
    return rows


def compare_rows(
    old_rows: dict[str, dict],
    new_rows: dict[str, dict],
    *,
    filter_name: str,
    mode: str,
) -> dict:
    if set(old_rows) != set(new_rows):
        raise ValueError(
            "Paired runs have different query IDs: "
            f"old_only={sorted(set(old_rows) - set(new_rows))}, "
            f"new_only={sorted(set(new_rows) - set(old_rows))}"
        )
    positive_ids = sorted(
        query_id
        for query_id, row in old_rows.items()
        if row.get("retrieval_judgment") == "positive_gold"
    )
    pairs = []
    for query_id in positive_ids:
        old_row = old_rows[query_id]
        new_row = new_rows[query_id]
        old_metrics = old_row["results"][filter_name][mode]
        new_metrics = new_row["results"][filter_name][mode]
        pairs.append(
            {
                "query_id": query_id,
                "canonical_id": old_row["canonical_id"],
                "regime": old_row["regime"],
                "query": old_row["query"],
                "old": {metric: old_metrics[metric] for metric in METRICS},
                "new": {metric: new_metrics[metric] for metric in METRICS},
            }
        )

    fixed = [p["query_id"] for p in pairs if not p["old"]["hit_at_5"] and p["new"]["hit_at_5"]]
    regressed = [
        p["query_id"] for p in pairs if p["old"]["hit_at_5"] and not p["new"]["hit_at_5"]
    ]
    mrr_improved = [p["query_id"] for p in pairs if p["new"]["mrr_at_5"] > p["old"]["mrr_at_5"]]
    mrr_regressed = [p["query_id"] for p in pairs if p["new"]["mrr_at_5"] < p["old"]["mrr_at_5"]]
    exact_metric_matches = sum(p["old"] == p["new"] for p in pairs)

    by_regime = {}
    for regime in REGIMES:
        regime_pairs = [pair for pair in pairs if pair["regime"] == regime]
        by_regime[regime] = {
            metric: {
                "old": mean(pair["old"][metric] for pair in regime_pairs),
                "new": mean(pair["new"][metric] for pair in regime_pairs),
                "delta": mean(pair["new"][metric] for pair in regime_pairs)
                - mean(pair["old"][metric] for pair in regime_pairs),
            }
            for metric in METRICS
        }
    return {
        "filter": filter_name,
        "mode": mode,
        "positive_pair_count": len(pairs),
        "exact_metric_match_count": exact_metric_matches,
        "fixed": fixed,
        "regressed": regressed,
        "mrr_improved": mrr_improved,
        "mrr_regressed": mrr_regressed,
        "by_regime": by_regime,
        "pairs": pairs,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Paired retrieval comparison",
        "",
        f"- old: `{report['old_run']}`",
        f"- new: `{report['new_run']}`",
        f"- configuration: filter=`{report['filter']}`, mode=`{report['mode']}`",
        (
            f"- positive pairs: {report['positive_pair_count']} | "
            f"exact metric matches: {report['exact_metric_match_count']}"
        ),
        (
            f"- Hit@5 fixed/regressed: {len(report['fixed'])}/{len(report['regressed'])} | "
            f"MRR improved/regressed: {len(report['mrr_improved'])}/{len(report['mrr_regressed'])}"
        ),
        "",
        "| regime | old Hit@5 | new Hit@5 | Δ | old MRR@5 | new MRR@5 | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for regime in REGIMES:
        hit = report["by_regime"][regime]["hit_at_5"]
        mrr = report["by_regime"][regime]["mrr_at_5"]
        lines.append(
            f"| {regime} | {hit['old']:.4f} | {hit['new']:.4f} | {hit['delta']:+.4f} | "
            f"{mrr['old']:.4f} | {mrr['new']:.4f} | {mrr['delta']:+.4f} |"
        )
    for title, key in (
        ("Hit@5 fixed", "fixed"),
        ("Hit@5 regressed", "regressed"),
        ("MRR improved", "mrr_improved"),
        ("MRR regressed", "mrr_regressed"),
    ):
        values = report[key]
        lines.extend(["", f"## {title}", "", ", ".join(f"`{value}`" for value in values) or "None."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old-run", required=True)
    parser.add_argument("--new-run", required=True)
    parser.add_argument("--filter", choices=("none", "query_parser"), default="query_parser")
    parser.add_argument("--mode", choices=("lexical", "dense", "hybrid"), default="lexical")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite paired report: {args.output_dir}")

    report = compare_rows(
        load_rows(args.old),
        load_rows(args.new),
        filter_name=args.filter,
        mode=args.mode,
    )
    report["old_run"] = args.old_run
    report["new_run"] = args.new_run
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
