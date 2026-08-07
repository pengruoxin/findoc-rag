"""Offline RRF fusion-weight sweep on benchmark-v2-retrieval-view.

Component rankings (lexical / dense) are computed once per query instance and
filter state; every weight combination is then fused offline, so the sweep is
fast and reproducible. Outputs per-weight summaries plus a per-regime best
selection analysis (development-only: best is chosen on the eval set itself).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from run_retrieval_variant_eval import (
    AGGREGATE_METRICS,
    FILTERS,
    build_filters,
    hit_at_k,
    ndcg_at_k,
    parse_for_filter,
    query_instances,
    reciprocal_rank,
)

from findoc_rag.corpus import resolve_current_index
from findoc_rag.indexing import reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEW = ROOT / "data/evaluation/benchmark-v2-retrieval-view.json"
DEFAULT_INDEX_ROOT = ROOT / "data/indexes/corpus"
WEIGHTS: tuple[tuple[float, float], ...] = (
    (1.0, 1.0),   # equal
    (2.0, 1.0),   # v1 baseline
    (3.0, 1.0),
    (4.0, 1.0),
    (1.0, 0.0),   # lexical only (RRF with zero dense weight)
    (0.0, 1.0),   # dense only
)
REGIMES = ("canonical", "ticker_or_finance_shorthand", "semantic_or_relative_time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/ranking/fusion-sweep-v1",
    )
    return parser.parse_args()


def metrics_for_hits(all_hits, gold: set[str], negatives: set[str], top_k: int) -> dict:
    hits = all_hits[:top_k]
    ranks = [hit.rank for hit in hits if hit.chunk.chunk_id in gold]
    relevant_in_top = sum(1 for hit in hits if hit.chunk.chunk_id in gold)
    return {
        "hit_at_5": hit_at_k(ranks, top_k),
        "mrr_at_5": reciprocal_rank(ranks),
        "recall_at_5": relevant_in_top / len(gold) if gold else 0.0,
        "precision_at_5": relevant_in_top / top_k,
        "ndcg_at_5": ndcg_at_k(gold, negatives, hits, top_k),
        "candidate_recall": bool(
            any(hit.chunk.chunk_id in gold for hit in all_hits)
        ),
        "avg_evidence_tokens_top5": (
            sum(hit.chunk.estimated_token_count for hit in hits) / len(hits)
            if hits
            else 0.0
        ),
        "negative_count_in_top5": sum(
            1 for hit in hits if hit.chunk.chunk_id in negatives
        ),
    }


def aggregate(rows: list[dict], by_canonical: bool) -> dict:
    if by_canonical:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["canonical_id"], []).append(row)
        metric_rows = [
            {metric: mean(item[metric] for item in group) for metric in AGGREGATE_METRICS}
            for group in grouped.values()
        ]
    else:
        metric_rows = rows
    return {metric: mean(row[metric] for row in metric_rows) for metric in AGGREGATE_METRICS}


def render_markdown(summary: dict, args: argparse.Namespace) -> str:
    lines = [
        "# RRF fusion-weight sweep v1",
        "",
        f"- dataset: `{summary['dataset_id']}` | index: `{summary['index_id']}`",
        (
            f"- top_k={args.top_k} | candidate_k={args.candidate_k} | "
            f"rrf_k={args.rrf_k} | positive instances: {summary['positive_instance_count']}"
        ),
        "- 指标为 partial-judgment（gold 相关、hard negatives 负例、其余 unjudged）。",
        "- per-regime best 是在评测集上选择的最优权重，属于 development-only 上限，不代表独立泛化结论。",
        "",
        "## Query-level Hit@5（query_parser 过滤）",
        "",
        "| weight | canonical | ticker | semantic/相对时间 |",
        "|---|---:|---:|---:|",
    ]
    for weight in summary["weights"]:
        label = f"{weight[0]}:{weight[1]}"
        row = summary["by_regime_query"][label]["query_parser"]
        lines.append(
            f"| {label} | {row['canonical']['hit_at_5']:.4f} | "
            f"{row['ticker_or_finance_shorthand']['hit_at_5']:.4f} | "
            f"{row['semantic_or_relative_time']['hit_at_5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Query-level MRR@5（query_parser 过滤）",
            "",
            "| weight | canonical | ticker | semantic/相对时间 |",
            "|---|---:|---:|---:|",
        ]
    )
    for weight in summary["weights"]:
        label = f"{weight[0]}:{weight[1]}"
        row = summary["by_regime_query"][label]["query_parser"]
        lines.append(
            f"| {label} | {row['canonical']['mrr_at_5']:.4f} | "
            f"{row['ticker_or_finance_shorthand']['mrr_at_5']:.4f} | "
            f"{row['semantic_or_relative_time']['mrr_at_5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-regime best（query_parser 过滤，development-only）",
            "",
            "| regime | best weight | Hit@5 | MRR@5 | NDCG@5 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for regime in REGIMES:
        best = summary["per_regime_best"]["query_parser"][regime]
        lines.append(
            f"| {regime} | {best['weight']} | {best['hit_at_5']:.4f} | "
            f"{best['mrr_at_5']:.4f} | {best['ndcg_at_5']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    view = json.loads(args.view.read_text(encoding="utf-8"))
    index = resolve_current_index(args.index_root)
    positive = [item for item in view["items"] if item["retrieval_judgment"] == "positive_gold"]

    # Compute component rankings once per instance / filter state.
    instances: list[dict] = []
    for item in positive:
        for instance in query_instances(item):
            variant = instance["variant"]
            as_of_date = variant.get("as_of_date") if variant else None
            resolved, companies, years = parse_for_filter(instance["query"], as_of_date)
            components: dict[str, dict] = {}
            for filter_name in FILTERS:
                filters = build_filters(companies, years) if filter_name == "query_parser" else None
                lexical = index.search_lexical(resolved, args.candidate_k, filters)
                dense = index.search_dense_batch(
                    [resolved], top_k=args.candidate_k, filters=[filters]
                )[0]
                components[filter_name] = {"lexical": lexical, "dense": dense}
            instances.append(
                {
                    "query_id": instance["query_id"],
                    "canonical_id": item["query_id"],
                    "regime": "canonical" if variant is None else variant["query_regime"],
                    "gold": set(item.get("gold_chunk_ids") or []),
                    "negatives": {
                        negative["chunk_id"]
                        for negative in item.get("hard_negatives") or []
                    },
                    "components": components,
                }
            )

    by_weight: dict[str, dict] = {}
    for lexical_weight, dense_weight in WEIGHTS:
        label = f"{lexical_weight}:{dense_weight}"
        per_filter: dict[str, list[dict]] = {fname: [] for fname in FILTERS}
        for instance in instances:
            for filter_name in FILTERS:
                comp = instance["components"][filter_name]
                if lexical_weight > 0 and dense_weight > 0:
                    fused = reciprocal_rank_fusion(
                        comp["lexical"],
                        comp["dense"],
                        top_k=args.candidate_k,
                        rrf_k=args.rrf_k,
                        lexical_weight=lexical_weight,
                        dense_weight=dense_weight,
                    )
                elif lexical_weight > 0:
                    fused = comp["lexical"]
                else:
                    fused = comp["dense"]
                per_filter[filter_name].append(
                    {
                        "query_id": instance["query_id"],
                        "canonical_id": instance["canonical_id"],
                        "regime": instance["regime"],
                        **metrics_for_hits(
                            fused,
                            instance["gold"],
                            instance["negatives"],
                            args.top_k,
                        ),
                    }
                )
        by_weight[label] = {
            "query_parser": per_filter["query_parser"],
            "none": per_filter["none"],
        }

    # Summarize per regime.
    by_regime_query: dict[str, dict] = {}
    by_regime_group: dict[str, dict] = {}
    for label, payload in by_weight.items():
        by_regime_query[label] = {
            filter_name: {
                regime: aggregate(
                    [row for row in rows if row["regime"] == regime],
                    by_canonical=False,
                )
                for regime in REGIMES
            }
            for filter_name, rows in payload.items()
        }
        by_regime_group[label] = {
            filter_name: {
                regime: aggregate(
                    [row for row in rows if row["regime"] == regime],
                    by_canonical=True,
                )
                for regime in REGIMES
            }
            for filter_name, rows in payload.items()
        }

    # Per-regime best on the eval set (development-only).
    per_regime_best: dict[str, dict[str, dict]] = {}
    for filter_name in FILTERS:
        per_regime_best[filter_name] = {}
        for regime in REGIMES:
            candidates = []
            for label, payload in by_regime_query.items():
                metrics = payload[filter_name][regime]
                candidates.append(
                    {
                        "weight": label,
                        "hit_at_5": metrics["hit_at_5"],
                        "mrr_at_5": metrics["mrr_at_5"],
                        "ndcg_at_5": metrics["ndcg_at_5"],
                    }
                )
            per_regime_best[filter_name][regime] = max(
                candidates, key=lambda item: (item["hit_at_5"], item["mrr_at_5"])
            )

    summary = {
        "run_id": "fusion-sweep-v1",
        "dataset_id": view["dataset_id"],
        "index_id": index.manifest.index_id,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "weights": [list(w) for w in WEIGHTS],
        "positive_instance_count": len(instances),
        "positive_group_count": len(positive),
        "by_regime_query": by_regime_query,
        "by_regime_group": by_regime_group,
        "per_regime_best": per_regime_best,
        "note": "per_regime_best is development-only: selected on the eval set itself",
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {output_dir}")
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "view": str(args.view),
                "index_root": str(args.index_root),
                "top_k": args.top_k,
                "candidate_k": args.candidate_k,
                "rrf_k": args.rrf_k,
                "weights": [list(w) for w in WEIGHTS],
                "metadata_filter_source": {"none": "none", "query_parser": "query_parser"},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        render_markdown(summary, args), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
