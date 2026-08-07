"""Variant-regime retrieval evaluation on benchmark-v2-retrieval-view.

Runs lexical / dense / hybrid retrieval over canonical + variant query
instances, with and without query-parser metadata filters, and reports
per-regime metrics. All experiment configuration is serialized next to the
results so runs are reproducible and hand-off friendly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date
from pathlib import Path
from statistics import mean

from findoc_rag.corpus import resolve_current_index
from findoc_rag.indexing import SearchFilters, reciprocal_rank_fusion
from findoc_rag.query_expansion import expand_query
from findoc_rag.query_rewriting import LLMQueryRewriter
from findoc_rag.time_utils import resolve_relative_time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEW = ROOT / "data/evaluation/benchmark-v2-retrieval-view.json"
DEFAULT_INDEX_ROOT = ROOT / "data/indexes/corpus"

KNOWN_TICKERS = {"600519": "贵州茅台", "600887": "伊利股份"}
KNOWN_COMPANIES = tuple(KNOWN_TICKERS.values())
YEAR_PATTERN = re.compile(r"20\d{2}")
MODES = ("lexical", "dense", "hybrid")
FILTERS = ("none", "query_parser")
AGGREGATE_METRICS = (
    "hit_at_5",
    "mrr_at_5",
    "recall_at_5",
    "precision_at_5",
    "ndcg_at_5",
    "candidate_recall",
    "negative_count_in_top5",
    "avg_evidence_tokens_top5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--lexical-weight", type=float, default=2.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument(
        "--expand-synonyms",
        action="store_true",
        help="apply deterministic financial synonym expansion before retrieval",
    )
    parser.add_argument(
        "--rewrite",
        choices=("none", "deterministic", "llm"),
        default="none",
        help="query rewrite mode: deterministic synonym table, LLM normalization, or none",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/ranking/variant-regime-v1",
    )
    return parser.parse_args()


class _DeterministicRewriter:
    """Adapter so the deterministic table satisfies the rewrite protocol."""

    @staticmethod
    def rewrite(query: str) -> str:
        return expand_query(query)


def make_rewriter(mode: str):
    if mode == "deterministic":
        return _DeterministicRewriter()
    if mode == "llm":
        return LLMQueryRewriter()
    return None


def query_instances(item: dict) -> list[dict]:
    instances = [{"query_id": item["query_id"], "query": item["query"], "variant": None}]
    for variant in item["query_variants"]:
        instances.append(
            {
                "query_id": f"{item['query_id']}::{variant['variant_id']}",
                "query": variant["query"],
                "variant": variant,
            }
        )
    return instances


def parse_for_filter(
    query: str,
    as_of_date: str | None,
) -> tuple[str, list[str], list[int]]:
    """Resolve relative time and extract company / year signals from text."""
    resolved = query
    if as_of_date:
        resolved, _ = resolve_relative_time(query, date.fromisoformat(as_of_date))
    companies = [company for company in KNOWN_COMPANIES if company in resolved]
    tickers = [ticker for ticker in KNOWN_TICKERS if ticker in resolved]
    companies.extend(KNOWN_TICKERS[ticker] for ticker in tickers)
    years = [int(year) for year in YEAR_PATTERN.findall(resolved)]
    return resolved, list(dict.fromkeys(companies)), years


def build_filters(companies: list[str], years: list[int]) -> SearchFilters | None:
    if not companies and not years:
        return None
    return SearchFilters(
        company_names=companies,
        report_years=years,
    )


def hit_at_k(ranks: list[int], k: int) -> int:
    return int(any(rank <= k for rank in ranks))


def reciprocal_rank(ranks: list[int]) -> float:
    return 1.0 / min(ranks) if ranks else 0.0


def ndcg_at_k(gold_set: set[str], negative_set: set[str], hits, k: int) -> float:
    gains = []
    for hit in hits[:k]:
        if hit.chunk.chunk_id in gold_set:
            gains.append(1.0)
        elif hit.chunk.chunk_id in negative_set:
            gains.append(0.0)
        else:
            gains.append(0.0)  # unjudged treated as non-relevant (partial judgment)
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_instance(
    index,
    instance: dict,
    item: dict,
    args: argparse.Namespace,
) -> dict:
    gold = set(item.get("gold_chunk_ids") or [])
    negatives = {negative["chunk_id"] for negative in item.get("hard_negatives") or []}
    variant = instance["variant"]
    as_of_date = variant.get("as_of_date") if variant else None
    resolved, companies, years = parse_for_filter(instance["query"], as_of_date)
    if getattr(args, "rewriter", None) is not None:
        resolved = args.rewriter.rewrite(resolved)
    elif getattr(args, "expand_synonyms", False):
        resolved = expand_query(resolved)

    per_filter: dict[str, dict] = {}
    for filter_name in FILTERS:
        filters = build_filters(companies, years) if filter_name == "query_parser" else None
        row: dict = {}
        for mode in MODES:
            if mode == "dense":
                dense_batch = index.search_dense_batch(
                    [resolved], top_k=args.candidate_k, filters=[filters]
                )
                hits = dense_batch[0][: args.top_k]
                candidate = dense_batch[0]
            elif mode == "lexical":
                hits = index.search_lexical(resolved, args.top_k, filters)
                candidate = index.search_lexical(resolved, args.candidate_k, filters)
            else:
                lexical = index.search_lexical(resolved, args.candidate_k, filters)
                dense = index.search_dense_batch(
                    [resolved], top_k=args.candidate_k, filters=[filters]
                )[0]
                candidate = reciprocal_rank_fusion(
                    lexical,
                    dense,
                    top_k=args.candidate_k,
                    rrf_k=args.rrf_k,
                    lexical_weight=args.lexical_weight,
                    dense_weight=args.dense_weight,
                )
                hits = candidate[: args.top_k]

            ranks = [
                hit.rank for hit in hits if hit.chunk.chunk_id in gold
            ]
            negative_ranks = [
                hit.rank for hit in hits if hit.chunk.chunk_id in negatives
            ]
            candidate_ranks = [
                hit.rank for hit in candidate if hit.chunk.chunk_id in gold
            ]
            relevant_in_top = sum(1 for hit in hits if hit.chunk.chunk_id in gold)
            evidence_tokens = [
                hit.chunk.estimated_token_count for hit in hits[: args.top_k]
            ]
            row[mode] = {
                "hit_at_5": hit_at_k(ranks, args.top_k),
                "mrr_at_5": reciprocal_rank(ranks),
                "recall_at_5": relevant_in_top / len(gold) if gold else 0.0,
                "precision_at_5": relevant_in_top / args.top_k,
                "ndcg_at_5": ndcg_at_k(gold, negatives, hits, args.top_k),
                "candidate_recall": bool(candidate_ranks),
                "avg_evidence_tokens_top5": (
                    sum(evidence_tokens) / len(evidence_tokens) if evidence_tokens else 0.0
                ),
                "first_gold_rank": min(ranks, default=None),
                "first_candidate_rank": min(candidate_ranks, default=None),
                "negative_count_in_top5": len(negative_ranks),
            }
        per_filter[filter_name] = row

    return {
        "query_id": instance["query_id"],
        "canonical_id": item["query_id"],
        "regime": "canonical" if variant is None else variant["query_regime"],
        "variant_types": variant.get("variant_types") if variant else [],
        "query": instance["query"],
        "resolved_query": resolved,
        "as_of_date": as_of_date,
        "answerability": item["answerability"],
        "retrieval_judgment": item["retrieval_judgment"],
        "company_signals": companies,
        "year_signals": years,
        "gold_count": len(gold),
        "hard_negative_count": len(negatives),
        "results": per_filter,
    }


def aggregate(rows: list[dict], filter_name: str, mode: str, by_canonical: bool) -> dict:
    if by_canonical:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["canonical_id"], []).append(row)
        metric_rows: list[dict] = []
        for group in grouped.values():
            metric_rows.append(
                {
                    metric: mean(item["results"][filter_name][mode][metric] for item in group)
                    for metric in AGGREGATE_METRICS
                }
            )
    else:
        metric_rows = [row["results"][filter_name][mode] for row in rows]
    return {
        metric: mean(row[metric] for row in metric_rows)
        for metric in AGGREGATE_METRICS
    }


def render_markdown(summary: dict, args: argparse.Namespace) -> str:
    lines = [
        "# Variant-regime retrieval evaluation v1",
        "",
        f"- dataset: `{summary['dataset_id']}` | index: `{summary['index_id']}`",
        (
            f"- top_k={args.top_k} | candidate_k={args.candidate_k} | "
            f"rrf_k={args.rrf_k} | weights lexical={args.lexical_weight}:dense={args.dense_weight}"
        ),
        (
            f"- positive instances: {summary['positive_instance_count']} "
            f"(canonical groups: {summary['positive_group_count']})"
        ),
        "",
        "指标均为 partial-judgment（gold 相关、hard negatives 负例、其余 unjudged）。",
        "",
        "## Query-level 平均（每个 query instance 一票）",
        "",
        "| regime | filter | mode | Hit@5 | MRR@5 | Recall@5 | Precision@5 | NDCG@5 | cand_recall | neg_in_top5 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for regime in summary["regimes"]:
        for filter_name in FILTERS:
            for mode in MODES:
                m = summary["by_regime"][regime][filter_name][mode]
                lines.append(
                    f"| {regime} | {filter_name} | {mode} | "
                    f"{m['hit_at_5']:.4f} | {m['mrr_at_5']:.4f} | {m['recall_at_5']:.4f} | "
                    f"{m['precision_at_5']:.4f} | {m['ndcg_at_5']:.4f} | "
                    f"{m['candidate_recall']:.4f} | {m['negative_count_in_top5']:.4f} |"
                )
    lines.extend(
        [
            "",
            "## Canonical-group 平均（先组内聚合再平均，防伪重复）",
            "",
            "| regime | filter | mode | Hit@5 | MRR@5 | Recall@5 | NDCG@5 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for regime in summary["regimes"]:
        for filter_name in FILTERS:
            for mode in MODES:
                m = summary["by_regime_group"][regime][filter_name][mode]
                lines.append(
                    f"| {regime} | {filter_name} | {mode} | "
                    f"{m['hit_at_5']:.4f} | {m['mrr_at_5']:.4f} | "
                    f"{m['recall_at_5']:.4f} | {m['ndcg_at_5']:.4f} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.rewriter = make_rewriter(args.rewrite)
    view = json.loads(args.view.read_text(encoding="utf-8"))
    index = resolve_current_index(args.index_root)
    positive = [item for item in view["items"] if item["retrieval_judgment"] == "positive_gold"]
    behavior = [
        item for item in view["items"]
        if item["retrieval_judgment"] in ("negative_no_positive_gold", "clarification_no_positive_gold")
    ]
    rows = [evaluate_instance(index, instance, item, args) for item in positive for instance in query_instances(item)]
    behavior_rows = [
        evaluate_instance(index, instance, item, args)
        for item in behavior
        for instance in query_instances(item)
    ]

    regimes = ["canonical", "ticker_or_finance_shorthand", "semantic_or_relative_time"]
    by_regime = {
        regime: {
            filter_name: {
                mode: aggregate(
                    [row for row in rows if row["regime"] == regime],
                    filter_name,
                    mode,
                    by_canonical=False,
                )
                for mode in MODES
            }
            for filter_name in FILTERS
        }
        for regime in regimes
    }
    by_regime_group = {
        regime: {
            filter_name: {
                mode: aggregate(
                    [row for row in rows if row["regime"] == regime],
                    filter_name,
                    mode,
                    by_canonical=True,
                )
                for mode in MODES
            }
            for filter_name in FILTERS
        }
        for regime in regimes
    }

    summary = {
        "run_id": "variant-regime-v1",
        "dataset_id": view["dataset_id"],
        "index_id": index.manifest.index_id,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "rrf_weights": {"lexical": args.lexical_weight, "dense": args.dense_weight},
        "metadata_filter_source": {"none": "none", "query_parser": "query_parser"},
        "positive_instance_count": len(rows),
        "positive_group_count": len(positive),
        "behavior_instance_count": len(behavior_rows),
        "regimes": regimes,
        "by_regime": by_regime,
        "by_regime_group": by_regime_group,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "view": str(args.view),
                "index_root": str(args.index_root),
                "top_k": args.top_k,
                "candidate_k": args.candidate_k,
                "rrf_k": args.rrf_k,
                "rrf_weights": {"lexical": args.lexical_weight, "dense": args.dense_weight},
                "metadata_filter_source": {"none": "none", "query_parser": "query_parser"},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "per_query.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows + behavior_rows),
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
