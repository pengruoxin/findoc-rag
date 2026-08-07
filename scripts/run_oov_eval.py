"""Out-of-vocabulary paraphrase retrieval evaluation.

Measures whether the system can retrieve gold evidence for paraphrases that
are NOT covered by the deterministic synonym table, with and without LLM
query rewriting. This is the honest test of semantic robustness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from run_retrieval_variant_eval import (
    AGGREGATE_METRICS,
    FILTERS,
    MODES,
    evaluate_instance,
    make_rewriter,
)

from findoc_rag.corpus import resolve_current_index

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOV = ROOT / "data/evaluation/oov-variants-v1.json"
DEFAULT_VIEW = ROOT / "data/evaluation/benchmark-v2-retrieval-view.json"
DEFAULT_INDEX_ROOT = ROOT / "data/indexes/corpus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oov", type=Path, default=DEFAULT_OOV)
    parser.add_argument("--view", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--lexical-weight", type=float, default=2.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--rewrite", choices=("none", "deterministic", "llm"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def aggregate(rows: list[dict], filter_name: str, mode: str) -> dict:
    return {
        metric: mean(row["results"][filter_name][mode][metric] for row in rows)
        for metric in AGGREGATE_METRICS
    }


def main() -> None:
    args = parse_args()
    oov = json.loads(args.oov.read_text(encoding="utf-8"))
    view = json.loads(args.view.read_text(encoding="utf-8"))
    index = resolve_current_index(args.index_root)
    by_id = {item["query_id"]: item for item in view["items"]}
    args.rewriter = make_rewriter(args.rewrite)

    rows = []
    for item in oov["items"]:
        gold_item = by_id[item["query_id"]]
        for number, query in enumerate(item["oov_queries"], start=1):
            instance = {
                "query_id": f"{item['query_id']}::oov{number}",
                "query": query,
                "variant": {
                    "query_regime": "oov",
                    "as_of_date": "2025-04-30",
                    "variant_types": ["oov"],
                },
            }
            rows.append(evaluate_instance(index, instance, gold_item, args))

    summary = {
        "run_id": f"oov-eval-{args.rewrite}",
        "dataset_id": oov["dataset_id"],
        "index_id": index.manifest.index_id,
        "rewrite_mode": args.rewrite,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "instance_count": len(rows),
        "question_count": len(oov["items"]),
        "by_filter_mode": {
            filter_name: {mode: aggregate(rows, filter_name, mode) for mode in MODES}
            for filter_name in FILTERS
        },
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {output_dir}")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "per_query.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
