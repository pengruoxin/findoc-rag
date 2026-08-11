"""Run Oracle-Context or Retrieved-Context generation evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

from findoc_rag.answer_generation import MAX_GENERATION_CONTEXTS, GroundedAnswerGenerator
from findoc_rag.chunking import estimate_tokens
from findoc_rag.corpus import resolve_current_index
from findoc_rag.documents.models import DocumentChunk
from findoc_rag.generation_evaluation import (
    GenerationEvaluationDataset,
    GenerationRunItem,
    score_generation_run_item,
)
from findoc_rag.indexing import SearchFilters, SearchHit
from findoc_rag.query_expansion import expand_query
from findoc_rag.query_rewriting import LLMQueryRewriter
from findoc_rag.time_utils import parse_as_of_date, resolve_relative_time

ROOT = Path(__file__).resolve().parents[1]
PROMPT_REVISION = "evidence-first-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lane",
        choices=("oracle_context", "retrieved_context", "robustness"),
        required=True,
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/benchmark-v2.json"))
    parser.add_argument("--index-root", type=Path, default=Path("data/indexes/corpus"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generation/runs"))
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument(
        "--api-model",
        default="deepseek-chat",
        help="model name sent to the API (--model is only a run label)",
    )
    parser.add_argument("--endpoint", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--require-remote", action="store_true")
    parser.add_argument(
        "--variant",
        choices=("canonical", "ticker_or_finance_shorthand", "semantic_or_relative_time", "all"),
        default="canonical",
        help="query instances to run: canonical, one variant regime, or all",
    )
    parser.add_argument(
        "--rewrite",
        choices=("none", "deterministic", "llm"),
        default="deterministic",
        help="retrieved-lane query rewrite mode (deterministic preserves the baseline)",
    )
    return parser.parse_args()


def code_revision() -> tuple[str, bool]:
    """Return (git HEAD sha, dirty flag) for controlled experiment records."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
        )
        return head, dirty
    except (subprocess.SubprocessError, OSError):
        return "unknown", False


def load_chunks() -> dict[str, DocumentChunk]:
    chunks: dict[str, DocumentChunk] = {}
    for path in (ROOT / "data/catalog/versions").glob("*/chunks.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            chunk = DocumentChunk.model_validate_json(line)
            chunks[chunk.chunk_id] = chunk
    return chunks


def oracle_hits(item, chunks: dict[str, DocumentChunk]) -> list[SearchHit]:
    hits = []
    for rank, chunk_id in enumerate(item.gold_chunk_ids, start=1):
        company_name = "贵州茅台" if chunk_id.startswith("5299f4940e2c:") else "伊利股份"
        enriched = chunks[chunk_id].model_copy(
            update={"company_name": company_name, "report_year": 2024}
        )
        hits.append(SearchHit(rank=rank, chunk=enriched, score=1.0))
    return hits


def robustness_hits(
    item,
    chunks: dict[str, DocumentChunk],
) -> tuple[list[SearchHit], list[str]]:
    """Interleave verified evidence with source-backed distractors."""
    gold = oracle_hits(item, chunks)
    negative_budget = MAX_GENERATION_CONTEXTS - len(gold)
    if negative_budget < 1:
        raise ValueError(
            f"Robustness case {item.query_id} leaves no room for a hard negative"
        )
    negatives: list[tuple[SearchHit, str]] = []
    for negative in item.hard_negatives[:negative_budget]:
        chunk = chunks[negative.chunk_id]
        company_name = (
            "贵州茅台" if negative.chunk_id.startswith("5299f4940e2c:") else "伊利股份"
        )
        enriched = chunk.model_copy(
            update={"company_name": company_name, "report_year": 2024}
        )
        negatives.append(
            (
                SearchHit(rank=1, chunk=enriched, score=1.1),
                f"hard_negative:{negative.negative_type}",
            )
        )

    ordered: list[tuple[SearchHit, str]] = []
    for index in range(max(len(gold), len(negatives))):
        if index < len(negatives):
            ordered.append(negatives[index])
        if index < len(gold):
            ordered.append((gold[index], "gold"))
    hits = [
        hit.model_copy(update={"rank": rank})
        for rank, (hit, _) in enumerate(ordered, start=1)
    ]
    return hits, [label for _, label in ordered]


def retrieved_hits(item, query, index) -> list[SearchHit]:
    filters = SearchFilters(
        company_names=item.company_names,
        report_years=item.report_years,
    )
    return index.search(
        query,
        top_k=MAX_GENERATION_CONTEXTS,
        mode="lexical",
        candidate_k=20,
        rrf_k=60,
        filters=filters if filters.active else None,
    )


def query_instances(item, variant_mode: str) -> list[dict]:
    """Yield canonical and/or variant query instances with stable IDs."""
    instances = []
    if variant_mode in ("canonical", "all"):
        instances.append({"query_id": item.query_id, "query": item.query, "variant": None})
    for variant in item.query_variants:
        if variant_mode == "all" or variant.query_regime == variant_mode:
            instances.append(
                {
                    "query_id": f"{item.query_id}::{variant.variant_id}",
                    "query": variant.query,
                    "variant": variant,
                }
            )
    return instances


def main() -> None:
    args = parse_args()
    dataset = GenerationEvaluationDataset.model_validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    api_key_set = bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if args.require_remote and not api_key_set:
        raise SystemExit("Remote generation requested, but DEEPSEEK_API_KEY is not set")
    generator = GroundedAnswerGenerator(
        model=args.api_model,
        endpoint=args.endpoint,
        enabled=api_key_set,
    )
    chunks = load_chunks()
    index = resolve_current_index(args.index_root) if args.lane == "retrieved_context" else None
    prompt_sha256 = hashlib.sha256(PROMPT_REVISION.encode()).hexdigest()
    run_suffix = "" if args.variant == "canonical" else f"-{args.variant}"
    run_id = f"{args.lane}-{dataset.dataset_id}-{args.model}{run_suffix}"
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True)
    rewriter = (
        LLMQueryRewriter(cache_path=run_dir / "rewrites.json")
        if args.rewrite == "llm"
        else None
    )

    evaluation_items = (
        [item for item in dataset.items if item.hard_negatives]
        if args.lane == "robustness"
        else dataset.items
    )
    if not evaluation_items:
        raise ValueError(f"No dataset items are eligible for lane {args.lane}")

    run_items: list[GenerationRunItem] = []
    scores = []
    for item in evaluation_items:
        for instance in query_instances(item, args.variant):
            started = perf_counter()
            as_of_date = parse_as_of_date(
                instance["variant"].as_of_date if instance["variant"] else None
            )
            resolved_query = instance["query"]
            search_query: str | None = None
            time_cues: list[str] = []
            try:
                if as_of_date is not None:
                    resolved_query, time_cues = resolve_relative_time(
                        instance["query"], as_of_date
                    )
                if args.lane == "oracle_context":
                    hits = oracle_hits(item, chunks)
                    context_labels = ["gold"] * len(hits)
                elif args.lane == "robustness":
                    hits, context_labels = robustness_hits(item, chunks)
                else:
                    search_query = resolved_query
                    if args.rewrite == "deterministic":
                        search_query = expand_query(resolved_query)
                    elif args.rewrite == "llm" and rewriter is not None:
                        search_query = rewriter.rewrite(resolved_query)
                    hits = retrieved_hits(item, search_query, index)
                    context_labels = ["retrieved"] * len(hits)
                answer = generator.generate(resolved_query, hits)
                is_remote = answer.provider in {"openai-compatible", "remote-abstention"}
                run_item = GenerationRunItem(
                    query_id=instance["query_id"],
                    response=answer.answer,
                    retrieved_contexts=[
                        hit.chunk.text[:1800] for hit in hits[:MAX_GENERATION_CONTEXTS]
                    ],
                    retrieved_chunk_ids=[
                        hit.chunk.chunk_id for hit in hits[:MAX_GENERATION_CONTEXTS]
                    ],
                    context_labels=context_labels[:MAX_GENERATION_CONTEXTS],
                    provider=answer.provider,
                    model=args.model if is_remote else "deterministic",
                    api_model=args.api_model if is_remote else None,
                    index_id=index.manifest.index_id if index else dataset.corpus_index_id,
                    prompt_sha256=prompt_sha256,
                    latency_ms=(perf_counter() - started) * 1000,
                    grounded=answer.grounded,
                    as_of_date=str(as_of_date) if as_of_date else None,
                    resolved_query=resolved_query,
                    search_query=(
                        search_query if args.lane == "retrieved_context" else None
                    ),
                    time_cues=time_cues,
                    context_tokens=sum(
                        estimate_tokens(text)
                        for text in (
                            hit.chunk.text[:1800]
                            for hit in hits[:MAX_GENERATION_CONTEXTS]
                        )
                    ),
                    observed_behavior=(
                        "clarify"
                        if answer.provider == "clarification"
                        else "answer"
                        if answer.grounded
                        else "abstain"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                run_item = GenerationRunItem(
                    query_id=instance["query_id"],
                    response="",
                    retrieved_contexts=[],
                    retrieved_chunk_ids=[],
                    context_labels=[],
                    provider="error",
                    model=args.model,
                    api_model=None,
                    index_id=index.manifest.index_id if index else dataset.corpus_index_id,
                    prompt_sha256=prompt_sha256,
                    latency_ms=(perf_counter() - started) * 1000,
                    grounded=False,
                    as_of_date=str(as_of_date) if as_of_date else None,
                    resolved_query=resolved_query,
                    search_query=search_query,
                    time_cues=time_cues,
                    observed_behavior="abstain",
                    error=str(exc),
                )
            run_items.append(run_item)
            if args.variant == "canonical":
                scores.append(score_generation_run_item(item, run_item))

    (run_dir / "items.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in run_items),
        encoding="utf-8",
    )
    (run_dir / "deterministic-scores.jsonl").write_text(
        "".join(score.model_dump_json() + "\n" for score in scores),
        encoding="utf-8",
    )
    strict_scores = [score for score in scores if score.strict_success_eligible]
    context_tokens = [
        item.context_tokens for item in run_items if item.context_tokens is not None
    ]
    latencies = sorted(item.latency_ms for item in run_items)
    p95_latency_ms = (
        latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        if latencies
        else None
    )
    revision, dirty = code_revision()
    summary = {
        "run_id": run_id,
        "dataset_id": dataset.dataset_id,
        "lane": args.lane,
        "model": args.model,
        "api_model": args.api_model,
        "rewrite_mode": args.rewrite,
        "code_revision": revision,
        "code_dirty": dirty,
        "variant_mode": args.variant,
        "remote_generation": api_key_set,
        "dataset_item_count": dataset.item_count,
        "item_count": len(run_items),
        "hard_negative_count": sum(
            len(item.hard_negatives) for item in evaluation_items
        ),
        "strict_success_eligible_count": len(strict_scores),
        "strict_success_rate": (
            sum(score.strict_success for score in strict_scores) / len(strict_scores)
            if strict_scores
            else None
        ),
        "expected_behavior_accuracy": (
            sum(score.expected_behavior_correct for score in scores) / len(scores)
            if scores
            else None
        ),
        "avg_context_tokens": (
            sum(context_tokens) / len(context_tokens) if context_tokens else None
        ),
        "p95_latency_ms": p95_latency_ms,
        "run_error_rate": sum(item.error is not None for item in run_items) / len(run_items),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
