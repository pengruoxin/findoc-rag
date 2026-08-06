"""Run Oracle-Context or Retrieved-Context generation evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from findoc_rag.answer_generation import MAX_GENERATION_CONTEXTS, GroundedAnswerGenerator
from findoc_rag.corpus import resolve_current_index
from findoc_rag.documents.models import DocumentChunk
from findoc_rag.generation_evaluation import (
    GenerationEvaluationDataset,
    GenerationRunItem,
    score_generation_run_item,
)
from findoc_rag.indexing import SearchFilters, SearchHit

ROOT = Path(__file__).resolve().parents[1]
PROMPT_REVISION = "evidence-first-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lane",
        choices=("oracle_context", "retrieved_context", "robustness"),
        required=True,
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/generation-eval-v1.json"))
    parser.add_argument("--index-root", type=Path, default=Path("data/indexes/corpus"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generation/runs"))
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--endpoint", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--require-remote", action="store_true")
    return parser.parse_args()


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


def retrieved_hits(item, index) -> list[SearchHit]:
    filters = SearchFilters(
        company_names=item.company_names,
        report_years=item.report_years,
    )
    return index.search(
        item.query,
        top_k=MAX_GENERATION_CONTEXTS,
        mode="hybrid",
        candidate_k=20,
        rrf_k=60,
        filters=filters if filters.active else None,
    )


def main() -> None:
    args = parse_args()
    dataset = GenerationEvaluationDataset.model_validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    api_key_set = bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if args.require_remote and not api_key_set:
        raise SystemExit("Remote generation requested, but DEEPSEEK_API_KEY is not set")
    generator = GroundedAnswerGenerator(
        model=args.model,
        endpoint=args.endpoint,
        enabled=api_key_set,
    )
    chunks = load_chunks()
    index = resolve_current_index(args.index_root) if args.lane == "retrieved_context" else None
    prompt_sha256 = hashlib.sha256(PROMPT_REVISION.encode()).hexdigest()
    run_id = f"{args.lane}-{dataset.dataset_id}-{args.model}"
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True)

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
        started = perf_counter()
        try:
            if args.lane == "oracle_context":
                hits = oracle_hits(item, chunks)
                context_labels = ["gold"] * len(hits)
            elif args.lane == "robustness":
                hits, context_labels = robustness_hits(item, chunks)
            else:
                hits = retrieved_hits(item, index)
                context_labels = ["retrieved"] * len(hits)
            answer = generator.generate(item.query, hits)
            run_item = GenerationRunItem(
                query_id=item.query_id,
                response=answer.answer,
                retrieved_contexts=[
                    hit.chunk.text[:1800] for hit in hits[:MAX_GENERATION_CONTEXTS]
                ],
                retrieved_chunk_ids=[
                    hit.chunk.chunk_id for hit in hits[:MAX_GENERATION_CONTEXTS]
                ],
                context_labels=context_labels[:MAX_GENERATION_CONTEXTS],
                provider=answer.provider,
                model=args.model if answer.provider == "openai-compatible" else "deterministic",
                index_id=index.manifest.index_id if index else dataset.corpus_index_id,
                prompt_sha256=prompt_sha256,
                latency_ms=(perf_counter() - started) * 1000,
                grounded=answer.grounded,
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
                query_id=item.query_id,
                response="",
                retrieved_contexts=[],
                retrieved_chunk_ids=[],
                context_labels=[],
                provider="error",
                model=args.model,
                index_id=index.manifest.index_id if index else dataset.corpus_index_id,
                prompt_sha256=prompt_sha256,
                latency_ms=(perf_counter() - started) * 1000,
                grounded=False,
                observed_behavior="abstain",
                error=str(exc),
            )
        run_items.append(run_item)
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
    summary = {
        "run_id": run_id,
        "dataset_id": dataset.dataset_id,
        "lane": args.lane,
        "model": args.model,
        "remote_generation": api_key_set,
        "dataset_item_count": dataset.item_count,
        "item_count": len(scores),
        "hard_negative_count": sum(
            len(item.hard_negatives) for item in evaluation_items
        ),
        "strict_success_eligible_count": len(strict_scores),
        "strict_success_rate": sum(score.strict_success for score in strict_scores)
        / len(strict_scores),
        "expected_behavior_accuracy": sum(score.expected_behavior_correct for score in scores)
        / len(scores),
        "run_error_rate": sum(item.error is not None for item in run_items) / len(run_items),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
