"""Run Oracle-Context or Retrieved-Context generation evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter

from findoc_rag.answer_generation import (
    MAX_GENERATION_CONTEXT_CHARS,
    MAX_GENERATION_CONTEXTS,
    GroundedAnswerGenerator,
)
from findoc_rag.benchmark_assets import benchmark_chunk_paths, validate_benchmark_lock
from findoc_rag.benchmark_migration import (
    resolve_evaluation_index_id,
    validate_migration_manifest,
)
from findoc_rag.chunking import estimate_tokens
from findoc_rag.corpus import resolve_current_index
from findoc_rag.documents.models import DocumentChunk
from findoc_rag.evaluation.reporting import build_generation_quality_report
from findoc_rag.generation_evaluation import (
    GenerationEvaluationDataset,
    GenerationRunItem,
    score_generation_run_item,
)
from findoc_rag.indexing import SearchFilters, SearchHit
from findoc_rag.provider_credentials import resolve_provider_api_key
from findoc_rag.query_expansion import expand_query
from findoc_rag.query_rewriting import LLMQueryRewriter
from findoc_rag.query_routing import route_finance_query
from findoc_rag.scope_routing import plan_candidate_budget, route_structured_evidence
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
    parser.add_argument(
        "--split",
        choices=("calibration", "dev", "frozen_test"),
        help="evaluate only one split and enforce its corpus_indexes binding",
    )
    parser.add_argument(
        "--migration-manifest",
        type=Path,
        help="validated benchmark migration binding for a replacement index",
    )
    parser.add_argument(
        "--migration-view",
        type=Path,
        default=Path("data/evaluation/benchmark-v2-retrieval-view.json"),
    )
    parser.add_argument(
        "--source-evidence",
        type=Path,
        default=Path("data/evaluation/benchmark-evidence-v1.jsonl"),
    )
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
        "--disable-remote",
        action="store_true",
        help="force deterministic/evidence-only generation even when a provider key exists",
    )
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
    parser.add_argument(
        "--oracle-metadata",
        action="store_true",
        help=(
            "diagnostic only: filter retrieved context with benchmark company/year metadata; "
            "the default derives filters only from the query"
        ),
    )
    args = parser.parse_args()
    if args.oracle_metadata and args.lane != "retrieved_context":
        parser.error("--oracle-metadata is only valid for --lane retrieved_context")
    if args.disable_remote and args.require_remote:
        parser.error("--disable-remote and --require-remote are mutually exclusive")
    return args


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


def code_fingerprint() -> str:
    """Hash executable project inputs so dirty runs remain distinguishable."""
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "configs").glob("*.toml")),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_chunks(index=None, chunk_ids: set[str] | None = None) -> dict[str, DocumentChunk]:
    if index is not None:
        requested = sorted(chunk_ids or set())
        resolved = index.resolve_chunks(requested)
        missing = [
            chunk_id for chunk_id, chunk in zip(requested, resolved, strict=True) if chunk is None
        ]
        if missing:
            raise ValueError(
                "Validated evaluation index is missing benchmark chunks: " + ", ".join(missing)
            )
        return {chunk.chunk_id: chunk for chunk in resolved if chunk is not None}
    chunks: dict[str, DocumentChunk] = {}
    for path in benchmark_chunk_paths(ROOT):
        for line in path.read_text(encoding="utf-8").splitlines():
            chunk = DocumentChunk.model_validate_json(line)
            chunks[chunk.chunk_id] = chunk
    return chunks


def oracle_hits(item, chunks: dict[str, DocumentChunk]) -> list[SearchHit]:
    hits = []
    for rank, chunk_id in enumerate(item.gold_chunk_ids, start=1):
        hits.append(SearchHit(rank=rank, chunk=chunks[chunk_id], score=1.0))
    return hits


def robustness_hits(
    item,
    chunks: dict[str, DocumentChunk],
) -> tuple[list[SearchHit], list[str]]:
    """Interleave verified evidence with source-backed distractors."""
    gold = oracle_hits(item, chunks)
    negative_budget = MAX_GENERATION_CONTEXTS - len(gold)
    if negative_budget < 1:
        raise ValueError(f"Robustness case {item.query_id} leaves no room for a hard negative")
    negatives: list[tuple[SearchHit, str]] = []
    for negative in item.hard_negatives[:negative_budget]:
        chunk = chunks[negative.chunk_id]
        negatives.append(
            (
                SearchHit(rank=1, chunk=chunk, score=1.1),
                f"hard_negative:{negative.negative_type}",
            )
        )

    ordered: list[tuple[SearchHit, str]] = []
    for index in range(max(len(gold), len(negatives))):
        if index < len(negatives):
            ordered.append(negatives[index])
        if index < len(gold):
            ordered.append((gold[index], "gold"))
    hits = [hit.model_copy(update={"rank": rank}) for rank, (hit, _) in enumerate(ordered, start=1)]
    return hits, [label for _, label in ordered]


def retrieved_hits(
    item,
    search_query: str,
    index,
    *,
    routing_query: str | None = None,
    oracle_metadata: bool = False,
) -> tuple[list[SearchHit], str]:
    if oracle_metadata:
        filters = SearchFilters(
            company_names=item.company_names,
            report_years=item.report_years,
        )
        filter_source = "oracle_metadata"
    else:
        route = route_finance_query(routing_query or search_query)
        filters = SearchFilters(
            company_names=route.company_names,
            report_years=route.report_years,
        )
        filter_source = "query_router" if filters.active else "none"
    route_query = routing_query or search_query
    _, budget = plan_candidate_budget(
        route_query,
        20,
        maximum_candidate_k=100,
        enabled=True,
    )
    candidates = index.search(
        search_query,
        top_k=budget.effective_candidate_k,
        mode="lexical",
        candidate_k=budget.effective_candidate_k,
        rrf_k=60,
        filters=filters if filters.active else None,
    )
    hits = route_structured_evidence(
        route_query,
        candidates,
        MAX_GENERATION_CONTEXTS,
    )
    return hits, filter_source


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
    if args.dataset.resolve() == (ROOT / "data/evaluation/benchmark-v2.json").resolve():
        validate_benchmark_lock(ROOT)
    dataset = GenerationEvaluationDataset.model_validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    selected_items = [
        item for item in dataset.items if args.split is None or item.split == args.split
    ]
    if not selected_items:
        raise ValueError(f"Dataset has no items for split {args.split!r}")
    selected_splits = {item.split for item in selected_items}
    selected_index_ids = {
        dataset.corpus_indexes.get(split, dataset.corpus_index_id) for split in selected_splits
    }
    if args.split is None and len(selected_index_ids) > 1:
        raise ValueError(
            "Dataset spans isolated indexes; pass --split to select one evaluation boundary"
        )
    api_key_set = not args.disable_remote and bool(resolve_provider_api_key(args.endpoint))
    if args.require_remote and not api_key_set:
        raise SystemExit(
            "Remote generation requested, but no key is configured for the endpoint host"
        )
    generator = GroundedAnswerGenerator(
        model=args.api_model,
        endpoint=args.endpoint,
        enabled=api_key_set,
    )
    index = resolve_current_index(args.index_root)
    migration = None
    if index is not None and args.migration_manifest is not None:
        migration = json.loads(args.migration_manifest.read_text(encoding="utf-8"))
        migration_result = validate_migration_manifest(
            migration,
            view_path=args.migration_view,
            source_evidence_path=args.source_evidence,
            target_index_root=args.index_root,
        )
        if not migration_result.ok:
            raise ValueError(
                "Benchmark migration validation failed: " + "; ".join(migration_result.errors)
            )
    if index is not None:
        expected_index_id = (
            dataset.corpus_indexes.get(args.split, dataset.corpus_index_id)
            if args.split
            else dataset.corpus_index_id
        )
        resolve_evaluation_index_id(
            view={"corpus_index_id": expected_index_id},
            index_id=index.manifest.index_id,
            migration_manifest=migration,
        )
    judged_chunk_ids = {
        chunk_id
        for item in selected_items
        for chunk_id in (
            *item.gold_chunk_ids,
            *(negative.chunk_id for negative in item.hard_negatives),
        )
    }
    chunks = load_chunks(index, judged_chunk_ids)
    prompt_sha256 = hashlib.sha256(PROMPT_REVISION.encode()).hexdigest()
    split_suffix = f"-{args.split}" if args.split else ""
    run_suffix = split_suffix if args.variant == "canonical" else f"{split_suffix}-{args.variant}"
    run_id = f"{args.lane}-{dataset.dataset_id}-{args.model}{run_suffix}"
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists and will not be overwritten: {run_dir}"
        )
    run_dir.mkdir(parents=True)
    rewriter = (
        LLMQueryRewriter(cache_path=run_dir / "rewrites.json") if args.rewrite == "llm" else None
    )

    evaluation_items = (
        [item for item in selected_items if item.hard_negatives]
        if args.lane == "robustness"
        else selected_items
    )
    if not evaluation_items:
        raise ValueError(f"No dataset items are eligible for lane {args.lane}")

    run_items: list[GenerationRunItem] = []
    scores = []
    filter_source_counts: dict[str, int] = {}
    for item in evaluation_items:
        for instance in query_instances(item, args.variant):
            started = perf_counter()
            filter_source = "not_applicable"
            as_of_date = parse_as_of_date(
                instance["variant"].as_of_date if instance["variant"] else None
            )
            resolved_query = instance["query"]
            search_query: str | None = None
            time_cues: list[str] = []
            try:
                if as_of_date is not None:
                    resolved_query, time_cues = resolve_relative_time(instance["query"], as_of_date)
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
                    hits, filter_source = retrieved_hits(
                        item,
                        search_query,
                        index,
                        routing_query=resolved_query,
                        oracle_metadata=args.oracle_metadata,
                    )
                    filter_source_counts[filter_source] = (
                        filter_source_counts.get(filter_source, 0) + 1
                    )
                    context_labels = ["retrieved"] * len(hits)
                answer = generator.generate(resolved_query, hits)
                is_remote = answer.provider in {"openai-compatible", "remote-abstention"}
                run_item = GenerationRunItem(
                    query_id=instance["query_id"],
                    response=answer.answer,
                    retrieved_contexts=[
                        hit.chunk.text[:MAX_GENERATION_CONTEXT_CHARS]
                        for hit in hits[:MAX_GENERATION_CONTEXTS]
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
                    search_query=(search_query if args.lane == "retrieved_context" else None),
                    filter_source=filter_source,
                    time_cues=time_cues,
                    context_tokens=sum(
                        estimate_tokens(text)
                        for text in (
                            hit.chunk.text[:MAX_GENERATION_CONTEXT_CHARS]
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
                    filter_source=filter_source,
                    time_cues=time_cues,
                    observed_behavior=None,
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
    context_tokens = [item.context_tokens for item in run_items if item.context_tokens is not None]
    latencies = sorted(item.latency_ms for item in run_items)
    p95_latency_ms = (
        latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None
    )
    revision, dirty = code_revision()
    remote_providers = {"openai-compatible", "remote-abstention"}
    remote_success_count = sum(
        item.provider in remote_providers and item.error is None for item in run_items
    )
    run_error_count = sum(item.error is not None for item in run_items)
    summary = {
        "run_id": run_id,
        "dataset_id": dataset.dataset_id,
        "source_index_id": dataset.corpus_index_id,
        "migration_id": migration.get("migration_id") if migration else None,
        "lane": args.lane,
        "split": args.split,
        "model": args.model,
        "api_model": args.api_model,
        "rewrite_mode": args.rewrite,
        "filter_source": (
            "not_applicable"
            if args.lane != "retrieved_context"
            else "oracle_metadata"
            if args.oracle_metadata
            else "query_derived"
        ),
        "filter_source_counts": filter_source_counts,
        "evidence_routing": "structured-table-v1",
        "candidate_budget_policy": "scope-adaptive-20-to-100",
        "code_revision": revision,
        "code_dirty": dirty,
        "code_fingerprint": code_fingerprint(),
        "variant_mode": args.variant,
        # Keep capability configuration separate from observed execution.  A
        # configured key does not prove that a remote request succeeded.
        "remote_configured": api_key_set,
        "remote_generation": remote_success_count > 0,
        "remote_success_count": remote_success_count,
        "dataset_item_count": dataset.item_count,
        "selected_dataset_item_count": len(selected_items),
        "item_count": len(run_items),
        "hard_negative_count": sum(len(item.hard_negatives) for item in evaluation_items),
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
        "run_error_count": run_error_count,
        "run_error_rate": run_error_count / len(run_items),
    }
    quality_report = build_generation_quality_report(dataset, run_items, scores)
    (run_dir / "quality-report.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["provider_counts"] = quality_report["overall"]["provider_counts"]
    summary["quality_report"] = "quality-report.json"
    summary["confidence_intervals"] = {
        "strict_success": quality_report["overall"]["strict_success"],
        "expected_behavior_accuracy": quality_report["overall"]["expected_behavior_accuracy"],
        "run_error_rate": quality_report["overall"]["run_error_rate"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_remote and run_error_count:
        raise SystemExit(
            "Remote generation run is incomplete: "
            f"{run_error_count}/{len(run_items)} items failed; artifacts were retained for audit"
        )


if __name__ == "__main__":
    main()
