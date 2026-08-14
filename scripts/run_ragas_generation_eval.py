"""Run the four RAGAS generation metrics against an immutable generation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.benchmark_migration import validate_migration_manifest
from findoc_rag.generation_evaluation import (
    GenerationEvaluationDataset,
    select_ragas_run_items,
)
from findoc_rag.provider_credentials import resolve_provider_api_key
from findoc_rag.ragas_coverage import (
    count_complete_metric_rows,
    summarize_metric_coverage,
)
from findoc_rag.ragas_runner import load_and_validate_run_manifest, load_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_jsonl", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/benchmark-v2.json"))
    parser.add_argument(
        "--migration-manifest",
        type=Path,
        help="validated benchmark migration binding used by the generation run",
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
    parser.add_argument("--target-index-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/generation/ragas-evaluation-v1.json"))
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--endpoint", default="https://api.deepseek.com")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = GenerationEvaluationDataset.model_validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    run_items = load_run(args.run_jsonl)
    run_summary, lane = load_and_validate_run_manifest(
        args.run_jsonl, dataset, run_items
    )
    allowed_index_ids = {dataset.corpus_index_id}
    migration_id = None
    if args.migration_manifest is not None:
        if args.target_index_root is None:
            raise ValueError("--target-index-root is required with --migration-manifest")
        migration = json.loads(args.migration_manifest.read_text(encoding="utf-8"))
        migration_result = validate_migration_manifest(
            migration,
            view_path=args.migration_view,
            source_evidence_path=args.source_evidence,
            target_index_root=args.target_index_root,
        )
        if not migration_result.ok:
            raise ValueError(
                "Benchmark migration validation failed: "
                + "; ".join(migration_result.errors)
            )
        migration_id = migration["migration_id"]
        if run_summary.get("migration_id") != migration_id:
            raise ValueError("Generation run migration_id does not match migration manifest")
        allowed_index_ids = {migration["target_index"]["index_id"]}
    elif run_summary.get("migration_id") is not None:
        raise ValueError("Migrated generation run requires --migration-manifest")
    answerable, selection = select_ragas_run_items(
        dataset,
        run_items,
        lane,
        allowed_index_ids=allowed_index_ids,
    )
    run = {item.query_id: item for item in run_items}

    api_key = resolve_provider_api_key(args.endpoint)
    if not api_key:
        raise SystemExit("A provider-specific API key is required for RAGAS LLM judging")

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import ContextRelevance, Faithfulness, LLMContextRecall, ResponseRelevancy

    samples = [
        SingleTurnSample(
            user_input=item.query,
            response=run[item.query_id].response,
            retrieved_contexts=run[item.query_id].retrieved_contexts,
            reference_contexts=[evidence.verbatim_quote for evidence in item.gold_evidence],
            reference=item.reference_answer,
        )
        for item in answerable
    ]
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=args.judge_model,
            api_key=api_key,
            base_url=args.endpoint,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=args.embedding_model)
    )
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        ContextRelevance(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=metrics,
        raise_exceptions=False,
        experiment_name=f"findoc-rag-{dataset.dataset_id}",
    )
    frame = result.to_pandas()
    rows = frame.to_dict(orient="records")
    for query_id, row in zip(selection.eligible_query_ids, rows, strict=True):
        row["query_id"] = query_id
    metric_names = [metric.name for metric in metrics]
    # --model is only a run label; the actual API model is what matters for
    # judging independence. Older runs without api_model fall back to the label.
    answer_models = sorted(
        {
            run[item.query_id].api_model or run[item.query_id].model
            for item in answerable
        }
    )
    remote_answer_items = [
        run[item.query_id]
        for item in answerable
        if run[item.query_id].provider in {"openai-compatible", "remote-abstention"}
    ]
    # Deterministic table/guardrail outputs intentionally have no API model.
    # Audit only records that actually claim a remote provider.
    api_model_recorded = all(item.api_model is not None for item in remote_answer_items)
    independent_judge = args.judge_model not in answer_models
    summary, metric_coverage = summarize_metric_coverage(rows, metric_names)
    # RAGAS may omit a metric column when every judge call for that metric
    # fails. Count from serialized rows so this remains an auditable 0 rather
    # than raising a pandas KeyError and losing the coverage report entirely.
    complete_row_count = count_complete_metric_rows(rows, metric_names)
    payload = {
        "run_id": run_summary["run_id"],
        "run_jsonl": str(args.run_jsonl),
        "dataset_id": dataset.dataset_id,
        "migration_id": migration_id,
        "code_revision": run_summary.get("code_revision"),
        "code_dirty": run_summary.get("code_dirty"),
        "code_fingerprint": run_summary.get("code_fingerprint"),
        "dataset_status": dataset.status,
        "independent_gold": dataset.independent_gold,
        "judge_provider": "deepseek-openai-compatible",
        "judge_model": args.judge_model,
        "answer_models": answer_models,
        "api_model_recorded": api_model_recorded,
        "independent_judge": independent_judge,
        "embedding_model": args.embedding_model,
        **selection.model_dump(),
        "metrics": summary,
        "metric_coverage": metric_coverage,
        "complete_row_count": complete_row_count,
        "complete_row_coverage": (
            complete_row_count / len(rows) if rows else 0.0
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if args.wandb_mode != "disabled":
        import wandb

        with wandb.init(
            project="findoc-rag-evaluation",
            job_type="ragas-evaluation",
            mode=args.wandb_mode,
            config={
                "dataset_id": dataset.dataset_id,
                "run_id": run_summary["run_id"],
                "lane": selection.lane,
                "eligible_count": selection.eligible_count,
                "coverage": selection.coverage,
                "judge_model": args.judge_model,
                "answer_models": answer_models,
                "api_model_recorded": api_model_recorded,
                "independent_judge": independent_judge,
                "embedding_model": args.embedding_model,
            },
        ) as run_handle:
            run_handle.log({key: value for key, value in summary.items() if value is not None})
            artifact = wandb.Artifact(dataset.dataset_id, type="generation-eval-dataset")
            artifact.add_file(str(args.dataset))
            run_handle.log_artifact(artifact)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
