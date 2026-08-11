"""Run the four RAGAS generation metrics against an immutable generation run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from findoc_rag.generation_evaluation import (
    GenerationEvaluationDataset,
    select_ragas_run_items,
)
from findoc_rag.ragas_runner import load_and_validate_run_manifest, load_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_jsonl", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/benchmark-v2.json"))
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
    answerable, selection = select_ragas_run_items(dataset, run_items, lane)
    run = {item.query_id: item for item in run_items}

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required for RAGAS LLM judging")

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
    api_model_recorded = all(
        run[item.query_id].api_model is not None for item in answerable
    )
    independent_judge = args.judge_model not in answer_models
    summary = {
        name: float(frame[name].dropna().mean()) if name in frame else None
        for name in metric_names
    }
    payload = {
        "run_id": run_summary["run_id"],
        "run_jsonl": str(args.run_jsonl),
        "dataset_id": dataset.dataset_id,
        "code_revision": run_summary.get("code_revision"),
        "code_dirty": run_summary.get("code_dirty"),
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
