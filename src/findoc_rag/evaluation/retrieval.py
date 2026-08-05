from collections.abc import Sequence

from findoc_rag.retrieval.base import Retriever
from findoc_rag.schemas import BenchmarkQuestion


def reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: set[str]) -> float:
    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in gold_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        raise ValueError("gold_ids must not be empty")
    return len(set(retrieved_ids[:k]) & gold_ids) / len(gold_ids)


def hit_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    """Return 1 when at least one gold document occurs in the first k results."""
    return float(bool(set(retrieved_ids[:k]) & gold_ids))


def evaluate_retriever(
    retriever: Retriever,
    questions: list[BenchmarkQuestion],
    top_k: int = 10,
) -> tuple[dict, list[dict]]:
    if not questions:
        raise ValueError("At least one question is required")

    cutoffs = [cutoff for cutoff in (1, 5, 10) if cutoff <= top_k]
    metric_totals = {
        metric: 0.0
        for cutoff in cutoffs
        for metric in (f"hit_at_{cutoff}", f"recall_at_{cutoff}")
    }
    metric_totals["mrr"] = 0.0
    results: list[dict] = []

    for question in questions:
        hits = retriever.search(question.question, top_k=top_k)
        retrieved_ids = [hit.document_id for hit in hits]
        gold_ids = set(question.gold_document_ids)
        per_question_metrics = {
            metric: value
            for cutoff in cutoffs
            for metric, value in (
                (f"hit_at_{cutoff}", hit_at_k(retrieved_ids, gold_ids, cutoff)),
                (f"recall_at_{cutoff}", recall_at_k(retrieved_ids, gold_ids, cutoff)),
            )
        }
        per_question_metrics["reciprocal_rank"] = reciprocal_rank(retrieved_ids, gold_ids)

        for name in metric_totals:
            source_name = "reciprocal_rank" if name == "mrr" else name
            metric_totals[name] += per_question_metrics[source_name]

        results.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "gold_document_ids": question.gold_document_ids,
                "hits": [
                    {"document_id": hit.document_id, "rank": hit.rank, "score": hit.score}
                    for hit in hits
                ],
                "metrics": per_question_metrics,
            }
        )

    summary = {
        "retriever": retriever.name,
        "question_count": len(questions),
        "top_k": top_k,
        "metrics": {
            name: total / len(questions) for name, total in metric_totals.items()
        },
    }
    return summary, results
