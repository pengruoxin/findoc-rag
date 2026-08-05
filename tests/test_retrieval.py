from findoc_rag.evaluation.retrieval import hit_at_k, recall_at_k, reciprocal_rank
from findoc_rag.retrieval.bm25 import BM25Retriever
from findoc_rag.schemas import CorpusDocument


def test_bm25_ranks_lexically_matching_document_first() -> None:
    documents = [
        CorpusDocument(
            document_id="revenue",
            source_document="report",
            page_number=1,
            text="Annual revenue increased to 500 million dollars.",
        ),
        CorpusDocument(
            document_id="employees",
            source_document="report",
            page_number=2,
            text="The company employed twelve thousand people.",
        ),
    ]

    hits = BM25Retriever(documents).search("What was annual revenue?", top_k=2)

    assert hits[0].document_id == "revenue"


def test_retrieval_metrics_use_gold_document_ids() -> None:
    retrieved = ["wrong", "gold-b", "gold-a"]
    gold = {"gold-a", "gold-b"}

    assert recall_at_k(retrieved, gold, 1) == 0.0
    assert recall_at_k(retrieved, gold, 2) == 0.5
    assert recall_at_k(retrieved, gold, 3) == 1.0
    assert hit_at_k(retrieved, gold, 1) == 0.0
    assert hit_at_k(retrieved, gold, 2) == 1.0
    assert reciprocal_rank(retrieved, gold) == 0.5
