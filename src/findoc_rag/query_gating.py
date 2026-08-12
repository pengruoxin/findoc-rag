"""Query-rewrite quality gate for the production query path.

An LLM rewrite can degrade retrieval (the canonical-lane experiment showed
``同比增幅 -> 同比增长率`` pushing the gold chunk out of top-5). The gate
compares the rewritten query against the deterministic baseline on the live
lexical index and falls back when the rewrite is both lower-scoring and
retrieves a largely different candidate set.
"""

from __future__ import annotations

from findoc_rag.indexing import PersistentIndex, SearchFilters


def select_best_query(
    index: PersistentIndex,
    llm_query: str,
    deterministic_query: str,
    filters: SearchFilters | None,
    *,
    top_k: int = 5,
    min_overlap: float = 0.4,
    score_ratio: float = 0.8,
) -> tuple[str, str]:
    """Return (selected_query, source) with source in {"llm", "deterministic"}."""
    det_hits = index.search_lexical(deterministic_query, top_k, filters)
    llm_hits = index.search_lexical(llm_query, top_k, filters)
    if not det_hits:
        return llm_query, "llm"
    if not llm_hits:
        return deterministic_query, "deterministic"

    det_ids = {hit.chunk.chunk_id for hit in det_hits}
    llm_ids = {hit.chunk.chunk_id for hit in llm_hits}
    union = det_ids | llm_ids
    overlap = len(det_ids & llm_ids) / len(union) if union else 0.0
    det_top1 = det_hits[0].score
    llm_top1 = llm_hits[0].score
    degraded = llm_top1 < det_top1 * score_ratio and overlap < min_overlap
    if degraded:
        return deterministic_query, "deterministic"
    return llm_query, "llm"
