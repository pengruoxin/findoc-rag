from threading import RLock
from typing import Protocol

from findoc_rag.indexing import SearchHit, searchable_chunk_text

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...


class CrossEncoderReranker:
    """Process-lifetime cached CrossEncoder with deterministic ranking metadata."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, batch_size: int = 16) -> None:
        self._model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._lock = RLock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:
                    raise RuntimeError(
                        "Reranker dependencies are missing. Run: uv sync --extra dense"
                    ) from exc
                self._model = CrossEncoder(self.model_name)
            return self._model

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [(query, searchable_chunk_text(hit.chunk)) for hit in hits]
        scores = self._get_model().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scored = sorted(
            zip(hits, (float(score) for score in scores), strict=True),
            key=lambda item: (-item[1], item[0].rank),
        )
        return [
            hit.model_copy(
                update={
                    "rank": rank,
                    "original_rank": hit.rank,
                    "rerank_score": score,
                    "rank_delta": hit.rank - rank,
                }
            )
            for rank, (hit, score) in enumerate(scored[:top_k], start=1)
        ]
