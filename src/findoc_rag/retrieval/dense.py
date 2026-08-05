import hashlib
import json
from pathlib import Path

import numpy as np

from findoc_rag.schemas import CorpusDocument, RetrievalHit

DEFAULT_MODEL = "intfloat/e5-small-v2"


def _corpus_fingerprint(documents: list[CorpusDocument], model_name: str) -> str:
    hasher = hashlib.sha256(model_name.encode("utf-8"))
    for document in documents:
        hasher.update(document.document_id.encode("utf-8"))
        hasher.update(document.text.encode("utf-8"))
    return hasher.hexdigest()[:16]


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        documents: list[CorpusDocument],
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path = Path("data/cache/embeddings"),
        batch_size: int = 16,
    ) -> None:
        if not documents:
            raise ValueError("DenseRetriever requires at least one document")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense dependencies are missing. Run: uv sync --extra dev --extra dense"
            ) from exc

        self.documents = documents
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        fingerprint = _corpus_fingerprint(documents, model_name)
        safe_model_name = model_name.replace("/", "--")
        self.cache_path = cache_dir / f"{safe_model_name}-{fingerprint}.npz"

        if self.cache_path.exists():
            cached = np.load(self.cache_path)
            cached_ids = json.loads(str(cached["document_ids"].item()))
            expected_ids = [document.document_id for document in documents]
            if cached_ids != expected_ids:
                raise ValueError("Dense embedding cache document order does not match the corpus")
            self.embeddings = cached["embeddings"]
        else:
            passages = [f"passage: {document.text}" for document in documents]
            self.embeddings = self.model.encode(
                passages,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.cache_path,
                embeddings=self.embeddings,
                document_ids=json.dumps([document.document_id for document in documents]),
            )

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_embedding = self.model.encode(
            [f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(-scores, kind="stable")[: min(top_k, len(self.documents))]
        return [
            RetrievalHit(
                document_id=self.documents[index].document_id,
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]
