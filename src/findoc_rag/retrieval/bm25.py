import re

import numpy as np
from rank_bm25 import BM25Okapi

from findoc_rag.schemas import CorpusDocument, RetrievalHit

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.&'-][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize English financial text for the first lexical baseline."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    name = "bm25"

    def __init__(self, documents: list[CorpusDocument]) -> None:
        if not documents:
            raise ValueError("BM25Retriever requires at least one document")
        self.documents = documents
        self._index = BM25Okapi([tokenize(document.text) for document in documents])

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        scores = self._index.get_scores(tokenize(query))
        ranked_indices = np.argsort(-scores, kind="stable")[: min(top_k, len(self.documents))]
        return [
            RetrievalHit(
                document_id=self.documents[index].document_id,
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]
