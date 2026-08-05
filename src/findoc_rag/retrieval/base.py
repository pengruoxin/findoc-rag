from typing import Protocol

from findoc_rag.schemas import RetrievalHit


class Retriever(Protocol):
    name: str

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]: ...
