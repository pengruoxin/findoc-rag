from pydantic import BaseModel, Field


class CorpusDocument(BaseModel):
    """One retrievable unit in the benchmark corpus."""

    document_id: str
    source_document: str
    page_number: int
    text: str
    metadata: dict[str, str | int] = Field(default_factory=dict)


class BenchmarkQuestion(BaseModel):
    """A question with gold answer and evidence identifiers."""

    question_id: str
    question: str
    answer: str
    gold_document_ids: list[str]
    question_type: str
    source_document: str
    justification: str


class RetrievalHit(BaseModel):
    """A ranked result returned by any retriever."""

    document_id: str
    score: float
    rank: int
