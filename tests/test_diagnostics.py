from pathlib import Path
from types import SimpleNamespace

from findoc_rag.diagnostics import (
    DocumentProfile,
    analyze_recall_failures,
    evaluate_diagnostic_dataset,
    generate_diagnostic_dataset,
)
from findoc_rag.documents.models import BoundingBox, DocumentChunk, ElementReference
from findoc_rag.indexing import SearchHit
from findoc_rag.registry import DocumentRegistry


def make_chunk(chunk_id: str, document_id: str, text: str, section: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        section_path=[section],
        page_start=1,
        page_end=1,
        element_references=[
            ElementReference(
                element_id=f"element-{chunk_id}",
                page_number=1,
                bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
            )
        ],
        character_count=len(text),
        estimated_token_count=len(text),
    )


class FakeIndex:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.manifest = SimpleNamespace(index_id="test-index", chunk_count=len(hits))

    def search(self, query: str, top_k: int, mode: str, candidate_k: int, filters=None):
        return self.hits[:top_k]

    def search_dense_batch(self, queries, top_k: int, filters=None):
        return [self.hits[:top_k] for _ in queries]

    def search_lexical(self, query: str, top_k: int, filters=None):
        return self.hits[:top_k]

    def _load_chunks(self, chunk_ids: list[str]):
        return {
            hit.chunk.chunk_id: hit.chunk
            for hit in self.hits
            if hit.chunk.chunk_id in chunk_ids
        }


def activate_chunks(
    registry: DocumentRegistry, tmp_path: Path, key: str, digest: str, chunks: list[DocumentChunk]
) -> None:
    source = tmp_path / f"{digest}.pdf"
    source.write_bytes(b"pdf")
    decision = registry.begin_ingestion(key, digest * 64, source)
    document_ir = tmp_path / f"{digest}.json"
    chunks_path = tmp_path / f"{digest}.jsonl"
    document_ir.write_text("{}", encoding="utf-8")
    chunks_path.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8"
    )
    registry.activate(decision.version.version_id, document_ir, chunks_path, len(chunks))


def test_generator_anchors_positive_and_labels_cross_company_negative(tmp_path: Path) -> None:
    first = make_chunk("first-1", "first", "营业收入 100", "主要会计数据和财务指标")
    second = make_chunk("second-1", "second", "营业收入 200", "主要会计数据和财务指标")
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    activate_chunks(registry, tmp_path, "company:first", "a", [first])
    activate_chunks(registry, tmp_path, "company:second", "b", [second])
    index = FakeIndex(
        [
            SearchHit(rank=1, chunk=first, score=1),
            SearchHit(rank=2, chunk=second, score=0.5),
        ]
    )

    dataset = generate_diagnostic_dataset(
        registry,
        index,
        [
            DocumentProfile(document_key="company:first", company="甲公司", year=2024),
            DocumentProfile(document_key="company:second", company="乙公司", year=2024),
        ],
        candidate_k=5,
    )

    query = next(item for item in dataset.queries if item.company == "甲公司")
    assert query.status == "accepted"
    assert any(item.label == "relevant" and item.chunk_id == "first-1" for item in query.judgments)
    assert any(
        item.negative_type == "wrong_company" and item.chunk_id == "second-1"
        for item in query.judgments
    )
    assert query.judgments[0].excerpt


def test_evaluation_uses_only_accepted_structural_gold(tmp_path: Path) -> None:
    first = make_chunk("first-1", "first", "营业收入 100", "主要会计数据和财务指标")
    second = make_chunk("second-1", "second", "营业收入 200", "主要会计数据和财务指标")
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    activate_chunks(registry, tmp_path, "company:first", "a", [first])
    activate_chunks(registry, tmp_path, "company:second", "b", [second])
    generation_index = FakeIndex(
        [SearchHit(rank=1, chunk=second, score=1), SearchHit(rank=2, chunk=first, score=0.5)]
    )
    dataset = generate_diagnostic_dataset(
        registry,
        generation_index,
        [
            DocumentProfile(document_key="company:first", company="甲公司", year=2024),
            DocumentProfile(document_key="company:second", company="乙公司", year=2024),
        ],
        candidate_k=5,
    )

    evaluation = evaluate_diagnostic_dataset(dataset, generation_index, top_k=5)

    assert evaluation.hit_at_k == 1
    assert evaluation.mrr_at_k == 0.75
    assert evaluation.results[0].first_relevant_rank == 2


def test_failure_analysis_identifies_candidate_budget_miss(tmp_path: Path) -> None:
    first = make_chunk("first-1", "first", "营业收入 100", "主要会计数据和财务指标")
    second = make_chunk("second-1", "second", "营业收入 200", "主要会计数据和财务指标")
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    activate_chunks(registry, tmp_path, "company:first", "a", [first])
    activate_chunks(registry, tmp_path, "company:second", "b", [second])
    index = FakeIndex(
        [SearchHit(rank=1, chunk=second, score=1), SearchHit(rank=2, chunk=first, score=0.5)]
    )
    dataset = generate_diagnostic_dataset(
        registry,
        index,
        [
            DocumentProfile(document_key="company:first", company="甲公司", year=2024),
            DocumentProfile(document_key="company:second", company="乙公司", year=2024),
        ],
        candidate_k=2,
    )
    evaluation = evaluate_diagnostic_dataset(dataset, index, mode="lexical", top_k=1)
    evaluation.results = [evaluation.results[0].model_copy(
        update={"candidate_recall": False, "candidate_first_rank": None, "effective_candidate_k": 1}
    )]
    evaluation.evaluated_query_count = 1

    report = analyze_recall_failures(dataset, evaluation, index)

    assert report.failure_count == 1
    assert report.failures[0].failure_type == "candidate_budget_too_small"
    assert report.failures[0].lexical_first_rank == 2
