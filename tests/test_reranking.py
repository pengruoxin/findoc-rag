from pathlib import Path

from findoc_rag.config import (
    ObservabilitySettings,
    RerankerSettings,
    RetrievalSettings,
    ScopeRoutingSettings,
)
from findoc_rag.documents.models import (
    BoundingBox,
    DocumentChunk,
    ElementReference,
    StructuredTable,
    StructuredTableCell,
)
from findoc_rag.indexing import PersistentIndex, SearchHit
from findoc_rag.reranking import CrossEncoderReranker
from findoc_rag.service import RetrievalService, SearchRequest
from findoc_rag.structured_tables import chunk_payload_sha256


def chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document",
        chunk_index=int(chunk_id[-1]),
        text=text,
        section_path=[],
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


class FakeCrossEncoder:
    def predict(self, pairs, batch_size: int, show_progress_bar: bool):
        assert batch_size == 2
        assert show_progress_bar is False
        return [0.1 if "quarterly" in passage else 0.9 for _, passage in pairs]


class FakeReranker:
    model_name = "test/reranker"

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        assert query == "annual revenue"
        reversed_hits = list(reversed(hits))
        return [
            hit.model_copy(
                update={
                    "rank": rank,
                    "original_rank": hit.rank,
                    "rerank_score": 1 / rank,
                    "rank_delta": hit.rank - rank,
                }
            )
            for rank, hit in enumerate(reversed_hits[:top_k], start=1)
        ]


def test_cross_encoder_reranker_preserves_original_rank_and_breaks_ties() -> None:
    reranker = CrossEncoderReranker("test/model", batch_size=2)
    reranker._model = FakeCrossEncoder()
    assert reranker._get_model() is reranker._get_model()
    hits = [
        SearchHit(rank=1, chunk=chunk("chunk-1", "quarterly revenue"), score=0.8),
        SearchHit(rank=2, chunk=chunk("chunk-2", "annual revenue"), score=0.7),
        SearchHit(rank=3, chunk=chunk("chunk-3", "annual revenue details"), score=0.6),
    ]

    ranked = reranker.rerank("annual revenue", hits, top_k=2)

    assert [hit.chunk.chunk_id for hit in ranked] == ["chunk-2", "chunk-3"]
    assert [hit.original_rank for hit in ranked] == [2, 3]
    assert [hit.rerank_score for hit in ranked] == [0.9, 0.9]
    assert [hit.rank_delta for hit in ranked] == [1, 1]


def test_service_traces_reranking_and_candidate_ranking_changes(tmp_path: Path) -> None:
    chunks = [chunk("chunk-1", "annual revenue"), chunk("chunk-2", "annual revenue details")]
    source = tmp_path / "chunks.jsonl"
    source.write_text("".join(item.model_dump_json() + "\n" for item in chunks), encoding="utf-8")
    index_dir = tmp_path / "index"
    PersistentIndex.build(index_dir, chunks, source)
    trace_db = tmp_path / "traces.sqlite3"
    service = RetrievalService(
        RetrievalSettings(
            index_dir=index_dir,
            default_mode="lexical",
            top_k=1,
            candidate_k=2,
        ),
        ObservabilitySettings(trace_db=trace_db),
        RerankerSettings(enabled=True, model="test/reranker"),
        reranker=FakeReranker(),
    )

    response = service.search(SearchRequest(query="annual revenue"), "request-1")
    trace = service.trace_store.get(response.trace_id)

    assert response.reranked is True
    assert response.reranker_model == "test/reranker"
    assert response.hits[0].original_rank == 2
    assert [stage.stage for stage in trace.stages] == ["lexical", "rerank"]
    assert trace.stages[-1].hits[0].original_rank == 2
    assert trace.stages[-1].hits[0].rank_delta == 1
    assert trace.reranked is True


def test_service_rejects_unconfigured_on_demand_reranking(tmp_path: Path) -> None:
    chunks = [chunk("chunk-1", "annual revenue")]
    source = tmp_path / "chunks.jsonl"
    source.write_text(chunks[0].model_dump_json() + "\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    PersistentIndex.build(index_dir, chunks, source)
    service = RetrievalService(
        RetrievalSettings(index_dir=index_dir, default_mode="lexical"),
        ObservabilitySettings(enabled=False),
    )

    try:
        service.search(SearchRequest(query="annual revenue", rerank=True), "request-1")
    except ValueError as exc:
        assert "no reranker is configured" in str(exc.__cause__)
    else:
        raise AssertionError("Expected reranking configuration error")


def test_service_records_scope_routing_stage(tmp_path: Path) -> None:
    audit = chunk("chunk-1", "营业收入 audit").model_copy(
        update={"section_path": ["关键审计事项 > 收入确认"]}
    )
    annual = chunk("chunk-2", "营业收入 annual").model_copy(
        update={"section_path": ["主要会计数据和财务指标"]}
    )
    source = tmp_path / "chunks.jsonl"
    source.write_text(audit.model_dump_json() + "\n" + annual.model_dump_json() + "\n")
    index_dir = tmp_path / "index"
    PersistentIndex.build(index_dir, [audit, annual], source)
    service = RetrievalService(
        RetrievalSettings(
            index_dir=index_dir, default_mode="lexical", top_k=1, candidate_k=2
        ),
        ObservabilitySettings(trace_db=tmp_path / "traces.sqlite3"),
        scope_settings=ScopeRoutingSettings(enabled=True),
    )

    response = service.search(SearchRequest(query="2024年营业收入是多少"), "request-scope")
    trace = service.trace_store.get(response.trace_id)

    assert response.inferred_scope == "annual_summary"
    assert response.hits[0].chunk.chunk_id == "chunk-2"
    assert [stage.stage for stage in trace.stages] == ["lexical", "scope", "structured"]
    assert trace.inferred_scope == "annual_summary"


def test_service_consumes_index_bound_structured_table_artifact_online(
    tmp_path: Path,
) -> None:
    plain = chunk("chunk-1", "各季度营业收入经营情况说明")
    verified = chunk("chunk-2", "2024年分季度主要财务数据")
    table = StructuredTable(
        table_id="chunk-2:quarterly",
        chunk_id=verified.chunk_id,
        chunk_sha256=chunk_payload_sha256(verified),
        table_type="quarterly",
        page_start=1,
        page_end=1,
        source="coordinate",
        cells=[
            StructuredTableCell(row="营业收入", column="第一季度", value="1")
        ],
    )
    source = tmp_path / "chunks.jsonl"
    source.write_text(
        plain.model_dump_json() + "\n" + verified.model_dump_json() + "\n",
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    PersistentIndex.build(
        index_dir,
        [plain, verified],
        source,
        structured_tables=[table],
    )
    service = RetrievalService(
        RetrievalSettings(
            index_dir=index_dir, default_mode="lexical", top_k=1, candidate_k=2
        ),
        ObservabilitySettings(trace_db=tmp_path / "traces.sqlite3"),
    )

    response = service.search(
        SearchRequest(query="2024年各季度营业收入是多少"),
        "request-structured",
    )
    trace = service.trace_store.get(response.trace_id)

    assert response.hits[0].chunk.chunk_id == "chunk-2"
    assert response.hits[0].chunk.structured_tables[0].table_id == table.table_id
    assert [stage.stage for stage in trace.stages] == ["lexical", "structured"]
    assert trace.stages[-1].hits[0].retrieval_rank is not None
