import sqlite3
from pathlib import Path

import numpy as np

from findoc_rag.documents.models import BoundingBox, DocumentChunk, ElementReference
from findoc_rag.indexing import (
    PersistentIndex,
    SearchFilters,
    SearchHit,
    reciprocal_rank_fusion,
    tokenize_for_search,
)


def chunk(chunk_id: str, text: str, section: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=int(chunk_id[-1]),
        text=text,
        section_path=[section],
        page_start=1,
        page_end=1,
        element_references=[
            ElementReference(
                element_id=f"element-{chunk_id}",
                page_number=1,
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100),
            )
        ],
        character_count=len(text),
        estimated_token_count=10,
    )


def test_mixed_language_tokenizer() -> None:
    tokens = tokenize_for_search("2024年营业收入 Revenue 100.5亿元")

    assert "2024" in tokens
    assert "营业" in tokens
    assert "业收" in tokens
    assert "revenue" in tokens
    assert "100.5" in tokens


def test_persistent_lexical_index_returns_full_chunk(tmp_path: Path) -> None:
    chunks = [
        chunk("c0", "公司营业收入达到一百亿元。", "主要财务指标"),
        chunk("c1", "研发人员数量持续增加。", "研发投入"),
        chunk("c2", "Operating cash flow increased.", "Cash Flow"),
    ]
    source = tmp_path / "chunks.jsonl"
    source.write_text("".join(item.model_dump_json() + "\n" for item in chunks), encoding="utf-8")

    built = PersistentIndex.build(tmp_path / "index", chunks, source)
    reopened = PersistentIndex(tmp_path / "index")
    hits = reopened.search("营业收入", mode="lexical", top_k=2, candidate_k=2)

    assert built.manifest.index_id == reopened.manifest.index_id
    assert hits[0].chunk.chunk_id == "c0"
    assert hits[0].chunk.element_references[0].element_id == "element-c0"


def test_metadata_filters_exclude_other_company_and_year(tmp_path: Path) -> None:
    first = chunk("c0", "annual revenue 100", "financial indicators").model_copy(
        update={"company_name": "甲公司", "report_year": 2024, "document_type": "annual"}
    )
    second = chunk("c1", "annual revenue 200", "financial indicators").model_copy(
        update={"company_name": "乙公司", "report_year": 2024, "document_type": "annual"}
    )
    source = tmp_path / "chunks.jsonl"
    source.write_text(first.model_dump_json() + "\n" + second.model_dump_json() + "\n")
    index = PersistentIndex.build(tmp_path / "index", [first, second], source)

    hits = index.search(
        "annual revenue",
        mode="lexical",
        top_k=5,
        candidate_k=5,
        filters=SearchFilters(company_names=["乙公司"], report_years=[2024]),
    )
    missing = index.search(
        "annual revenue",
        mode="lexical",
        top_k=5,
        candidate_k=5,
        filters=SearchFilters(company_names=["乙公司"], report_years=[2023]),
    )

    assert [hit.chunk.chunk_id for hit in hits] == ["c1"]
    assert missing == []


def test_index_detects_manifest_database_count_mismatch(tmp_path: Path) -> None:
    chunks = [chunk("c0", "营业收入。", "主要财务指标")]
    source = tmp_path / "chunks.jsonl"
    source.write_text(chunks[0].model_dump_json() + "\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    PersistentIndex.build(index_dir, chunks, source)
    with sqlite3.connect(index_dir / "lexical.sqlite3") as connection:
        connection.execute("DELETE FROM chunks")

    try:
        PersistentIndex(index_dir)
    except ValueError as error:
        assert "chunk counts disagree" in str(error)
    else:
        raise AssertionError("A corrupt index should not open")


def test_rrf_rewards_agreement_between_retrievers() -> None:
    first = chunk("c0", "first", "section")
    second = chunk("c1", "second", "section")
    third = chunk("c2", "third", "section")
    lexical = [
        SearchHit(rank=1, chunk=first, score=5, lexical_rank=1, lexical_score=5),
        SearchHit(rank=2, chunk=second, score=4, lexical_rank=2, lexical_score=4),
    ]
    dense = [
        SearchHit(rank=1, chunk=third, score=0.9, dense_rank=1, dense_score=0.9),
        SearchHit(rank=2, chunk=first, score=0.8, dense_rank=2, dense_score=0.8),
    ]

    fused = reciprocal_rank_fusion(lexical, dense, top_k=3)

    assert fused[0].chunk.chunk_id == "c0"
    assert fused[0].lexical_rank == 1
    assert fused[0].dense_rank == 2


def test_dense_model_is_loaded_once_per_open_index(tmp_path: Path, monkeypatch) -> None:
    chunks = [
        chunk("c0", "营业收入增长。", "主要财务指标"),
        chunk("c1", "研发人员增加。", "研发投入"),
    ]
    source = tmp_path / "chunks.jsonl"
    source.write_text("".join(item.model_dump_json() + "\n" for item in chunks), encoding="utf-8")
    load_calls: list[str] = []

    class FakeModel:
        def encode(self, texts, **_kwargs):
            vectors = []
            for text in texts:
                vectors.append([1.0, 0.0] if "营业" in text else [0.0, 1.0])
            return np.asarray(vectors, dtype=np.float32)

    def fake_loader(model_name: str):
        load_calls.append(model_name)
        return FakeModel()

    monkeypatch.setattr("findoc_rag.indexing._load_sentence_transformer", fake_loader)
    index_dir = tmp_path / "dense-index"
    PersistentIndex.build(index_dir, chunks, source, dense_model="fake-e5")
    reopened = PersistentIndex(index_dir)

    reopened.search_dense("营业收入", top_k=1)
    reopened.search_dense("营业收入", top_k=1)

    assert load_calls == ["fake-e5", "fake-e5"]


def test_incremental_dense_build_reuses_unchanged_chunk_embeddings(
    tmp_path: Path, monkeypatch
) -> None:
    first_chunks = [
        chunk("c0", "营业收入增长。", "主要财务指标"),
        chunk("c1", "研发人员增加。", "研发投入"),
    ]
    second_chunks = [
        first_chunks[0],
        chunk("c2", "经营现金流增长。", "现金流"),
    ]
    load_calls: list[str] = []

    class FakeModel:
        def encode(self, texts, **_kwargs):
            return np.asarray(
                [[1.0, float(index)] for index, _text in enumerate(texts)],
                dtype=np.float32,
            )

    def fake_loader(model_name: str):
        load_calls.append(model_name)
        return FakeModel()

    monkeypatch.setattr("findoc_rag.indexing._load_sentence_transformer", fake_loader)
    first_source = tmp_path / "first.jsonl"
    first_source.write_text(
        "".join(item.model_dump_json() + "\n" for item in first_chunks), encoding="utf-8"
    )
    first_index = PersistentIndex.build(
        tmp_path / "index-1", first_chunks, first_source, dense_model="fake-e5"
    )
    second_source = tmp_path / "second.jsonl"
    second_source.write_text(
        "".join(item.model_dump_json() + "\n" for item in second_chunks), encoding="utf-8"
    )

    second_index = PersistentIndex.build(
        tmp_path / "index-2",
        second_chunks,
        second_source,
        dense_model="fake-e5",
        reuse_dense_from=first_index,
    )

    assert second_index.manifest.parent_index_id == first_index.manifest.index_id
    assert second_index.manifest.reused_embedding_count == 1
    assert second_index.manifest.encoded_embedding_count == 1
    assert load_calls == ["fake-e5", "fake-e5"]
