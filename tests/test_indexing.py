import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from findoc_rag.documents.models import (
    BoundingBox,
    DocumentChunk,
    ElementReference,
    StructuredTable,
    StructuredTableCell,
)
from findoc_rag.indexing import (
    PersistentIndex,
    SearchFilters,
    SearchHit,
    _dense_text,
    reciprocal_rank_fusion,
    tokenize_for_search,
)
from findoc_rag.structured_tables import chunk_payload_sha256


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


def quarterly_table(source: DocumentChunk) -> StructuredTable:
    return StructuredTable(
        table_id=f"{source.chunk_id}:quarterly",
        chunk_id=source.chunk_id,
        chunk_sha256=chunk_payload_sha256(source),
        table_type="quarterly",
        page_start=source.page_start,
        page_end=source.page_end,
        source="coordinate",
        cells=[
            StructuredTableCell(
                row="营业收入", column="第一季度", value="100.00"
            )
        ],
    )


def test_mixed_language_tokenizer() -> None:
    tokens = tokenize_for_search("2024年营业收入 Revenue 100.5亿元")

    assert "2024" in tokens
    assert "营业" in tokens
    assert "业收" in tokens
    assert "revenue" in tokens
    assert "100.5" in tokens


def test_tokenizer_preserves_financial_terms() -> None:
    tokens = tokenize_for_search("2024年营业收入和经营活动产生的现金流量净额")
    assert "营业收入" in tokens
    assert "经营活动产生的现金流量净额" in tokens


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


def test_structured_table_sidecar_enriches_hits_without_changing_chunk_identity(
    tmp_path: Path,
) -> None:
    source_chunk = chunk("c0", "第一季度 营业收入 100.00", "分季度主要财务数据")
    source = tmp_path / "chunks.jsonl"
    serialized_chunk = source_chunk.model_dump_json()
    source.write_text(serialized_chunk + "\n", encoding="utf-8")
    plain = PersistentIndex.build(tmp_path / "plain", [source_chunk], source)
    enriched = PersistentIndex.build(
        tmp_path / "enriched",
        [source_chunk],
        source,
        structured_tables=[quarterly_table(source_chunk)],
    )

    [hit] = enriched.search("营业收入", mode="lexical", top_k=1, candidate_k=1)

    assert plain.manifest.index_id == enriched.manifest.index_id
    assert hit.chunk.structured_tables[0].cells[0].value == "100.00"
    assert hit.chunk.model_dump_json() == serialized_chunk
    assert "structured_tables" not in hit.chunk.model_dump()


def test_schema1_structured_table_sidecar_remains_readable(tmp_path: Path) -> None:
    source_chunk = chunk("c0", "第一季度 营业收入 100.00", "分季度主要财务数据")
    source = tmp_path / "chunks.jsonl"
    source.write_text(source_chunk.model_dump_json() + "\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    PersistentIndex.build(
        index_dir,
        [source_chunk],
        source,
        structured_tables=[quarterly_table(source_chunk)],
    )
    manifest_path = index_dir / "manifest.json"
    artifact_path = index_dir / "structured_tables.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "structured_table_schema_version": 1,
            "structured_table_generator": "coordinate-safe-v2",
        }
    )
    artifact.update({"schema_version": 1, "generator": "coordinate-safe-v2"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    reopened = PersistentIndex(index_dir)

    [hit] = reopened.search("营业收入", mode="lexical", top_k=1, candidate_k=1)
    assert hit.chunk.structured_tables[0].cells[0].value_bbox is None


def test_structured_table_build_rejects_wrong_source_chunk_hash(tmp_path: Path) -> None:
    source_chunk = chunk("c0", "营业收入 100.00", "分季度主要财务数据")
    source = tmp_path / "chunks.jsonl"
    source.write_text(source_chunk.model_dump_json() + "\n", encoding="utf-8")
    table = quarterly_table(source_chunk).model_copy(
        update={"chunk_sha256": "0" * 64}
    )

    with pytest.raises(ValueError, match="source chunk hash mismatch"):
        PersistentIndex.build(
            tmp_path / "index", [source_chunk], source, structured_tables=[table]
        )


def test_structured_table_content_tampering_fails_closed(tmp_path: Path) -> None:
    source_chunk = chunk("c0", "营业收入 100.00", "分季度主要财务数据")
    source = tmp_path / "chunks.jsonl"
    source.write_text(source_chunk.model_dump_json() + "\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    PersistentIndex.build(
        index_dir, [source_chunk], source, structured_tables=[quarterly_table(source_chunk)]
    )
    tables_path = index_dir / "structured_tables.jsonl"
    tables_path.write_text(
        tables_path.read_text(encoding="utf-8").replace("100.00", "999.00"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content digest mismatch"):
        PersistentIndex(index_dir)


def test_structured_table_manifest_index_binding_fails_closed(tmp_path: Path) -> None:
    source_chunk = chunk("c0", "营业收入 100.00", "分季度主要财务数据")
    source = tmp_path / "chunks.jsonl"
    source.write_text(source_chunk.model_dump_json() + "\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    PersistentIndex.build(
        index_dir, [source_chunk], source, structured_tables=[quarterly_table(source_chunk)]
    )
    artifact_path = index_dir / "structured_tables.manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["index_id"] = "wrong-index"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="index ID mismatch"):
        PersistentIndex(index_dir)


def test_legacy_index_without_structured_table_artifact_still_opens(
    tmp_path: Path,
) -> None:
    source_chunk = chunk("c0", "营业收入 100.00", "主要财务指标")
    source = tmp_path / "chunks.jsonl"
    source.write_text(source_chunk.model_dump_json() + "\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    PersistentIndex.build(index_dir, [source_chunk], source)
    (index_dir / "structured_tables.jsonl").unlink()
    (index_dir / "structured_tables.manifest.json").unlink()
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in tuple(manifest):
        if field.startswith("structured_table"):
            manifest.pop(field)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reopened = PersistentIndex(index_dir)

    assert reopened.manifest.structured_table_count == 0
    assert reopened.resolve_chunks(["c0"])[0].structured_tables == []


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
    assert index.list_company_names() == ["乙公司", "甲公司"]
    assert index.list_report_years() == [2024]
    assert index.list_company_report_years() == {
        "乙公司": [2024],
        "甲公司": [2024],
    }


def test_index_propagates_statement_scope_from_explicit_section_boundaries(
    tmp_path: Path,
) -> None:
    chunks = [
        chunk("c0", "七、合并财务报表项目注释", "财务报告"),
        chunk("c1", "61、营业收入和营业成本", "(1)营业收入和营业成本情况"),
        chunk("c2", "十九、母公司财务报表主要项目注释", "财务报告"),
        chunk("c3", "4、营业收入和营业成本", "(1)营业收入和营业成本情况"),
    ]
    source = tmp_path / "chunks.jsonl"
    source.write_text(
        "".join(item.model_dump_json() + "\n" for item in chunks), encoding="utf-8"
    )
    index = PersistentIndex.build(tmp_path / "index", chunks, source)

    resolved = index.resolve_chunks(["c1", "c3"])

    assert resolved[0].statement_scope == "consolidated"
    assert resolved[1].statement_scope == "parent"
    assert "statement_scope" not in resolved[0].model_dump()


def test_page_window_is_bounded_and_never_crosses_documents(tmp_path: Path) -> None:
    chunks = [
        chunk("c0", "page nine", "section").model_copy(
            update={"page_start": 9, "page_end": 9}
        ),
        chunk("c1", "page ten", "section").model_copy(
            update={"page_start": 10, "page_end": 10}
        ),
        chunk("c2", "page eleven", "section").model_copy(
            update={"page_start": 11, "page_end": 11}
        ),
        chunk("c3", "page twelve", "section").model_copy(
            update={"page_start": 12, "page_end": 12}
        ),
        chunk("c4", "other document page ten", "section").model_copy(
            update={
                "document_id": "doc-2",
                "chunk_index": 0,
                "page_start": 10,
                "page_end": 10,
            }
        ),
    ]
    source = tmp_path / "chunks.jsonl"
    source.write_text(
        "".join(item.model_dump_json() + "\n" for item in chunks), encoding="utf-8"
    )
    index = PersistentIndex.build(tmp_path / "index", chunks, source)

    window = index.page_window("c2", before_pages=1, after_pages=0)

    assert [item.chunk_id for item in window] == ["c1", "c2"]
    assert {item.document_id for item in window} == {"doc-1"}
    with pytest.raises(KeyError, match="Unknown anchor chunk"):
        index.page_window("missing")


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


def test_dense_batch_search_encodes_queries_once(tmp_path: Path) -> None:
    chunks = [chunk("c0", "annual revenue", "annual"), chunk("c1", "quarterly revenue", "quarterly")]
    source = tmp_path / "chunks.jsonl"
    source.write_text("".join(item.model_dump_json() + "\n" for item in chunks), encoding="utf-8")
    index = PersistentIndex.build(tmp_path / "index", chunks, source)
    index.manifest.dense_model = "fake-model"
    index._dense_embeddings = np.eye(2, dtype=np.float32)
    index._dense_chunk_ids = ["c0", "c1"]
    calls: list[int] = []

    class BatchModel:
        def encode(self, texts, **_kwargs):
            calls.append(len(texts))
            return np.eye(len(texts), 2, dtype=np.float32)

    index._dense_model = BatchModel()
    results = index.search_dense_batch(["annual", "quarterly"], top_k=1)

    assert calls == [2]
    assert [result[0].chunk.chunk_id for result in results] == ["c0", "c1"]


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


def test_dense_text_e5_uses_kind_prefix() -> None:
    assert _dense_text("营收", "intfloat/multilingual-e5-small", "query") == "query: 营收"
    assert _dense_text("营收", "intfloat/multilingual-e5-small", "passage") == "passage: 营收"


def test_dense_text_bge_zh_uses_query_instruction() -> None:
    assert (
        _dense_text("营收", "BAAI/bge-small-zh-v1.5", "query")
        == "为这个句子生成表示以用于检索相关文章：营收"
    )
    assert _dense_text("营收", "BAAI/bge-small-zh-v1.5", "passage") == "营收"


def test_dense_text_bge_m3_keeps_plain_text() -> None:
    assert _dense_text("营收", "BAAI/bge-m3", "query") == "营收"
    assert _dense_text("营收", "BAAI/bge-m3", "passage") == "营收"


def test_dense_text_unknown_model_keeps_plain_text() -> None:
    assert _dense_text("营收", "some-other-model", "query") == "营收"
