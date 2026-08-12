"""Tests for the query-rewrite quality gate."""

from __future__ import annotations

from pathlib import Path

from findoc_rag.documents.models import BoundingBox, DocumentChunk, ElementReference
from findoc_rag.indexing import PersistentIndex
from findoc_rag.query_gating import select_best_query


def _build_index(tmp_path: Path) -> PersistentIndex:
    chunks = [
        DocumentChunk(
            chunk_id="chunk-revenue",
            document_id="doc-0",
            chunk_index=0,
            text="贵州茅台2024年营业收入为170,899,152,276.34元，比上年同期增减15.71%",
            section_path=["主要会计数据"],
            page_start=5,
            page_end=5,
            element_references=[
                ElementReference(
                    element_id="element-0",
                    page_number=5,
                    bbox=BoundingBox(x0=10, y0=20, x1=200, y1=80),
                )
            ],
            character_count=60,
            estimated_token_count=60,
        ),
        DocumentChunk(
            chunk_id="chunk-profit",
            document_id="doc-0",
            chunk_index=1,
            text="贵州茅台2024年净利润同比增长率为15.71%",
            section_path=["主要财务指标"],
            page_start=5,
            page_end=5,
            element_references=[
                ElementReference(
                    element_id="element-1",
                    page_number=5,
                    bbox=BoundingBox(x0=10, y0=90, x1=200, y1=150),
                )
            ],
            character_count=40,
            estimated_token_count=40,
        ),
    ]
    source = tmp_path / "chunks.jsonl"
    source.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    return PersistentIndex.build(index_dir, chunks, source)


def test_gate_falls_back_when_llm_rewrite_degrades(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    selected, source = select_best_query(
        index,
        llm_query="贵州茅台2024年同比增长率是多少",
        deterministic_query="贵州茅台2024年营业收入比上年同期增减是多少",
        filters=None,
        top_k=1,
    )
    assert source == "deterministic"
    assert "比上年同期增减" in selected


def test_gate_keeps_llm_rewrite_when_evidence_is_unchanged(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    selected, source = select_best_query(
        index,
        llm_query="贵州茅台2024年营业收入及比上年同期增减是多少",
        deterministic_query="贵州茅台2024年营业收入比上年同期增减是多少",
        filters=None,
        top_k=1,
    )
    assert source == "llm"
    assert "营业收入及比上年同期增减" in selected
