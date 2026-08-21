from pathlib import Path

import pytest
from pydantic import ValidationError

from findoc_rag.agent_tasks import EvidenceMemory, add_evidence
from findoc_rag.documents.models import (
    BoundingBox,
    DocumentChunk,
    DocumentElement,
    DocumentPage,
    ElementReference,
    ParsedDocument,
    PdfLine,
    PdfSpan,
    StructuredTable,
    StructuredTableCell,
)
from findoc_rag.indexing import SearchHit
from findoc_rag.structured_tables import build_structured_tables, chunk_payload_sha256
from findoc_rag.table_cell_proof import (
    TableCellGeometryProof,
    build_table_cell_proofs,
)


def _chunk(*, with_geometry: bool = True) -> DocumentChunk:
    chunk = DocumentChunk(
        chunk_id="chunk-table",
        document_id="document-table",
        chunk_index=0,
        text="主要会计数据 单位：亿元 2024年 营业收入 100.00",
        section_path=["主要会计数据"],
        page_start=8,
        page_end=8,
        element_references=[
            ElementReference(
                element_id="element-table",
                page_number=8,
                bbox=BoundingBox(x0=30, y0=80, x1=500, y1=200),
            )
        ],
        character_count=40,
        estimated_token_count=30,
    )
    chunk.structured_tables = [
        StructuredTable(
            table_id="chunk-table:annual_data",
            chunk_id=chunk.chunk_id,
            chunk_sha256=chunk_payload_sha256(chunk),
            table_type="annual_data",
            page_start=8,
            page_end=8,
            unit="亿元",
            source="coordinate" if with_geometry else "text",
            cells=[
                StructuredTableCell(
                    row="营业收入",
                    row_index=1,
                    column="2024年",
                    column_index=1,
                    value="100.00",
                    page_number=8 if with_geometry else None,
                    value_bbox=(
                        BoundingBox(x0=200, y0=120, x1=260, y1=132)
                        if with_geometry
                        else None
                    ),
                    coordinate_space=(
                        "pymupdf_unrotated_page" if with_geometry else None
                    ),
                )
            ],
        )
    ]
    return chunk


def test_coordinate_cell_proof_is_chunk_bound_and_tamper_evident() -> None:
    [proof] = build_table_cell_proofs(_chunk())

    assert proof.geometry_status == "coordinate"
    assert proof.page_number == 8
    assert proof.row_index == 1
    assert proof.column_index == 1
    assert proof.value_bbox == BoundingBox(x0=200, y0=120, x1=260, y1=132)

    tampered = proof.model_dump()
    tampered["value"] = "900.00"
    with pytest.raises(ValidationError, match="binding SHA-256 mismatch"):
        TableCellGeometryProof.model_validate(tampered)


def test_text_fallback_is_explicitly_not_a_geometry_proof() -> None:
    [proof] = build_table_cell_proofs(_chunk(with_geometry=False))

    assert proof.geometry_status == "text_only"
    assert proof.page_number is None
    assert proof.value_bbox is None


def test_wrong_sidecar_chunk_hash_fails_before_agent_memory() -> None:
    chunk = _chunk()
    chunk.structured_tables[0].chunk_sha256 = "0" * 64

    with pytest.raises(ValueError, match="source chunk SHA-256 mismatch"):
        build_table_cell_proofs(chunk)


def test_agent_evidence_persists_table_cell_proofs(tmp_path: Path) -> None:
    chunk = _chunk()
    memory = EvidenceMemory(index_id="index-proof")

    added = add_evidence(
        memory,
        "task:extract",
        [SearchHit(rank=1, chunk=chunk, score=1.0)],
    )

    assert added == 1
    assert len(memory.items[0].table_cell_proofs) == 1
    serialized = memory.model_dump_json()
    restored = EvidenceMemory.model_validate_json(serialized)
    assert restored == memory


def test_structured_table_builder_retains_numeric_cell_pdf_geometry() -> None:
    text = (
        "营业收入和营业成本情况 单位：亿元 本期发生额 收入 成本 "
        "上期发生额 收入 成本 主营业务 100.00 20.00 90.00 18.00"
    )
    chunk = DocumentChunk(
        chunk_id="chunk-note",
        document_id="document-note",
        chunk_index=0,
        text=text,
        section_path=["营业收入和营业成本情况"],
        page_start=3,
        page_end=3,
        element_references=[
            ElementReference(
                element_id="element-note",
                page_number=3,
                bbox=BoundingBox(x0=40, y0=40, x1=560, y1=140),
            )
        ],
        character_count=len(text),
        estimated_token_count=len(text),
    )
    lines = [
        PdfLine(
            bbox=BoundingBox(x0=40, y0=40, x1=240, y1=50),
            spans=[
                PdfSpan(
                    text="营业收入和营业成本情况",
                    bbox=BoundingBox(x0=40, y0=40, x1=240, y1=50),
                )
            ],
        ),
        PdfLine(
            bbox=BoundingBox(x0=200, y0=70, x1=560, y1=80),
            spans=[
                PdfSpan(text="收入", bbox=BoundingBox(x0=200, y0=70, x1=250, y1=80)),
                PdfSpan(text="成本", bbox=BoundingBox(x0=300, y0=70, x1=350, y1=80)),
                PdfSpan(text="收入", bbox=BoundingBox(x0=400, y0=70, x1=450, y1=80)),
                PdfSpan(text="成本", bbox=BoundingBox(x0=500, y0=70, x1=550, y1=80)),
            ],
        ),
        PdfLine(
            bbox=BoundingBox(x0=40, y0=100, x1=560, y1=112),
            spans=[
                PdfSpan(
                    text="主营业务", bbox=BoundingBox(x0=40, y0=100, x1=120, y1=112)
                ),
                PdfSpan(
                    text="100.00", bbox=BoundingBox(x0=200, y0=100, x1=250, y1=112)
                ),
                PdfSpan(
                    text="20.00", bbox=BoundingBox(x0=300, y0=100, x1=350, y1=112)
                ),
                PdfSpan(
                    text="90.00", bbox=BoundingBox(x0=400, y0=100, x1=450, y1=112)
                ),
                PdfSpan(
                    text="18.00", bbox=BoundingBox(x0=500, y0=100, x1=550, y1=112)
                ),
            ],
        ),
    ]
    document = ParsedDocument(
        document_id=chunk.document_id,
        source_path="note.pdf",
        filename="note.pdf",
        content_sha256="d" * 64,
        page_count=3,
        pages=[
            DocumentPage(
                page_number=page_number,
                width=600,
                height=800,
                elements=(
                    [
                        DocumentElement(
                            element_id="element-note",
                            element_type="text",
                            text=text,
                            bbox=BoundingBox(x0=40, y0=40, x1=560, y1=140),
                            reading_order=0,
                            lines=lines,
                        )
                    ]
                    if page_number == 3
                    else []
                ),
                extracted_character_count=len(text) if page_number == 3 else 0,
                image_count=0,
                needs_ocr=False,
            )
            for page_number in range(1, 4)
        ],
        parser="pymupdf",
        parser_version="test",
    )

    [table] = build_structured_tables([chunk], {chunk.document_id: document})

    assert table.source == "coordinate"
    assert [cell.row_index for cell in table.cells] == [1, 1, 1, 1]
    assert [cell.column_index for cell in table.cells] == [1, 2, 3, 4]
    assert all(cell.page_number == 3 for cell in table.cells)
    assert table.cells[0].value_bbox == BoundingBox(x0=200, y0=100, x1=250, y1=112)
