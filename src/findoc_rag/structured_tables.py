"""Build deterministic, chunk-bound structured-table sidecar artifacts.

The sidecar is deliberately excluded from ``DocumentChunk`` serialization so
table improvements do not silently rewrite corpus snapshots, chunk hashes, or
benchmark-bound retrieval identity.  Each record instead carries the exact
source chunk hash and is bound to an index by the index-local artifact manifest.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

from pydantic import BaseModel, Field

from findoc_rag.documents.models import (
    BoundingBox,
    DocumentChunk,
    ParsedDocument,
    StructuredTable,
    StructuredTableCell,
)
from findoc_rag.table_reconstruction import (
    ExtractedCell,
    TableType,
    blocks_from_document_ir,
    detect_unit,
    extract_cells,
    normalize_label,
    normalize_value,
    reconstruct_cells,
    select_table_cells,
)

STRUCTURED_TABLE_SCHEMA_VERSION = 2
STRUCTURED_TABLE_GENERATOR = "coordinate-safe-v4-pdf-validated"
SUPPORTED_STRUCTURED_TABLE_ARTIFACTS = {
    (1, "coordinate-safe-v2"),
    (2, "coordinate-safe-v3-geometry"),
    (STRUCTURED_TABLE_SCHEMA_VERSION, STRUCTURED_TABLE_GENERATOR),
}
TABLE_TYPES: tuple[TableType, ...] = (
    "quarterly",
    "note_cost",
    "segment",
    "annual_data",
    "concentration",
)

_SECTION_NAMES = {
    "分行业": "主营业务分行业情况",
    "分产品": "主营业务分产品情况",
    "分地区": "主营业务分地区情况",
    "分销售模式": "主营业务分销售模式情况",
}


class StructuredTableArtifactManifest(BaseModel):
    schema_version: int = STRUCTURED_TABLE_SCHEMA_VERSION
    generator: str = STRUCTURED_TABLE_GENERATOR
    index_id: str
    source_chunk_sha256: str
    tables_sha256: str
    table_count: int = Field(ge=0)
    cell_count: int = Field(ge=0)


def serialize_structured_tables(tables: list[StructuredTable]) -> str:
    return "".join(table.model_dump_json() + "\n" for table in tables)


def structured_tables_sha256(tables: list[StructuredTable]) -> str:
    return hashlib.sha256(serialize_structured_tables(tables).encode()).hexdigest()


def load_structured_tables(text: str) -> list[StructuredTable]:
    return [StructuredTable.model_validate_json(line) for line in text.splitlines() if line.strip()]


def chunk_payload_sha256(chunk: DocumentChunk) -> str:
    """Hash only persisted chunk fields, excluding runtime enrichments."""
    return hashlib.sha256(chunk.model_dump_json().encode("utf-8")).hexdigest()


def infer_table_types(text: str) -> list[TableType]:
    """Detect supported regulatory table families without evaluation gold."""
    compact = re.sub(r"\s+", "", text)
    detected: list[TableType] = []
    if all(name in compact for name in ("第一季度", "第二季度", "第三季度", "第四季度")):
        detected.append("quarterly")
    # Generic notes such as 投资收益 also contain 本期/上期发生额 plus the
    # words 收入 and 成本 in row prose. Requiring the regulatory section
    # title prevents adjacent two-column notes from being mislabeled as the
    # four-column revenue/cost table.
    if (
        "营业收入和营业成本" in compact
        and "本期发生额" in compact
        and "上期发生额" in compact
        and "收入" in compact
        and "成本" in compact
    ):
        detected.append("note_cost")
    if any(
        marker in compact
        for marker in (
            "主营业务分行业情况",
            "主营业务分产品情况",
            "主营业务分地区情况",
            "主营业务分销售模式情况",
        )
    ):
        detected.append("segment")
    if "主要会计数据" in compact and len(set(re.findall(r"20\d{2}年", compact))) >= 3:
        detected.append("annual_data")
    if "前五名客户销售额" in compact or "前五名供应商采购额" in compact:
        detected.append("concentration")
    return detected


def _canonical_section(section: str) -> str:
    return _SECTION_NAMES.get(section, section)


def _restore_segment_sections(cells: list[ExtractedCell], text: str) -> list[ExtractedCell]:
    """Recover section labels that geometry loses across a page boundary."""
    text_sections = {
        (normalize_label(cell.row), cell.column, normalize_value(cell.value)): cell.section
        for cell in extract_cells(text, "segment")
        if cell.section
    }
    return [
        ExtractedCell(
            row=cell.row,
            column=cell.column,
            value=cell.value,
            section=cell.section
            or text_sections.get(
                (normalize_label(cell.row), cell.column, normalize_value(cell.value)),
                "",
            ),
            page_number=cell.page_number,
            value_bbox=cell.value_bbox,
        )
        for cell in cells
    ]


def _attach_unique_coordinate_geometry(
    selected_cells: list[ExtractedCell],
    coordinate_cells: list[ExtractedCell],
) -> list[ExtractedCell]:
    """Bind text-selected cells only when one exact coordinate cell agrees."""

    candidates: dict[tuple[str, str, str, str], list[ExtractedCell]] = {}
    for cell in coordinate_cells:
        key = (
            _canonical_section(cell.section),
            normalize_label(cell.row),
            cell.column,
            normalize_value(cell.value),
        )
        candidates.setdefault(key, []).append(cell)
    bound: list[ExtractedCell] = []
    for cell in selected_cells:
        if cell.page_number is not None and cell.value_bbox is not None:
            bound.append(cell)
            continue
        key = (
            _canonical_section(cell.section),
            normalize_label(cell.row),
            cell.column,
            normalize_value(cell.value),
        )
        matches = [
            candidate
            for candidate in candidates.get(key, [])
            if candidate.page_number is not None and candidate.value_bbox is not None
        ]
        if len(matches) != 1:
            bound.append(cell)
            continue
        match = matches[0]
        bound.append(
            ExtractedCell(
                row=cell.row,
                column=cell.column,
                value=cell.value,
                section=cell.section,
                page_number=match.page_number,
                value_bbox=match.value_bbox,
            )
        )
    return bound


def build_structured_tables(
    chunks: list[DocumentChunk],
    documents: Mapping[str, ParsedDocument] | None = None,
) -> list[StructuredTable]:
    """Build safe cells from persisted IR v2, with conservative text fallback."""
    records: list[StructuredTable] = []
    document_map = documents or {}
    for chunk in chunks:
        table_types = infer_table_types(chunk.text)
        if not table_types:
            continue
        document = document_map.get(chunk.document_id)
        has_geometry = bool(
            document
            and any(
                element.lines
                for page in document.pages[chunk.page_start - 1 : chunk.page_end]
                for element in page.elements
            )
        )
        blocks = (
            blocks_from_document_ir(
                document,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
            )
            if document is not None and has_geometry
            else []
        )
        region = chunk.section_path[-1] if chunk.section_path else ""
        for table_type in table_types:
            coordinate_cells = (
                reconstruct_cells(blocks, table_type, region=region) if blocks else []
            )
            selection = select_table_cells(coordinate_cells, chunk.text, table_type)
            if not selection.cells:
                continue
            selected_cells = (
                _restore_segment_sections(selection.cells, chunk.text)
                if table_type == "segment"
                else selection.cells
            )
            geometry_candidates = (
                _restore_segment_sections(coordinate_cells, chunk.text)
                if table_type == "segment"
                else coordinate_cells
            )
            selected_cells = _attach_unique_coordinate_geometry(
                selected_cells,
                geometry_candidates,
            )
            row_indices: dict[tuple[str, str], int] = {}
            column_indices: dict[str, int] = {}
            cells: list[StructuredTableCell] = []
            for cell in selected_cells:
                row_key = (_canonical_section(cell.section), normalize_label(cell.row))
                row_index = row_indices.setdefault(row_key, len(row_indices) + 1)
                column_index = column_indices.setdefault(cell.column, len(column_indices) + 1)
                has_geometry = cell.page_number is not None and cell.value_bbox is not None
                cells.append(
                    StructuredTableCell(
                        row=cell.row,
                        column=cell.column,
                        value=cell.value,
                        section=row_key[0],
                        row_index=row_index,
                        column_index=column_index,
                        page_number=cell.page_number if has_geometry else None,
                        value_bbox=(
                            BoundingBox(
                                x0=cell.value_bbox[0],
                                y0=cell.value_bbox[1],
                                x1=cell.value_bbox[2],
                                y1=cell.value_bbox[3],
                            )
                            if has_geometry and cell.value_bbox is not None
                            else None
                        ),
                        coordinate_space=("pymupdf_unrotated_page" if has_geometry else None),
                    )
                )
            records.append(
                StructuredTable(
                    table_id=f"{chunk.chunk_id}:{table_type}",
                    chunk_id=chunk.chunk_id,
                    chunk_sha256=chunk_payload_sha256(chunk),
                    table_type=table_type,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    unit=detect_unit(chunk.text),
                    source=selection.source,
                    selection_reasons=list(selection.reasons),
                    cells=cells,
                )
            )
    return records
