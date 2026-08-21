"""Chunk-bound table-cell geometry proofs for auditable Agent evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from findoc_rag.documents.models import BoundingBox, DocumentChunk
from findoc_rag.structured_tables import chunk_payload_sha256

TableType = Literal[
    "quarterly",
    "note_cost",
    "segment",
    "annual_data",
    "concentration",
]


def _binding_payload(proof: TableCellGeometryProof) -> dict:
    return {
        "schema_version": proof.schema_version,
        "table_id": proof.table_id,
        "table_type": proof.table_type,
        "chunk_id": proof.chunk_id,
        "chunk_sha256": proof.chunk_sha256,
        "table_source": proof.table_source,
        "page_start": proof.page_start,
        "page_end": proof.page_end,
        "unit": proof.unit,
        "section": proof.section,
        "row": proof.row,
        "row_index": proof.row_index,
        "column": proof.column,
        "column_index": proof.column_index,
        "value": proof.value,
        "geometry_status": proof.geometry_status,
        "page_number": proof.page_number,
        "value_bbox": (
            proof.value_bbox.model_dump(mode="json")
            if proof.value_bbox is not None
            else None
        ),
        "coordinate_space": proof.coordinate_space,
    }


def table_cell_binding_sha256(proof: TableCellGeometryProof) -> str:
    canonical = json.dumps(
        _binding_payload(proof),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TableCellGeometryProof(BaseModel):
    """One logical row/column/value cell bound to its exact chunk and PDF bbox."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    table_id: str
    table_type: TableType
    chunk_id: str
    chunk_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    table_source: Literal["coordinate", "text"]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    unit: str = ""
    section: str = ""
    row: str
    row_index: int | None = Field(default=None, ge=1)
    column: str
    column_index: int | None = Field(default=None, ge=1)
    value: str
    geometry_status: Literal["coordinate", "text_only"]
    page_number: int | None = Field(default=None, ge=1)
    value_bbox: BoundingBox | None = None
    coordinate_space: str | None = None
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_binding(self) -> TableCellGeometryProof:
        if self.page_end < self.page_start:
            raise ValueError("Proof page_end must not precede page_start")
        if self.geometry_status == "coordinate":
            if (
                self.page_number is None
                or self.value_bbox is None
                or self.coordinate_space is None
            ):
                raise ValueError("Coordinate proof requires page, bbox, and coordinate space")
            if not self.page_start <= self.page_number <= self.page_end:
                raise ValueError("Coordinate proof page falls outside the table page range")
        elif any(
            value is not None
            for value in (self.page_number, self.value_bbox, self.coordinate_space)
        ):
            raise ValueError("Text-only proof must not claim PDF coordinates")
        if self.binding_sha256 != table_cell_binding_sha256(self):
            raise ValueError("Table-cell proof binding SHA-256 mismatch")
        return self


def build_table_cell_proofs(chunk: DocumentChunk) -> list[TableCellGeometryProof]:
    """Validate sidecar bindings and convert every table cell into a proof."""

    chunk_sha256 = chunk_payload_sha256(chunk)
    proofs: list[TableCellGeometryProof] = []
    for table in chunk.structured_tables:
        if table.chunk_id != chunk.chunk_id:
            raise ValueError("Structured table references a different chunk ID")
        if table.chunk_sha256 != chunk_sha256:
            raise ValueError("Structured table source chunk SHA-256 mismatch")
        for cell in table.cells:
            has_geometry = (
                cell.page_number is not None
                and cell.value_bbox is not None
                and cell.coordinate_space is not None
            )
            payload = {
                "table_id": table.table_id,
                "table_type": table.table_type,
                "chunk_id": table.chunk_id,
                "chunk_sha256": table.chunk_sha256,
                "table_source": table.source,
                "page_start": table.page_start,
                "page_end": table.page_end,
                "unit": table.unit,
                "section": cell.section,
                "row": cell.row,
                "row_index": cell.row_index,
                "column": cell.column,
                "column_index": cell.column_index,
                "value": cell.value,
                "geometry_status": "coordinate" if has_geometry else "text_only",
                "page_number": cell.page_number if has_geometry else None,
                "value_bbox": cell.value_bbox if has_geometry else None,
                "coordinate_space": cell.coordinate_space if has_geometry else None,
                "binding_sha256": "0" * 64,
            }
            unbound = TableCellGeometryProof.model_construct(**payload)
            payload["binding_sha256"] = table_cell_binding_sha256(unbound)
            proofs.append(TableCellGeometryProof.model_validate(payload))
    return proofs
