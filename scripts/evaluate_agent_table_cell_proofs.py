"""Evaluate P4-G table cell/geometry proofs without opening the frozen split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from findoc_rag.corpus import collect_active_chunks, collect_active_documents
from findoc_rag.indexing import PersistentIndex
from findoc_rag.registry import DocumentRegistry
from findoc_rag.structured_tables import build_structured_tables, load_structured_tables
from findoc_rag.table_cell_proof import build_table_cell_proofs

DEFAULT_REGISTRY = Path("data/catalog/agent-hard-v3/registry.sqlite3")
DEFAULT_INDEX = Path("data/indexes/agent-hard-v3/development/generations/20260820T055043-3f82e702")
DEFAULT_OUTPUT = Path("reports/agent/agent-p4g-table-cell-geometry-proof-v1.json")
ALLOWED_SPLITS = {"calibration", "dev"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_cells(tables) -> set[tuple[str, str, str, str, str]]:
    return {
        (table.table_id, cell.section, cell.row, cell.column, cell.value)
        for table in tables
        for cell in table.cells
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--baseline-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    registry_path = args.registry.resolve()
    registry = DocumentRegistry(registry_path)
    versions = [
        version
        for version in registry.active_versions()
        if version.metadata.get("benchmark_split") in ALLOWED_SPLITS
    ]
    if not versions or any(
        version.metadata.get("benchmark_split") not in ALLOWED_SPLITS for version in versions
    ):
        raise ValueError("Evaluation may read only calibration and dev versions")

    chunks, version_ids = collect_active_chunks(registry, ALLOWED_SPLITS)
    documents = collect_active_documents(registry, ALLOWED_SPLITS)
    rebuilt_tables = build_structured_tables(chunks, documents)

    baseline_index = PersistentIndex(args.baseline_index)
    baseline_path = baseline_index.directory / "structured_tables.jsonl"
    baseline_tables = load_structured_tables(baseline_path.read_text(encoding="utf-8"))
    baseline_semantic = _semantic_cells(baseline_tables)
    rebuilt_semantic = _semantic_cells(rebuilt_tables)
    if baseline_semantic != rebuilt_semantic:
        missing = len(baseline_semantic - rebuilt_semantic)
        added = len(rebuilt_semantic - baseline_semantic)
        raise AssertionError(
            f"Cell semantics changed while adding geometry: missing={missing}, added={added}"
        )

    tables_by_chunk: dict[str, list] = defaultdict(list)
    for table in rebuilt_tables:
        tables_by_chunk[table.chunk_id].append(table)
    proofs = []
    for chunk in chunks:
        tables = tables_by_chunk.get(chunk.chunk_id, [])
        if not tables:
            continue
        enriched = chunk.model_copy(update={"structured_tables": tables})
        proofs.extend(build_table_cell_proofs(enriched))

    coordinate_cells = [
        cell for table in rebuilt_tables if table.source == "coordinate" for cell in table.cells
    ]
    geometry_cells = [
        cell
        for cell in coordinate_cells
        if cell.page_number is not None
        and cell.value_bbox is not None
        and cell.coordinate_space is not None
    ]
    all_geometry_cells = [
        cell
        for table in rebuilt_tables
        for cell in table.cells
        if cell.page_number is not None
        and cell.value_bbox is not None
        and cell.coordinate_space is not None
    ]
    text_source_geometry_cells = [
        cell
        for table in rebuilt_tables
        if table.source == "text"
        for cell in table.cells
        if cell.page_number is not None
        and cell.value_bbox is not None
        and cell.coordinate_space is not None
    ]
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    text_only_items = []
    for table in rebuilt_tables:
        chunk = chunks_by_id[table.chunk_id]
        for cell in table.cells:
            if cell.value_bbox is not None:
                continue
            text_only_items.append(
                {
                    "document_key": chunk.document_key,
                    "company_name": chunk.company_name,
                    "report_year": chunk.report_year,
                    "chunk_id": table.chunk_id,
                    "table_id": table.table_id,
                    "table_type": table.table_type,
                    "page_start": table.page_start,
                    "page_end": table.page_end,
                    "section": cell.section,
                    "row": cell.row,
                    "row_index": cell.row_index,
                    "column": cell.column,
                    "column_index": cell.column_index,
                    "value": cell.value,
                }
            )
    indexed_cells = [
        cell
        for table in rebuilt_tables
        for cell in table.cells
        if cell.row_index is not None and cell.column_index is not None
    ]
    coordinate_proofs = [proof for proof in proofs if proof.geometry_status == "coordinate"]
    baseline_geometry_cells = sum(
        cell.value_bbox is not None for table in baseline_tables for cell in table.cells
    )
    total_cells = sum(len(table.cells) for table in rebuilt_tables)
    geometry_breakdown: dict[str, dict[str, int]] = {}
    for table in rebuilt_tables:
        bucket = geometry_breakdown.setdefault(
            table.table_type,
            {"table_count": 0, "cell_count": 0, "geometry_cell_count": 0},
        )
        bucket["table_count"] += 1
        bucket["cell_count"] += len(table.cells)
        bucket["geometry_cell_count"] += sum(cell.value_bbox is not None for cell in table.cells)
    for bucket in geometry_breakdown.values():
        bucket["text_only_cell_count"] = bucket["cell_count"] - bucket["geometry_cell_count"]
    checks = {
        "semantic_cells_unchanged": baseline_semantic == rebuilt_semantic,
        "every_cell_has_tamper_evident_proof": len(proofs) == total_cells,
        "every_cell_has_logical_row_column_index": len(indexed_cells) == total_cells,
        "every_coordinate_cell_has_pdf_bbox": len(geometry_cells) == len(coordinate_cells),
        "coordinate_proof_count_matches_geometry_cells": len(coordinate_proofs)
        == len(all_geometry_cells),
        "frozen_split_not_loaded": all(
            version.metadata.get("benchmark_split") in ALLOWED_SPLITS for version in versions
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"P4-G contract failed: {checks}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1",
        "status": "complete",
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "evaluation": "P4-G table cell/geometry proof on calibration+dev PDF IR",
        "source": {
            "registry_path": str(registry_path.relative_to(Path.cwd().resolve())).replace(
                "\\", "/"
            ),
            "registry_sha256": _sha256(registry_path),
            "baseline_index_id": baseline_index.manifest.index_id,
            "baseline_generation_path": str(
                baseline_index.directory.relative_to(Path.cwd().resolve())
            ).replace("\\", "/"),
            "baseline_manifest_sha256": _sha256(baseline_index.directory / "manifest.json"),
            "baseline_schema_version": (baseline_index.manifest.structured_table_schema_version),
            "baseline_generator": baseline_index.manifest.structured_table_generator,
            "version_ids": version_ids,
            "splits": sorted(ALLOWED_SPLITS),
            "document_count": len(documents),
            "chunk_count": len(chunks),
        },
        "baseline": {
            "table_count": len(baseline_tables),
            "cell_count": sum(len(table.cells) for table in baseline_tables),
            "geometry_cell_count": baseline_geometry_cells,
            "geometry_cell_rate": (
                baseline_geometry_cells / sum(len(table.cells) for table in baseline_tables)
            ),
        },
        "p4g": {
            "table_count": len(rebuilt_tables),
            "cell_count": total_cells,
            "semantic_cell_exact_count": len(baseline_semantic & rebuilt_semantic),
            "semantic_cell_exact_rate": len(baseline_semantic & rebuilt_semantic)
            / len(baseline_semantic),
            "logical_index_count": len(indexed_cells),
            "logical_index_rate": len(indexed_cells) / total_cells,
            "coordinate_source_cell_count": len(coordinate_cells),
            "coordinate_geometry_cell_count": len(geometry_cells),
            "coordinate_geometry_coverage": len(geometry_cells) / len(coordinate_cells),
            "text_source_geometry_cell_count": len(text_source_geometry_cells),
            "total_geometry_cell_count": len(all_geometry_cells),
            "total_geometry_coverage": len(all_geometry_cells) / total_cells,
            "text_only_cell_count": total_cells - len(all_geometry_cells),
            "text_only_items": text_only_items,
            "binding_validated_proof_count": len(proofs),
            "proof_coverage": len(proofs) / total_cells,
            "geometry_breakdown": geometry_breakdown,
            "checks": checks,
        },
        "runtime_cost": {
            "model_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "frozen_test_opened": False,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "P4-G cell proofs: "
        f"{len(proofs)}/{total_cells}; coordinate geometry: "
        f"{len(geometry_cells)}/{len(coordinate_cells)}"
    )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
