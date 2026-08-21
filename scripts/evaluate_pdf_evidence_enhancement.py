"""Evaluate bounded PDF evidence enhancement without opening the frozen split."""

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
from findoc_rag.structured_tables import (
    STRUCTURED_TABLE_GENERATOR,
    build_structured_tables,
    load_structured_tables,
)
from findoc_rag.table_cell_proof import build_table_cell_proofs
from findoc_rag.visual_inspection import PdfRegionInspector

DEFAULT_REGISTRY = Path("data/catalog/agent-hard-v3/registry.sqlite3")
DEFAULT_BASELINE_INDEX = Path(
    "data/indexes/agent-hard-v3/development/generations/20260820T055043-3f82e702"
)
DEFAULT_HARD_CASES = Path("data/evaluation/pdf-evidence-hard-cases-v1.json")
DEFAULT_PRIOR_REPORT = Path("reports/agent/agent-p4g-table-cell-geometry-proof-v1.json")
DEFAULT_SOURCE_MANIFEST = Path("data/evaluation/agent-hard-v3-source-manifest.json")
DEFAULT_OUTPUT = Path("reports/agent/pdf-evidence-enhancement-v1.json")
DEFAULT_REGION_DIRECTORY = Path("reports/agent/pdf-evidence-regions-v1")
ALLOWED_SPLITS = {"calibration", "dev"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_key(document_key: str | None, table, cell) -> tuple[str, ...]:
    return (
        document_key or "",
        table.table_type,
        cell.section,
        cell.row,
        cell.column,
        cell.value,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--baseline-index",
        type=Path,
        default=DEFAULT_BASELINE_INDEX,
    )
    parser.add_argument("--hard-cases", type=Path, default=DEFAULT_HARD_CASES)
    parser.add_argument(
        "--prior-report",
        type=Path,
        default=DEFAULT_PRIOR_REPORT,
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--region-directory",
        type=Path,
        default=DEFAULT_REGION_DIRECTORY,
    )
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
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    new_tables = build_structured_tables(chunks, documents)

    baseline_index = PersistentIndex(args.baseline_index)
    baseline_tables = load_structured_tables(
        (baseline_index.directory / "structured_tables.jsonl").read_text(encoding="utf-8")
    )
    hard_cases = json.loads(args.hard_cases.read_text(encoding="utf-8"))
    prior_report = json.loads(args.prior_report.read_text(encoding="utf-8"))
    if set(hard_cases["allowed_splits"]) != ALLOWED_SPLITS:
        raise ValueError("Hard-case fixture must be calibration/dev only")

    false_table_keys = {
        (item["chunk_id"], item["table_type"]) for item in hard_cases["false_positive_tables"]
    }
    baseline_false_tables = [
        table for table in baseline_tables if (table.chunk_id, table.table_type) in false_table_keys
    ]
    new_false_tables = [
        table for table in new_tables if (table.chunk_id, table.table_type) in false_table_keys
    ]
    expected_false_cells = sum(
        item["baseline_cell_count"] for item in hard_cases["false_positive_tables"]
    )

    baseline_semantics = {
        _semantic_key(
            chunks_by_id.get(table.chunk_id).document_key
            if table.chunk_id in chunks_by_id
            else None,
            table,
            cell,
        )
        for table in baseline_tables
        if (table.chunk_id, table.table_type) not in false_table_keys
        and table.table_type != "annual_data"
        for cell in table.cells
    }
    new_semantics = {
        _semantic_key(chunks_by_id[table.chunk_id].document_key, table, cell)
        for table in new_tables
        for cell in table.cells
    }

    tables_by_chunk: dict[str, list] = defaultdict(list)
    for table in new_tables:
        tables_by_chunk[table.chunk_id].append(table)
    proofs = []
    proof_by_expected_key = {}
    for chunk in chunks:
        chunk_tables = tables_by_chunk.get(chunk.chunk_id, [])
        if not chunk_tables:
            continue
        enriched = chunk.model_copy(update={"structured_tables": chunk_tables})
        for proof in build_table_cell_proofs(enriched):
            proofs.append(proof)
            proof_by_expected_key[
                (
                    chunk.document_key,
                    proof.row,
                    proof.column,
                    proof.value,
                )
            ] = proof

    expected_cells = {tuple(item) for item in hard_cases["expected_cells"]}
    exact_expected_cells = expected_cells & set(proof_by_expected_key)
    expected_coordinate_cells = {
        key
        for key in expected_cells
        if key in proof_by_expected_key
        and proof_by_expected_key[key].geometry_status == "coordinate"
    }

    inspector = PdfRegionInspector(
        args.source_manifest.resolve(),
        workspace=Path.cwd(),
    )
    region_directory = args.region_directory.resolve()
    region_proofs = []
    for item in hard_cases["region_proof_cases"]:
        key = tuple(item)
        proof = proof_by_expected_key.get(key)
        if proof is None:
            raise AssertionError(f"Missing region-proof case: {key}")
        region_proofs.append(
            inspector.render_table_cell_region(
                str(key[0]),
                proof,
                output_directory=region_directory,
            )
        )

    total_cells = sum(len(table.cells) for table in new_tables)
    geometry_cells = sum(
        cell.value_bbox is not None for table in new_tables for cell in table.cells
    )
    checks = {
        "baseline_contains_all_adjudicated_false_tables": len(baseline_false_tables)
        == len(false_table_keys),
        "baseline_false_cell_count_matches_fixture": sum(
            len(table.cells) for table in baseline_false_tables
        )
        == expected_false_cells,
        "adjudicated_false_tables_removed": not new_false_tables,
        "unaffected_baseline_cells_preserved": baseline_semantics <= new_semantics,
        "manual_expected_cells_exact": exact_expected_cells == expected_cells,
        "manual_expected_cells_have_coordinates": expected_coordinate_cells == expected_cells,
        "every_new_cell_has_geometry": geometry_cells == total_cells,
        "every_new_cell_has_tamper_evident_proof": len(proofs) == total_cells,
        "bounded_region_cases_rendered": len(region_proofs)
        == len(hard_cases["region_proof_cases"]),
        "bounded_region_area_contract": all(
            proof.rendered_area_ratio <= 0.2 for proof in region_proofs
        ),
        "frozen_split_not_loaded": all(
            version.metadata.get("benchmark_split") in ALLOWED_SPLITS for version in versions
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"PDF evidence enhancement contract failed: {checks}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1",
        "status": "complete",
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "evaluation": "PDF evidence enhancement on calibration+dev",
        "source": {
            "registry_path": registry_path.relative_to(Path.cwd()).as_posix(),
            "registry_sha256": _sha256(registry_path),
            "baseline_index_id": baseline_index.manifest.index_id,
            "baseline_generation_path": baseline_index.directory.relative_to(Path.cwd()).as_posix(),
            "hard_case_fixture": args.hard_cases.as_posix(),
            "hard_case_fixture_sha256": _sha256(args.hard_cases),
            "prior_report": args.prior_report.as_posix(),
            "prior_report_sha256": _sha256(args.prior_report),
            "source_manifest": args.source_manifest.as_posix(),
            "source_manifest_sha256": _sha256(args.source_manifest),
            "version_ids": version_ids,
            "splits": sorted(ALLOWED_SPLITS),
            "document_count": len(documents),
            "chunk_count": len(chunks),
        },
        "baseline": {
            "table_count": len(baseline_tables),
            "cell_count": sum(len(table.cells) for table in baseline_tables),
            "geometry_cell_count": sum(
                cell.value_bbox is not None for table in baseline_tables for cell in table.cells
            ),
            "adjudicated_false_table_count": len(baseline_false_tables),
            "adjudicated_false_cell_count": sum(
                len(table.cells) for table in baseline_false_tables
            ),
        },
        "previous_p4g": {
            "table_count": prior_report["p4g"]["table_count"],
            "cell_count": prior_report["p4g"]["cell_count"],
            "geometry_cell_count": prior_report["p4g"]["total_geometry_cell_count"],
            "text_only_cell_count": prior_report["p4g"]["text_only_cell_count"],
            "frozen_test_opened": prior_report["frozen_test_opened"],
        },
        "enhanced": {
            "generator": STRUCTURED_TABLE_GENERATOR,
            "table_count": len(new_tables),
            "cell_count": total_cells,
            "geometry_cell_count": geometry_cells,
            "geometry_coverage": geometry_cells / total_cells,
            "text_only_cell_count": total_cells - geometry_cells,
            "proof_count": len(proofs),
            "adjudicated_false_table_count": len(new_false_tables),
            "unaffected_baseline_cell_count": len(baseline_semantics),
            "unaffected_baseline_cell_preserved_count": len(baseline_semantics & new_semantics),
            "manual_expected_cell_count": len(expected_cells),
            "manual_expected_cell_exact_count": len(exact_expected_cells),
            "region_proof_count": len(region_proofs),
            "region_area_ratio_max": max(proof.rendered_area_ratio for proof in region_proofs),
            "region_proofs": [proof.model_dump(mode="json") for proof in region_proofs],
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
        "PDF evidence enhancement: "
        f"false tables {len(baseline_false_tables)} -> {len(new_false_tables)}; "
        f"geometry {geometry_cells}/{total_cells}; "
        f"manual hard cells {len(exact_expected_cells)}/{len(expected_cells)}"
    )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
