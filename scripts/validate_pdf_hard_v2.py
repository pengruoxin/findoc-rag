"""Validate the complex-PDF benchmark inventory without opening frozen gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.pdf_complex_benchmark import PdfHardBenchmarkManifest
from findoc_rag.pdf_evaluation import PdfExtractionBenchmark, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/pdf-hard-v2/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pdf-extraction/pdf-hard-v2-inventory.json"),
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def validate_inventory(manifest_path: Path, workspace: Path) -> dict:
    manifest = PdfHardBenchmarkManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    loaded_sources: dict[str, dict] = {}
    source_results: list[dict] = []
    for source in manifest.sources:
        benchmark_path = (workspace / source.benchmark_path).resolve(strict=True)
        pdf_path = (workspace / source.pdf_path).resolve(strict=True)
        actual_benchmark_sha256 = file_sha256(benchmark_path)
        actual_pdf_sha256 = file_sha256(pdf_path)
        if source.source_format == "legacy_extraction_benchmark":
            benchmark = PdfExtractionBenchmark.model_validate_json(
                benchmark_path.read_text(encoding="utf-8")
            )
            declared_pdf_sha256 = benchmark.pdf_sha256
            page_cases = {page.case_id: [page.page_number] for page in benchmark.pages}
            page_groups = {
                group.group_id: group.page_numbers for group in benchmark.evaluation_groups
            }
            available_pages = {page.page_number for page in benchmark.pages}
            question_count = len(benchmark.table_questions)
        else:
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            declared_pdf_sha256 = benchmark["pdf_sha256"]
            page_cases = {
                page["candidate_id"]: [page["challenge_page_number"]]
                for page in benchmark["pages"]
            }
            page_groups = {}
            available_pages = {
                page["challenge_page_number"] for page in benchmark["pages"]
            }
            question_count = 0
        source_valid = (
            actual_benchmark_sha256 == source.benchmark_sha256
            and actual_pdf_sha256 == source.pdf_sha256
            and declared_pdf_sha256 == source.pdf_sha256
        )
        source_results.append(
            {
                "source_id": source.source_id,
                "source_format": source.source_format,
                "source_valid": source_valid,
                "benchmark_path": source.benchmark_path,
                "benchmark_sha256": actual_benchmark_sha256,
                "pdf_path": source.pdf_path,
                "pdf_sha256": actual_pdf_sha256,
                "page_count": len(available_pages),
                "question_count": question_count,
            }
        )
        loaded_sources[source.source_id] = {
            "available_pages": available_pages,
            "page_cases": page_cases,
            "page_groups": page_groups,
        }

    case_results: list[dict] = []
    for case in manifest.cases:
        source_index = loaded_sources[case.source_id]
        available_pages = source_index["available_pages"]
        expected_pages: list[int]
        source_reference_valid: bool
        if case.source_case_id:
            expected_pages = source_index["page_cases"].get(case.source_case_id, [])
            source_reference_valid = bool(expected_pages)
        else:
            expected_pages = source_index["page_groups"].get(case.source_group_id, [])
            source_reference_valid = bool(expected_pages)
        pages_valid = set(case.page_numbers) <= available_pages and case.page_numbers == expected_pages
        case_results.append(
            {
                "case_id": case.case_id,
                "source_id": case.source_id,
                "source_reference_valid": source_reference_valid,
                "pages_valid": pages_valid,
                "source_kind": case.source_kind,
                "primary_stratum": case.primary_stratum,
                "split": case.split,
                "table_count": case.table_count,
                "counts_toward_target": case.counts_toward_target,
            }
        )

    quotas = manifest.quota_report()
    remaining_tables = sum(
        value
        for stratum in quotas.values()
        for name, value in stratum.items()
        if name.endswith("_remaining")
    )
    source_integrity = all(item["source_valid"] for item in source_results)
    case_integrity = all(
        item["source_reference_valid"] and item["pages_valid"] for item in case_results
    )
    controlled_raster_cases = sum(
        case.source_kind == "controlled_rasterization" for case in manifest.cases
    )
    genuine_scan_cases = sum(
        case.source_kind == "genuine_scanned_pdf" for case in manifest.cases
    )
    return {
        "dataset_id": manifest.dataset_id,
        "schema_version": manifest.schema_version,
        "manifest_status": manifest.status,
        "source_integrity": source_integrity,
        "case_integrity": case_integrity,
        "ready_for_formal_freeze": (
            source_integrity and case_integrity and remaining_tables == 0
        ),
        "gold_loaded": False,
        "frozen_gold_boundary": "manifest_only_no_frozen_gold_access",
        "legacy_case_count": sum(
            case.split == "legacy_regression" for case in manifest.cases
        ),
        "controlled_raster_case_count": controlled_raster_cases,
        "genuine_scan_case_count": genuine_scan_cases,
        "remaining_formal_table_slots": remaining_tables,
        "quota_report": quotas,
        "sources": source_results,
        "cases": case_results,
    }


def main() -> None:
    args = parse_args()
    workspace = Path.cwd().resolve()
    manifest_path = args.manifest.resolve(strict=True)
    report = validate_inventory(manifest_path, workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["source_integrity"] or not report["case_integrity"]:
        raise SystemExit(1)
    if args.require_complete and not report["ready_for_formal_freeze"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
