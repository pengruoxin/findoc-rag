"""Retry only unresolved scan pages with red-seal-suppressing OCR."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from findoc_rag.documents.pdf import (
    PdfExtractionConfig,
    parse_pdf,
    replace_parsed_document_pages,
)
from findoc_rag.pdf_evaluation import file_sha256
from findoc_rag.pdf_scan_evaluation import (
    evaluate_page_probes,
    load_scan_probe_benchmark,
    summarize_scan_lane,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(
            "data/evaluation/pdf-hard-v2/genuine-scan-provisional-probes-v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/pdf-extraction/pdf-hard-v2-adaptive-red-channel-ocr-v1.json"
        ),
    )
    parser.add_argument("--base-dpi", type=int, default=180)
    parser.add_argument("--retry-dpi", type=int, default=240)
    return parser.parse_args()


def _page_reports(document, expected_by_page: dict) -> list[dict]:
    return [
        evaluate_page_probes(page, expected_by_page[page.page_number])
        for page in document.pages
    ]


def _unresolved_pages(page_reports: list[dict]) -> set[int]:
    return {
        page["page_number"]
        for page in page_reports
        if any(
            prediction["probe_type"] == "row_value"
            and not prediction["structured_cell_recoverable"]
            for prediction in page["predictions"]
        )
    }


def main() -> None:
    args = parse_args()
    benchmark_path = args.benchmark.resolve(strict=True)
    benchmark, pdf_path = load_scan_probe_benchmark(benchmark_path)
    expected_by_page = {page.page_number: page for page in benchmark.pages}

    started = time.perf_counter()
    base = parse_pdf(
        pdf_path,
        PdfExtractionConfig(
            mode="auto", ocr_backend="rapidocr", ocr_dpi=args.base_dpi
        ),
    )
    base_elapsed_ms = (time.perf_counter() - started) * 1000
    base_pages = _page_reports(base, expected_by_page)
    retry_pages = _unresolved_pages(base_pages)

    retry_elapsed_ms = 0.0
    final = base
    if retry_pages:
        retry_started = time.perf_counter()
        retry = parse_pdf(
            pdf_path,
            PdfExtractionConfig(
                mode="force",
                ocr_backend="rapidocr-red-channel",
                ocr_dpi=args.retry_dpi,
                ocr_page_numbers=sorted(retry_pages),
            ),
        )
        retry_elapsed_ms = (time.perf_counter() - retry_started) * 1000
        final = replace_parsed_document_pages(base, retry, retry_pages)

    final_pages = _page_reports(final, expected_by_page)
    base_metrics = summarize_scan_lane(base_pages, base_elapsed_ms)
    final_metrics = summarize_scan_lane(
        final_pages, base_elapsed_ms + retry_elapsed_ms
    )
    report = {
        "schema_version": "1",
        "run_id": "pdf-hard-v2-adaptive-red-channel-ocr-v1",
        "evaluation_status": "provisional_development_not_formal_gold",
        "benchmark_path": args.benchmark.as_posix(),
        "benchmark_sha256": file_sha256(benchmark_path),
        "pdf_path": benchmark.pdf_path,
        "pdf_sha256": benchmark.pdf_sha256,
        "policy": {
            "base_backend": "rapidocr",
            "base_dpi": args.base_dpi,
            "retry_backend": "rapidocr-red-channel",
            "retry_dpi": args.retry_dpi,
            "retry_trigger": "page_has_unresolved_structured_cell",
        },
        "retry_page_numbers": sorted(retry_pages),
        "retry_page_rate": len(retry_pages) / len(final.pages),
        "base_metrics": base_metrics,
        "final_metrics": final_metrics,
        "deltas": {
            name: final_metrics[name] - base_metrics[name]
            for name in (
                "strict_probe_recall",
                "same_row_association_recall",
                "structured_cell_recall",
                "ocr_coordinate_bounds_rate",
                "elapsed_ms",
            )
        },
        "pages": final_pages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"retry_pages={len(retry_pages)}/{len(final.pages)}")
    print(
        "structured_cell_recall="
        f"{base_metrics['structured_cell_recall']:.3f}->"
        f"{final_metrics['structured_cell_recall']:.3f}"
    )
    print(f"elapsed_ms={final_metrics['elapsed_ms']:.1f}")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
