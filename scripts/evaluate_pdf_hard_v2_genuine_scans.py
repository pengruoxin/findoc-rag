"""Run native and RapidOCR lanes on provisional genuine-scan probes."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from findoc_rag.pdf_evaluation import file_sha256
from findoc_rag.pdf_scan_evaluation import (
    compare_scan_lanes,
    evaluate_scan_lane,
    load_scan_probe_benchmark,
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
            "reports/pdf-extraction/pdf-hard-v2-genuine-scan-baseline-v1.json"
        ),
    )
    parser.add_argument("--ocr-backend", default="rapidocr")
    parser.add_argument("--ocr-dpi", type=int, default=180)
    parser.add_argument("--legacy-rotated-geometry", action="store_true")
    parser.add_argument("--disable-hierarchical-headers", action="store_true")
    parser.add_argument("--disable-wrapped-row-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_path = args.benchmark.resolve(strict=True)
    benchmark, pdf_path = load_scan_probe_benchmark(benchmark_path)
    evaluation_options = {
        "use_display_geometry": not args.legacy_rotated_geometry,
        "allow_hierarchical_headers": not args.disable_hierarchical_headers,
        "allow_wrapped_row_labels": not args.disable_wrapped_row_labels,
    }
    native = evaluate_scan_lane(
        pdf_path, benchmark, lane="native", **evaluation_options
    )
    hybrid = evaluate_scan_lane(
        pdf_path,
        benchmark,
        lane="hybrid",
        ocr_backend=args.ocr_backend,
        ocr_dpi=args.ocr_dpi,
        **evaluation_options,
    )
    report = {
        "schema_version": "1",
        "run_id": "pdf-hard-v2-genuine-scan-baseline-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_status": "provisional_development_not_formal_gold",
        "benchmark_path": args.benchmark.as_posix(),
        "benchmark_sha256": file_sha256(benchmark_path),
        "pdf_path": benchmark.pdf_path,
        "pdf_sha256": benchmark.pdf_sha256,
        "counts_toward_formal_target": False,
        "evaluation_options": evaluation_options,
        "lanes": {"native": native, "hybrid": hybrid},
        "hybrid_minus_native": compare_scan_lanes(native, hybrid),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, lane in report["lanes"].items():
        metrics = lane["metrics"]
        print(
            f"{name}: strict_probe_recall={metrics['strict_probe_recall']:.3f} "
            f"same_row={metrics['same_row_association_recall']:.3f} "
            f"coordinate_bounds={metrics['ocr_coordinate_bounds_rate']:.3f} "
            f"elapsed_ms={metrics['elapsed_ms']:.1f}"
        )
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
