"""Validate a second-review packet against provisional PDF table gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.pdf_evaluation import PdfExtractionBenchmark
from findoc_rag.pdf_review import PdfReviewPacket, evaluate_pdf_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/pdf-cross-page-v1/benchmark.json"),
    )
    parser.add_argument(
        "--review-packet",
        type=Path,
        default=Path("data/evaluation/pdf-cross-page-v1/blind-review-packet.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pdf-extraction/pdf-cross-page-v1-review-status.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = PdfExtractionBenchmark.model_validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    packet = PdfReviewPacket.model_validate_json(
        args.review_packet.read_text(encoding="utf-8")
    )
    report = evaluate_pdf_review(benchmark, packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"status={report['status']}")
    print(f"completed={report['completed_count']}/{report['question_count']}")
    print(f"independent_gold_ready={report['independent_gold_ready']}")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
