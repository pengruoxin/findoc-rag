"""Run native and hybrid PDF extraction lanes and write an auditable report."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from findoc_rag.pdf_evaluation import run_pdf_extraction_benchmark, write_evaluation_report
from findoc_rag.pdf_table_interpretation import DeepSeekTableInterpreter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/pdf-extraction-v1/benchmark.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pdf-extraction/pdf-extraction-v1-baseline.json"),
    )
    parser.add_argument("--lanes", default="native,hybrid")
    parser.add_argument("--ocr-backend", default="rapidocr")
    parser.add_argument("--ocr-dpi", type=int, default=180)
    parser.add_argument(
        "--deepseek-table",
        action="store_true",
        help="interpret all table questions once per eligible page with DeepSeek text API",
    )
    parser.add_argument(
        "--require-deepseek-table",
        action="store_true",
        help="fail instead of recording not_run when the provider key is unavailable",
    )
    parser.add_argument(
        "--deepseek-model",
        default=os.getenv("FINDOC_RAG_ANSWER_MODEL", "deepseek-chat"),
    )
    parser.add_argument(
        "--deepseek-endpoint",
        default=os.getenv(
            "FINDOC_RAG_ANSWER_ENDPOINT",
            "https://api.deepseek.com/chat/completions",
        ),
    )
    parser.add_argument("--min-table-evidence-characters", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lanes = tuple(lane.strip() for lane in args.lanes.split(",") if lane.strip())
    unsupported = set(lanes) - {"native", "hybrid"}
    if unsupported:
        raise SystemExit(f"Unsupported lanes: {sorted(unsupported)}")
    deepseek_requested = args.deepseek_table or args.require_deepseek_table
    table_interpreter = None
    if deepseek_requested:
        candidate = DeepSeekTableInterpreter(
            model=args.deepseek_model,
            endpoint=args.deepseek_endpoint,
        )
        if candidate.available:
            table_interpreter = candidate
        elif args.require_deepseek_table:
            raise SystemExit(
                "DeepSeek table evaluation requires the provider-bound API key"
            )
    report = run_pdf_extraction_benchmark(
        args.dataset,
        lanes=lanes,
        ocr_backend=args.ocr_backend,
        ocr_dpi=args.ocr_dpi,
        table_interpreter=table_interpreter,
        table_interpretation_requested=deepseek_requested,
        min_table_evidence_characters=args.min_table_evidence_characters,
    )
    write_evaluation_report(report, args.output)
    for lane, lane_report in report["lanes"].items():
        overall = lane_report["overall"]
        print(
            f"{lane}: similarity={overall['mean_text_similarity']:.4f} "
            f"numeric_recall={overall['mean_numeric_recall']:.4f} "
            f"route_accuracy={overall['route_accuracy']:.4f} "
            f"table_recoverability="
            f"{lane_report['table_structure']['overall'].get('recoverable_rate', 0):.4f} "
            f"elapsed_ms={lane_report['elapsed_ms']:.1f}"
        )
        interpretation = lane_report["table_interpretation"]
        if "overall" in interpretation:
            print(
                f"{lane}/deepseek-table: "
                f"value_accuracy={interpretation['overall']['value_accuracy']:.4f} "
                f"strict_cell_accuracy="
                f"{interpretation['overall']['strict_cell_accuracy']:.4f} "
                f"input_tokens={interpretation['total_input_tokens']} "
                f"output_tokens={interpretation['total_output_tokens']}"
            )
        elif interpretation["status"] != "not_requested":
            print(
                f"{lane}/deepseek-table: status={interpretation['status']} "
                f"reason={interpretation.get('reason', '')}"
            )
    if report["comparison"] is not None:
        comparison = report["comparison"]
        print(
            "hybrid-native: "
            f"similarity={comparison['text_similarity_delta']:+.4f} "
            f"numeric_recall={comparison['numeric_recall_delta']:+.4f} "
            f"cer={comparison['character_error_rate_delta']:+.4f}"
        )
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
