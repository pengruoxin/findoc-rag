"""Evaluate a DeepSeek text fallback only on locally unresolved scan cells."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from findoc_rag.documents.pdf import PdfExtractionConfig, parse_pdf
from findoc_rag.pdf_evaluation import file_sha256
from findoc_rag.pdf_scan_evaluation import (
    evaluate_page_probes,
    load_scan_probe_benchmark,
)
from findoc_rag.pdf_table_interpretation import (
    DeepSeekTableInterpreter,
    TableQuestion,
    normalize_table_value,
    serialize_layout_page,
    table_values_equal,
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
            "reports/pdf-extraction/pdf-hard-v2-deepseek-high-risk-fallback-v1.json"
        ),
    )
    parser.add_argument("--ocr-dpi", type=int, default=180)
    parser.add_argument("--model", default="deepseek-chat")
    return parser.parse_args()


def unresolved_cell_predictions(page_report: dict) -> list[dict]:
    return [
        prediction
        for prediction in page_report["predictions"]
        if prediction["probe_type"] == "row_value"
        and not prediction["structured_cell_recoverable"]
    ]


def score_fallback_answers(
    predictions: list[dict], answers: list[dict], evidence: str
) -> list[dict]:
    expected = {prediction["probe_id"]: prediction for prediction in predictions}
    returned = {answer["question_id"]: answer for answer in answers}
    results: list[dict] = []
    for question_id, prediction in expected.items():
        answer = returned.get(question_id)
        status = answer["status"] if answer is not None else "missing"
        value = answer.get("value", "") if answer is not None else ""
        answered = status == "answered" and bool(value)
        value_correct = answered and table_values_equal(
            prediction["expected_value"], value
        )
        evidence_value_supported = answered and normalize_table_value(value) in (
            normalize_table_value(evidence)
        )
        # This lane is triggered precisely because local row/column proof is
        # absent.  A model-proposed value may help review, but cannot replace
        # that missing proof and is never auto-accepted.
        results.append(
            {
                "probe_id": question_id,
                "row_label": prediction["row_label"],
                "column_label": prediction["column_label"],
                "expected_value": prediction["expected_value"],
                "status": status,
                "returned_value": value,
                "value_correct": value_correct,
                "evidence_value_supported": evidence_value_supported,
                "decision": (
                    "manual_review"
                    if answered and evidence_value_supported
                    else "insufficient_evidence"
                ),
                "auto_accepted": False,
            }
        )
    return results


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    benchmark_path = args.benchmark.resolve(strict=True)
    benchmark, pdf_path = load_scan_probe_benchmark(benchmark_path)
    interpreter = DeepSeekTableInterpreter(model=args.model)
    if not interpreter.available:
        raise SystemExit("DEEPSEEK_API_KEY is required for the high-risk fallback run")

    document = parse_pdf(
        pdf_path,
        PdfExtractionConfig(mode="auto", ocr_backend="rapidocr", ocr_dpi=args.ocr_dpi),
    )
    expected_by_page = {page.page_number: page for page in benchmark.pages}
    total_cell_probes = sum(
        probe.probe_type == "row_value"
        for page in benchmark.pages
        for probe in page.probes
    )
    structured_recoverable_count = 0
    page_reports: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0
    fallback_results: list[dict] = []
    for page in document.pages:
        expected_page = expected_by_page[page.page_number]
        local_report = evaluate_page_probes(page, expected_page)
        row_predictions = [
            prediction
            for prediction in local_report["predictions"]
            if prediction["probe_type"] == "row_value"
        ]
        structured_recoverable_count += sum(
            prediction["structured_cell_recoverable"]
            for prediction in row_predictions
        )
        unresolved = unresolved_cell_predictions(local_report)
        if not unresolved:
            continue
        unresolved_ids = {prediction["probe_id"] for prediction in unresolved}
        probes = [
            probe for probe in expected_page.probes if probe.probe_id in unresolved_ids
        ]
        questions = [
            TableQuestion(
                question_id=probe.probe_id,
                question=(
                    f"请提取“{probe.row_label}”行、“{probe.column_label}”列的原始数值；"
                    "无法确认行列关系时返回 insufficient_evidence。"
                ),
                expected_value=probe.expected_value or "",
                row_label=probe.row_label or "",
                column_label=probe.column_label or "",
            )
            for probe in probes
        ]
        evidence = serialize_layout_page(page, max_characters=8_000)
        batch = interpreter.interpret_page(questions, evidence)
        total_input_tokens += batch.input_tokens or 0
        total_output_tokens += batch.output_tokens or 0
        scored = score_fallback_answers(
            unresolved,
            [answer.model_dump(mode="json") for answer in batch.answers],
            evidence,
        )
        fallback_results.extend(scored)
        page_reports.append(
            {
                "page_number": page.page_number,
                "candidate_id": expected_page.candidate_id,
                "local_unresolved_count": len(unresolved),
                "question_count": len(questions),
                "prompt_sha256": batch.prompt_sha256,
                "input_tokens": batch.input_tokens,
                "output_tokens": batch.output_tokens,
                "elapsed_ms": batch.elapsed_ms,
                "results": scored,
            }
        )

    fallback_count = len(fallback_results)
    correct_candidates = sum(item["value_correct"] for item in fallback_results)
    supported_candidates = sum(
        item["evidence_value_supported"] for item in fallback_results
    )
    report = {
        "schema_version": "1",
        "run_id": "pdf-hard-v2-deepseek-high-risk-fallback-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_status": "provisional_development_not_formal_gold",
        "benchmark_path": args.benchmark.as_posix(),
        "benchmark_sha256": file_sha256(benchmark_path),
        "pdf_path": benchmark.pdf_path,
        "pdf_sha256": benchmark.pdf_sha256,
        "policy": {
            "trigger": "local_structured_cell_recoverable_false",
            "scope": "unresolved_cells_only",
            "fallback_provider": interpreter.provider,
            "fallback_model": interpreter.model,
            "fallback_prompt_revision": interpreter.prompt_revision,
            "fallback_auto_accept_allowed": False,
            "reason": "model output cannot replace missing local row-column proof",
        },
        "metrics": {
            "page_count": len(document.pages),
            "fallback_page_count": len(page_reports),
            "fallback_page_rate": _ratio(len(page_reports), len(document.pages)),
            "cell_probe_count": total_cell_probes,
            "local_structured_recoverable_count": structured_recoverable_count,
            "local_structured_recall": _ratio(
                structured_recoverable_count, total_cell_probes
            ),
            "fallback_question_count": fallback_count,
            "fallback_cell_rate": _ratio(fallback_count, total_cell_probes),
            "fallback_answer_rate": _ratio(
                sum(item["status"] == "answered" for item in fallback_results),
                fallback_count,
            ),
            "fallback_candidate_value_accuracy": _ratio(
                correct_candidates, fallback_count
            ),
            "fallback_evidence_value_support_rate": _ratio(
                supported_candidates, fallback_count
            ),
            "candidate_coverage_if_human_confirms": _ratio(
                structured_recoverable_count + correct_candidates,
                total_cell_probes,
            ),
            "unsafe_auto_accept_rate": 0.0,
            "manual_review_count": sum(
                item["decision"] == "manual_review" for item in fallback_results
            ),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
        "pages": page_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics = report["metrics"]
    print(f"fallback_pages={metrics['fallback_page_count']}/{metrics['page_count']}")
    print(
        f"fallback_cells={metrics['fallback_question_count']}/"
        f"{metrics['cell_probe_count']}"
    )
    print(
        "candidate_value_accuracy="
        f"{metrics['fallback_candidate_value_accuracy']:.3f}"
    )
    print(f"unsafe_auto_accept_rate={metrics['unsafe_auto_accept_rate']:.3f}")
    print(
        f"tokens={metrics['total_input_tokens']}+{metrics['total_output_tokens']}"
    )
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
