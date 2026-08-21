"""Independent, layered evaluation for native-first PDF extraction."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from findoc_rag.documents.models import ParsedDocument
from findoc_rag.documents.pdf import PdfExtractionConfig, parse_pdf
from findoc_rag.documents.routing import ExtractionRoute
from findoc_rag.pdf_table_interpretation import (
    TableInterpreter,
    TableQuestion,
    normalize_table_label,
    score_table_fact_pages,
    serialize_layout_pages,
    table_values_equal,
)

ChallengeType = Literal["native_text", "scanned_page", "mixed_page"]
GroupChallengeType = Literal[
    "native_text",
    "scanned_page",
    "mixed_page",
    "native_cross_page",
    "scanned_cross_page",
]
EvaluationLane = Literal["native", "hybrid"]
NUMBER = re.compile(r"[-+]?\d[\d,，]*(?:\.\d+)?%?")


class PdfBenchmarkPage(BaseModel):
    case_id: str
    page_number: int = Field(ge=1)
    challenge_type: ChallengeType
    expected_route: ExtractionRoute
    reference_text: str = Field(min_length=1)
    source_document: str
    source_page_number: int = Field(ge=1)


class PdfEvaluationGroup(BaseModel):
    group_id: str
    page_numbers: list[int] = Field(min_length=1)
    challenge_type: GroupChallengeType

    @model_validator(mode="after")
    def validate_page_numbers(self) -> PdfEvaluationGroup:
        if len(set(self.page_numbers)) != len(self.page_numbers):
            raise ValueError("Evaluation group page numbers must be unique")
        if self.page_numbers != sorted(self.page_numbers):
            raise ValueError("Evaluation group page numbers must be in ascending order")
        return self


class PdfExtractionBenchmark(BaseModel):
    schema_version: Literal["1", "2", "3"] = "2"
    dataset_id: str
    pdf_path: str
    pdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_policy: Literal["independent_native_source_page"]
    pages: list[PdfBenchmarkPage]
    table_questions: list[TableQuestion] = Field(default_factory=list)
    evaluation_groups: list[PdfEvaluationGroup] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pages(self) -> PdfExtractionBenchmark:
        page_numbers = [page.page_number for page in self.pages]
        case_ids = [page.case_id for page in self.pages]
        if len(set(page_numbers)) != len(page_numbers):
            raise ValueError("Benchmark page numbers must be unique")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Benchmark case IDs must be unique")
        question_ids = [question.question_id for question in self.table_questions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("Benchmark table question IDs must be unique")
        group_ids = [group.group_id for group in self.evaluation_groups]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("Benchmark evaluation group IDs must be unique")
        available_pages = set(page_numbers)
        for group in self.evaluation_groups:
            missing = set(group.page_numbers) - available_pages
            if missing:
                raise ValueError(
                    f"Evaluation group {group.group_id} references missing pages: "
                    f"{sorted(missing)}"
                )
            for question in self.table_questions:
                if question.target_page_offset >= len(group.page_numbers):
                    raise ValueError(
                        f"Question {question.question_id} targets offset "
                        f"{question.target_page_offset}, outside group {group.group_id}"
                    )
        return self


class PdfPageExtractionScore(BaseModel):
    case_id: str
    page_number: int
    challenge_type: ChallengeType
    expected_route: ExtractionRoute
    actual_route: ExtractionRoute
    route_correct: bool
    reference_characters: int = Field(ge=0)
    extracted_characters: int = Field(ge=0)
    character_error_rate: float = Field(ge=0)
    text_similarity: float = Field(ge=0, le=1)
    numeric_precision: float = Field(ge=0, le=1)
    numeric_recall: float = Field(ge=0, le=1)
    numeric_f1: float = Field(ge=0, le=1)
    ocr_attempted: bool
    ocr_succeeded: bool
    native_element_count: int = Field(ge=0)
    ocr_element_count: int = Field(ge=0)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def normalize_extraction_text(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def edit_distance(reference: str, candidate: str) -> int:
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, candidate_character in enumerate(candidate, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_character != candidate_character),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, candidate: str) -> float:
    normalized_reference = normalize_extraction_text(reference)
    normalized_candidate = normalize_extraction_text(candidate)
    if not normalized_reference:
        return float(bool(normalized_candidate))
    return edit_distance(normalized_reference, normalized_candidate) / len(normalized_reference)


def _numbers(text: str) -> Counter[str]:
    normalized = unicodedata.normalize("NFKC", text).replace("，", ",")
    return Counter(match.group(0).replace(",", "") for match in NUMBER.finditer(normalized))


def numeric_scores(reference: str, candidate: str) -> tuple[float, float, float]:
    expected = _numbers(reference)
    actual = _numbers(candidate)
    matched = sum((expected & actual).values())
    expected_count = sum(expected.values())
    actual_count = sum(actual.values())
    precision = matched / actual_count if actual_count else float(expected_count == 0)
    recall = matched / expected_count if expected_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def page_text(document: ParsedDocument, page_number: int) -> str:
    page = document.pages[page_number - 1]
    elements = sorted(
        (element for element in page.elements if element.element_type == "text"),
        key=lambda element: element.reading_order,
    )
    return "\n".join(element.text for element in elements if element.text.strip())


def _aggregate(scores: list[PdfPageExtractionScore]) -> dict:
    if not scores:
        return {"page_count": 0}
    ocr_cases = [score for score in scores if score.expected_route in {"partial_ocr", "full_ocr"}]
    return {
        "page_count": len(scores),
        "mean_character_error_rate": mean(score.character_error_rate for score in scores),
        "mean_text_similarity": mean(score.text_similarity for score in scores),
        "mean_numeric_precision": mean(score.numeric_precision for score in scores),
        "mean_numeric_recall": mean(score.numeric_recall for score in scores),
        "mean_numeric_f1": mean(score.numeric_f1 for score in scores),
        "route_accuracy": mean(float(score.route_correct) for score in scores),
        "ocr_required_page_count": len(ocr_cases),
        "ocr_success_rate": (
            mean(float(score.ocr_succeeded) for score in ocr_cases) if ocr_cases else None
        ),
    }


def _aggregate_table_structure(items: list[dict]) -> dict:
    if not items:
        return {"fact_count": 0}
    boolean_fields = (
        "section_found",
        "section_active",
        "section_carryover_correct",
        "row_found",
        "value_found",
        "row_value_same_row",
        "column_header_found",
        "column_aligned",
        "recoverable",
    )
    carryover_items = [
        item for item in items if item["requires_previous_page_context"]
    ]
    return {
        "fact_count": len(items),
        **{
            f"{field}_rate": mean(float(item[field]) for item in items)
            for field in boolean_fields
        },
        "previous_page_context_required_count": len(carryover_items),
        "previous_page_context_success_rate": (
            mean(float(item["section_carried_from_previous_page"]) for item in carryover_items)
            if carryover_items
            else None
        ),
    }


def _evaluation_groups(benchmark: PdfExtractionBenchmark) -> list[PdfEvaluationGroup]:
    if benchmark.evaluation_groups:
        return benchmark.evaluation_groups
    return [
        PdfEvaluationGroup(
            group_id=page.case_id,
            page_numbers=[page.page_number],
            challenge_type=page.challenge_type,
        )
        for page in benchmark.pages
    ]


def _evaluate_table_structure(
    benchmark: PdfExtractionBenchmark, document: ParsedDocument
) -> dict:
    items: list[dict] = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for group in _evaluation_groups(benchmark):
        pages = [document.pages[number - 1] for number in group.page_numbers]
        for question in benchmark.table_questions:
            score = {
                "group_id": group.group_id,
                "page_numbers": group.page_numbers,
                "challenge_type": group.challenge_type,
                "requires_previous_page_context": (
                    question.requires_previous_page_context
                ),
                **score_table_fact_pages(pages, question),
            }
            items.append(score)
            groups[group.challenge_type].append(score)
    return {
        "annotation_statuses": sorted(
            {question.annotation_status for question in benchmark.table_questions}
        ),
        "overall": _aggregate_table_structure(items),
        "by_challenge": {
            name: _aggregate_table_structure(values)
            for name, values in sorted(groups.items())
        },
        "items": items,
    }


def _labels_equal(expected: str, actual: str) -> bool:
    return normalize_table_label(expected) == normalize_table_label(actual)


def _aggregate_table_answers(items: list[dict], *, attempted_only: bool = False) -> dict:
    selected = [item for item in items if item["remote_attempted"]] if attempted_only else items
    if not selected:
        return {"question_count": 0}
    return {
        "question_count": len(selected),
        "answered_rate": mean(float(item["answered"]) for item in selected),
        "value_accuracy": mean(float(item["value_exact"]) for item in selected),
        "strict_cell_accuracy": mean(
            float(item["strict_cell_exact"]) for item in selected
        ),
    }


def _evaluate_table_interpretation(
    benchmark: PdfExtractionBenchmark,
    document: ParsedDocument,
    interpreter: TableInterpreter,
    *,
    min_evidence_characters: int,
) -> dict:
    items: list[dict] = []
    group_runs: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0
    evaluation_groups = _evaluation_groups(benchmark)
    for group in evaluation_groups:
        pages = [document.pages[number - 1] for number in group.page_numbers]
        evidence = serialize_layout_pages(pages)
        compact_characters = len(normalize_extraction_text(evidence))
        remote_attempted = compact_characters >= min_evidence_characters
        answers_by_id = {}
        group_run = {
            "group_id": group.group_id,
            "page_numbers": group.page_numbers,
            "evidence_characters": compact_characters,
            "remote_attempted": remote_attempted,
        }
        if not remote_attempted:
            group_run["status"] = "skipped_insufficient_evidence"
        else:
            try:
                batch = interpreter.interpret_page(benchmark.table_questions, evidence)
                answers_by_id = {answer.question_id: answer for answer in batch.answers}
                total_input_tokens += batch.input_tokens or 0
                total_output_tokens += batch.output_tokens or 0
                group_run.update(
                    {
                        "status": "completed",
                        "elapsed_ms": batch.elapsed_ms,
                        "prompt_sha256": batch.prompt_sha256,
                        "input_tokens": batch.input_tokens,
                        "output_tokens": batch.output_tokens,
                    }
                )
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                group_run.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        group_runs.append(group_run)

        for question in benchmark.table_questions:
            answer = answers_by_id.get(question.question_id)
            answered = answer is not None and answer.status == "answered"
            value_exact = answered and table_values_equal(
                question.expected_value, answer.value
            )
            unit_exact = answered and _labels_equal(question.expected_unit, answer.unit)
            row_exact = answered and _labels_equal(question.row_label, answer.row_label)
            column_exact = answered and _labels_equal(
                question.column_label, answer.column_label
            )
            section_exact = answered and _labels_equal(
                question.section_label, answer.section_label
            )
            items.append(
                {
                    "group_id": group.group_id,
                    "page_numbers": group.page_numbers,
                    "challenge_type": group.challenge_type,
                    "question_id": question.question_id,
                    "remote_attempted": remote_attempted,
                    "answered": answered,
                    "value_exact": value_exact,
                    "unit_exact": unit_exact,
                    "row_exact": row_exact,
                    "column_exact": column_exact,
                    "section_exact": section_exact,
                    "strict_cell_exact": (
                        value_exact
                        and unit_exact
                        and row_exact
                        and column_exact
                        and section_exact
                    ),
                    "predicted": answer.model_dump(mode="json") if answer else None,
                }
            )

    errors = [group for group in group_runs if group["status"] == "error"]
    completed = [group for group in group_runs if group["status"] == "completed"]
    status = "completed" if not errors else "partial_error"
    return {
        "status": status,
        "provider": interpreter.provider,
        "model": interpreter.model,
        "endpoint": interpreter.endpoint,
        "prompt_revision": interpreter.prompt_revision,
        "batching": "one_request_per_eligible_page_group",
        "min_evidence_characters": min_evidence_characters,
        "requested_group_count": len(evaluation_groups),
        "completed_group_count": len(completed),
        "error_group_count": len(errors),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "overall": _aggregate_table_answers(items),
        "attempted_only": _aggregate_table_answers(items, attempted_only=True),
        "groups": group_runs,
        "items": items,
    }


def evaluate_lane(
    benchmark: PdfExtractionBenchmark,
    pdf_path: Path,
    lane: EvaluationLane,
    *,
    ocr_backend: str = "rapidocr",
    ocr_dpi: int = 180,
    table_interpreter: TableInterpreter | None = None,
    table_interpretation_status: dict | None = None,
    min_table_evidence_characters: int = 300,
) -> dict:
    config = PdfExtractionConfig(
        mode="disabled" if lane == "native" else "auto",
        ocr_backend=ocr_backend,
        ocr_dpi=ocr_dpi,
    )
    started = time.perf_counter()
    document = parse_pdf(pdf_path, config)
    elapsed_ms = (time.perf_counter() - started) * 1000
    scores: list[PdfPageExtractionScore] = []
    for expected in benchmark.pages:
        page = document.pages[expected.page_number - 1]
        extracted = page_text(document, expected.page_number)
        cer = character_error_rate(expected.reference_text, extracted)
        precision, recall, f1 = numeric_scores(expected.reference_text, extracted)
        scores.append(
            PdfPageExtractionScore(
                case_id=expected.case_id,
                page_number=expected.page_number,
                challenge_type=expected.challenge_type,
                expected_route=expected.expected_route,
                actual_route=page.extraction_route,
                route_correct=page.extraction_route == expected.expected_route,
                reference_characters=len(normalize_extraction_text(expected.reference_text)),
                extracted_characters=len(normalize_extraction_text(extracted)),
                character_error_rate=cer,
                text_similarity=max(0.0, 1.0 - min(1.0, cer)),
                numeric_precision=precision,
                numeric_recall=recall,
                numeric_f1=f1,
                ocr_attempted=page.ocr_attempted,
                ocr_succeeded=page.ocr_succeeded,
                native_element_count=sum(
                    element.extraction_source == "native"
                    and element.element_type == "text"
                    for element in page.elements
                ),
                ocr_element_count=sum(
                    element.extraction_source == "ocr" for element in page.elements
                ),
            )
        )

    groups: dict[str, list[PdfPageExtractionScore]] = defaultdict(list)
    for score in scores:
        groups[score.challenge_type].append(score)
    table_structure = _evaluate_table_structure(benchmark, document)
    if table_interpreter is not None and benchmark.table_questions:
        table_interpretation = _evaluate_table_interpretation(
            benchmark,
            document,
            table_interpreter,
            min_evidence_characters=min_table_evidence_characters,
        )
    elif table_interpretation_status is not None:
        table_interpretation = table_interpretation_status
    else:
        table_interpretation = {"status": "not_requested"}
    return {
        "lane": lane,
        "config": config.model_dump(mode="json"),
        "elapsed_ms": elapsed_ms,
        "overall": _aggregate(scores),
        "by_challenge": {name: _aggregate(values) for name, values in sorted(groups.items())},
        "pages": [score.model_dump(mode="json") for score in scores],
        "table_structure": table_structure,
        "table_interpretation": table_interpretation,
    }


def run_pdf_extraction_benchmark(
    dataset_path: Path,
    *,
    lanes: tuple[EvaluationLane, ...] = ("native", "hybrid"),
    ocr_backend: str = "rapidocr",
    ocr_dpi: int = 180,
    table_interpreter: TableInterpreter | None = None,
    table_interpretation_requested: bool = False,
    min_table_evidence_characters: int = 300,
) -> dict:
    benchmark = PdfExtractionBenchmark.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )
    pdf_path = (dataset_path.parent / benchmark.pdf_path).resolve(strict=True)
    actual_hash = file_sha256(pdf_path)
    if actual_hash != benchmark.pdf_sha256:
        raise ValueError(
            f"Benchmark PDF hash mismatch: expected {benchmark.pdf_sha256}, got {actual_hash}"
        )
    table_interpretation_status = None
    if table_interpretation_requested and table_interpreter is None:
        table_interpretation_status = {
            "status": "not_run",
            "reason": "missing_provider_api_key",
        }
    lane_reports = {
        lane: evaluate_lane(
            benchmark,
            pdf_path,
            lane,
            ocr_backend=ocr_backend,
            ocr_dpi=ocr_dpi,
            table_interpreter=table_interpreter,
            table_interpretation_status=table_interpretation_status,
            min_table_evidence_characters=min_table_evidence_characters,
        )
        for lane in lanes
    }
    comparison = None
    if "native" in lane_reports and "hybrid" in lane_reports:
        native = lane_reports["native"]["overall"]
        hybrid = lane_reports["hybrid"]["overall"]
        comparison = {
            "text_similarity_delta": (
                hybrid["mean_text_similarity"] - native["mean_text_similarity"]
            ),
            "numeric_recall_delta": (
                hybrid["mean_numeric_recall"] - native["mean_numeric_recall"]
            ),
            "character_error_rate_delta": (
                hybrid["mean_character_error_rate"]
                - native["mean_character_error_rate"]
            ),
            "elapsed_ms_delta": (
                lane_reports["hybrid"]["elapsed_ms"]
                - lane_reports["native"]["elapsed_ms"]
            ),
        }
        native_table = lane_reports["native"]["table_structure"]["overall"]
        hybrid_table = lane_reports["hybrid"]["table_structure"]["overall"]
        comparison["table_recoverability_delta"] = (
            hybrid_table.get("recoverable_rate", 0)
            - native_table.get("recoverable_rate", 0)
        )
        native_interpretation = lane_reports["native"]["table_interpretation"]
        hybrid_interpretation = lane_reports["hybrid"]["table_interpretation"]
        if (
            "overall" in native_interpretation
            and "overall" in hybrid_interpretation
        ):
            comparison["deepseek_strict_cell_accuracy_delta"] = (
                hybrid_interpretation["overall"]["strict_cell_accuracy"]
                - native_interpretation["overall"]["strict_cell_accuracy"]
            )
    return {
        "schema_version": "3",
        "dataset_id": benchmark.dataset_id,
        "dataset_path": dataset_path.resolve().as_posix(),
        "dataset_sha256": hashlib.sha256(
            dataset_path.read_bytes()
        ).hexdigest(),
        "pdf_sha256": actual_hash,
        "lanes": lane_reports,
        "comparison": comparison,
    }


def write_evaluation_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
