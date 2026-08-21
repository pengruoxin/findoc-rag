"""Human-review accounting for provisional PDF table gold labels."""

from __future__ import annotations

from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from findoc_rag.pdf_evaluation import PdfExtractionBenchmark
from findoc_rag.pdf_table_interpretation import (
    normalize_table_label,
    table_values_equal,
)


class PdfReviewItem(BaseModel):
    question_id: str
    question: str
    source_page: int = Field(ge=1)
    challenge_page: int = Field(ge=1)
    row_label: str | None = None
    column_label: str | None = None
    value: str | None = None
    unit: str | None = None
    section_label: str | None = None
    decision: Literal["accept", "needs_adjudication"] | None = None
    notes: str | None = None

    @property
    def completed(self) -> bool:
        return self.decision is not None and all(
            value is not None
            for value in (
                self.row_label,
                self.column_label,
                self.value,
                self.unit,
                self.section_label,
            )
        )


class PdfReviewPacket(BaseModel):
    schema_version: Literal["1"] = "1"
    dataset_id: str
    purpose: Literal[
        "blind_second_reviewer_annotation", "second_reviewer_verification"
    ]
    source_document: str
    source_pages: list[int]
    instructions: list[str]
    challenge_pdf: str
    reviewer_id: str | None = None
    reviewer_independence_attestation: bool | None = None
    review_method: Literal["blind_reannotation", "candidate_verification"] | None = None
    reviewed_at: str | None = None
    items: list[PdfReviewItem]

    @model_validator(mode="after")
    def validate_question_ids(self) -> PdfReviewPacket:
        question_ids = [item.question_id for item in self.items]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Review packet question IDs must be unique")
        return self


def _labels_equal(expected: str, actual: str | None) -> bool:
    return actual is not None and normalize_table_label(expected) == normalize_table_label(
        actual
    )


def evaluate_pdf_review(
    benchmark: PdfExtractionBenchmark, packet: PdfReviewPacket
) -> dict:
    """Compare a completed human-review packet without mutating the benchmark."""

    expected = {question.question_id: question for question in benchmark.table_questions}
    actual = {item.question_id: item for item in packet.items}
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(
            f"Review packet question mismatch: missing={missing}, unexpected={unexpected}"
        )

    completed_items = [item for item in packet.items if item.completed]
    if len(completed_items) != len(packet.items) or not packet.reviewer_id:
        return {
            "status": "awaiting_second_reviewer",
            "dataset_id": benchmark.dataset_id,
            "question_count": len(packet.items),
            "completed_count": len(completed_items),
            "reviewer_id_present": bool(packet.reviewer_id),
            "independence_attested": packet.reviewer_independence_attestation is True,
            "review_method": packet.review_method,
            "human_verification_complete": False,
            "blind_reannotation_complete": False,
            "independent_gold_ready": False,
        }

    results: list[dict] = []
    for question_id, question in expected.items():
        item = actual[question_id]
        value_exact = item.value is not None and table_values_equal(
            question.expected_value, item.value
        )
        row_exact = _labels_equal(question.row_label, item.row_label)
        column_exact = _labels_equal(question.column_label, item.column_label)
        unit_exact = _labels_equal(question.expected_unit, item.unit)
        section_exact = _labels_equal(question.section_label, item.section_label)
        hard_agreement = all(
            (value_exact, row_exact, column_exact, unit_exact, section_exact)
        )
        results.append(
            {
                "question_id": question_id,
                "decision": item.decision,
                "value_exact": value_exact,
                "row_exact": row_exact,
                "column_exact": column_exact,
                "unit_exact": unit_exact,
                "section_exact": section_exact,
                "hard_agreement": hard_agreement,
                "needs_adjudication": (
                    item.decision == "needs_adjudication" or not hard_agreement
                ),
            }
        )

    adjudication_ids = [
        item["question_id"] for item in results if item["needs_adjudication"]
    ]
    independence_attested = packet.reviewer_independence_attestation is True
    if not independence_attested:
        status = "missing_independence_attestation"
    elif packet.review_method is None:
        status = "missing_review_method"
    elif adjudication_ids:
        status = "needs_adjudication"
    else:
        status = "complete"
    field_names = ("value_exact", "row_exact", "column_exact", "unit_exact", "section_exact")
    return {
        "status": status,
        "dataset_id": benchmark.dataset_id,
        "question_count": len(results),
        "completed_count": len(results),
        "reviewer_id": packet.reviewer_id,
        "reviewed_at": packet.reviewed_at,
        "independence_attested": independence_attested,
        "review_method": packet.review_method,
        "agreement_interpretation": (
            "independent_blind_reannotation"
            if packet.review_method == "blind_reannotation"
            else "candidate_verification_not_blind_reannotation"
        ),
        "hard_agreement_rate": mean(float(item["hard_agreement"]) for item in results),
        "field_agreement": {
            name: mean(float(item[name]) for item in results) for name in field_names
        },
        "adjudication_question_ids": adjudication_ids,
        "human_verification_complete": status == "complete",
        "blind_reannotation_complete": (
            status == "complete" and packet.review_method == "blind_reannotation"
        ),
        "independent_gold_ready": status == "complete",
        "items": results,
    }
