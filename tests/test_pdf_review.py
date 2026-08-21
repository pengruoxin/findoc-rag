from findoc_rag.pdf_evaluation import (
    PdfBenchmarkPage,
    PdfExtractionBenchmark,
)
from findoc_rag.pdf_review import PdfReviewItem, PdfReviewPacket, evaluate_pdf_review
from findoc_rag.pdf_table_interpretation import TableQuestion


def _benchmark() -> PdfExtractionBenchmark:
    return PdfExtractionBenchmark(
        dataset_id="review-test",
        pdf_path="challenge.pdf",
        pdf_sha256="0" * 64,
        reference_policy="independent_native_source_page",
        pages=[
            PdfBenchmarkPage(
                case_id="native",
                page_number=1,
                challenge_type="native_text",
                expected_route="native",
                reference_text="固定资产 100",
                source_document="test",
                source_page_number=10,
            )
        ],
        table_questions=[
            TableQuestion(
                question_id="fixed-assets",
                question="固定资产期末余额是多少？",
                expected_value="100.00",
                expected_unit="元",
                row_label="固定资产",
                column_label="期末余额",
                section_label="项目列示",
            )
        ],
    )


def _packet(item: PdfReviewItem) -> PdfReviewPacket:
    return PdfReviewPacket(
        dataset_id="review-test",
        purpose="blind_second_reviewer_annotation",
        source_document="test",
        source_pages=[10],
        instructions=["blind"],
        challenge_pdf="challenge.pdf",
        reviewer_id="reviewer-2",
        reviewer_independence_attestation=True,
        review_method="blind_reannotation",
        reviewed_at="2026-08-19T00:00:00Z",
        items=[item],
    )


def test_incomplete_pdf_review_never_claims_independent_gold() -> None:
    packet = _packet(
        PdfReviewItem(
            question_id="fixed-assets",
            question="固定资产期末余额是多少？",
            source_page=10,
            challenge_page=1,
        )
    )

    report = evaluate_pdf_review(_benchmark(), packet)

    assert report["status"] == "awaiting_second_reviewer"
    assert report["independent_gold_ready"] is False


def test_completed_pdf_review_uses_hard_field_agreement() -> None:
    packet = _packet(
        PdfReviewItem(
            question_id="fixed-assets",
            question="固定资产期末余额是多少？",
            source_page=10,
            challenge_page=1,
            row_label="固定资产",
            column_label="期末余额",
            value="100",
            unit="元",
            section_label="项目列示",
            decision="accept",
        )
    )

    report = evaluate_pdf_review(_benchmark(), packet)

    assert report["status"] == "complete"
    assert report["hard_agreement_rate"] == 1.0
    assert report["human_verification_complete"] is True
    assert report["blind_reannotation_complete"] is True
    assert report["independent_gold_ready"] is True


def test_candidate_verification_is_complete_but_not_reported_as_blind() -> None:
    packet = _packet(
        PdfReviewItem(
            question_id="fixed-assets",
            question="固定资产期末余额是多少？",
            source_page=10,
            challenge_page=1,
            row_label="固定资产",
            column_label="期末余额",
            value="100",
            unit="元",
            section_label="项目列示",
            decision="accept",
        )
    ).model_copy(update={"review_method": "candidate_verification"})

    report = evaluate_pdf_review(_benchmark(), packet)

    assert report["status"] == "complete"
    assert report["human_verification_complete"] is True
    assert report["blind_reannotation_complete"] is False
    assert report["agreement_interpretation"] == (
        "candidate_verification_not_blind_reannotation"
    )
