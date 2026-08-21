"""Build a real cross-page table benchmark plus a human-review record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf

from findoc_rag.pdf_evaluation import (
    PdfBenchmarkPage,
    PdfEvaluationGroup,
    PdfExtractionBenchmark,
    file_sha256,
)
from findoc_rag.pdf_table_interpretation import TableQuestion

SOURCE_DOCUMENT = "cninfo:600690:annual:2024"
SOURCE_PAGES = (184, 185, 186)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-source-manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation/pdf-cross-page-v1"),
    )
    parser.add_argument("--render-dpi", type=int, default=180)
    return parser.parse_args()


def resolve_source(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        document
        for document in manifest["documents"]
        if document["document_key"] == SOURCE_DOCUMENT
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest entry for {SOURCE_DOCUMENT}")
    return Path(matches[0]["local_file"]).resolve(strict=True)


def table_questions() -> list[TableQuestion]:
    questions = [
        TableQuestion(
            question_id="lease-original-closing-production-equipment",
            question="使用权资产表中，账面原值期末余额里的生产设备金额是多少？",
            expected_value="424,335,480.27",
            expected_unit="元",
            row_label="5.期末余额",
            column_label="生产设备",
            section_label="一、账面原值",
            target_page_offset=0,
        ),
        TableQuestion(
            question_id="lease-depreciation-opening-transport-equipment",
            question="使用权资产表中，累计折旧期初余额里的运输设备金额是多少？",
            expected_value="133,592,322.52",
            expected_unit="元",
            row_label="1.期初余额",
            column_label="运输设备",
            section_label="二、累计折旧",
            target_page_offset=1,
        ),
        TableQuestion(
            question_id="lease-depreciation-closing-building",
            question="使用权资产表中，累计折旧期末余额里的房屋及建筑物金额是多少？",
            expected_value="2,728,106,200.79",
            expected_unit="元",
            row_label="5.期末余额",
            column_label="房屋及建筑物",
            section_label="二、累计折旧",
            target_page_offset=1,
        ),
        TableQuestion(
            question_id="lease-carrying-closing-transport-equipment",
            question="使用权资产表中，期末账面价值里的运输设备金额是多少？",
            expected_value="146,521,021.87",
            expected_unit="元",
            row_label="1.期末账面价值",
            column_label="运输设备",
            section_label="四、账面价值",
            target_page_offset=1,
        ),
        TableQuestion(
            question_id="lease-original-closing-total",
            question="使用权资产表中，账面原值期末余额的合计是多少？",
            expected_value="9,158,716,606.73",
            expected_unit="元",
            row_label="5.期末余额",
            column_label="合计",
            section_label="一、账面原值",
            target_page_offset=1,
        ),
        TableQuestion(
            question_id="lease-depreciation-closing-total",
            question="使用权资产表中，累计折旧期末余额的合计是多少？",
            expected_value="3,316,847,042.37",
            expected_unit="元",
            row_label="5.期末余额",
            column_label="合计",
            section_label="二、累计折旧",
            target_page_offset=2,
            requires_previous_page_context=True,
        ),
        TableQuestion(
            question_id="lease-carrying-closing-total",
            question="使用权资产表中，期末账面价值的合计是多少？",
            expected_value="5,841,869,564.36",
            expected_unit="元",
            row_label="1.期末账面价值",
            column_label="合计",
            section_label="四、账面价值",
            target_page_offset=2,
        ),
    ]
    return [
        question.model_copy(update={"annotation_status": "human_verified"})
        for question in questions
    ]


def _review_packet(questions: list[TableQuestion]) -> dict:
    return {
        "schema_version": "1",
        "dataset_id": "pdf-cross-page-v1",
        "purpose": "second_reviewer_verification",
        "source_document": SOURCE_DOCUMENT,
        "source_pages": list(SOURCE_PAGES),
        "instructions": [
            "工作区用户复核 benchmark.json 中的候选标注。",
            "本记录属于 candidate_verification，不等同于独立盲标。",
            "工作区用户于 2026-08-20 确认全部 7 条候选标注正确。",
        ],
        "challenge_pdf": "challenge-pages.pdf",
        "reviewer_id": "workspace_user",
        "reviewer_independence_attestation": True,
        "review_method": "candidate_verification",
        "reviewed_at": "2026-08-20",
        "items": [
            {
                "question_id": question.question_id,
                "question": question.question,
                "source_page": SOURCE_PAGES[question.target_page_offset],
                "challenge_page": question.target_page_offset + 1,
                "row_label": question.row_label,
                "column_label": question.column_label,
                "value": question.expected_value,
                "unit": question.expected_unit,
                "section_label": question.section_label,
                "decision": "accept",
                "notes": "工作区用户确认候选标注复核无误。",
            }
            for question in questions
        ],
    }


def build_benchmark(source_path: Path, output_dir: Path, *, render_dpi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    challenge_path = output_dir / "challenge-pages.pdf"
    dataset_path = output_dir / "benchmark.json"
    review_path = output_dir / "blind-review-packet.json"
    source_indexes = [number - 1 for number in SOURCE_PAGES]
    references: list[str] = []

    with pymupdf.open(source_path) as source:
        if max(source_indexes) >= source.page_count:
            raise ValueError("Configured source page is outside the source PDF")
        challenge = pymupdf.open()
        for source_index in source_indexes:
            source_page = source[source_index]
            references.append(source_page.get_text("text", sort=True).strip())
            page = challenge.new_page(
                width=source_page.rect.width, height=source_page.rect.height
            )
            page.show_pdf_page(page.rect, source, source_index)
        for source_index in source_indexes:
            source_page = source[source_index]
            pixmap = source_page.get_pixmap(dpi=render_dpi, alpha=False)
            page = challenge.new_page(
                width=source_page.rect.width, height=source_page.rect.height
            )
            page.insert_image(page.rect, stream=pixmap.tobytes("png"))
        challenge.set_metadata(
            {
                "title": "FinDocRAG real cross-page table challenge v1",
                "subject": (
                    f"Exact and rasterized derivatives of {SOURCE_DOCUMENT} "
                    f"pages {SOURCE_PAGES[0]}-{SOURCE_PAGES[-1]}"
                ),
            }
        )
        challenge.save(challenge_path, garbage=4, deflate=True, no_new_id=True)
        challenge.close()

    questions = table_questions()
    pages = []
    for offset, (source_page, reference) in enumerate(
        zip(SOURCE_PAGES, references, strict=True), start=1
    ):
        pages.append(
            PdfBenchmarkPage(
                case_id=f"native-cross-page-{offset}",
                page_number=offset,
                challenge_type="native_text",
                expected_route="native",
                reference_text=reference,
                source_document=SOURCE_DOCUMENT,
                source_page_number=source_page,
            )
        )
        pages.append(
            PdfBenchmarkPage(
                case_id=f"rasterized-cross-page-{offset}",
                page_number=offset + len(SOURCE_PAGES),
                challenge_type="scanned_page",
                expected_route="full_ocr",
                reference_text=reference,
                source_document=SOURCE_DOCUMENT,
                source_page_number=source_page,
            )
        )
    pages.sort(key=lambda page: page.page_number)

    benchmark = PdfExtractionBenchmark(
        schema_version="3",
        dataset_id="pdf-cross-page-v1",
        pdf_path=challenge_path.name,
        pdf_sha256=file_sha256(challenge_path),
        reference_policy="independent_native_source_page",
        pages=pages,
        table_questions=questions,
        evaluation_groups=[
            PdfEvaluationGroup(
                group_id="native-cross-page",
                page_numbers=[1, 2, 3],
                challenge_type="native_cross_page",
            ),
            PdfEvaluationGroup(
                group_id="rasterized-cross-page",
                page_numbers=[4, 5, 6],
                challenge_type="scanned_cross_page",
            ),
        ],
    )
    dataset_path.write_text(benchmark.model_dump_json(indent=2) + "\n", encoding="utf-8")
    review_path.write_text(
        json.dumps(_review_packet(questions), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset_path


def main() -> None:
    args = parse_args()
    source_path = resolve_source(args.source_manifest)
    dataset_path = build_benchmark(
        source_path,
        args.output_dir,
        render_dpi=args.render_dpi,
    )
    benchmark = PdfExtractionBenchmark.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )
    print(f"dataset={dataset_path.resolve()}")
    print(f"pdf_sha256={benchmark.pdf_sha256}")
    print(f"pages={len(benchmark.pages)}")
    print(f"groups={len(benchmark.evaluation_groups)}")
    print(f"table_questions={len(benchmark.table_questions)}")


if __name__ == "__main__":
    main()
