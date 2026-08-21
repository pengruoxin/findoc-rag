"""Build a reproducible native/scanned/mixed PDF extraction challenge set."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pymupdf

from findoc_rag.pdf_evaluation import (
    PdfBenchmarkPage,
    PdfExtractionBenchmark,
    file_sha256,
)
from findoc_rag.pdf_table_interpretation import TableQuestion

FIXED_ASSET_DOCUMENT = "cninfo:600519:annual:2024"
FIXED_ASSET_PAGE = 98


def fixed_asset_questions(
    source_document: str, source_page_number: int
) -> list[TableQuestion]:
    """Return provisional hard labels only for the audited source page."""

    if (
        source_document != FIXED_ASSET_DOCUMENT
        or source_page_number != FIXED_ASSET_PAGE
    ):
        return []
    return [
        TableQuestion(
            question_id="fixed-assets-closing-balance",
            question="项目列示中，固定资产的期末余额是多少？",
            expected_value="21,871,446,747.14",
            expected_unit="元",
            row_label="固定资产",
            column_label="期末余额",
            section_label="项目列示",
        ),
        TableQuestion(
            question_id="fixed-assets-opening-balance",
            question="项目列示中，固定资产的期初余额是多少？",
            expected_value="19,909,280,655.97",
            expected_unit="元",
            row_label="固定资产",
            column_label="期初余额",
            section_label="项目列示",
        ),
        TableQuestion(
            question_id="gross-value-opening-house",
            question="固定资产情况表中，账面原值期初余额里的房屋及建筑物金额是多少？",
            expected_value="29,020,638,616.45",
            expected_unit="元",
            row_label="1.期初余额",
            column_label="房屋及建筑物",
            section_label="一、账面原值",
        ),
        TableQuestion(
            question_id="gross-value-current-increase-total",
            question="固定资产情况表中，账面原值本期增加金额的合计是多少？",
            expected_value="3,696,992,365.12",
            expected_unit="元",
            row_label="2.本期增加金额",
            column_label="合计",
            section_label="一、账面原值",
        ),
        TableQuestion(
            question_id="depreciation-closing-machinery",
            question="固定资产情况表中，累计折旧期末余额里的机器设备金额是多少？",
            expected_value="1,802,069,462.69",
            expected_unit="元",
            row_label="4.期末余额",
            column_label="机器设备",
            section_label="二、累计折旧",
        ),
        TableQuestion(
            question_id="carrying-value-closing-total",
            question="固定资产情况表中，期末账面价值的合计是多少？",
            expected_value="21,871,446,747.14",
            expected_unit="元",
            row_label="1.期末账面价值",
            column_label="合计",
            section_label="四、账面价值",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pdf", type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-source-manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/evaluation/pdf-extraction-v1")
    )
    parser.add_argument("--source-page", type=int, help="One-based source page override")
    parser.add_argument("--render-dpi", type=int, default=180)
    return parser.parse_args()


def resolve_source(args: argparse.Namespace) -> tuple[Path, str]:
    if args.source_pdf is not None:
        return args.source_pdf.resolve(strict=True), args.source_pdf.name
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    document = manifest["documents"][0]
    path = Path(document["local_file"]).resolve(strict=True)
    return path, document["document_key"]


def select_source_page(document: pymupdf.Document, override: int | None) -> int:
    if override is not None:
        if override < 1 or override > document.page_count:
            raise ValueError(f"Source page {override} is outside 1-{document.page_count}")
        return override - 1
    candidates: list[tuple[int, int, int]] = []
    for index, page in enumerate(document):
        text = page.get_text("text", sort=True).strip()
        compact_count = len("".join(text.split()))
        numeric_count = len(re.findall(r"\d[\d,.%]*", text))
        if 500 <= compact_count <= 1800 and numeric_count >= 8:
            candidates.append((numeric_count, -compact_count, index))
    if not candidates:
        raise ValueError("No medium-size, number-bearing source page was found")
    return max(candidates)[2]


def build_benchmark(
    source_path: Path,
    source_document: str,
    output_dir: Path,
    *,
    source_page_override: int | None = None,
    render_dpi: int = 180,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "challenge-pages.pdf"
    dataset_path = output_dir / "benchmark.json"
    marker = "MIXED PAGE NATIVE MARKER"

    with pymupdf.open(source_path) as source:
        source_index = select_source_page(source, source_page_override)
        source_page = source[source_index]
        reference = source_page.get_text("text", sort=True).strip()
        pixmap = source_page.get_pixmap(dpi=render_dpi, alpha=False)
        rendered = pixmap.tobytes("png")
        challenge = pymupdf.open()

        native_page = challenge.new_page(
            width=source_page.rect.width, height=source_page.rect.height
        )
        native_page.show_pdf_page(native_page.rect, source, source_index)

        scanned_page = challenge.new_page(
            width=source_page.rect.width, height=source_page.rect.height
        )
        scanned_page.insert_image(scanned_page.rect, stream=rendered)

        mixed_page = challenge.new_page(
            width=source_page.rect.width, height=source_page.rect.height
        )
        mixed_page.insert_image(mixed_page.rect, stream=rendered)
        mixed_page.insert_text((18, 18), marker, fontsize=8, overlay=True)

        challenge.set_metadata(
            {
                "title": "FinDocRAG PDF extraction challenge v1",
                "subject": f"Derived from {source_document} page {source_index + 1}",
            }
        )
        challenge.save(pdf_path, garbage=4, deflate=True, no_new_id=True)
        challenge.close()

    benchmark = PdfExtractionBenchmark(
        schema_version="2",
        dataset_id="pdf-extraction-v1",
        pdf_path=pdf_path.name,
        pdf_sha256=file_sha256(pdf_path),
        reference_policy="independent_native_source_page",
        pages=[
            PdfBenchmarkPage(
                case_id="native-text-control",
                page_number=1,
                challenge_type="native_text",
                expected_route="native",
                reference_text=reference,
                source_document=source_document,
                source_page_number=source_index + 1,
            ),
            PdfBenchmarkPage(
                case_id="rasterized-scan",
                page_number=2,
                challenge_type="scanned_page",
                expected_route="full_ocr",
                reference_text=reference,
                source_document=source_document,
                source_page_number=source_index + 1,
            ),
            PdfBenchmarkPage(
                case_id="mixed-native-and-scan",
                page_number=3,
                challenge_type="mixed_page",
                expected_route="partial_ocr",
                reference_text=f"{marker}\n{reference}",
                source_document=source_document,
                source_page_number=source_index + 1,
            ),
        ],
        table_questions=fixed_asset_questions(source_document, source_index + 1),
    )
    dataset_path.write_text(benchmark.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return dataset_path


def main() -> None:
    args = parse_args()
    source_path, source_document = resolve_source(args)
    dataset_path = build_benchmark(
        source_path,
        source_document,
        args.output_dir,
        source_page_override=args.source_page,
        render_dpi=args.render_dpi,
    )
    benchmark = PdfExtractionBenchmark.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )
    print(f"dataset={dataset_path.resolve()}")
    print(f"pdf_sha256={benchmark.pdf_sha256}")
    print(f"cases={len(benchmark.pages)}")
    print(f"table_questions={len(benchmark.table_questions)}")
    print(f"source_page={benchmark.pages[0].source_page_number}")


if __name__ == "__main__":
    main()
