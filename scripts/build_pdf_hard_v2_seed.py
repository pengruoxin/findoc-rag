"""Build a provenance-bound PDF from visually reviewed development candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf

from findoc_rag.pdf_evaluation import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("data/evaluation/pdf-hard-v2/candidate-visual-review-v1.json"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("data/evaluation/pdf-hard-v2/development-candidates.pdf"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/evaluation/pdf-hard-v2/development-candidates.json"),
    )
    return parser.parse_args()


def build_seed(review_path: Path, output_pdf: Path) -> dict:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review["review_status"] != "assistant_visual_review_provisional":
        raise ValueError("Unexpected candidate review status")
    selected = [item for item in review["items"] if item["selected"]]
    if not selected:
        raise ValueError("No visually reviewed candidates were selected")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output = pymupdf.open()
    pages: list[dict] = []
    try:
        for challenge_page_number, item in enumerate(selected, start=1):
            source_path = Path(item["local_file"]).resolve(strict=True)
            if file_sha256(source_path) != item["source_sha256"]:
                raise ValueError(f"Source PDF SHA-256 mismatch: {source_path}")
            with pymupdf.open(source_path) as source:
                source_index = item["source_page_number"] - 1
                if source_index < 0 or source_index >= source.page_count:
                    raise ValueError(f"Source page outside PDF: {item['candidate_id']}")
                output.insert_pdf(source, from_page=source_index, to_page=source_index)
            pages.append(
                {
                    "candidate_id": item["candidate_id"],
                    "challenge_page_number": challenge_page_number,
                    "document_key": item["document_key"],
                    "source_file": item["local_file"],
                    "source_sha256": item["source_sha256"],
                    "source_page_number": item["source_page_number"],
                    "assigned_split": item["assigned_split"],
                    "primary_stratum": item["primary_stratum"],
                    "additional_strata": item["additional_strata"],
                    "annotation_status": "unannotated",
                    "counts_toward_target": False,
                }
            )
        output.save(output_pdf, garbage=4, deflate=True)
    finally:
        output.close()
    with pymupdf.open(output_pdf) as check:
        if check.page_count != len(pages):
            raise ValueError("Built seed PDF page count does not match selected candidates")
    return {
        "schema_version": "1",
        "dataset_id": "pdf-hard-v2-development-candidates",
        "status": "assistant_visual_reviewed_unannotated",
        "review_source": review_path.as_posix(),
        "review_sha256": file_sha256(review_path),
        "pdf_path": output_pdf.as_posix(),
        "pdf_sha256": file_sha256(output_pdf),
        "page_count": len(pages),
        "counts_toward_target": False,
        "frozen_gold_present": False,
        "pages": pages,
    }


def main() -> None:
    args = parse_args()
    report = build_seed(args.review.resolve(strict=True), args.output_pdf)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"pages={report['page_count']}")
    print(f"pdf_sha256={report['pdf_sha256']}")
    print(f"pdf={args.output_pdf.resolve()}")
    print(f"manifest={args.output_manifest.resolve()}")


if __name__ == "__main__":
    main()
