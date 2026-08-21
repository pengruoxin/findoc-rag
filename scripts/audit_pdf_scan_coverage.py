"""Audit a PDF corpus for genuine image-dominant, low-text table pages."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pymupdf

TABLE_TERMS = ("项目", "合计", "期末余额", "期初余额", "本期", "上期")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-source-manifest.json"),
    )
    parser.add_argument(
        "--visual-review",
        type=Path,
        default=Path("data/evaluation/pdf-scan-visual-review-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pdf-extraction/pdf-scan-coverage-audit-v1.json"),
    )
    parser.add_argument("--max-native-characters", type=int, default=250)
    parser.add_argument("--min-image-coverage", type=float, default=0.45)
    return parser.parse_args()


def _image_coverage(page: pymupdf.Page, raw: dict) -> tuple[int, float, float]:
    page_area = page.rect.width * page.rect.height
    ratios = []
    for block in raw.get("blocks", []):
        if block.get("type") != 1:
            continue
        x0, y0, x1, y1 = block["bbox"]
        ratios.append(max(0, x1 - x0) * max(0, y1 - y0) / page_area)
    return len(ratios), sum(ratios), max(ratios, default=0.0)


def run_audit(
    manifest_path: Path,
    visual_review_path: Path,
    *,
    max_native_characters: int,
    min_image_coverage: float,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    visual_review = json.loads(visual_review_path.read_text(encoding="utf-8"))
    decisions = {
        (item["document_key"], item["page_number"]): item
        for item in visual_review["items"]
    }
    candidates: list[dict] = []
    page_count = 0
    for document in manifest["documents"]:
        path = Path(document["local_file"]).resolve(strict=True)
        with pymupdf.open(path) as pdf:
            page_count += pdf.page_count
            for page_index, page in enumerate(pdf, start=1):
                text = page.get_text("text", sort=True)
                compact_characters = len("".join(text.split()))
                raw = page.get_text("dict", sort=True)
                image_count, coverage_sum, coverage_max = _image_coverage(page, raw)
                if (
                    compact_characters >= max_native_characters
                    or max(coverage_sum, coverage_max) < min_image_coverage
                ):
                    continue
                key = (document["document_key"], page_index)
                decision = decisions.get(key)
                candidates.append(
                    {
                        "document_key": document["document_key"],
                        "page_number": page_index,
                        "native_characters": compact_characters,
                        "numeric_token_count": len(re.findall(r"\d[\d,.%]*", text)),
                        "table_term_count": sum(text.count(term) for term in TABLE_TERMS),
                        "image_count": image_count,
                        "image_coverage_sum": round(coverage_sum, 4),
                        "image_coverage_max": round(coverage_max, 4),
                        "visual_review": decision,
                    }
                )

    unreviewed = [item for item in candidates if item["visual_review"] is None]
    eligible = [
        item
        for item in candidates
        if item["visual_review"]
        and item["visual_review"]["eligible_scanned_table"]
    ]
    return {
        "schema_version": "1",
        "source_manifest": manifest_path.resolve().as_posix(),
        "visual_review_source": visual_review_path.resolve().as_posix(),
        "visual_review_status": visual_review["review_status"],
        "thresholds": {
            "max_native_characters": max_native_characters,
            "min_image_coverage": min_image_coverage,
        },
        "document_count": len(manifest["documents"]),
        "page_count": page_count,
        "image_dominant_low_text_candidate_count": len(candidates),
        "visually_reviewed_candidate_count": len(candidates) - len(unreviewed),
        "unreviewed_candidate_count": len(unreviewed),
        "eligible_genuine_scanned_table_count": len(eligible),
        "conclusion": (
            "no_eligible_genuine_scanned_table_in_current_corpus"
            if not eligible and not unreviewed
            else "candidate_follow_up_required"
        ),
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    report = run_audit(
        args.source_manifest,
        args.visual_review,
        max_native_characters=args.max_native_characters,
        min_image_coverage=args.min_image_coverage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"documents={report['document_count']}")
    print(f"pages={report['page_count']}")
    print(f"candidates={report['image_dominant_low_text_candidate_count']}")
    print(f"eligible_scanned_tables={report['eligible_genuine_scanned_table_count']}")
    print(f"conclusion={report['conclusion']}")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
