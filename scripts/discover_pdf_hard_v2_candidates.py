"""Discover non-frozen complex-table pages for human annotation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pymupdf

from findoc_rag.pdf_candidate_discovery import build_page_signals, classify_page
from findoc_rag.pdf_evaluation import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-source-manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pdf-extraction/pdf-hard-v2-candidates.json"),
    )
    parser.add_argument(
        "--allowed-splits", nargs="+", default=["calibration", "dev"]
    )
    parser.add_argument("--top-per-kind", type=int, default=12)
    parser.add_argument("--render-top-per-kind", type=int, default=2)
    parser.add_argument(
        "--render-dir", type=Path, default=Path("tmp/pdfs/pdf-hard-v2-candidates")
    )
    return parser.parse_args()


def _image_profile(page: pymupdf.Page, raw: dict) -> tuple[int, float]:
    page_area = page.rect.width * page.rect.height
    ratios = []
    for block in raw.get("blocks", []):
        if block.get("type") != 1 or page_area <= 0:
            continue
        x0, y0, x1, y1 = block["bbox"]
        ratios.append(max(0.0, x1 - x0) * max(0.0, y1 - y0) / page_area)
    return len(ratios), max(ratios, default=0.0)


def discover_candidates(
    manifest_path: Path, *, allowed_splits: set[str], top_per_kind: int
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_documents = [
        document for document in manifest["documents"] if document["split"] in allowed_splits
    ]
    excluded_documents = [
        document for document in manifest["documents"] if document["split"] not in allowed_splits
    ]
    by_kind: dict[str, list[dict]] = defaultdict(list)
    page_count = 0
    for document in selected_documents:
        path = Path(document["local_file"]).resolve(strict=True)
        if file_sha256(path) != document["sha256"]:
            raise ValueError(f"Source PDF SHA-256 mismatch: {path}")
        with pymupdf.open(path) as pdf:
            page_count += pdf.page_count
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text", sort=True)
                raw = page.get_text("dict", sort=True)
                span_texts = [
                    span.get("text", "")
                    for block in raw.get("blocks", [])
                    if block.get("type") == 0
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ]
                span_items = [
                    (
                        span.get("text", ""),
                        (span["bbox"][0] + span["bbox"][2]) / 2,
                        (span["bbox"][1] + span["bbox"][3]) / 2,
                    )
                    for block in raw.get("blocks", [])
                    if block.get("type") == 0
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ]
                image_count, image_coverage_max = _image_profile(page, raw)
                signals = build_page_signals(
                    page_number=page_number,
                    text=text,
                    span_texts=span_texts,
                    span_items=span_items,
                    drawing_count=len(page.get_drawings()),
                    image_count=image_count,
                    image_coverage_max=image_coverage_max,
                    rotation=page.rotation,
                )
                for candidate in classify_page(signals):
                    by_kind[candidate.kind].append(
                        {
                            "candidate_id": (
                                f"{document['document_key']}:p{page_number}:{candidate.kind}"
                            ),
                            "document_key": document["document_key"],
                            "company_name": document["company_name"],
                            "report_year": document["report_year"],
                            "source_split": document["split"],
                            "local_file": document["local_file"],
                            "source_sha256": document["sha256"],
                            "page_number": page_number,
                            "kind": candidate.kind,
                            "score": round(candidate.score, 4),
                            "reason_codes": candidate.reason_codes,
                            "signals": signals.model_dump(mode="json"),
                        }
                    )
    selected: dict[str, list[dict]] = {}
    for kind, candidates in sorted(by_kind.items()):
        candidates.sort(
            key=lambda item: (-item["score"], item["document_key"], item["page_number"])
        )
        selected[kind] = candidates[:top_per_kind]
    return {
        "schema_version": "1",
        "purpose": "candidate_discovery_only_not_gold",
        "source_manifest": manifest_path.resolve().as_posix(),
        "source_manifest_sha256": file_sha256(manifest_path),
        "allowed_splits": sorted(allowed_splits),
        "excluded_splits": sorted({document["split"] for document in excluded_documents}),
        "frozen_documents_opened": False,
        "document_count": len(selected_documents),
        "page_count": page_count,
        "candidate_counts": {kind: len(values) for kind, values in selected.items()},
        "candidates": selected,
    }


def render_candidates(report: dict, render_dir: Path, top_per_kind: int) -> list[str]:
    render_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []
    seen: set[tuple[str, int]] = set()
    for candidates in report["candidates"].values():
        for candidate in candidates[:top_per_kind]:
            key = (candidate["local_file"], candidate["page_number"])
            if key in seen:
                continue
            seen.add(key)
            safe_key = candidate["document_key"].replace(":", "-")
            output = render_dir / f"{safe_key}-p{candidate['page_number']}.png"
            with pymupdf.open(Path(candidate["local_file"]).resolve(strict=True)) as pdf:
                page = pdf[candidate["page_number"] - 1]
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                pixmap.save(output)
            rendered.append(output.resolve().as_posix())
    return rendered


def main() -> None:
    args = parse_args()
    report = discover_candidates(
        args.source_manifest.resolve(strict=True),
        allowed_splits=set(args.allowed_splits),
        top_per_kind=args.top_per_kind,
    )
    report["rendered_pages"] = render_candidates(
        report, args.render_dir, args.render_top_per_kind
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"documents={report['document_count']}")
    print(f"pages={report['page_count']}")
    print(f"candidate_counts={report['candidate_counts']}")
    print(f"rendered_pages={len(report['rendered_pages'])}")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
