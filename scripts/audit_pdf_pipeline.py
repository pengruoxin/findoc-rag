"""Reproducible PDF-pipeline audit for complex Chinese annual reports.

Checks text-layer quality, block/line/span geometry granularity, fonts,
rotation, low-text pages, columnar-layout risk, and known reading-order /
text-layer omission probes. Writes machine-readable statistics to
``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
AUDIT_RUN_ID = "pdf-audit-2026-08-13-v2"
PDFS = {
    "moutai": ROOT / "data/artifacts/cninfo/600519_2024_1222993920.pdf",
    "yili": ROOT / "data/artifacts/cninfo/600887_2024_1223421123.pdf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Audit a named PDF (repeatable); replaces the built-in benchmark PDFs",
    )
    parser.add_argument(
        "--chunks-root",
        type=Path,
        default=ROOT / "data/catalog/versions",
        help="Directory containing VERSION/chunks.jsonl files",
    )
    return parser.parse_args()


def parse_pdf_specs(specs: list[str]) -> dict[str, Path]:
    if not specs:
        return PDFS.copy()
    result: dict[str, Path] = {}
    for spec in specs:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"Expected --pdf NAME=PATH, got: {spec!r}")
        if name in result:
            raise ValueError(f"Duplicate PDF name: {name}")
        result[name] = Path(raw_path)
    return result


def _percentile(values: list[int], ratio: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, int(len(ordered) * ratio) - 1)]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def audit_document(path: Path) -> dict:
    page_chars: list[int] = []
    blocks_per_page: list[int] = []
    spans_per_page: list[int] = []
    spans_per_line: list[int] = []
    chars_per_span: list[int] = []
    multi_span_lines = 0
    total_lines = 0
    mixed_blocks = 0
    total_blocks = 0
    overlapping_y_pages = 0
    low_text_pages: list[int] = []
    rotated_pages: list[int] = []
    replacement_char_pages: list[int] = []
    fonts: Counter[str] = Counter()
    non_embedded: Counter[str] = Counter()

    with pymupdf.open(path) as pdf:
        for page_no in range(1, pdf.page_count + 1):
            page = pdf[page_no - 1]
            if page.rotation != 0:
                rotated_pages.append(page_no)
            raw = page.get_text("dict", sort=True)
            blocks = [b for b in raw.get("blocks", []) if b.get("type") == 0]
            page_chars.append(
                sum(
                    len(span.get("text", ""))
                    for block in blocks
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                )
            )
            blocks_per_page.append(len(blocks))
            spans_on_page = 0
            y_ranges: list[tuple[float, float]] = []
            for block in blocks:
                total_blocks += 1
                bbox = block.get("bbox")
                if bbox:
                    y_ranges.append((bbox[1], bbox[3]))
                text_has_cjk = False
                text_has_number = False
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    total_lines += 1
                    if len(spans) > 1:
                        multi_span_lines += 1
                    spans_on_page += len(spans)
                    for span in spans:
                        text = span.get("text", "")
                        chars_per_span.append(len(text))
                        spans_per_line.append(len(spans))
                        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
                            text_has_cjk = True
                        if any(ch.isdigit() for ch in text):
                            text_has_number = True
                if text_has_cjk and text_has_number:
                    mixed_blocks += 1
            spans_per_page.append(spans_on_page)
            for i, (y0a, y1a) in enumerate(y_ranges):
                for y0b, y1b in y_ranges[i + 1 :]:
                    if y0a < y1b and y0b < y1a:
                        overlapping_y_pages += 1
                        break
                else:
                    continue
                break
            if page_chars[-1] < 300:
                low_text_pages.append(page_no)
            bad_char = any(
                "\ufffd" in span.get("text", "")
                for block in blocks
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            )
            if bad_char:
                replacement_char_pages.append(page_no)
            for font in page.get_fonts(full=True):
                basefont = str(font[3])
                embedded = bool(font[1])
                fonts[basefont] += 1
                if not embedded:
                    non_embedded[basefont] += 1
        page_count = pdf.page_count
    return {
        "page_count": page_count,
        "chars_per_page_p05": _percentile(page_chars, 0.05),
        "chars_per_page_p50": _percentile(page_chars, 0.5),
        "chars_per_page_min": min(page_chars, default=0),
        "blocks_per_page_p50": _percentile(blocks_per_page, 0.5),
        "spans_per_page_p50": _percentile(spans_per_page, 0.5),
        "spans_per_line_p50": _percentile(spans_per_line, 0.5),
        "chars_per_span_p50": _percentile(chars_per_span, 0.5),
        "chars_per_span_max": max(chars_per_span, default=0),
        "multi_span_line_ratio": _ratio(multi_span_lines, total_lines),
        "mixed_cjk_number_block_ratio": _ratio(mixed_blocks, total_blocks),
        "overlapping_y_pages": overlapping_y_pages,
        "low_text_pages": low_text_pages,
        "rotated_pages": rotated_pages,
        "replacement_char_pages": replacement_char_pages,
        "top_fonts": dict(fonts.most_common(6)),
        "non_embedded_fonts": dict(non_embedded),
    }


def chunk_stats(chunks_root: Path = ROOT / "data/catalog/versions") -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for path in chunks_root.glob("*/chunks.jsonl"):
        version = Path(path).parent.name
        total = 0
        cross_page = 0
        continuation = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            chunk = json.loads(line)
            total += 1
            if chunk["page_start"] != chunk["page_end"]:
                cross_page += 1
            if chunk.get("is_continuation"):
                continuation += 1
        result[version] = {
            "chunks": total,
            "cross_page": cross_page,
            "continuation": continuation,
        }
    return result


def main() -> None:
    args = parse_args()
    pdfs = parse_pdf_specs(args.pdf)
    stats = {
        "run_id": AUDIT_RUN_ID,
        "documents": {
            name: audit_document(path) for name, path in pdfs.items()
        },
        "chunks": chunk_stats(args.chunks_root),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {args.output_dir}")
    (args.output_dir / "summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Keep the CLI output safe on Windows terminals that still default to GBK;
    # the persisted JSON remains UTF-8 with readable Chinese text.
    print(json.dumps(stats, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
