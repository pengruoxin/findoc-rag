"""Evaluate coordinate-level table reconstruction against table-eval gold.

For every annotated table the script feeds the raw pymupdf blocks of the
chunk's pages (whole pages, no region cropping) to
``findoc_rag.table_reconstruction.reconstruct_cells`` and scores the result
with the same cell-triple rule used by ``evaluate_table_extraction.py``.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pymupdf

from findoc_rag.table_reconstruction import (
    blocks_from_pymupdf_dict,
    normalize_label,
    normalize_value,
    reconstruct_cells,
)

ROOT = Path(__file__).resolve().parents[1]

PDFS = {
    "5299f4940e2ce4e91084b73dc457d558b9d335fa76fbfee6227e4254eb7f4a30": (
        ROOT / "data/artifacts/cninfo/600519_2024_1222993920.pdf"
    ),
    "a82a81e52f52da3cd1b7f38ded08625dc18e3d4522b15d3ef76bf921e54c1f43": (
        ROOT / "data/artifacts/cninfo/600887_2024_1223421123.pdf"
    ),
}

DEFAULT_DATA = (
    ROOT / "data/evaluation/table-eval-v1.json",
    ROOT / "data/evaluation/table-eval-concentration-v1.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=list(DEFAULT_DATA))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_chunks() -> dict[str, dict]:
    chunks: dict[str, dict] = {}
    for path in glob.glob(str(ROOT / "data/catalog/versions/*/chunks.jsonl")):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            chunk = json.loads(line)
            chunks[chunk["chunk_id"]] = chunk
    return chunks


def cell_key(row: str, column: str, value: str) -> tuple[str, str, str]:
    return (normalize_label(row), column, normalize_value(value))


def score_table(table: dict, chunks: dict[str, dict]) -> dict:
    chunk = chunks[table["chunk_id"]]
    pdf_path = PDFS.get(chunk["document_id"].removeprefix("sha256:"))
    if pdf_path is None or not pdf_path.is_file():
        raise FileNotFoundError(f"Missing PDF for {table['table_id']}: {pdf_path}")
    pdf = pymupdf.open(pdf_path)
    model_blocks = []
    for page_no in range(chunk["page_start"], chunk["page_end"] + 1):
        page_raw = pdf[page_no - 1].get_text("dict", sort=True)
        model_blocks.extend(
            blocks_from_pymupdf_dict(page_raw, page=page_no)
        )
    region = chunk.get("section_path", [""])[-1] or ""
    predicted = reconstruct_cells(model_blocks, table["table_type"], region=region)
    pred_keys = {cell_key(c.row, c.column, c.value) for c in predicted}
    gold = {cell_key(cell["row"], cell["column"], cell["value"]) for cell in table["cells"]}
    correct = len(pred_keys & gold)
    return {
        "table_id": table["table_id"],
        "table_type": table["table_type"],
        "pages": f"{chunk['page_start']}-{chunk['page_end']}",
        "gold_cells": len(gold),
        "predicted_cells": len(pred_keys),
        "correct_cells": correct,
        "precision": correct / len(pred_keys) if pred_keys else 0.0,
        "recall": correct / len(gold) if gold else 0.0,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# 坐标级表格重建评测（整页块输入，无区域裁剪）",
        "",
        f"- 数据集：{', '.join(str(p.name) for p in DEFAULT_DATA)}",
        "- 输入：chunk 覆盖页的完整 pymupdf blocks；无表格区域裁剪（P0 最严苛口径）",
        "",
        "| table | 类型 | 页 | gold | pred | hit | P | R |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["tables"]:
        lines.append(
            f"| {row['table_id']} | {row['table_type']} | {row['pages']} | "
            f"{row['gold_cells']} | {row['predicted_cells']} | {row['correct_cells']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                f"合计：gold={summary['gold_cells_total']} "
                f"hit={summary['correct_cells_total']} "
                f"Recall={summary['recall_total']:.4f}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    chunks = load_chunks()
    tables: list[dict] = []
    for dataset_path in args.data:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        tables.extend(dataset["tables"])
    results = [score_table(table, chunks) for table in tables]
    gold_total = sum(row["gold_cells"] for row in results)
    correct_total = sum(row["correct_cells"] for row in results)
    summary = {
        "run_id": "coordinate-smoke-v1",
        "input_mode": "whole-pages-no-cropping",
        "table_count": len(results),
        "gold_cells_total": gold_total,
        "correct_cells_total": correct_total,
        "recall_total": correct_total / gold_total if gold_total else 0.0,
        "tables": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {args.output_dir}")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
