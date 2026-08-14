"""Evaluate coordinate-level table reconstruction against table-eval gold.

For every annotated table the script feeds the raw pymupdf blocks of the
chunk's pages (whole pages, no region cropping) to
``findoc_rag.table_reconstruction.reconstruct_cells`` and scores the result
with the same cell-triple rule used by ``evaluate_table_extraction.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pymupdf

from findoc_rag.benchmark_assets import benchmark_chunk_paths
from findoc_rag.table_reconstruction import (
    blocks_from_document_ir,
    blocks_from_pymupdf_dict,
    normalize_label,
    normalize_value,
    reconstruct_cells,
    select_table_cells,
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
    parser.add_argument(
        "--input-mode",
        choices=("pdf", "ir"),
        default="pdf",
        help="replay geometry from source PDFs or persisted document IR v2",
    )
    parser.add_argument(
        "--documents-root",
        type=Path,
        default=ROOT / "data/catalog/versions",
        help="root containing persisted document.json files for --input-mode ir",
    )
    parser.add_argument(
        "--candidate-policy",
        choices=("raw", "safe"),
        default="safe",
        help="score raw coordinate output or the chunk-grounded production selector",
    )
    return parser.parse_args()


def load_chunks() -> dict[str, dict]:
    chunks: dict[str, dict] = {}
    for path in benchmark_chunk_paths(ROOT):
        for line in path.read_text(encoding="utf-8").splitlines():
            chunk = json.loads(line)
            chunks[chunk["chunk_id"]] = chunk
    return chunks


def cell_key(row: str, column: str, value: str) -> tuple[str, str, str]:
    return (normalize_label(row), column, normalize_value(value))


def load_documents(root: Path) -> tuple[dict[str, object], list[dict]]:
    documents: dict[str, object] = {}
    provenance: list[dict] = []
    for path in sorted(root.rglob("document.json")):
        from findoc_rag.documents.models import ParsedDocument

        document = ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        if document.document_id in documents:
            raise ValueError(
                "Multiple persisted IR versions share document ID "
                f"{document.document_id!r} under {root}; pass a version-scoped "
                "--documents-root"
            )
        documents[document.document_id] = document
        manifest_path = path.parent / "ingestion-manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        provenance.append(
            {
                "document_id": document.document_id,
                "version_id": manifest.get("version_id"),
                "content_sha256": manifest.get("content_sha256"),
                "processing_fingerprint": manifest.get("processing_fingerprint"),
                "ir_schema_version": manifest.get("processing_components", {}).get(
                    "ir_schema_version"
                ),
                "document_ir_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "geometry_element_count": sum(
                    bool(element.lines)
                    for page in document.pages
                    for element in page.elements
                ),
                "relative_path": path.relative_to(root).as_posix(),
            }
        )
    return documents, provenance


def score_table(
    table: dict,
    chunks: dict[str, dict],
    *,
    input_mode: str = "pdf",
    documents: dict[str, object] | None = None,
    candidate_policy: str = "safe",
) -> dict:
    chunk = chunks[table["chunk_id"]]
    if input_mode == "ir":
        document = (documents or {}).get(chunk["document_id"])
        if document is None:
            raise FileNotFoundError(
                f"Missing persisted document IR for {table['table_id']}: {chunk['document_id']}"
            )
        if not any(element.lines for page in document.pages for element in page.elements):
            raise ValueError(
                f"Persisted document IR has no line/span geometry for {table['table_id']}; "
                "reprocess the PDF or pass --documents-root pointing to IR v2"
            )
        model_blocks = blocks_from_document_ir(
            document,  # type: ignore[arg-type]
            page_start=chunk["page_start"],
            page_end=chunk["page_end"],
        )
    else:
        pdf_path = PDFS.get(chunk["document_id"].removeprefix("sha256:"))
        if pdf_path is None or not pdf_path.is_file():
            raise FileNotFoundError(f"Missing PDF for {table['table_id']}: {pdf_path}")
        model_blocks = []
        with pymupdf.open(pdf_path) as pdf:
            for page_no in range(chunk["page_start"], chunk["page_end"] + 1):
                page_raw = pdf[page_no - 1].get_text("dict", sort=True)
                model_blocks.extend(
                    blocks_from_pymupdf_dict(page_raw, page=page_no)
                )
    region = chunk.get("section_path", [""])[-1] or ""
    predicted = reconstruct_cells(model_blocks, table["table_type"], region=region)
    selection = select_table_cells(predicted, chunk["text"], table["table_type"])
    if candidate_policy == "safe":
        predicted = selection.cells
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
        "selected_source": selection.source if candidate_policy == "safe" else "coordinate_raw",
        "dropped_coordinate_cells": (
            selection.dropped_coordinate_cells if candidate_policy == "safe" else 0
        ),
        "selection_reasons": list(selection.reasons) if candidate_policy == "safe" else [],
        "precision": correct / len(pred_keys) if pred_keys else 0.0,
        "recall": correct / len(gold) if gold else 0.0,
    }


def render_markdown(summary: dict) -> str:
    input_description = {
        "source-pdf-whole-pages-no-cropping": (
            "chunk 覆盖页的完整 pymupdf blocks"
        ),
        "persisted-ir-v2-whole-pages-no-cropping": (
            "chunk 覆盖页的持久化 Document IR v2 line/span geometry"
        ),
    }.get(summary["input_mode"], summary["input_mode"])
    lines = [
        "# 坐标级表格重建评测（整页块输入，无区域裁剪）",
        "",
        f"- 数据集：{', '.join(str(p.name) for p in DEFAULT_DATA)}",
        f"- 输入：{input_description}；无表格区域裁剪（P0 最严苛口径）",
    ]
    for item in summary.get("document_ir_provenance", []):
        lines.append(
            "- IR 版本："
            f"`{item['version_id']}` / `{item['document_id']}` / "
            f"processing `{item['processing_fingerprint']}` / "
            f"document SHA `{item['document_ir_sha256']}`"
        )
    lines.extend(
        [
            "",
            "| table | 类型 | 页 | gold | pred | hit | P | R |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
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
    if args.input_mode == "ir":
        documents, document_ir_provenance = load_documents(args.documents_root)
    else:
        documents, document_ir_provenance = None, []
    tables: list[dict] = []
    for dataset_path in args.data:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        tables.extend(dataset["tables"])
    results = [
        score_table(
            table,
            chunks,
            input_mode=args.input_mode,
            documents=documents,
            candidate_policy=args.candidate_policy,
        )
        for table in tables
    ]
    gold_total = sum(row["gold_cells"] for row in results)
    correct_total = sum(row["correct_cells"] for row in results)
    summary = {
        "run_id": f"coordinate-smoke-v1-{args.input_mode}",
        "input_mode": (
            "persisted-ir-v2-whole-pages-no-cropping"
            if args.input_mode == "ir"
            else "source-pdf-whole-pages-no-cropping"
        ),
        "candidate_policy": args.candidate_policy,
        "source_document_ids": sorted(
            {chunks[table["chunk_id"]]["document_id"] for table in tables}
        ),
        "document_ir_provenance": document_ir_provenance,
        "table_count": len(results),
        "gold_cells_total": gold_total,
        "correct_cells_total": correct_total,
        "predicted_cells_total": sum(row["predicted_cells"] for row in results),
        "precision_total": (
            correct_total / sum(row["predicted_cells"] for row in results)
            if any(row["predicted_cells"] for row in results)
            else 0.0
        ),
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
