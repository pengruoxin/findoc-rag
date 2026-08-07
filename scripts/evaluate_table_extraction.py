"""Cell-level table extraction evaluation on table-eval-v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.table_extraction import (
    extract_cells,
    normalize_label,
    normalize_value,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/evaluation/table-eval-v1.json"
DEFAULT_OUTPUT = ROOT / "reports/ranking/table-eval-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--chunks", type=Path, nargs="*")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_chunks(paths: list[Path] | None) -> dict[str, str]:
    chunks: dict[str, str] = {}
    candidates = paths or sorted(
        (ROOT / "data/catalog/versions").glob("*/chunks.jsonl")
    )
    for path in candidates:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunk = json.loads(line)
                chunks[chunk["chunk_id"]] = chunk["text"]
    return chunks


def gold_key(cell: dict) -> tuple[str, str, str]:
    return (
        normalize_label(cell["row"]),
        cell["column"],
        normalize_value(cell["value"]),
    )


def eval_table(table: dict, text: str) -> dict:
    gold = {gold_key(cell) for cell in table["cells"]}
    predicted = extract_cells(text, table["table_type"])
    pred_set = {
        (normalize_label(cell.row), cell.column, normalize_value(cell.value))
        for cell in predicted
    }
    correct = len(pred_set & gold)
    precision = correct / len(pred_set) if pred_set else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    wrong_rows = sorted(
        {
            normalize_label(cell.row)
            for cell in predicted
            if (normalize_label(cell.row), cell.column, normalize_value(cell.value))
            not in gold
        }
    )
    return {
        "table_id": table["table_id"],
        "table_type": table["table_type"],
        "gold_cells": len(gold),
        "predicted_cells": len(pred_set),
        "correct_cells": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "wrong_rows": wrong_rows[:10],
        "implemented": table["table_type"] == "quarterly",
    }


def main() -> None:
    args = parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    chunks = load_chunks(args.chunks)
    results = []
    for table in data["tables"]:
        text = chunks.get(table["chunk_id"], "")
        if not text:
            raise SystemExit(f"missing chunk {table['chunk_id']}")
        results.append(eval_table(table, text))

    by_type: dict[str, list[dict]] = {}
    for row in results:
        by_type.setdefault(row["table_type"], []).append(row)
    type_summary = {}
    for table_type, rows in sorted(by_type.items()):
        implemented = rows[0]["implemented"]
        type_summary[table_type] = {
            "implemented": implemented,
            "table_count": len(rows),
            "gold_cells": sum(row["gold_cells"] for row in rows),
            "correct_cells": sum(row["correct_cells"] for row in rows),
            "precision": (
                sum(row["correct_cells"] for row in rows)
                / sum(row["predicted_cells"] for row in rows)
                if implemented
                else 0.0
            ),
            "recall": (
                sum(row["correct_cells"] for row in rows)
                / sum(row["gold_cells"] for row in rows)
                if implemented
                else 0.0
            ),
        }

    summary = {
        "run_id": "table-eval-v1",
        "dataset_id": data["dataset_id"],
        "table_count": len(results),
        "gold_cells_total": sum(row["gold_cells"] for row in results),
        "by_table_type": type_summary,
        "tables": results,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {output_dir}")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 表格抽取评测 v1（table-eval-v1）",
        "",
        f"- 数据集：`{data['dataset_id']}` | 表数：{len(results)} | 标注单元格：{summary['gold_cells_total']}",
        "- 匹配规则：单元格三元组 (行标签归一化, 列头, 数值归一化) 完全一致才算对。",
        "",
        "| 表型 | 实现 | 表数 | gold cells | 正确 cells | Precision | Recall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for table_type, row in sorted(type_summary.items()):
        lines.append(
            f"| {table_type} | {'✅' if row['implemented'] else '⬜ 未实现'} | "
            f"{row['table_count']} | {row['gold_cells']} | {row['correct_cells']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 逐表结果",
            "",
            "| table | 类型 | gold | 预测 | 正确 | Precision | Recall | 错误行（归一化） |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in results:
        lines.append(
            f"| {row['table_id']} | {row['table_type']} | {row['gold_cells']} | "
            f"{row['predicted_cells']} | {row['correct_cells']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {'; '.join(row['wrong_rows'][:5]) or '-'} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
