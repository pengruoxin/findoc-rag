"""Evaluate production query routing signals (P2-12).

For every routing item the script applies the same inference functions used by
``/v1/query`` (``infer_finance_filters`` + deterministic query preparation) and
reports missed / extra / exact-match rates for company and year signals.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from findoc_rag.api import infer_finance_filters, prepare_finance_query
from findoc_rag.time_utils import resolve_relative_time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/evaluation/query-routing-v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = json.loads(args.data.read_text(encoding="utf-8"))
    as_of = date.fromisoformat(dataset.get("default_as_of_date", "2025-04-30"))
    rows: list[dict] = []
    for item in dataset["items"]:
        resolved_base, _ = resolve_relative_time(item["query"], as_of)
        got_companies, got_years = infer_finance_filters(resolved_base)
        resolved = prepare_finance_query(
            resolved_base, as_of_date=as_of, rewrite_mode="deterministic"
        )
        expected_companies = list(item["expected_companies"])
        expected_years = list(item["expected_years"])
        company_miss = sorted(set(expected_companies) - set(got_companies))
        company_extra = sorted(set(got_companies) - set(expected_companies))
        year_miss = sorted(set(expected_years) - set(got_years))
        year_extra = sorted(set(got_years) - set(expected_years))
        exact = not (company_miss or company_extra or year_miss or year_extra)
        rows.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "resolved_query": resolved,
                "expected_companies": expected_companies,
                "expected_years": expected_years,
                "got_companies": got_companies,
                "got_years": got_years,
                "company_miss": company_miss,
                "company_extra": company_extra,
                "year_miss": year_miss,
                "year_extra": year_extra,
                "exact": exact,
            }
        )
    total = len(rows)
    summary = {
        "run_id": "query-routing-v1",
        "dataset_id": dataset["dataset_id"],
        "as_of_date": as_of.isoformat(),
        "item_count": total,
        "exact_match_rate": sum(row["exact"] for row in rows) / total,
        "company_miss_rate": sum(bool(row["company_miss"]) for row in rows) / total,
        "company_extra_rate": sum(bool(row["company_extra"]) for row in rows) / total,
        "year_miss_rate": sum(bool(row["year_miss"]) for row in rows) / total,
        "year_extra_rate": sum(bool(row["year_extra"]) for row in rows) / total,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {args.output_dir}")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# 查询路由评测 v1（query-routing-v1）",
                "",
                f"- 数据集：`{dataset['dataset_id']}` | 样本：{total} | "
                f"as_of_date：{as_of.isoformat()}",
                "",
                "| 指标 | 值 |",
                "|---|---:|",
                f"| 精确匹配率 | {summary['exact_match_rate']:.4f} |",
                f"| 公司漏过滤率（该过滤没过滤） | {summary['company_miss_rate']:.4f} |",
                f"| 公司错/过过滤率 | {summary['company_extra_rate']:.4f} |",
                f"| 年份漏过滤率 | {summary['year_miss_rate']:.4f} |",
                f"| 年份错/过过滤率 | {summary['year_extra_rate']:.4f} |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
