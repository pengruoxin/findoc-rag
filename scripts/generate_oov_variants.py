"""Generate out-of-vocabulary query variants for robustness evaluation.

OOV variants must not contain the canonical filing wording nor any alias
already covered by the deterministic synonym table, so they measure real
semantic robustness instead of synonym-table recall.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEW = ROOT / "data/evaluation/benchmark-v2-retrieval-view.json"
DEFAULT_OUTPUT = ROOT / "data/evaluation/oov-variants-v1.json"

BANNED_TERMS = (
    "营业收入",
    "营业总收入",
    "毛利率",
    "净资产收益率",
    "净利润",
    "现金流量",
    "主要风险",
    "可能面对的风险",
    "前五名客户",
    "计划实现",
    "比上年同期增减",
    "营业成本",
    "营收",
    "毛利水平",
    "净资产回报率",
    "前五大客户",
    "同比增幅",
    "一定实现",
)

TARGET_QUERY_IDS = (
    "moutai_revenue_yoy",
    "moutai_roe",
    "moutai_product_margin",
    "moutai_cashflow_change",
    "yili_product_margin",
    "customer_concentration_comparison",
    "moutai_disclosed_risks",
    "yili_quarterly_net_profit",
    "moutai_production_sales_inventory",
    "yili_inventory_policy",
    "moutai_annual_deducted_profit",
    "yili_consolidated_parent_revenue",
)

PROMPT = (
    "把下面的财务问句改写成 3 条普通用户会说的口语问法，要求：\n"
    "1. 意思不变，仍然指向同一个财务信息；\n"
    "2. 不能出现以下年报术语或常见缩写：{banned}；\n"
    "3. 每条用完全不同的说法；\n"
    "4. 只输出 JSON 数组，例如 [\"问法1\", \"问法2\", \"问法3\"]，不要解释。\n"
    "原问句：{query}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="deepseek-chat")
    return parser.parse_args()


def rewrite_variants(query: str, banned: str, api_key: str, model: str) -> list[str]:
    payload = {
        "model": model,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": "你是中文口语改写助手，只输出 JSON。"},
            {"role": "user", "content": PROMPT.format(query=query, banned=banned)},
        ],
    }
    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=httpx.Timeout(60.0, connect=20.0),
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```(?:json)?|```$", "", content).strip()
            parsed = json.loads(content)
            if isinstance(parsed, list) and parsed:
                return [str(item) for item in parsed[:3]]
        except (httpx.TransportError, httpx.HTTPStatusError, json.JSONDecodeError, KeyError, ValueError):
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return []


def main() -> None:
    args = parse_args()
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required")
    view = json.loads(args.view.read_text(encoding="utf-8"))
    by_id = {item["query_id"]: item for item in view["items"]}
    banned = "、".join(BANNED_TERMS)
    items = []
    rejected = 0
    for query_id in TARGET_QUERY_IDS:
        item = by_id[query_id]
        variants = rewrite_variants(item["query"], banned, api_key, args.model)
        clean = [
            variant
            for variant in variants
            if not any(term in variant for term in BANNED_TERMS)
        ]
        rejected += len(variants) - len(clean)
        items.append(
            {
                "query_id": query_id,
                "canonical_query": item["query"],
                "oov_queries": clean,
            }
        )
        print(f"{query_id}: {len(variants)} generated, {len(clean)} kept")
        for variant in clean:
            print(f"    - {variant}")
        time.sleep(0.5)
    payload = {
        "schema_version": 1,
        "dataset_id": "oov-variants-v1",
        "source": "deepseek-chat generated, program-validated against synonym table",
        "items": items,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"total oov variants: {sum(len(i['oov_queries']) for i in items)}, rejected: {rejected}")


if __name__ == "__main__":
    main()
