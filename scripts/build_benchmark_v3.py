"""Build the assistant-curated calibration/development portion of benchmark v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from findoc_rag.documents.models import DocumentChunk
from findoc_rag.generation_evaluation import GenerationEvaluationDataset
from findoc_rag.io import read_jsonl
from findoc_rag.registry import DocumentRegistry

CALIBRATION_QUERY_IDS = {
    "moutai_revenue_yoy",
    "moutai_cashflow_change",
    "moutai_product_margin",
    "moutai_production_sales_inventory",
    "moutai_rd_composition",
    "u_moutai_2025_actual_revenue",
    "yili_annual_deducted_profit",
    "yili_cashflow_change",
    "yili_product_margin",
    "yili_liquid_milk_inventory",
    "yili_inventory_policy",
    "u_yili_top_customer_names",
}


DEV_SPECS: list[dict[str, Any]] = [
    {
        "prefix": "midea_2023",
        "company_id": "000333",
        "company_name": "美的集团",
        "year": 2023,
        "document_key": "cninfo:000333:annual:2023",
        "items": [
            {
                "suffix": "revenue_yoy",
                "query": "美的集团2023年营业收入及同比增幅是多少？",
                "category": "single_fact",
                "difficulty": "easy",
                "chunk_ids": ["22f4915d91ad:c14:a4f5d0e777bc4061"],
                "facts": [
                    ("revenue", "营业收入", "372037280", "number", "千元", None),
                    ("yoy", "营业收入同比增幅", "8.18", "percentage", "%", None),
                ],
                "answer": "美的集团2023年营业收入为372,037,280千元，同比增长8.18%[1]。",
            },
            {
                "suffix": "product_mix",
                "query": "美的集团2023年暖通空调和消费电器收入及收入占比分别是多少？",
                "category": "multi_fact_table",
                "difficulty": "medium",
                "chunk_ids": ["22f4915d91ad:c121:9f8a232bc0261e80"],
                "facts": [
                    ("hvac_revenue", "暖通空调收入", "161110843", "number", "千元", None),
                    ("hvac_share", "暖通空调收入占比", "43.31", "percentage", "%", None),
                    ("appliance_revenue", "消费电器收入", "134691669", "number", "千元", None),
                    ("appliance_share", "消费电器收入占比", "36.20", "percentage", "%", None),
                ],
                "answer": "2023年暖通空调收入161,110,843千元、占43.31%；消费电器收入134,691,669千元、占36.20%[1]。",
            },
            {
                "suffix": "rd_profile",
                "query": "美的集团2023年研发投入、研发投入占营收比例和研发人员数量是多少？",
                "category": "multi_fact_table",
                "difficulty": "medium",
                "chunk_ids": ["22f4915d91ad:c127:b2d5e94ed334a2a3"],
                "facts": [
                    ("rd_spend", "研发投入", "14583311", "number", "千元", None),
                    ("rd_ratio", "研发投入占营业收入比例", "3.92", "percentage", "%", None),
                    ("rd_people", "研发人员数量", "23242", "number", "人", None),
                ],
                "answer": "2023年研发投入14,583,311千元，占营业收入3.92%，研发人员23,242人[1]。",
            },
            {
                "suffix": "inventory_increase",
                "query": "美的集团家用电器行业2023年末库存量较2022年末增加了多少万台/套？",
                "category": "calculation",
                "difficulty": "medium",
                "chunk_ids": ["22f4915d91ad:c123:96d27ef9eacdab35"],
                "facts": [
                    (
                        "inventory_increase",
                        "库存量增加额",
                        "544.51",
                        "number",
                        "万台/套",
                        "9251.29 - 8706.78 = 544.51",
                    )
                ],
                "answer": "2023年末库存量较2022年末增加544.51万台/套（9,251.29－8,706.78）[1]。",
            },
            {
                "suffix": "key_audit_matter",
                "query": "美的集团2023年为何将暖通空调及消费电器收入确认列为关键审计事项？",
                "category": "narrative",
                "difficulty": "hard",
                "chunk_ids": ["22f4915d91ad:c328:8a7fa526a12b7ec1"],
                "facts": [
                    (
                        "kam_reason",
                        "关键审计事项原因",
                        "销售渠道多、境内外客户众多、销量巨大且相关收入金额重大，需要投入大量审计资源",
                        "text",
                        None,
                        None,
                    )
                ],
                "answer": "由于销售渠道多、境内外客户众多、销量巨大，且暖通空调及消费电器收入金额重大，审计需要投入大量资源，因此被列为关键审计事项[1]。",
            },
            {
                "suffix": "future_actual",
                "query": "仅依据美的集团2023年年报，2024年实际营业收入是多少？",
                "category": "unanswerable",
                "difficulty": "easy",
                "abstention_reason": "2023年年报没有披露2024年实际营业收入。",
            },
        ],
    },
    {
        "prefix": "midea_2024",
        "company_id": "000333",
        "company_name": "美的集团",
        "year": 2024,
        "document_key": "cninfo:000333:annual:2024",
        "items": [
            {
                "suffix": "revenue_yoy",
                "query": "美的集团2024年营业收入及同比增幅是多少？",
                "category": "single_fact",
                "difficulty": "easy",
                "chunk_ids": ["b17a9b9b84bc:c13:c9373c45dc02197f"],
                "facts": [
                    ("revenue", "营业收入", "407149600", "number", "千元", None),
                    ("yoy", "营业收入同比增幅", "9.44", "percentage", "%", None),
                ],
                "answer": "美的集团2024年营业收入为407,149,600千元，同比增长9.44%[1]。",
            },
            {
                "suffix": "product_mix",
                "query": "美的集团2024年智能家居业务和商业及工业解决方案收入及占比分别是多少？",
                "category": "multi_fact_table",
                "difficulty": "medium",
                "chunk_ids": ["b17a9b9b84bc:c113:c875fb5428de663c"],
                "facts": [
                    ("smart_home_revenue", "智能家居业务收入", "269532353", "number", "千元", None),
                    ("smart_home_share", "智能家居业务收入占比", "66.20", "percentage", "%", None),
                    ("bi_revenue", "商业及工业解决方案收入", "104496253", "number", "千元", None),
                    ("bi_share", "商业及工业解决方案收入占比", "25.67", "percentage", "%", None),
                ],
                "answer": "2024年智能家居业务收入269,532,353千元、占66.20%；商业及工业解决方案收入104,496,253千元、占25.67%[1]。",
            },
            {
                "suffix": "rd_profile",
                "query": "美的集团2024年研发投入、研发投入占营收比例和研发人员数量是多少？",
                "category": "multi_fact_table",
                "difficulty": "medium",
                "chunk_ids": ["b17a9b9b84bc:c119:4a358375ae716b15"],
                "facts": [
                    ("rd_spend", "研发投入", "16232771", "number", "千元", None),
                    ("rd_ratio", "研发投入占营业收入比例", "3.99", "percentage", "%", None),
                    ("rd_people", "研发人员数量", "23693", "number", "人", None),
                ],
                "answer": "2024年研发投入16,232,771千元，占营业收入3.99%，研发人员23,693人[1]。",
            },
            {
                "suffix": "inventory_increase",
                "query": "美的集团家用电器行业2024年末库存量较2023年末增加了多少万台/套，披露的原因是什么？",
                "category": "calculation",
                "difficulty": "hard",
                "chunk_ids": ["b17a9b9b84bc:c115:18aa82b8ae68d293"],
                "facts": [
                    (
                        "inventory_increase",
                        "库存量增加额",
                        "3899.67",
                        "number",
                        "万台/套",
                        "13150.96 - 9251.29 = 3899.67",
                    ),
                    (
                        "inventory_reason",
                        "库存增长原因",
                        "临近春节假期提前备货",
                        "text",
                        None,
                        None,
                    ),
                ],
                "answer": "库存量增加3,899.67万台/套（13,150.96－9,251.29）；公司称主要因为临近春节假期提前备货[1]。",
            },
            {
                "suffix": "key_audit_matter",
                "query": "美的集团2024年为何将智能家居业务收入确认列为关键审计事项？",
                "category": "narrative",
                "difficulty": "hard",
                "chunk_ids": ["b17a9b9b84bc:c318:04196d85a2df25a5"],
                "facts": [
                    (
                        "kam_reason",
                        "关键审计事项原因",
                        "销售渠道多、境内外客户众多、销量巨大且智能家居业务收入金额重大，需要投入大量审计资源",
                        "text",
                        None,
                        None,
                    )
                ],
                "answer": "由于销售渠道多、境内外客户众多、销量巨大，且智能家居业务收入金额重大，需要投入大量审计资源，因此被列为关键审计事项[1]。",
            },
            {
                "suffix": "future_actual",
                "query": "仅依据美的集团2024年年报，2025年实际净利润是多少？",
                "category": "unanswerable",
                "difficulty": "easy",
                "abstention_reason": "2024年年报没有披露2025年实际净利润。",
            },
        ],
    },
    {
        "prefix": "shenhua_2023",
        "company_id": "601088",
        "company_name": "中国神华",
        "year": 2023,
        "document_key": "cninfo:601088:annual:2023",
        "items": [
            {
                "suffix": "revenue_yoy",
                "query": "中国神华2023年营业收入及同比变动是多少？",
                "category": "single_fact",
                "difficulty": "easy",
                "chunk_ids": ["451aaacb1779:c14:ce981af7411444ad"],
                "facts": [
                    ("revenue", "营业收入", "343074", "number", "百万元", None),
                    ("yoy", "营业收入同比降幅", "0.4", "percentage", "%", None),
                ],
                "answer": "中国神华2023年营业收入为343,074百万元，同比下降0.4%[1]。",
            },
            {
                "suffix": "coal_source_margin",
                "query": "中国神华2023年自产煤和外购煤销售收入及毛利率分别是多少？",
                "category": "multi_fact_table",
                "difficulty": "medium",
                "chunk_ids": ["451aaacb1779:c62:eb71ea11864d12cd"],
                "facts": [
                    ("own_revenue", "自产煤销售收入", "178242", "number", "百万元", None),
                    ("own_margin", "自产煤毛利率", "46.6", "percentage", "%", None),
                    ("purchased_revenue", "外购煤销售收入", "84626", "number", "百万元", None),
                    ("purchased_margin", "外购煤毛利率", "2.4", "percentage", "%", None),
                ],
                "answer": "2023年自产煤销售收入178,242百万元、毛利率46.6%；外购煤销售收入84,626百万元、毛利率2.4%[1]。",
            },
            {
                "suffix": "rd_profile",
                "query": "中国神华2023年研发投入、占营收比例和研发人员数量是多少？",
                "category": "multi_fact_table",
                "difficulty": "medium",
                "chunk_ids": ["451aaacb1779:c50:6500b05d70ae355a"],
                "facts": [
                    ("rd_spend", "研发投入", "4453", "number", "百万元", None),
                    ("rd_ratio", "研发投入占营业收入比例", "1.3", "percentage", "%", None),
                    ("rd_people", "研发人员数量", "3030", "number", "人", None),
                ],
                "answer": "2023年研发投入4,453百万元，占营业收入1.3%，研发人员3,030人[1]。",
            },
            {
                "suffix": "coal_sales_production_gap",
                "query": "中国神华2023年煤炭销售量比生产量多多少百万吨？",
                "category": "calculation",
                "difficulty": "medium",
                "chunk_ids": ["451aaacb1779:c44:b3258e79ece8b63a"],
                "facts": [
                    (
                        "sales_production_gap",
                        "煤炭销售量与生产量之差",
                        "125.5",
                        "number",
                        "百万吨",
                        "450.0 - 324.5 = 125.5",
                    )
                ],
                "answer": "煤炭销售量比生产量多125.5百万吨（450.0－324.5）[1]。",
            },
            {
                "suffix": "coal_revenue_recognition",
                "query": "中国神华2023年煤炭销售收入何时确认，该收入约占营业收入多少？",
                "category": "accounting_policy",
                "difficulty": "hard",
                "chunk_ids": ["451aaacb1779:c296:39e88b0d81b5d1bf"],
                "facts": [
                    (
                        "recognition",
                        "煤炭销售收入确认时点",
                        "客户取得煤炭商品控制权时",
                        "text",
                        None,
                        None,
                    ),
                    ("revenue_share", "煤炭销售收入占营业收入比例", "65", "percentage", "%", None),
                ],
                "answer": "煤炭销售收入在客户取得煤炭商品控制权时确认；2023年度该收入约占营业收入65%[1]。",
            },
            {
                "suffix": "future_actual",
                "query": "仅依据中国神华2023年年报，2024年实际商品煤产量是多少？",
                "category": "unanswerable",
                "difficulty": "easy",
                "abstention_reason": "2023年年报没有披露2024年实际商品煤产量。",
            },
        ],
    },
    {
        "prefix": "shenhua_2024",
        "company_id": "601088",
        "company_name": "中国神华",
        "year": 2024,
        "document_key": "cninfo:601088:annual:2024",
        "items": [
            {
                "suffix": "revenue_yoy",
                "query": "中国神华2024年营业收入及同比变动是多少？",
                "category": "single_fact",
                "difficulty": "easy",
                "chunk_ids": ["4b3a8db26ae2:c14:df539193657dc9c8"],
                "facts": [
                    ("revenue", "营业收入", "338375", "number", "百万元", None),
                    ("yoy", "营业收入同比降幅", "1.4", "percentage", "%", None),
                ],
                "answer": "中国神华2024年营业收入为338,375百万元，同比下降1.4%[1]。",
            },
            {
                "suffix": "coal_source_margin",
                "query": "中国神华2024年自产煤和外购煤销售收入及毛利率分别是多少？",
                "category": "multi_fact_table",
                "difficulty": "medium",
                "chunk_ids": ["4b3a8db26ae2:c58:eeb18ccce26e633a"],
                "facts": [
                    ("own_revenue", "自产煤销售收入", "172442", "number", "百万元", None),
                    ("own_margin", "自产煤毛利率", "44.5", "percentage", "%", None),
                    ("purchased_revenue", "外购煤销售收入", "86373", "number", "百万元", None),
                    ("purchased_margin", "外购煤毛利率", "1.8", "percentage", "%", None),
                ],
                "answer": "2024年自产煤销售收入172,442百万元、毛利率44.5%；外购煤销售收入86,373百万元、毛利率1.8%[1]。",
            },
            {
                "suffix": "rd_profile",
                "query": "中国神华2024年研发投入、占营收比例和研发人员数量是多少？",
                "category": "multi_fact_table",
                "difficulty": "medium",
                "chunk_ids": ["4b3a8db26ae2:c46:34bfc5d19800eda0"],
                "facts": [
                    ("rd_spend", "研发投入", "4148", "number", "百万元", None),
                    ("rd_ratio", "研发投入占营业收入比例", "1.2", "percentage", "%", None),
                    ("rd_people", "研发人员数量", "3628", "number", "人", None),
                ],
                "answer": "2024年研发投入4,148百万元，占营业收入1.2%，研发人员3,628人[1]。",
            },
            {
                "suffix": "coal_sales_production_gap",
                "query": "中国神华2024年煤炭销售量比生产量多多少百万吨？",
                "category": "calculation",
                "difficulty": "medium",
                "chunk_ids": ["4b3a8db26ae2:c41:b60346edfa3f49d4"],
                "facts": [
                    (
                        "sales_production_gap",
                        "煤炭销售量与生产量之差",
                        "132.2",
                        "number",
                        "百万吨",
                        "459.3 - 327.1 = 132.2",
                    )
                ],
                "answer": "煤炭销售量比生产量多132.2百万吨（459.3－327.1）[1]。",
            },
            {
                "suffix": "coal_revenue_recognition",
                "query": "中国神华2024年煤炭销售收入何时确认，该收入约占营业收入多少？",
                "category": "accounting_policy",
                "difficulty": "hard",
                "chunk_ids": ["4b3a8db26ae2:c308:2587b8ea4ccf9b82"],
                "facts": [
                    (
                        "recognition",
                        "煤炭销售收入确认时点",
                        "客户取得煤炭商品控制权时",
                        "text",
                        None,
                        None,
                    ),
                    ("revenue_share", "煤炭销售收入占营业收入比例", "65", "percentage", "%", None),
                ],
                "answer": "煤炭销售收入在客户取得煤炭商品控制权时确认；2024年度该收入约占营业收入65%[1]。",
            },
            {
                "suffix": "future_actual",
                "query": "仅依据中国神华2024年年报，2025年实际营业收入是多少？",
                "category": "unanswerable",
                "difficulty": "easy",
                "abstention_reason": "2024年年报没有披露2025年实际营业收入。",
            },
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("development", "frozen", "combined"),
        default="development",
    )
    parser.add_argument("--specs", type=Path)
    parser.add_argument(
        "--development-dataset",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-development.json"),
    )
    parser.add_argument("--legacy", type=Path, default=Path("data/evaluation/benchmark-v2.json"))
    parser.add_argument(
        "--version-manifest",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-version-manifest.json"),
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("data/catalog/benchmark-v3/registry.sqlite3")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluation/benchmark-v3-development.json")
    )
    return parser.parse_args()


def _variants(query_id: str, query: str, company_id: str) -> list[dict]:
    return [
        {
            "variant_id": f"{query_id}:ticker",
            "query": query.replace("美的集团", company_id).replace("中国神华", company_id),
            "variant_types": ["ticker_or_finance_shorthand"],
            "query_regime": "ticker_or_finance_shorthand",
        },
        {
            "variant_id": f"{query_id}:semantic",
            "query": f"请从年报证据回答：{query}",
            "variant_types": ["semantic_paraphrase"],
            "query_regime": "semantic_paraphrase",
        },
    ]


def _annotation(source_hash: str, notes: str) -> dict:
    return {
        "created_by": "assistant_curated",
        "review_status": "assistant_verified",
        "confidence": "high",
        "source_pdf_sha256": [source_hash],
        "notes": notes,
        "human_reviews": [],
    }


def _dev_item(
    group: dict[str, Any],
    spec: dict[str, Any],
    chunks: dict[str, DocumentChunk],
    version: dict[str, Any],
) -> dict:
    query_id = f"{group['prefix']}_{spec['suffix']}"
    chunk_ids = spec.get("chunk_ids") or []
    answerable = bool(chunk_ids)
    facts = []
    for fact_id, predicate, value, value_type, unit, derivation in spec.get("facts") or []:
        facts.append(
            {
                "fact_id": fact_id,
                "description": predicate,
                "subject": group["company_name"],
                "predicate": predicate,
                "required": True,
                "canonical_value": value,
                "value_type": value_type,
                "acceptable_values": [value],
                "unit": unit,
                "currency": "CNY" if unit in {"千元", "百万元"} else None,
                "period": f"FY{group['year']}",
                "scope": "annual_report",
                "tolerance": "0",
                "derivation": derivation,
                "evidence_chunk_ids": chunk_ids,
            }
        )
    evidence = []
    for position, chunk_id in enumerate(chunk_ids, start=1):
        chunk = chunks[chunk_id]
        evidence.append(
            {
                "evidence_id": f"{query_id}:e{position}",
                "chunk_id": chunk_id,
                "document_version_id": version["version_id"],
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_path": chunk.section_path,
                "verbatim_quote": chunk.text,
                "supports_fact_ids": [fact["fact_id"] for fact in facts],
                "pdf_visual_verified": False,
            }
        )
    return {
        "query_id": query_id,
        "family_id": query_id,
        "split": group.get("split", "dev"),
        "query": spec["query"],
        "company_ids": [group["company_id"]],
        "company_names": [group["company_name"]],
        "company_aliases": [group["company_name"], group["company_id"]],
        "report_years": [group["year"]],
        "category": spec["category"],
        "difficulty": spec["difficulty"],
        "answerability": "answerable" if answerable else "unanswerable",
        "reference_answer": spec.get("answer") or "年报未披露该未来实际数据，应拒绝回答。",
        "expected_facts": facts,
        "gold_chunk_ids": chunk_ids,
        "gold_evidence": evidence,
        "hard_negatives": [],
        "tags": ["benchmark_v3", "assistant_curated", group["document_key"]],
        "query_variants": _variants(query_id, spec["query"], group["company_id"]),
        "answer_contract": {
            "expected_behavior": "answer" if answerable else "abstain",
            "required_format": "short" if answerable else "abstention",
            "require_citations": answerable,
            "require_units": answerable,
            "forbid_external_knowledge": True,
        },
        "required_citation_count": len(chunk_ids),
        "abstention_reason": spec.get("abstention_reason"),
        "annotation": _annotation(
            version["content_sha256"],
            "Assistant-authored and source-text verified; pending PDF visual review and two independent human approvals.",
        ),
        "notes": "",
    }


def main() -> None:
    args = parse_args()
    legacy = json.loads(args.legacy.read_text(encoding="utf-8"))
    version_manifest = json.loads(args.version_manifest.read_text(encoding="utf-8"))
    versions = {item["document_key"]: item for item in version_manifest["documents"]}
    development_index_id = version_manifest["indexes"]["development"]["index_id"]
    frozen_index_id = version_manifest["indexes"]["frozen_test"]["index_id"]

    registry = DocumentRegistry(args.registry)
    chunks: dict[str, DocumentChunk] = {}
    for version in registry.active_versions():
        for chunk in read_jsonl(Path(version.chunks_path), DocumentChunk):
            chunks[chunk.chunk_id] = chunk

    items: list[dict] = []
    groups: list[dict[str, Any]] = []
    if args.mode == "development":
        for legacy_item in legacy["items"]:
            if legacy_item["query_id"] not in CALIBRATION_QUERY_IDS:
                continue
            item = json.loads(json.dumps(legacy_item, ensure_ascii=False))
            item["split"] = "calibration"
            document_key = (
                "cninfo:600519:annual:2024"
                if "600519" in item["company_ids"]
                else "cninfo:600887:annual:2024"
            )
            version = versions[document_key]
            for evidence in item.get("gold_evidence") or []:
                evidence["document_version_id"] = version["version_id"]
                evidence["pdf_visual_verified"] = False
            item["annotation"] = _annotation(
                version["content_sha256"],
                "Migrated from assistant-curated benchmark-v2; pending renewed PDF visual review and two independent human approvals.",
            )
            items.append(item)
        groups = DEV_SPECS
    else:
        if args.specs is None:
            raise SystemExit("--specs is required for frozen and combined modes")
        groups = json.loads(args.specs.read_text(encoding="utf-8"))["groups"]
        if args.mode == "combined":
            development = GenerationEvaluationDataset.model_validate_json(
                args.development_dataset.read_text(encoding="utf-8")
            )
            items = [item.model_dump(mode="json") for item in development.items]

    for group in groups:
        version = versions[group["document_key"]]
        items.extend(_dev_item(group, spec, chunks, version) for spec in group["items"])

    index_id = frozen_index_id if args.mode == "frozen" else development_index_id
    dataset_id = {
        "development": "benchmark-v3-development",
        "frozen": "benchmark-v3-frozen",
        "combined": "benchmark-v3",
    }[args.mode]

    payload = {
        "schema_version": 3,
        "chunk_schema_version": 3,
        "dataset_id": dataset_id,
        "corpus_index_id": index_id,
        "corpus_indexes": {
            "calibration": version_manifest["indexes"]["calibration"]["index_id"],
            "dev": development_index_id,
            "frozen_test": frozen_index_id,
        },
        "independent_gold": False,
        "reviewer": "assistant source-text verified; pending independent human review",
        "status": "assistant_curated_provisional",
        "tracks": ["oracle_context", "retrieved_context", "robustness"],
        "item_count": len(items),
        "items": items,
    }
    dataset = GenerationEvaluationDataset.model_validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dataset.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dataset.item_count} items to {args.output}")


if __name__ == "__main__":
    main()
