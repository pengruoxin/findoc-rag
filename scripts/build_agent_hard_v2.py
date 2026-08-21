"""Build the document-blind Agent hard-v2 candidate set from existing source gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from findoc_rag.agent_evaluation import (
    AgentHardCase,
    AgentHardDataset,
    AgentHardEvidenceSource,
    AgentHardExpectedFact,
)

UNSEEN_DOCUMENTS = {
    "cninfo:600519:annual:2024",
    "cninfo:600887:annual:2024",
    "cninfo:000333:annual:2023",
    "cninfo:000333:annual:2024",
    "cninfo:601088:annual:2024",
}

FUTURE_SOURCE_BY_QUERY = {
    "u_moutai_2025_actual_revenue": "cninfo:600519:annual:2024",
    "u_yili_top_customer_names": "cninfo:600887:annual:2024",
    "midea_2023_future_actual": "cninfo:000333:annual:2023",
    "midea_2024_future_actual": "cninfo:000333:annual:2024",
    "shenhua_2023_future_actual": "cninfo:601088:annual:2023",
    "shenhua_2024_future_actual": "cninfo:601088:annual:2024",
    "haier_2023_future_actual": "cninfo:600690:annual:2023",
    "haier_2024_future_actual": "cninfo:600690:annual:2024",
    "cypc_2023_future_actual": "cninfo:600900:annual:2023",
    "cypc_2024_future_actual": "cninfo:600900:annual:2024",
}

CROSS_DOCUMENT_SPECS = [
    {
        "case_id": "blind_midea_2023_2024_revenue_compare",
        "query": "比较美的集团2023年和2024年营业收入及同比增幅。",
        "source_ids": ["midea_2023_revenue_yoy", "midea_2024_revenue_yoy"],
        "expected_target_ids": [
            "company:美的集团:year:2023",
            "company:美的集团:year:2024",
        ],
        "challenge_types": ["cross_document", "multi_period", "multi_fact"],
    },
    {
        "case_id": "blind_midea_2023_2024_rd_compare",
        "query": "比较美的集团2023年和2024年研发投入、研发投入占营收比例及研发人员数量。",
        "source_ids": ["midea_2023_rd_profile", "midea_2024_rd_profile"],
        "expected_target_ids": [
            "company:美的集团:year:2023",
            "company:美的集团:year:2024",
        ],
        "challenge_types": ["cross_document", "multi_period", "dense_table"],
    },
    {
        "case_id": "blind_moutai_yili_cashflow_compare",
        "query": "比较贵州茅台和伊利股份2024年经营活动现金流量净额、同比增幅及各自变动原因。",
        "source_ids": ["moutai_cashflow_change", "yili_cashflow_change"],
        "expected_target_ids": [
            "company:贵州茅台:year:2024",
            "company:伊利股份:year:2024",
        ],
        "challenge_types": ["cross_company", "multi_fact", "narrative_evidence"],
    },
    {
        "case_id": "blind_midea_shenhua_2024_revenue_compare",
        "query": "比较美的集团和中国神华2024年营业收入及同比变动。",
        "source_ids": ["midea_2024_revenue_yoy", "shenhua_2024_revenue_yoy"],
        "expected_target_ids": [
            "company:美的集团:year:2024",
            "company:中国神华:year:2024",
        ],
        "challenge_types": ["cross_company", "unit_mismatch", "multi_fact"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("data/evaluation/benchmark-v3.json"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-source-manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/agent-hard-v2.json"),
    )
    return parser.parse_args()


def _source_document_key(item: dict[str, Any], chunks_by_id: dict[str, dict]) -> str:
    future_source = FUTURE_SOURCE_BY_QUERY.get(item["query_id"])
    if future_source is not None:
        return future_source
    document_keys = {
        chunks_by_id[chunk_id]["document_key"] for chunk_id in item["gold_chunk_ids"]
    }
    if len(document_keys) != 1:
        raise ValueError(
            f"Expected one source document for {item['query_id']}: {document_keys}"
        )
    return document_keys.pop()


def _expected_facts(item: dict[str, Any], *, prefix: str = "") -> list[AgentHardExpectedFact]:
    return [
        AgentHardExpectedFact(
            fact_id=f"{prefix}{fact['fact_id']}",
            description=fact["description"],
            acceptable_values=list(
                dict.fromkeys(
                    [
                        *fact.get("acceptable_values", []),
                        str(fact["canonical_value"]),
                    ]
                )
            ),
            unit=fact.get("unit"),
        )
        for fact in item["expected_facts"]
    ]


def _challenge_types(item: dict[str, Any]) -> list[str]:
    category = item["category"]
    challenges = {
        "single_fact": ["document_blind", "exact_value"],
        "multi_fact_table": ["document_blind", "dense_table", "multi_fact"],
        "narrative": ["document_blind", "narrative_evidence", "multi_fact"],
        "accounting_policy": ["document_blind", "accounting_scope", "narrative_evidence"],
        "calculation": ["document_blind", "arithmetic_reconciliation"],
        "unanswerable": ["document_blind", "safe_abstention", "temporal_scope"],
    }.get(category, ["document_blind", category])
    pages = {page for evidence in item["gold_evidence"] for page in range(
        evidence["page_start"], evidence["page_end"] + 1
    )}
    if len(pages) > 1:
        challenges.append("cross_page_evidence")
    return challenges


def _evidence_source(
    document_key: str,
    items: list[dict[str, Any]],
    manifest_by_key: dict[str, dict],
) -> AgentHardEvidenceSource:
    pages = sorted(
        {
            page
            for item in items
            for evidence in item["gold_evidence"]
            for page in range(evidence["page_start"], evidence["page_end"] + 1)
            if any(
                chunk_id in item["gold_chunk_ids"]
                for chunk_id in [evidence["chunk_id"]]
            )
        }
    )
    return AgentHardEvidenceSource(
        document_key=document_key,
        local_file=manifest_by_key[document_key]["local_file"],
        pages=pages,
    )


def _load_index_chunks(index_root: Path) -> dict[str, dict]:
    current = json.loads((index_root / "current.json").read_text(encoding="utf-8"))
    generation = index_root / current["generation_path"]
    import sqlite3

    with sqlite3.connect(generation / "lexical.sqlite3") as connection:
        rows = connection.execute("SELECT chunk_id, payload_json FROM chunks").fetchall()
    return {chunk_id: json.loads(payload) for chunk_id, payload in rows}


def build_dataset(
    source_dataset: Path,
    source_manifest: Path,
    *,
    index_root: Path = Path("data/indexes/benchmark-v3"),
) -> AgentHardDataset:
    source = json.loads(source_dataset.read_text(encoding="utf-8"))
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest_by_key = {
        document["document_key"]: document for document in manifest["documents"]
    }
    chunks_by_id = _load_index_chunks(index_root)
    selected: list[tuple[dict[str, Any], str]] = []
    for item in source["items"]:
        document_key = _source_document_key(item, chunks_by_id)
        if document_key in UNSEEN_DOCUMENTS:
            selected.append((item, document_key))
    if len(selected) != 30:
        raise ValueError(f"Expected 30 document-local cases, found {len(selected)}")

    cases: list[AgentHardCase] = []
    source_item_by_id = {item["query_id"]: item for item, _ in selected}
    source_key_by_id = {item["query_id"]: key for item, key in selected}
    for item, document_key in selected:
        task_type = "calculate" if item["category"] == "calculation" else "extract"
        expected_behavior = (
            "abstain" if item["answerability"] == "unanswerable" else "answer"
        )
        cases.append(
            AgentHardCase(
                case_id=f"blind_{item['query_id']}",
                task_type=task_type,
                query=item["query"],
                challenge_types=_challenge_types(item),
                expected_behavior=expected_behavior,
                expected_target_ids=[],
                expected_facts=_expected_facts(item),
                evidence_sources=[
                    _evidence_source(
                        document_key,
                        [item],
                        manifest_by_key,
                    )
                ],
                gold_rationale=(
                    item.get("reference_answer")
                    or item.get("abstention_reason")
                    or "Source benchmark expected behavior."
                ),
                annotation_status="assistant_verified_provisional",
            )
        )

    for spec in CROSS_DOCUMENT_SPECS:
        items = [source_item_by_id[item_id] for item_id in spec["source_ids"]]
        document_keys = [source_key_by_id[item_id] for item_id in spec["source_ids"]]
        facts = [
            fact
            for item in items
            for fact in _expected_facts(item, prefix=f"{item['query_id']}:")
        ]
        cases.append(
            AgentHardCase(
                case_id=spec["case_id"],
                task_type="compare",
                query=spec["query"],
                challenge_types=spec["challenge_types"],
                expected_behavior="answer",
                expected_target_ids=spec["expected_target_ids"],
                expected_facts=facts,
                evidence_sources=[
                    _evidence_source(
                        document_key,
                        [
                            item
                            for item, key in zip(items, document_keys, strict=True)
                            if key == document_key
                        ],
                        manifest_by_key,
                    )
                    for document_key in dict.fromkeys(document_keys)
                ],
                gold_rationale="；".join(item["reference_answer"] for item in items),
                annotation_status="assistant_verified_provisional_composite",
            )
        )

    return AgentHardDataset(
        schema_version="1",
        dataset_id="agent-hard-v2-document-blind-candidate",
        description=(
            "Thirty document-local and four cross-document tasks from five annual reports "
            "not used to develop the P0/P1 Agent tools. Gold is inherited from benchmark-v3 "
            "assistant-verified source evidence and remains provisional pending independent review."
        ),
        index_scope="benchmark-v3 full immutable index; five Agent-unseen source documents",
        source_manifest=source_manifest.as_posix(),
        gold_policy=(
            "Gold values and source pages are inherited before this Agent baseline from "
            "benchmark-v3; no model judge and no hard-v2 gold enters Agent prompts."
        ),
        cases=cases,
    )


def main() -> int:
    args = parse_args()
    dataset = build_dataset(args.source_dataset, args.source_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dataset.model_dump_json(indent=2) + "\n", encoding="utf-8")
    task_counts: dict[str, int] = {}
    for case in dataset.cases:
        task_counts[case.task_type] = task_counts.get(case.task_type, 0) + 1
    print(f"cases={len(dataset.cases)}")
    print(f"task_counts={json.dumps(task_counts, ensure_ascii=False, sort_keys=True)}")
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
