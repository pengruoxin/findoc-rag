"""Prepare retrieval and page-text evidence for hard-v3 gold annotation.

The workspace is annotation-only. It never calls DeepSeek and is not consumed
by the evaluated Agent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf

from findoc_rag.corpus import resolve_current_index
from findoc_rag.indexing import SearchFilters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/agent-hard-v3-questions.json"),
    )
    parser.add_argument(
        "--review-packet",
        type=Path,
        default=Path("data/evaluation/agent-hard-v3-question-review.json"),
    )
    parser.add_argument(
        "--index-root", type=Path, default=Path("data/indexes/agent-hard-v3")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tmp/pdfs/agent-hard-v3-gold-workspace")
    )
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def _page_text(path: Path, page_number: int) -> str:
    with pymupdf.open(path) as pdf:
        return pdf[page_number - 1].get_text("text", sort=True).strip()


def main() -> None:
    args = parse_args()
    dataset = json.loads(args.questions.read_text(encoding="utf-8"))
    review = json.loads(args.review_packet.read_text(encoding="utf-8"))
    review_by_id = {item["case_id"]: item for item in review["items"]}
    index_names = {
        "calibration": "calibration",
        "dev": "development",
        "frozen_test": "frozen_test",
    }
    indexes = {
        split: resolve_current_index(args.index_root / name)
        for split, name in index_names.items()
    }
    grouped: dict[str, list[dict]] = {}
    for question in dataset["questions"]:
        review_item = review_by_id[question["case_id"]]
        source_keys = review_item["source_document_keys"]
        hits = []
        if question["expected_behavior"] == "answer":
            results = indexes[question["split"]].search(
                question["query"],
                top_k=args.top_k,
                mode="lexical",
                candidate_k=max(args.top_k, 50),
                filters=SearchFilters(document_keys=source_keys),
            )
            hits = [
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "document_key": hit.chunk.document_key,
                    "page_start": hit.chunk.page_start,
                    "page_end": hit.chunk.page_end,
                    "section_path": hit.chunk.section_path,
                    "score": hit.score,
                    "text": hit.chunk.text,
                }
                for hit in results
            ]
        candidate_pages = []
        for source in review_item["evidence_candidates"]:
            for candidate in source["candidate_pages"][:4]:
                candidate_pages.append(
                    {
                        "document_key": source["document_key"],
                        "local_file": source["local_file"],
                        "page_number": candidate["page_number"],
                        "matched_probes": candidate["matched_probes"],
                        "text": _page_text(
                            Path(source["local_file"]), candidate["page_number"]
                        ),
                    }
                )
        grouped.setdefault(question["company_name"], []).append(
            {
                **question,
                "source_document_keys": source_keys,
                "candidate_pages": candidate_pages,
                "retrieval_hits": hits,
                "annotation": {
                    "expected_facts": [],
                    "evidence_sources": [],
                    "gold_rationale": "",
                    "derivation": None,
                    "status": "unannotated",
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1",
        "dataset_id": dataset["dataset_id"],
        "question_payload_sha256": dataset["question_payload_sha256"],
        "model_assistance": "none",
        "evaluated_agent_called": False,
        "companies": [],
    }
    for company_name, items in grouped.items():
        security_code = items[0]["company_ids"][0]
        path = args.output_dir / f"{security_code}-{company_name}.json"
        path.write_text(
            json.dumps(
                {
                    "company_name": company_name,
                    "security_code": security_code,
                    "items": items,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest["companies"].append(
            {
                "company_name": company_name,
                "security_code": security_code,
                "item_count": len(items),
                "path": path.resolve().as_posix(),
            }
        )
        print(f"prepared {company_name}: {len(items)} cases")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"workspace={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
