"""Validate hard-v3 question freezing, source isolation, and review readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from findoc_rag.evaluation.governance import (
    BenchmarkCorpusPlan,
    GovernancePolicy,
    audit_corpus_plan,
    audit_prepared_corpus,
)
from findoc_rag.ingestion import file_sha256

FORBIDDEN_BLIND_FIELDS = (
    '"expected_facts"',
    '"acceptable_values"',
    '"gold_rationale"',
    '"evidence_sources"',
)


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
        "--plan", type=Path, default=Path("data/evaluation/agent-hard-v3-corpus-plan.json")
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/evaluation/agent-hard-v3-source-manifest.json"),
    )
    parser.add_argument(
        "--version-manifest",
        type=Path,
        default=Path("data/evaluation/agent-hard-v3-version-manifest.json"),
    )
    parser.add_argument(
        "--old-plan",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-corpus-plan.json"),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/evaluation-governance-p0.json"),
    )
    parser.add_argument(
        "--pdf-audit",
        type=Path,
        default=Path("reports/pdf-extraction/agent-hard-v3-source-audit/summary.json"),
    )
    parser.add_argument(
        "--scan-audit",
        type=Path,
        default=Path("reports/pdf-extraction/agent-hard-v3-scan-coverage-audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/agent/agent-hard-v3-question-audit.json"),
    )
    return parser.parse_args()


def _normalize_query(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _collect_queries(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "query" and isinstance(item, str):
                result.append(item)
            else:
                result.extend(_collect_queries(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_collect_queries(item))
    return result


def _question_hash(questions: list[dict]) -> str:
    payload = [
        {
            "case_id": item["case_id"],
            "task_type": item["task_type"],
            "agent_command": item["agent_command"],
            "query": item["query"],
        }
        for item in questions
    ]
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _check(condition: bool, label: str, blockers: list[str], checks: dict) -> None:
    checks[label] = condition
    if not condition:
        blockers.append(label)


def main() -> None:
    args = parse_args()
    raw_questions = args.questions.read_text(encoding="utf-8")
    bank = json.loads(raw_questions)
    review = json.loads(args.review_packet.read_text(encoding="utf-8"))
    plan = BenchmarkCorpusPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    version_manifest = json.loads(args.version_manifest.read_text(encoding="utf-8"))
    old_plan = BenchmarkCorpusPlan.model_validate_json(
        args.old_plan.read_text(encoding="utf-8")
    )
    policy_payload = json.loads(args.policy.read_text(encoding="utf-8"))
    policy_payload["split_strategy"] = plan.split_strategy
    policy = GovernancePolicy.model_validate(policy_payload)
    pdf_audit = json.loads(args.pdf_audit.read_text(encoding="utf-8"))
    scan_audit = json.loads(args.scan_audit.read_text(encoding="utf-8"))

    questions = bank["questions"]
    review_items = review["items"]
    blockers: list[str] = []
    checks: dict[str, bool] = {}
    warnings: list[str] = []

    _check(len(questions) == 96, "question_count_is_96", blockers, checks)
    _check(
        len({item["case_id"] for item in questions}) == len(questions),
        "case_ids_are_unique",
        blockers,
        checks,
    )
    normalized_queries = [_normalize_query(item["query"]) for item in questions]
    _check(
        len(set(normalized_queries)) == len(questions),
        "question_text_is_unique",
        blockers,
        checks,
    )
    _check(
        not any(field in raw_questions for field in FORBIDDEN_BLIND_FIELDS),
        "blind_questions_contain_no_gold_fields",
        blockers,
        checks,
    )
    _check(
        _question_hash(questions) == bank["question_payload_sha256"],
        "question_hash_matches",
        blockers,
        checks,
    )
    _check(
        review["question_payload_sha256"] == bank["question_payload_sha256"],
        "review_packet_matches_question_hash",
        blockers,
        checks,
    )
    _check(
        {item["case_id"] for item in review_items}
        == {item["case_id"] for item in questions},
        "review_packet_case_membership_matches",
        blockers,
        checks,
    )

    task_counts = Counter(item["task_type"] for item in questions)
    command_counts = Counter(item["agent_command"] for item in questions)
    behavior_counts = Counter(item["expected_behavior"] for item in questions)
    split_counts = Counter(item["split"] for item in questions)
    _check(
        task_counts == {"extract": 56, "compare": 24, "calculate": 16},
        "task_type_distribution_matches",
        blockers,
        checks,
    )
    _check(
        command_counts
        == {"extract": 48, "compare": 16, "calculate": 16, "verify": 8, "clarify": 8},
        "agent_command_distribution_matches",
        blockers,
        checks,
    )
    _check(
        behavior_counts == {"answer": 80, "abstain": 16},
        "answerability_distribution_matches",
        blockers,
        checks,
    )
    _check(
        split_counts == {"calibration": 24, "dev": 24, "frozen_test": 48},
        "split_distribution_matches",
        blockers,
        checks,
    )
    _check(
        len({item["company_ids"][0] for item in questions}) == 8,
        "company_count_is_8",
        blockers,
        checks,
    )

    plan_audit = audit_corpus_plan(plan, policy)
    prepared_audit = audit_prepared_corpus(plan, source_manifest, version_manifest)
    _check(
        plan_audit.ready_for_acquisition,
        "sealed_plan_passes_governance",
        blockers,
        checks,
    )
    _check(
        prepared_audit.ready_for_annotation,
        "prepared_corpus_passes_governance",
        blockers,
        checks,
    )
    new_keys = {item.document_key for item in plan.documents}
    old_keys = {item.document_key for item in old_plan.documents}
    new_companies = {item.security_code for item in plan.documents}
    old_companies = {item.security_code for item in old_plan.documents}
    _check(
        not new_keys.intersection(old_keys),
        "source_documents_do_not_overlap_old_benchmark",
        blockers,
        checks,
    )
    _check(
        not new_companies.intersection(old_companies),
        "companies_do_not_overlap_old_benchmark",
        blockers,
        checks,
    )

    source_records = {
        item["document_key"]: item for item in source_manifest["documents"]
    }
    source_hash_valid = all(
        Path(item["local_file"]).is_file()
        and file_sha256(Path(item["local_file"])) == item["sha256"]
        for item in source_records.values()
    )
    _check(source_hash_valid, "source_pdf_hashes_match", blockers, checks)
    _check(
        len({item["sha256"] for item in source_records.values()}) == 16,
        "source_pdf_hashes_are_unique",
        blockers,
        checks,
    )

    page_count_by_key: dict[str, int] = {}
    for key, item in source_records.items():
        with pymupdf.open(item["local_file"]) as pdf:
            page_count_by_key[key] = pdf.page_count
    evidence_candidates_valid = True
    for item in review_items:
        if item["structural_status"] == "not_applicable":
            continue
        if not item["evidence_candidates"]:
            evidence_candidates_valid = False
            continue
        for source in item["evidence_candidates"]:
            candidates = source["candidate_pages"]
            if not candidates:
                evidence_candidates_valid = False
                continue
            if any(
                candidate["page_number"] < 1
                or candidate["page_number"] > page_count_by_key[source["document_key"]]
                for candidate in candidates
            ):
                evidence_candidates_valid = False
    _check(
        evidence_candidates_valid,
        "all_answerable_questions_have_valid_candidate_pages",
        blockers,
        checks,
    )
    _check(
        Counter(item["structural_status"] for item in review_items)
        == {"probe_pages_found": 79, "visual_page_confirmed": 1, "not_applicable": 16},
        "question_structure_checks_complete",
        blockers,
        checks,
    )

    page_total = sum(
        document["page_count"] for document in pdf_audit["documents"].values()
    )
    _check(page_total == 4690, "pdf_page_audit_covers_4690_pages", blockers, checks)
    _check(
        not any(
            document["replacement_char_pages"]
            for document in pdf_audit["documents"].values()
        ),
        "pdf_audit_has_no_replacement_character_pages",
        blockers,
        checks,
    )
    _check(
        scan_audit["conclusion"]
        == "no_eligible_genuine_scanned_table_in_current_corpus",
        "scanned_table_claim_is_not_overstated",
        blockers,
        checks,
    )
    _check(
        scan_audit["unreviewed_candidate_count"] == 0,
        "all_scan_candidates_were_visually_reviewed",
        blockers,
        checks,
    )

    prior_queries: set[str] = set()
    prior_paths: list[str] = []
    for path in sorted(Path("data/evaluation").glob("agent-hard-v[12]*.json")):
        if path == args.questions:
            continue
        prior_paths.append(path.as_posix())
        payload = json.loads(path.read_text(encoding="utf-8"))
        prior_queries.update(_normalize_query(value) for value in _collect_queries(payload))
    query_overlap = sorted(set(normalized_queries).intersection(prior_queries))
    _check(
        not query_overlap,
        "questions_do_not_exactly_overlap_prior_agent_sets",
        blockers,
        checks,
    )
    if not prior_paths:
        warnings.append("No prior agent-hard-v1/v2 files were found for exact query auditing")

    report = {
        "schema_version": "1",
        "dataset_id": bank["dataset_id"],
        "question_payload_sha256": bank["question_payload_sha256"],
        "ready_for_gold_annotation": not blockers,
        "ready_for_agent_evaluation": False,
        "agent_evaluation_blockers": [
            "independent gold facts are not frozen",
            "exact evidence pages are not frozen",
            "two independent human approvals are not complete",
            "clarify command scoring is not implemented",
        ],
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
        "statistics": {
            "question_count": len(questions),
            "company_count": len(new_companies),
            "source_document_count": len(new_keys),
            "pdf_page_count": page_total,
            "task_type_counts": dict(sorted(task_counts.items())),
            "agent_command_counts": dict(sorted(command_counts.items())),
            "expected_behavior_counts": dict(sorted(behavior_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "scan_candidate_count": scan_audit[
                "image_dominant_low_text_candidate_count"
            ],
            "eligible_genuine_scanned_table_count": scan_audit[
                "eligible_genuine_scanned_table_count"
            ],
        },
        "prior_question_sources_checked": prior_paths,
        "exact_prior_query_overlap_count": len(query_overlap),
        "corpus_plan_audit": plan_audit.model_dump(mode="json"),
        "prepared_corpus_audit": prepared_audit.model_dump(mode="json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"ready_for_gold_annotation={report['ready_for_gold_annotation']}")
    print(f"ready_for_agent_evaluation={report['ready_for_agent_evaluation']}")
    print(f"checks={sum(checks.values())}/{len(checks)}")
    print(f"blockers={blockers}")
    print(f"report={args.output.resolve()}")
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
