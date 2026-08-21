"""Validate the document-blind Agent hard-v2 candidate before any capability tuning."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from findoc_rag.agent_evaluation import AgentHardDataset, validate_agent_hard_sources

EXPECTED_DOCUMENTS = {
    "cninfo:600519:annual:2024",
    "cninfo:600887:annual:2024",
    "cninfo:000333:annual:2023",
    "cninfo:000333:annual:2024",
    "cninfo:601088:annual:2024",
}
P0_P1_DEVELOPMENT_DOCUMENTS = {
    "cninfo:601088:annual:2023",
    "cninfo:600690:annual:2023",
    "cninfo:600690:annual:2024",
    "cninfo:600900:annual:2023",
    "cninfo:600900:annual:2024",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/agent-hard-v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/validation/agent-hard-v2-candidate.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = AgentHardDataset.model_validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    workspace = Path.cwd()
    source_validation = validate_agent_hard_sources(dataset, workspace=workspace)
    errors = list(source_validation.errors)
    case_ids = [case.case_id for case in dataset.cases]
    if len(case_ids) != len(set(case_ids)):
        errors.append("case IDs are not unique")
    if len(dataset.cases) != 34:
        errors.append(f"expected 34 cases, found {len(dataset.cases)}")

    referenced_documents = {
        source.document_key
        for case in dataset.cases
        for source in case.evidence_sources
    }
    if referenced_documents != EXPECTED_DOCUMENTS:
        errors.append(
            "document-blind source set mismatch: "
            f"{sorted(referenced_documents ^ EXPECTED_DOCUMENTS)}"
        )
    overlap = referenced_documents & P0_P1_DEVELOPMENT_DOCUMENTS
    if overlap:
        errors.append(f"P0/P1 development-document leakage: {sorted(overlap)}")

    local_file_by_document = {
        source.document_key: source.local_file
        for case in dataset.cases
        for source in case.evidence_sources
    }
    page_count_by_document: dict[str, int] = {}
    for document_key, local_file in local_file_by_document.items():
        with pymupdf.open(workspace / local_file) as document:
            page_count_by_document[document_key] = document.page_count

    for case in dataset.cases:
        if case.expected_behavior == "answer" and not case.expected_facts:
            errors.append(f"answer case has no facts: {case.case_id}")
        if case.expected_behavior == "abstain" and case.expected_facts:
            errors.append(f"abstain case has facts: {case.case_id}")
        if case.task_type == "compare" and not case.expected_target_ids:
            errors.append(f"compare case has no expected targets: {case.case_id}")
        if case.task_type != "compare" and case.expected_target_ids:
            errors.append(f"non-compare case has expected targets: {case.case_id}")
        if not case.annotation_status.startswith("assistant_verified_provisional"):
            errors.append(f"case overstates annotation status: {case.case_id}")
        for source in case.evidence_sources:
            page_count = page_count_by_document[source.document_key]
            invalid_pages = [page for page in source.pages if not 1 <= page <= page_count]
            if invalid_pages:
                errors.append(
                    f"invalid evidence pages for {case.case_id}: {invalid_pages}"
                )

    task_counts = Counter(case.task_type for case in dataset.cases)
    behavior_counts = Counter(case.expected_behavior for case in dataset.cases)
    challenge_counts = Counter(
        challenge for case in dataset.cases for challenge in case.challenge_types
    )
    payload = {
        "schema_version": "1",
        "dataset_id": dataset.dataset_id,
        "validated_at": datetime.now(UTC).isoformat(),
        "valid": not errors,
        "ready_for_external_claims": False,
        "external_claim_blockers": [
            "gold is assistant-verified provisional",
            "independent human double review is incomplete",
            "no true scanned annual-report source is present in the five-document set",
        ],
        "case_count": len(dataset.cases),
        "document_count": len(referenced_documents),
        "referenced_documents": sorted(referenced_documents),
        "p0_p1_document_overlap": sorted(overlap),
        "task_counts": dict(sorted(task_counts.items())),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "challenge_counts": dict(sorted(challenge_counts.items())),
        "source_validation": source_validation.model_dump(mode="json"),
        "page_count_by_document": page_count_by_document,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"valid={str(payload['valid']).lower()}")
    print(f"cases={payload['case_count']}")
    print(f"documents={payload['document_count']}")
    print(f"output={args.output.resolve()}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
