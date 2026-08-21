"""Validate hard-v3 provisional gold without running the evaluated Agent."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from findoc_rag.agent_evaluation import AgentHardDataset, validate_agent_hard_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/agent-hard-v3.json"),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/agent-hard-v3-questions.json"),
    )
    parser.add_argument(
        "--spec-dir",
        type=Path,
        default=Path("data/evaluation/agent-hard-v3-gold-specs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/agent/agent-hard-v3-gold-audit.json"),
    )
    return parser.parse_args()


def _check(condition: bool, label: str, blockers: list[str], checks: dict) -> None:
    checks[label] = condition
    if not condition:
        blockers.append(label)


def _expected_behavior(question: dict[str, Any]) -> str:
    if question["expected_response_mode"] == "clarify":
        return "clarify"
    return question["expected_behavior"]


def main() -> int:
    args = parse_args()
    dataset_bytes = args.dataset.read_bytes()
    dataset = AgentHardDataset.model_validate_json(dataset_bytes)
    bank = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = bank["questions"]
    cases = dataset.cases
    blockers: list[str] = []
    checks: dict[str, bool] = {}
    warnings = [
        "Gold remains assistant-source-verified provisional until independent double review.",
        "No DeepSeek/Agent baseline may be called by this validator.",
    ]

    _check(len(cases) == 96, "case_count_is_96", blockers, checks)
    _check(
        [case.case_id for case in cases] == [item["case_id"] for item in questions],
        "case_order_and_membership_match_questions",
        blockers,
        checks,
    )
    _check(
        all(
            case.query == question["query"]
            and case.task_type == question["task_type"]
            and case.challenge_types == question["challenge_types"]
            for case, question in zip(cases, questions, strict=True)
        ),
        "blind_fields_match_frozen_questions",
        blockers,
        checks,
    )
    _check(
        all(
            case.expected_behavior == _expected_behavior(question)
            for case, question in zip(cases, questions, strict=True)
        ),
        "answer_abstain_clarify_mapping_matches",
        blockers,
        checks,
    )
    behavior_counts = Counter(case.expected_behavior for case in cases)
    _check(
        behavior_counts == {"answer": 80, "abstain": 8, "clarify": 8},
        "behavior_distribution_matches",
        blockers,
        checks,
    )
    answer_cases = [case for case in cases if case.expected_behavior == "answer"]
    nonanswer_cases = [case for case in cases if case.expected_behavior != "answer"]
    _check(
        all(case.expected_facts and case.evidence_sources for case in answer_cases),
        "all_answer_cases_have_facts_and_sources",
        blockers,
        checks,
    )
    _check(
        all(not case.expected_facts and not case.evidence_sources for case in nonanswer_cases),
        "nonanswer_cases_do_not_invent_gold_evidence",
        blockers,
        checks,
    )
    _check(
        all(
            len({fact.fact_id for fact in case.expected_facts})
            == len(case.expected_facts)
            for case in cases
        ),
        "fact_ids_are_unique_within_each_case",
        blockers,
        checks,
    )
    _check(
        all(
            case.annotation_status == "assistant_source_verified_provisional"
            for case in cases
        ),
        "annotation_status_is_honestly_provisional",
        blockers,
        checks,
    )

    source_validation = validate_agent_hard_sources(dataset, workspace=Path.cwd())
    _check(
        source_validation.valid,
        "source_manifest_paths_and_sha256_match",
        blockers,
        checks,
    )
    page_counts: dict[str, int] = {}
    manifest = json.loads(
        (Path.cwd() / dataset.source_manifest).read_text(encoding="utf-8")
    )
    for document in manifest["documents"]:
        with pymupdf.open(document["local_file"]) as pdf:
            page_counts[document["document_key"]] = pdf.page_count
    _check(
        all(
            source.pages
            and all(1 <= page <= page_counts[source.document_key] for page in source.pages)
            for case in answer_cases
            for source in case.evidence_sources
        ),
        "all_gold_pages_are_in_pdf_range",
        blockers,
        checks,
    )
    _check(
        all(
            source.document_key.split(":")[1] == case.case_id.split("_")[1]
            for case in answer_cases
            for source in case.evidence_sources
        ),
        "gold_sources_match_case_company",
        blockers,
        checks,
    )
    _check(
        all(
            bool(case.expected_target_ids)
            == (case.task_type == "compare" and case.expected_behavior == "answer")
            for case in cases
        ),
        "plan_target_gold_only_applies_to_answerable_compare_cases",
        blockers,
        checks,
    )

    spec_files = sorted(args.spec_dir.glob("*.json"))
    spec_case_ids: list[str] = []
    spec_review_flags: list[bool] = []
    for path in spec_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec_case_ids.extend(item["case_id"] for item in payload["cases"])
        spec_review_flags.append(payload["independent_double_review_complete"])
    _check(len(spec_files) == 8, "eight_company_spec_files_exist", blockers, checks)
    _check(
        Counter(spec_case_ids) == Counter(case.case_id for case in cases),
        "company_specs_exactly_cover_dataset",
        blockers,
        checks,
    )
    _check(
        not any(spec_review_flags),
        "independent_review_is_not_falsely_claimed",
        blockers,
        checks,
    )

    report = {
        "schema_version": "1",
        "dataset": args.dataset.as_posix(),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "question_payload_sha256": bank["question_payload_sha256"],
        "model_assistance": "none",
        "evaluated_agent_called": False,
        "checks": checks,
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "blockers": blockers,
        "warnings": warnings,
        "statistics": {
            "case_count": len(cases),
            "behavior_counts": dict(sorted(behavior_counts.items())),
            "expected_fact_count": sum(len(case.expected_facts) for case in cases),
            "evidence_source_count": sum(len(case.evidence_sources) for case in cases),
            "evidence_page_reference_count": sum(
                len(source.pages)
                for case in cases
                for source in case.evidence_sources
            ),
        },
        "source_validation": source_validation.model_dump(mode="json"),
        "ready_for_provisional_baseline": not blockers,
        "ready_for_frozen_test_claim": False,
        "frozen_test_claim_blocker": "independent_double_review_pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ready_for_provisional_baseline={not blockers}")
    print("ready_for_frozen_test_claim=False")
    print(f"checks={sum(checks.values())}/{len(checks)}")
    print(f"blockers={json.dumps(blockers, ensure_ascii=False)}")
    print(f"report={args.output.resolve()}")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
