from __future__ import annotations

from findoc_rag.evaluation.governance import (
    BenchmarkCorpusPlan,
    GovernancePolicy,
    audit_benchmark_governance,
    audit_corpus_plan,
    audit_prepared_corpus,
)


def _review(reviewer_id: str) -> dict:
    return {
        "reviewer_id": reviewer_id,
        "verdict": "approve",
        "query_semantics_verified": True,
        "evidence_verified": True,
        "reference_answer_verified": True,
        "variants_verified": True,
    }


def _item(query_id: str, split: str, document_id: str, company: str, year: int) -> dict:
    return {
        "query_id": query_id,
        "family_id": query_id,
        "split": split,
        "answerability": "answerable",
        "company_ids": [company],
        "report_years": [year],
        "gold_evidence": [{"document_version_id": document_id}],
        "annotation": {
            "review_status": "human_verified",
            "human_reviews": [_review("reviewer-a"), _review("reviewer-b")],
        },
    }


def _policy(**updates) -> GovernancePolicy:
    payload = {
        "policy_id": "test-policy",
        "split_strategy": "document_blind",
        "document_blind_required": True,
        "required_human_reviewers": 2,
        "minimum_companies": 2,
        "minimum_report_years": 2,
        "minimum_frozen_test_items": 1,
        "require_variant_review": True,
    }
    payload.update(updates)
    return GovernancePolicy.model_validate(payload)


def test_document_blind_double_reviewed_benchmark_is_ready() -> None:
    benchmark = {
        "dataset_id": "blind-v1",
        "independent_gold": True,
        "status": "human_frozen",
        "items": [
            _item("q-dev", "dev", "doc-a", "000001", 2023),
            _item("q-test", "frozen_test", "doc-b", "000002", 2024),
        ],
    }

    report = audit_benchmark_governance(benchmark, _policy())

    assert report.ready_for_external_claims
    assert report.stats.fully_reviewed_item_count == 2
    assert not report.document_split_overlaps


def test_document_and_company_year_leakage_are_blockers() -> None:
    benchmark = {
        "dataset_id": "leaky-v1",
        "independent_gold": True,
        "status": "human_frozen",
        "items": [
            _item("q-dev", "dev", "doc-a", "000001", 2024),
            _item("q-test", "frozen_test", "doc-a", "000001", 2024),
        ],
    }

    report = audit_benchmark_governance(
        benchmark,
        _policy(minimum_companies=1, minimum_report_years=1),
    )

    assert not report.ready_for_external_claims
    assert report.document_split_overlaps[0].unit_id == "doc-a"
    assert report.entity_period_split_overlaps[0].unit_id == "000001:2024"


def test_legacy_human_verified_label_does_not_replace_independent_reviews() -> None:
    item = _item("q-test", "frozen_test", "doc-a", "000001", 2024)
    item["annotation"].pop("human_reviews")
    benchmark = {
        "dataset_id": "legacy-review-v1",
        "independent_gold": True,
        "status": "human_frozen",
        "items": [item],
    }

    report = audit_benchmark_governance(
        benchmark,
        _policy(minimum_companies=1, minimum_report_years=1),
    )

    assert not report.ready_for_external_claims
    assert report.stats.fully_reviewed_item_count == 0
    assert any("independent approvals" in blocker for blocker in report.blockers)


def test_company_blind_corpus_plan_is_ready_before_acquisition() -> None:
    documents = []
    for split, company in (
        ("calibration", "000001"),
        ("dev", "000002"),
        ("frozen_test", "000003"),
    ):
        for year in (2023, 2024):
            documents.append(
                {
                    "document_key": f"cninfo:{company}:annual:{year}",
                    "security_code": company,
                    "company_name": company,
                    "report_year": year,
                    "split": split,
                    "question_target": 12 if split == "frozen_test" else 1,
                }
            )
    plan = BenchmarkCorpusPlan.model_validate(
        {
            "plan_id": "test-plan",
            "sealed_at": "2026-08-19T00:00:00+08:00",
            "selection_protocol": "allocated_before_question_authoring_or_rule_tuning",
            "documents": documents,
        }
    )

    report = audit_corpus_plan(
        plan,
        _policy(minimum_companies=3, minimum_frozen_test_items=24),
    )

    assert report.ready_for_acquisition
    assert report.stats.split_question_targets["frozen_test"] == 24


def test_corpus_plan_rejects_company_leakage() -> None:
    plan = BenchmarkCorpusPlan.model_validate(
        {
            "plan_id": "leaky-plan",
            "sealed_at": "2026-08-19T00:00:00+08:00",
            "selection_protocol": "allocated_before_question_authoring_or_rule_tuning",
            "documents": [
                {
                    "document_key": "cninfo:000001:annual:2023",
                    "security_code": "000001",
                    "company_name": "A",
                    "report_year": 2023,
                    "split": "calibration",
                },
                {
                    "document_key": "cninfo:000001:annual:2024",
                    "security_code": "000001",
                    "company_name": "A",
                    "report_year": 2024,
                    "split": "frozen_test",
                    "question_target": 24,
                },
                {
                    "document_key": "cninfo:000002:annual:2024",
                    "security_code": "000002",
                    "company_name": "B",
                    "report_year": 2024,
                    "split": "dev",
                },
            ],
        }
    )

    report = audit_corpus_plan(
        plan,
        _policy(minimum_companies=2, minimum_frozen_test_items=24),
    )

    assert not report.ready_for_acquisition
    assert report.company_split_overlaps[0].unit_id == "000001"


def test_prepared_corpus_audit_enforces_split_index_membership() -> None:
    plan = BenchmarkCorpusPlan.model_validate(
        {
            "plan_id": "prepared-plan",
            "sealed_at": "2026-08-19T00:00:00+08:00",
            "selection_protocol": "allocated_before_question_authoring_or_rule_tuning",
            "documents": [
                {
                    "document_key": "cninfo:000001:annual:2024",
                    "security_code": "000001",
                    "company_name": "A",
                    "report_year": 2024,
                    "split": "calibration",
                },
                {
                    "document_key": "cninfo:000002:annual:2024",
                    "security_code": "000002",
                    "company_name": "B",
                    "report_year": 2024,
                    "split": "dev",
                },
                {
                    "document_key": "cninfo:000003:annual:2023",
                    "security_code": "000003",
                    "company_name": "C",
                    "report_year": 2023,
                    "split": "frozen_test",
                },
            ],
        }
    )
    source = {
        "plan_id": "prepared-plan",
        "documents": [
            {
                **item.model_dump(mode="json"),
                "sha256": f"hash-{item.security_code}",
            }
            for item in plan.documents
        ],
    }
    versions = {
        "plan_id": "prepared-plan",
        "documents": [
            {
                "document_key": item.document_key,
                "split": item.split,
                "content_sha256": f"hash-{item.security_code}",
                "version_id": f"version-{item.security_code}",
                "status": "active",
            }
            for item in plan.documents
        ],
        "indexes": {
            "calibration": {
                "index_id": "cal",
                "active_version_ids": ["version-000001"],
            },
            "development": {
                "index_id": "dev",
                "active_version_ids": ["version-000001", "version-000002"],
            },
            "frozen_test": {
                "index_id": "test",
                "active_version_ids": ["version-000003"],
            },
        },
    }

    ready = audit_prepared_corpus(plan, source, versions)
    versions["indexes"]["development"]["active_version_ids"].append("version-000003")
    leaked = audit_prepared_corpus(plan, source, versions)

    assert ready.ready_for_annotation
    assert not leaked.ready_for_annotation
    assert any("split isolation" in blocker for blocker in leaked.blockers)
