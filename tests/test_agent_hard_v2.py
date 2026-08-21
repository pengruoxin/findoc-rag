from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from findoc_rag.agent_evaluation import AgentHardDataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/evaluation/agent-hard-v2.json"
VALIDATION_PATH = ROOT / "reports/validation/agent-hard-v2-candidate.json"
ANALYSIS_PATH = ROOT / "reports/agent/agent-hard-v2-p2a-failure-analysis.json"
P2B1_SUMMARY_PATH = ROOT / "reports/agent/agent-hard-v2-p2b1-summary.json"

P0_P1_DOCUMENTS = {
    "cninfo:601088:annual:2023",
    "cninfo:600690:annual:2023",
    "cninfo:600690:annual:2024",
    "cninfo:600900:annual:2023",
    "cninfo:600900:annual:2024",
}


def test_agent_hard_v2_is_a_document_holdout_candidate() -> None:
    dataset = AgentHardDataset.model_validate_json(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    documents = {
        source.document_key
        for case in dataset.cases
        for source in case.evidence_sources
    }
    assert len(dataset.cases) == 34
    assert len(documents) == 5
    assert not documents & P0_P1_DOCUMENTS
    assert Counter(case.task_type for case in dataset.cases) == {
        "extract": 26,
        "calculate": 4,
        "compare": 4,
    }
    assert Counter(case.expected_behavior for case in dataset.cases) == {
        "answer": 29,
        "abstain": 5,
    }
    assert all(
        case.annotation_status.startswith("assistant_verified_provisional")
        for case in dataset.cases
    )


def test_agent_hard_v2_validation_keeps_external_claim_gate_closed() -> None:
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))

    assert validation["valid"] is True
    assert validation["ready_for_external_claims"] is False
    assert validation["case_count"] == 34
    assert validation["document_count"] == 5
    assert validation["p0_p1_document_overlap"] == []
    assert validation["source_validation"]["valid"] is True


def test_agent_hard_v2_analysis_preserves_raw_and_audited_scores() -> None:
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))

    assert analysis["valid"] is True
    assert analysis["ready_for_external_claims"] is False
    assert analysis["raw_machine_metrics"]["end_to_end_case_pass_rate"] == 0.5
    audited = analysis["assistant_source_audited_metrics"]
    assert audited["case_pass_count"] == 24
    assert audited["case_count"] == 34
    assert audited["executed_case_pass_count"] == 24
    assert audited["executed_case_count"] == 30
    assert audited["supported_fact_match_count"] == 75
    assert audited["supported_fact_count"] == 80
    summary = analysis["manual_audit_summary"]
    assert summary["reviewed_failure_count"] == 17
    assert summary["verdict_counts"] == {
        "evaluator_false_negative": 7,
        "genuine_agent_failure": 10,
    }


def test_agent_hard_v2_p2b1_summary_records_paired_gain_without_regression() -> None:
    summary = json.loads(P2B1_SUMMARY_PATH.read_text(encoding="utf-8"))

    assert summary["valid"] is True
    assert [run["run_id"] for run in summary["runs"]] == [
        "p2a_baseline",
        "p2b1a_document_fact_period",
        "p2b1b_document_year_syntax",
    ]
    final = summary["runs"][-1]
    assert final["machine_metrics"]["end_to_end_case_pass_rate"] == 21 / 34
    assert final["machine_metrics"]["behavior_accuracy"] == 1.0
    assert final["machine_metrics"]["safe_abstention_accuracy"] == 1.0
    assert final["assistant_source_audited_metrics"]["case_pass_count"] == 28
    assert final["assistant_source_audited_metrics"]["case_count"] == 34
    paired = summary["paired_comparisons"][-1]
    assert paired["net_case_delta"] == 4
    assert paired["regressed_case_ids"] == []
