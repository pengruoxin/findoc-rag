from __future__ import annotations

from findoc_rag.evaluation.reporting import (
    build_generation_quality_report,
    clustered_bootstrap_interval,
    wilson_interval,
)
from findoc_rag.generation_evaluation import (
    DeterministicCaseScore,
    GenerationEvaluationDataset,
    GenerationRunItem,
)


def test_clustered_bootstrap_is_deterministic_and_bounded() -> None:
    observations = [("doc-a", 1.0), ("doc-a", 1.0), ("doc-b", 0.0)]

    first = clustered_bootstrap_interval(observations, samples=200, seed=7)
    second = clustered_bootstrap_interval(observations, samples=200, seed=7)

    assert first == second
    assert first is not None
    assert 0 <= first.low <= first.high <= 1
    assert first.cluster_count == 2


def test_wilson_interval_does_not_treat_small_perfect_sample_as_certain() -> None:
    interval = wilson_interval(12, 12)

    assert interval is not None
    assert interval.low < 1
    assert interval.high == 1


def _dataset() -> GenerationEvaluationDataset:
    common = {
        "family_id": "family",
        "company_ids": ["000001"],
        "company_names": ["测试公司"],
        "company_aliases": ["测试公司"],
        "report_years": [2024],
        "difficulty": "easy",
        "answerability": "answerable",
        "reference_answer": "收入为1元[1]。",
        "expected_facts": [
            {
                "fact_id": "revenue",
                "description": "收入",
                "subject": "测试公司",
                "predicate": "营业收入",
                "canonical_value": "1",
                "value_type": "number",
                "acceptable_values": ["1"],
                "unit": "元",
                "period": "FY2024",
                "scope": "summary",
                "evidence_chunk_ids": ["chunk-1"],
            }
        ],
        "gold_chunk_ids": ["chunk-1"],
        "gold_evidence": [
            {
                "evidence_id": "e1",
                "chunk_id": "chunk-1",
                "document_version_id": "doc-a",
                "page_start": 1,
                "page_end": 1,
                "section_path": ["summary"],
                "verbatim_quote": "收入为1元",
                "supports_fact_ids": ["revenue"],
            }
        ],
        "answer_contract": {
            "expected_behavior": "answer",
            "required_format": "short",
        },
        "required_citation_count": 1,
        "annotation": {
            "created_by": "assistant_curated",
            "review_status": "assistant_verified",
            "confidence": "high",
            "source_pdf_sha256": [],
        },
    }
    items = [
        {**common, "query_id": "q1", "split": "dev", "query": "问题1", "category": "single_fact"},
        {
            **common,
            "query_id": "q2",
            "split": "frozen_test",
            "query": "问题2",
            "category": "calculation",
        },
    ]
    return GenerationEvaluationDataset.model_validate(
        {
            "dataset_id": "test-v1",
            "corpus_index_id": "index-1",
            "reviewer": "test",
            "status": "assistant_curated_provisional",
            "tracks": ["retrieved_context"],
            "item_count": 2,
            "items": items,
        }
    )


def _run(query_id: str, provider: str, *, error: str | None = None) -> GenerationRunItem:
    return GenerationRunItem(
        query_id=query_id,
        response="收入为1元[1]。" if error is None else "",
        retrieved_contexts=["收入为1元"] if error is None else [],
        retrieved_chunk_ids=["chunk-1"] if error is None else [],
        provider=provider,
        model="test",
        index_id="index-1",
        prompt_sha256="0" * 64,
        latency_ms=1,
        grounded=error is None,
        observed_behavior="answer" if error is None else None,
        filter_source="query_router",
        error=error,
    )


def _score(query_id: str, success: bool) -> DeterministicCaseScore:
    return DeterministicCaseScore(
        query_id=query_id,
        expected_behavior_correct=success,
        strict_success=success,
        gold_fact_recall=1 if success else 0,
        numeric_accuracy=1 if success else 0,
        unit_accuracy=1 if success else 0,
        context_recall=1 if success else 0,
        citation_validity=1 if success else 0,
        false_abstention=not success,
    )


def test_quality_report_stratifies_provider_route_and_split() -> None:
    report = build_generation_quality_report(
        _dataset(),
        [_run("q1", "deterministic-table"), _run("q2", "error", error="timeout")],
        [_score("q1", True), _score("q2", False)],
    )

    assert report["overall"]["provider_counts"] == {
        "deterministic-table": 1,
        "error": 1,
    }
    assert report["overall"]["strict_success"]["rate"] == 0.5
    assert report["overall"]["strict_success"]["wilson_ci"]["low"] < 0.5
    assert report["overall"]["run_error_rate"]["rate"] == 0.5
    assert set(report["strata"]["provider"]) == {"deterministic-table", "error"}
    assert set(report["strata"]["split"]) == {"dev", "frozen_test"}
    assert set(report["strata"]["filter_source"]) == {"query_router"}
