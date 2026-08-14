from pathlib import Path

import pytest
from pydantic import ValidationError

from findoc_rag.benchmark_assets import benchmark_chunk_paths, validate_benchmark_lock
from findoc_rag.generation_evaluation import (
    GenerationEvaluationDataset,
    GenerationJudgment,
    GenerationRunItem,
    aggregate_generation_judgments,
    score_generation_run_item,
    to_ragas_oracle_rows,
    validate_generation_dataset,
)


def test_generation_metrics_are_aggregated() -> None:
    item = GenerationJudgment(
        query_id="q1",
        answer="营业收入为测试值[1]。",
        faithfulness=1,
        answer_relevancy=1,
        context_relevancy=0.8,
        context_recall=0.9,
        source="human",
        rationale="答案由引用证据直接支持。",
        citation_ids=["c1"],
    )
    result = aggregate_generation_judgments([item])
    assert result.faithfulness == 1
    assert result.context_recall == 0.9


def test_llm_judge_requires_model_provenance() -> None:
    with pytest.raises(ValidationError):
        GenerationJudgment(
            query_id="q1",
            answer="测试答案",
            faithfulness=1,
            answer_relevancy=1,
            context_relevancy=1,
            context_recall=1,
            source="llm_judge",
            rationale="测试",
        )


def test_generation_dataset_is_source_verifiable() -> None:
    root = Path(__file__).resolve().parents[1]
    validate_benchmark_lock(root)
    report = validate_generation_dataset(
        root / "data/evaluation/generation-eval-v1.json",
        [root / "data/evaluation/benchmark-evidence-v1.jsonl"],
    )
    assert report.item_count == 48
    assert report.answerability_counts == {
        "answerable": 37,
        "unanswerable": 9,
        "needs_clarification": 2,
    }
    assert report.fact_count == 120
    assert report.split_counts == {
        "calibration": 12,
        "dev": 24,
        "frozen_test": 12,
    }
    assert report.family_count == 40
    assert report.unique_gold_chunk_count == 35
    assert report.robustness_item_count == 29
    assert report.robustness_split_counts["frozen_test"] == 12
    assert report.hard_negative_count == 53
    assert report.warning_count == 0


def test_clean_clone_benchmark_assets_are_self_contained() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = benchmark_chunk_paths(root)
    assert root / "data/evaluation/benchmark-evidence-v1.jsonl" in paths
    evidence_only = [path for path in paths if path.name == "benchmark-evidence-v1.jsonl"]
    report = validate_generation_dataset(
        root / "data/evaluation/generation-eval-v1.json", evidence_only
    )
    assert report.unique_gold_chunk_count == 35
    assert report.hard_negative_count == 53


def test_ragas_export_excludes_abstention_cases() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = GenerationEvaluationDataset.model_validate_json(
        (root / "data/evaluation/generation-eval-v1.json").read_text(encoding="utf-8")
    )
    rows = to_ragas_oracle_rows(dataset)
    assert len(rows) == 37
    assert all(row["reference_contexts"] for row in rows)


def test_deterministic_score_normalizes_financial_numbers() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = GenerationEvaluationDataset.model_validate_json(
        (root / "data/evaluation/generation-eval-v1.json").read_text(encoding="utf-8")
    )
    item = next(case for case in dataset.items if case.query_id == "moutai_revenue_yoy")
    run = GenerationRunItem(
        query_id=item.query_id,
        response="营业收入为170,899,152,276.34元，同比增长15.71%[1]。",
        retrieved_contexts=[item.gold_evidence[0].verbatim_quote],
        retrieved_chunk_ids=[item.gold_chunk_ids[0]],
        provider="test",
        model="test",
        index_id=dataset.corpus_index_id,
        prompt_sha256="0" * 64,
        latency_ms=1,
        grounded=True,
    )
    score = score_generation_run_item(item, run)
    assert score.strict_success
    assert score.numeric_accuracy == 1
    assert score.context_recall == 1


def test_unanswerable_case_scores_correct_abstention() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = GenerationEvaluationDataset.model_validate_json(
        (root / "data/evaluation/generation-eval-v1.json").read_text(encoding="utf-8")
    )
    item = next(case for case in dataset.items if case.query_id == "u_moutai_2025_actual_revenue")
    run = GenerationRunItem(
        query_id=item.query_id,
        response="当前证据不足，无法回答。",
        retrieved_contexts=[],
        retrieved_chunk_ids=[],
        provider="abstention",
        model="none",
        index_id=dataset.corpus_index_id,
        prompt_sha256="0" * 64,
        latency_ms=1,
        grounded=False,
    )
    assert score_generation_run_item(item, run).strict_success


@pytest.mark.parametrize(
    ("query_id", "observed_behavior"),
    [
        ("u_moutai_2025_actual_revenue", "abstain"),
        ("u_moutai_profit_ambiguous", "clarify"),
        ("moutai_revenue_yoy", "answer"),
    ],
)
def test_run_error_never_receives_behavior_or_strict_success(
    query_id: str,
    observed_behavior: str,
) -> None:
    """Abstain, clarify, and answer contracts all fail on infrastructure errors."""
    root = Path(__file__).resolve().parents[1]
    dataset = GenerationEvaluationDataset.model_validate_json(
        (root / "data/evaluation/benchmark-v2.json").read_text(encoding="utf-8")
    )
    item = next(case for case in dataset.items if case.query_id == query_id)
    run = GenerationRunItem(
        query_id=item.query_id,
        response="",
        retrieved_contexts=[],
        retrieved_chunk_ids=[],
        provider="error",
        model="test",
        index_id=dataset.corpus_index_id,
        prompt_sha256="0" * 64,
        latency_ms=1,
        grounded=False,
        observed_behavior=observed_behavior,
        error="timeout",
    )

    score = score_generation_run_item(item, run)

    assert not score.expected_behavior_correct
    assert not score.strict_success


def test_strict_success_requires_all_required_gold_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = GenerationEvaluationDataset.model_validate_json(
        (root / "data/evaluation/benchmark-v2.json").read_text(encoding="utf-8")
    )
    item = next(case for case in dataset.items if case.query_id == "revenue_cross_company")
    run = GenerationRunItem(
        query_id=item.query_id,
        response=item.reference_answer,
        retrieved_contexts=["无关上下文一。", "无关上下文二。"],
        retrieved_chunk_ids=["unrelated:chunk:1", "unrelated:chunk:2"],
        provider="test",
        model="test",
        index_id=dataset.corpus_index_id,
        prompt_sha256="0" * 64,
        latency_ms=1,
        grounded=True,
        observed_behavior="answer",
    )

    score = score_generation_run_item(item, run)

    assert score.gold_fact_recall == 1
    assert score.context_recall == 0
    assert not score.strict_success


def test_numeric_facts_are_bound_to_their_companies() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = GenerationEvaluationDataset.model_validate_json(
        (root / "data/evaluation/benchmark-v2.json").read_text(encoding="utf-8")
    )
    item = next(case for case in dataset.items if case.query_id == "revenue_cross_company")
    run = GenerationRunItem(
        query_id=item.query_id,
        response=(
            "贵州茅台为115,393,310,976.69元[1]，"
            "伊利股份为170,899,152,276.34元[2]；"
            "伊利股份高55,505,841,299.65元。"
        ),
        retrieved_contexts=[evidence.verbatim_quote for evidence in item.gold_evidence],
        retrieved_chunk_ids=item.gold_chunk_ids,
        provider="test",
        model="test",
        index_id=dataset.corpus_index_id,
        prompt_sha256="0" * 64,
        latency_ms=1,
        grounded=True,
        observed_behavior="answer",
    )

    score = score_generation_run_item(item, run)

    assert score.context_recall == 1
    assert score.numeric_accuracy < 1
    assert not score.strict_success


def test_numeric_binding_does_not_cross_chinese_clause_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = GenerationEvaluationDataset.model_validate_json(
        (root / "data/evaluation/benchmark-v2.json").read_text(encoding="utf-8")
    )
    item = next(case for case in dataset.items if case.query_id == "rd_cross_company")
    run = GenerationRunItem(
        query_id=item.query_id,
        response=(
            "贵州茅台2024年研发投入合计为695,376,735.81元[1]，"
            "伊利股份2024年研发投入合计为869,976,531.80元[2]。"
            "伊利股份更高，差额为174,599,795.99元。"
        ),
        retrieved_contexts=[evidence.verbatim_quote for evidence in item.gold_evidence],
        retrieved_chunk_ids=item.gold_chunk_ids,
        provider="test",
        model="test",
        index_id=dataset.corpus_index_id,
        prompt_sha256="0" * 64,
        latency_ms=1,
        grounded=True,
        observed_behavior="answer",
    )

    score = score_generation_run_item(item, run)

    assert score.numeric_accuracy == 1
    assert score.strict_success


def test_citation_contract_uses_unique_gold_contexts() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = GenerationEvaluationDataset.model_validate_json(
        (root / "data/evaluation/generation-eval-v1.json").read_text(encoding="utf-8")
    )
    for item in dataset.items:
        if item.answerability == "answerable":
            assert item.required_citation_count == len(set(item.gold_chunk_ids))


def test_quarterly_reconciliation_reference_covers_its_full_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = GenerationEvaluationDataset.model_validate_json(
        (root / "data/evaluation/generation-eval-v1.json").read_text(encoding="utf-8")
    )
    item = next(
        case for case in dataset.items if case.query_id == "yili_quarterly_profit_reconcile"
    )
    run = GenerationRunItem(
        query_id=item.query_id,
        response=item.reference_answer,
        retrieved_contexts=["gold context"] * len(item.gold_chunk_ids),
        retrieved_chunk_ids=item.gold_chunk_ids,
        context_labels=["gold"] * len(item.gold_chunk_ids),
        provider="reference-self-check",
        model="deterministic",
        index_id=dataset.corpus_index_id,
        prompt_sha256="0" * 64,
        latency_ms=1,
        grounded=True,
        observed_behavior="answer",
    )
    score = score_generation_run_item(item, run)
    assert score.gold_fact_recall == 1
    assert score.context_recall == 1
    assert score.strict_success


@pytest.mark.parametrize("unit_style", ["natural", "slash"])
def test_per_share_unit_accepts_equivalent_word_orders(unit_style: str) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = GenerationEvaluationDataset.model_validate_json(
        (root / "data/evaluation/generation-eval-v1.json").read_text(encoding="utf-8")
    )
    item = next(case for case in dataset.items if case.query_id == "yili_dividend_timing")
    response = item.reference_answer
    if unit_style == "slash":
        response = response.replace("每股1.20元", "1.20元/股").replace(
            "每股1.22元", "1.22元／股"
        )
    run = GenerationRunItem(
        query_id=item.query_id,
        response=response,
        retrieved_contexts=["gold context"] * len(item.gold_chunk_ids),
        retrieved_chunk_ids=item.gold_chunk_ids,
        context_labels=["gold"] * len(item.gold_chunk_ids),
        provider="reference-self-check",
        model="deterministic",
        index_id=dataset.corpus_index_id,
        prompt_sha256="0" * 64,
        latency_ms=1,
        grounded=True,
        observed_behavior="answer",
    )

    assert score_generation_run_item(item, run).unit_accuracy == 1


def test_yuan_unit_does_not_match_ten_thousand_yuan() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = GenerationEvaluationDataset.model_validate_json(
        (root / "data/evaluation/generation-eval-v1.json").read_text(encoding="utf-8")
    )
    item = next(case for case in dataset.items if case.query_id == "moutai_revenue_yoy")
    run = GenerationRunItem(
        query_id=item.query_id,
        response="营业收入为170,899,152,276.34万元，同比增长15.71%[1]。",
        retrieved_contexts=["gold context"],
        retrieved_chunk_ids=item.gold_chunk_ids,
        context_labels=["gold"],
        provider="test",
        model="deterministic",
        index_id=dataset.corpus_index_id,
        prompt_sha256="0" * 64,
        latency_ms=1,
        grounded=True,
        observed_behavior="answer",
    )
    score = score_generation_run_item(item, run)

    assert score.numeric_accuracy == 1
    assert score.unit_accuracy == 0.5
    assert not score.strict_success
