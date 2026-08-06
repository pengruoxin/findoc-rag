import json
from pathlib import Path

import pytest

from findoc_rag.generation_evaluation import (
    GenerationEvaluationDataset,
    GenerationRunItem,
    select_ragas_run_items,
)
from scripts.run_ragas_generation_eval import load_and_validate_run_manifest

ROOT = Path(__file__).resolve().parents[1]


def _dataset() -> GenerationEvaluationDataset:
    return GenerationEvaluationDataset.model_validate_json(
        (ROOT / "data/evaluation/generation-eval-v1.json").read_text(encoding="utf-8")
    )


def _valid_run_items(
    dataset: GenerationEvaluationDataset,
    lane: str,
) -> list[GenerationRunItem]:
    cases = (
        [item for item in dataset.items if item.hard_negatives]
        if lane == "robustness"
        else dataset.items
    )
    output = []
    for item in cases:
        behavior = item.answer_contract.expected_behavior
        answered = behavior == "answer"
        output.append(
            GenerationRunItem(
                query_id=item.query_id,
                response="有证据支持的测试回答[1]。" if answered else "测试拒答或澄清。",
                retrieved_contexts=["测试上下文"] if answered else [],
                retrieved_chunk_ids=["test-chunk"] if answered else [],
                context_labels=["gold"] if answered else [],
                provider="test",
                model="test-model",
                index_id=dataset.corpus_index_id,
                prompt_sha256="0" * 64,
                latency_ms=1,
                grounded=answered,
                observed_behavior=behavior,
            )
        )
    return output


def test_oracle_ragas_selection_requires_and_reports_full_coverage() -> None:
    dataset = _dataset()
    eligible, audit = select_ragas_run_items(
        dataset, _valid_run_items(dataset, "oracle_context"), "oracle_context"
    )

    assert len(eligible) == 37
    assert audit.scope_policy == "full_dataset"
    assert audit.run_item_count == dataset.item_count == 48
    assert audit.dataset_answerable_count == audit.eligible_count == 37
    assert audit.coverage == 1
    assert audit.lane_coverage == 1
    assert len(audit.excluded_non_answerable_query_ids) == 11


def test_robustness_ragas_selection_uses_exact_hard_negative_subset() -> None:
    dataset = _dataset()
    eligible, audit = select_ragas_run_items(
        dataset, _valid_run_items(dataset, "robustness"), "robustness"
    )

    assert len(eligible) == audit.eligible_count == 18
    assert audit.scope_policy == "hard_negative_subset"
    assert audit.lane_query_count == audit.run_item_count == 29
    assert audit.dataset_answerable_count == 37
    assert audit.coverage == pytest.approx(18 / 37)
    assert audit.lane_coverage == 1
    assert audit.lane_query_ids == [
        item.query_id for item in dataset.items if item.hard_negatives
    ]


def test_ragas_selection_rejects_duplicate_and_unknown_query_ids() -> None:
    dataset = _dataset()
    run_items = _valid_run_items(dataset, "oracle_context")

    with pytest.raises(ValueError, match="duplicate query IDs"):
        select_ragas_run_items(dataset, [*run_items, run_items[0]], "oracle_context")

    unknown = run_items[0].model_copy(update={"query_id": "unknown-query"})
    with pytest.raises(ValueError, match="absent from the dataset"):
        select_ragas_run_items(dataset, [*run_items[1:], unknown], "oracle_context")


def test_ragas_selection_enforces_lane_scope_and_audits_behavior_mismatches() -> None:
    dataset = _dataset()
    full_run = _valid_run_items(dataset, "retrieved_context")
    with pytest.raises(ValueError, match="query scope does not match"):
        select_ragas_run_items(dataset, full_run[:-1], "retrieved_context")

    robustness_run = _valid_run_items(dataset, "robustness")
    outside = next(item for item in full_run if item.query_id not in {x.query_id for x in robustness_run})
    with pytest.raises(ValueError, match="out_of_lane"):
        select_ragas_run_items(dataset, [*robustness_run, outside], "robustness")

    answer_position = next(
        index
        for index, item in enumerate(dataset.items)
        if item.answerability == "answerable"
    )
    mismatch = full_run[answer_position].model_copy(
        update={"observed_behavior": "abstain", "grounded": False}
    )
    mismatched_run = list(full_run)
    mismatched_run[answer_position] = mismatch
    eligible, audit = select_ragas_run_items(
        dataset, mismatched_run, "retrieved_context"
    )
    assert len(eligible) == audit.dataset_answerable_count == 37
    assert audit.eligible_count == 37
    assert audit.coverage == 1
    assert audit.behavior_mismatch_query_ids == [mismatch.query_id]


def test_ragas_selection_rejects_run_errors_and_wrong_index() -> None:
    dataset = _dataset()
    run_items = _valid_run_items(dataset, "oracle_context")

    failed = list(run_items)
    failed[0] = failed[0].model_copy(update={"error": "backend timeout"})
    with pytest.raises(ValueError, match="failed items"):
        select_ragas_run_items(dataset, failed, "oracle_context")

    wrong_index = list(run_items)
    wrong_index[0] = wrong_index[0].model_copy(update={"index_id": "stale-index"})
    with pytest.raises(ValueError, match="other than the dataset corpus index"):
        select_ragas_run_items(dataset, wrong_index, "oracle_context")

    inconsistent = list(run_items)
    inconsistent[0] = inconsistent[0].model_copy(update={"grounded": False})
    with pytest.raises(ValueError, match="inconsistent with grounded"):
        select_ragas_run_items(dataset, inconsistent, "oracle_context")


def test_run_manifest_is_bound_to_dataset_lane_and_jsonl(tmp_path: Path) -> None:
    dataset = _dataset()
    run_items = _valid_run_items(dataset, "robustness")
    run_dir = tmp_path / "robustness-test-run"
    run_dir.mkdir()
    run_jsonl = run_dir / "items.jsonl"
    summary = {
        "run_id": run_dir.name,
        "dataset_id": dataset.dataset_id,
        "lane": "robustness",
        "dataset_item_count": dataset.item_count,
        "item_count": len(run_items),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )

    loaded, lane = load_and_validate_run_manifest(run_jsonl, dataset, run_items)
    assert loaded["run_id"] == run_dir.name
    assert lane == "robustness"

    summary["dataset_id"] = "stale-dataset"
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="dataset_id does not match"):
        load_and_validate_run_manifest(run_jsonl, dataset, run_items)
