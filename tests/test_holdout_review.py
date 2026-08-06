import json
from pathlib import Path

from findoc_rag.holdout import AssistantReviewSet, load_holdout_eval


def test_assistant_review_set_is_provisional_and_complete() -> None:
    path = Path(__file__).parents[1] / "data/diagnostics/holdout-assistant-review-v1.json"
    review = AssistantReviewSet.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert review.reviewer == "assistant"
    assert review.independent_gold is False
    assert len(review.items) == 16
    assert len(review.benchmark_items()) == 16
    assert review.validate_chunk_ids({"5299f4940e2c:c8:81a5543345db6ac2"})
    assert review.validate_chunk_ids({item.gold_chunk_ids[0] for item in review.items}) == {}
    assert len(load_holdout_eval(path.parent / "holdout-eval-v2.json")) == 16
