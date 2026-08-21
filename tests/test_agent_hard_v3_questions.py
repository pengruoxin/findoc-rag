import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "evaluation" / "agent-hard-v3-questions.json"
REVIEW = ROOT / "data" / "evaluation" / "agent-hard-v3-question-review.json"


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


def test_hard_v3_question_bank_is_frozen_and_balanced() -> None:
    dataset = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = dataset["questions"]

    assert dataset["status"] == "questions_frozen_gold_pending"
    assert len(questions) == 96
    assert len({item["case_id"] for item in questions}) == 96
    assert len({item["query"] for item in questions}) == 96
    assert len({item["company_ids"][0] for item in questions}) == 8
    assert Counter(item["task_type"] for item in questions) == {
        "extract": 56,
        "compare": 24,
        "calculate": 16,
    }
    assert Counter(item["expected_behavior"] for item in questions) == {
        "answer": 80,
        "abstain": 16,
    }
    assert Counter(item["split"] for item in questions) == {
        "calibration": 24,
        "dev": 24,
        "frozen_test": 48,
    }
    assert _question_hash(questions) == dataset["question_payload_sha256"]


def test_hard_v3_review_packet_matches_questions_without_frozen_gold() -> None:
    dataset = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))

    assert review["question_payload_sha256"] == dataset["question_payload_sha256"]
    assert review["independent_gold"] is False
    assert {item["case_id"] for item in review["items"]} == {
        item["case_id"] for item in dataset["questions"]
    }
    assert Counter(item["structural_status"] for item in review["items"]) == {
        "probe_pages_found": 79,
        "visual_page_confirmed": 1,
        "not_applicable": 16,
    }
    assert all(item["annotation_status"] == "gold_pending" for item in review["items"])


def test_hard_v3_blind_question_file_contains_no_reference_answers() -> None:
    raw = QUESTIONS.read_text(encoding="utf-8")

    for forbidden in (
        '"expected_facts"',
        '"acceptable_values"',
        '"gold_rationale"',
        '"evidence_sources"',
    ):
        assert forbidden not in raw
