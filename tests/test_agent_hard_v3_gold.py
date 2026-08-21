from pathlib import Path

from scripts.build_agent_hard_v3_gold import GOLD, build_dataset

QUESTIONS = Path("data/evaluation/agent-hard-v3-questions.json")
MANIFEST = Path("data/evaluation/agent-hard-v3-source-manifest.json")


def test_hard_v3_provisional_gold_has_full_behavior_coverage() -> None:
    dataset = build_dataset(QUESTIONS, MANIFEST)

    counts = {
        behavior: sum(
            case.expected_behavior == behavior for case in dataset.cases
        )
        for behavior in ("answer", "abstain", "clarify")
    }

    assert len(GOLD) == 80
    assert len(dataset.cases) == 96
    assert counts == {"answer": 80, "abstain": 8, "clarify": 8}


def test_hard_v3_visual_inequality_is_not_flattened_to_an_exact_value() -> None:
    dataset = build_dataset(QUESTIONS, MANIFEST)
    by_id = {case.case_id: case for case in dataset.cases}
    ratio_case = by_id["v3_002352_calc_vehicle_rail_ratio"]
    ratio_fact = next(
        fact for fact in ratio_case.expected_facts if fact.fact_id == "ratio"
    )

    assert ratio_case.evidence_sources[0].pages == [33]
    assert ">416.67" in ratio_fact.acceptable_values
    assert "416.67" not in ratio_fact.acceptable_values


def test_hard_v3_ambiguous_scope_requires_clarification() -> None:
    dataset = build_dataset(QUESTIONS, MANIFEST)
    case = next(
        case
        for case in dataset.cases
        if case.case_id == "v3_000063_ambiguous_scope"
    )

    assert case.expected_behavior == "clarify"
    assert case.expected_facts == []
    assert case.evidence_sources == []
