import hashlib
import json
from pathlib import Path

from findoc_rag.agent_evaluation import (
    AgentHardCase,
    AgentHardDataset,
    AgentHardExpectedFact,
    diagnose_agent_requirements,
    score_agent_hard_case,
    validate_agent_hard_sources,
)
from findoc_rag.agent_tasks import (
    AgentTaskRequest,
    AtomicFactRequirement,
    CompareTaskController,
)
from findoc_rag.answer_generation import ClaimCitation, GeneratedAnswer
from tests.test_agent_tasks import EvidenceOnlyGenerator, FakeSearchBackend, _hit


def _case() -> AgentHardCase:
    return AgentHardCase.model_validate(
        {
            "case_id": "hard-test",
            "task_type": "compare",
            "query": "比较甲公司和乙公司2024年营业收入",
            "challenge_types": ["target_completeness"],
            "expected_behavior": "answer",
            "expected_target_ids": [
                "company:甲公司:year:2024",
                "company:乙公司:year:2024",
            ],
            "expected_facts": [
                {
                    "fact_id": "revenue",
                    "description": "营业收入",
                    "acceptable_values": ["100"],
                    "unit": "元",
                }
            ],
            "evidence_sources": [
                {
                    "document_key": "cninfo:test:annual:2024",
                    "local_file": "test.pdf",
                    "pages": [1],
                }
            ],
            "gold_rationale": "test",
            "annotation_status": "test",
        }
    )


def test_hard_case_scorer_does_not_credit_evidence_only_as_an_answer() -> None:
    backend = FakeSearchBackend(
        {"甲公司": [_hit("甲公司", "chunk-a")], "乙公司": [_hit("乙公司", "chunk-b")]}
    )
    trace = CompareTaskController(
        backend,
        EvidenceOnlyGenerator(),
        available_companies=["甲公司", "乙公司"],
    ).run(AgentTaskRequest(query=_case().query, top_k=1))

    score = score_agent_hard_case(_case(), trace)

    assert score.plan_target_exact is True
    assert score.behavior_correct is False
    assert score.fact_accuracy == 0.0
    assert score.case_pass is False


def test_hard_dataset_sources_are_bound_to_manifest_sha256(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"exact-pdf-bytes")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    manifest = {
        "documents": [
            {
                "document_key": "cninfo:test:annual:2024",
                "local_file": "source.pdf",
                "sha256": digest,
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    case = _case().model_dump(mode="json")
    case["evidence_sources"][0]["local_file"] = "source.pdf"
    dataset = AgentHardDataset.model_validate(
        {
            "dataset_id": "source-test",
            "description": "test",
            "index_scope": "test",
            "source_manifest": "manifest.json",
            "gold_policy": "test",
            "cases": [case],
        }
    )

    validation = validate_agent_hard_sources(dataset, workspace=tmp_path)

    assert validation.valid is True
    assert validation.verified_document_count == 1


def test_budget_exhaustion_is_not_credited_as_a_calibrated_abstention() -> None:
    backend = FakeSearchBackend({"甲公司": [], "乙公司": []})
    trace = CompareTaskController(
        backend,
        EvidenceOnlyGenerator(),
        available_companies=["甲公司", "乙公司"],
    ).run(AgentTaskRequest(query=_case().query, top_k=1))
    budget_trace = trace.model_copy(
        update={
            "stop_reason": "model_budget_exhausted",
            "result": trace.result.model_copy(
                update={
                    "outcome": "abstain",
                    "answer": GeneratedAnswer(
                        answer="模型未在预算内完成。",
                        citations=[],
                        provider="deepseek-agent-budget",
                        grounded=False,
                    ),
                }
            ),
        }
    )
    abstention_case = _case().model_copy(
        update={"expected_behavior": "abstain", "expected_facts": []}
    )

    score = score_agent_hard_case(abstention_case, budget_trace)

    assert score.behavior_correct is False
    assert score.case_pass is False


def test_clarification_is_scored_separately_from_abstention() -> None:
    query = "请比较这家公司，但先确认公司和年度。"
    trace = CompareTaskController(
        FakeSearchBackend({}),
        EvidenceOnlyGenerator(),
        available_companies=["甲公司", "乙公司"],
    ).run(AgentTaskRequest(query=query, top_k=1))
    clarification_case = _case().model_copy(
        update={
            "query": query,
            "expected_behavior": "clarify",
            "expected_facts": [],
            "evidence_sources": [],
            "expected_target_ids": [],
        }
    )

    score = score_agent_hard_case(clarification_case, trace)

    assert trace.result.outcome == "clarify"
    assert score.behavior_correct is True
    assert score.case_pass is True


def test_percentage_before_a_year_keeps_a_valid_numeric_boundary() -> None:
    backend = FakeSearchBackend(
        {"甲公司": [_hit("甲公司", "chunk-a")], "乙公司": [_hit("乙公司", "chunk-b")]}
    )
    trace = CompareTaskController(
        backend,
        EvidenceOnlyGenerator(),
        available_companies=["甲公司", "乙公司"],
    ).run(AgentTaskRequest(query=_case().query, top_k=1))
    trace = trace.model_copy(
        update={
            "result": trace.result.model_copy(
                update={
                    "outcome": "answer",
                    "answer": GeneratedAnswer(
                        answer="2024年毛利率为22.31%，2023年毛利率为20.00%。",
                        citations=[],
                        provider="test",
                        grounded=True,
                    ),
                }
            )
        }
    )
    case = _case().model_copy(
        update={
            "expected_facts": [
                AgentHardExpectedFact(
                    fact_id="margin",
                    description="2024年毛利率",
                    acceptable_values=["22.31%"],
                    unit="%",
                )
            ]
        }
    )

    score = score_agent_hard_case(case, trace)

    assert score.fact_scores[0].matched is True


def test_atomic_requirement_diagnostics_are_gold_independent() -> None:
    trace = CompareTaskController(
        FakeSearchBackend({"甲公司": [], "乙公司": []}),
        EvidenceOnlyGenerator(),
        available_companies=["甲公司", "乙公司"],
    ).run(AgentTaskRequest(query=_case().query, top_k=1))
    requirements = [
        AtomicFactRequirement(
            requirement_id="r1",
            description="集团归母净利润",
            subject="甲公司",
            subject_scope="group",
            fact_period="2024年",
            evidence_type="table_value",
        ),
        AtomicFactRequirement(
            requirement_id="r2",
            description="寿险新业务价值",
            subject="寿险",
            subject_scope="business_segment",
            fact_period="2024年",
            evidence_type="table_value",
        ),
    ]
    trace = trace.model_copy(
        update={
            "plan": trace.plan.model_copy(
                update={"fact_requirements": requirements}
            ),
            "result": trace.result.model_copy(
                update={
                    "requirement_claims": {"r1": ["归母净利润为100元"]},
                    "requirement_evidence": {"r1": ["chunk-a"]},
                    "requirement_scope_validated": {"r1": True},
                    "answer": GeneratedAnswer(
                        answer="归母净利润为100元[1]",
                        citations=[],
                        provider="test",
                        grounded=True,
                        claim_citations=[
                            ClaimCitation(
                                claim="归母净利润为100元",
                                citation_ordinals=[1],
                            )
                        ],
                    ),
                }
            ),
        }
    )

    diagnostics = diagnose_agent_requirements(trace)

    assert diagnostics.applicable is True
    assert diagnostics.task_requirement_coverage == 0.5
    assert diagnostics.requirement_evidence_coverage == 0.5
    assert diagnostics.scope_validation_rate == 0.5
    assert diagnostics.claim_citation_completeness == 1.0
