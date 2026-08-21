import json
from datetime import UTC, datetime

from findoc_rag.agent_tasks import (
    AgentEvidence,
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskTrace,
    AtomicFactRequirement,
    EvidenceMemory,
    SufficiencyDecision,
)
from findoc_rag.answer_generation import Citation, ClaimCitation, GeneratedAnswer
from findoc_rag.deepseek_agent import ModelToolCall, ToolModelResponse
from findoc_rag.evidence_verifier import (
    EvidenceVerifierAgent,
    evaluate_claim_risk_gate,
)
from findoc_rag.indexing import SearchFilters


class ScriptedModel:
    provider = "test"
    model = "test-model"
    endpoint = "https://example.invalid"
    available = True

    def __init__(self, responses: list[ToolModelResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[dict], list[dict]]] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> ToolModelResponse:
        self.calls.append((messages, tools))
        return self.responses.pop(0)


def _response(tool: str, arguments: dict) -> ToolModelResponse:
    return ToolModelResponse(
        content=None,
        tool_calls=[
            ModelToolCall(
                call_id=f"call-{tool}",
                name=tool,
                arguments=json.dumps(arguments, ensure_ascii=False),
            )
        ],
        finish_reason="tool_calls",
        input_tokens=100,
        output_tokens=20,
        elapsed_ms=5,
    )


def _support_proofs(
    requirement_ids: list[str],
    *,
    claim_overrides: dict[str, str] | None = None,
) -> list[dict]:
    claim_overrides = claim_overrides or {}
    return [
        {
            "requirement_id": requirement_id,
            "claim": claim_overrides.get(
                requirement_id,
                f"甲公司2024年风险事项{requirement_id.removeprefix('r')}",
            ),
            "evidence_quotes": [
                {
                    "evidence_chunk_id": "chunk-audit",
                    "quote": (
                        "风险事项1包括模型验证"
                        if requirement_id == "r1"
                        else f"风险事项{requirement_id.removeprefix('r')}"
                    ),
                }
            ],
        }
        for requirement_id in requirement_ids
    ]


def _trace(requirement_count: int = 4) -> AgentTaskTrace:
    requirements = [
        AtomicFactRequirement(
            requirement_id=f"r{index}",
            description=f"风险事项{index}",
            subject="甲公司",
            subject_scope="document",
            fact_period="2024年",
            evidence_type=(
                "audit_response"
                if requirement_count >= 4 and index == 1
                else "narrative"
            ),
            candidate_evidence_chunk_ids=["chunk-audit"],
        )
        for index in range(1, requirement_count + 1)
    ]
    memory = EvidenceMemory(
        index_id="index-test",
        items=[
            AgentEvidence(
                chunk_id="chunk-audit",
                content_sha256="0" * 64,
                target_ids=["task:extract"],
                document_id="document-a",
                document_key="document-a",
                company_name="甲公司",
                report_year=2024,
                page_start=10,
                page_end=10,
                section_path=["关键审计事项"],
                excerpt="风险事项1包括模型验证；风险事项2、风险事项3和风险事项4均已执行。",
            )
        ],
    )
    claims = {f"r{index}": [f"甲公司2024年风险事项{index}"] for index in range(1, requirement_count + 1)}
    requirement_evidence = {
        f"r{index}": ["chunk-audit"] for index in range(1, requirement_count + 1)
    }
    citation = Citation(
        ordinal=1,
        chunk_id="chunk-audit",
        page_start=10,
        page_end=10,
        section_path=["关键审计事项"],
        excerpt=memory.items[0].excerpt,
    )
    return AgentTaskTrace(
        task_id="a" * 32,
        task_type="extract",
        runtime="deepseek_tool_calling",
        status="completed",
        stop_reason="sufficient_evidence",
        query="请完整列出甲公司2024年关键审计事项",
        index_id="index-test",
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        rounds_completed=4,
        plan=AgentTaskPlan(
            task_type="extract",
            document_scope=SearchFilters(
                company_names=["甲公司"], report_years=[2024]
            ),
            document_year=2024,
            fact_periods=[2024],
            fact_requirements=requirements,
        ),
        tool_calls=[],
        evidence_memory=memory,
        sufficiency=SufficiencyDecision(
            status="sufficient",
            evidence_count_by_target={"task:extract": 1},
            gaps=[],
        ),
        result=AgentTaskResult(
            outcome="answer",
            answer=GeneratedAnswer(
                answer="\n".join(
                    f"甲公司2024年风险事项{index}[1]"
                    for index in range(1, requirement_count + 1)
                ),
                citations=[citation],
                provider="test",
                claim_citations=[
                    ClaimCitation(
                        claim=f"甲公司2024年风险事项{index}",
                        citation_ordinals=[1],
                    )
                    for index in range(1, requirement_count + 1)
                ],
            ),
            target_evidence={},
            requirement_claims=claims,
            requirement_evidence=requirement_evidence,
            requirement_scope_validated={
                f"r{index}": True for index in range(1, requirement_count + 1)
            },
        ),
    )


def _non_audit_trace(requirement_count: int = 4) -> AgentTaskTrace:
    trace = _trace(requirement_count=requirement_count)
    for requirement in trace.plan.fact_requirements:
        requirement.evidence_type = "narrative"
    return trace


def test_simple_extraction_stays_on_single_agent_fast_path() -> None:
    model = ScriptedModel([])

    reviewed = EvidenceVerifierAgent(
        model, optimizer_model=model, min_requirements=4
    ).review(_trace(requirement_count=1))

    assert reviewed.result.outcome == "answer"
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.final_decision == "not_routed"
    assert reviewed.claim_risk_gate is not None
    assert reviewed.claim_risk_gate.status == "pass"
    assert model.calls == []


def test_local_gate_rejects_an_explicit_cross_company_claim_without_model_cost() -> None:
    trace = _trace()
    claims = {
        key: list(values) for key, values in trace.result.requirement_claims.items()
    }
    claims["r1"][0] = claims["r1"][0].replace("甲公司", "乙公司")
    trace.result.requirement_claims = claims
    model = ScriptedModel([])

    reviewed = EvidenceVerifierAgent(
        model,
        optimizer_model=model,
        known_companies=["甲公司", "乙公司"],
    ).review(trace)

    assert reviewed.result.outcome == "abstain"
    assert reviewed.stop_reason == "claim_risk_gate_rejected"
    assert reviewed.claim_risk_gate is not None
    assert reviewed.claim_risk_gate.status == "reject"
    assert reviewed.claim_risk_gate.findings[0].check == "subject_conflict"
    assert model.calls == []


def test_local_gate_rejects_unsupported_number_and_period() -> None:
    trace = _trace()
    claims = {
        key: list(values) for key, values in trace.result.requirement_claims.items()
    }
    claims["r1"][0] = "甲公司2022年风险事项1金额999亿元"
    trace.result.requirement_claims = claims

    gate = evaluate_claim_risk_gate(trace, known_companies=["甲公司"])

    assert gate.status == "reject"
    assert {
        "period_conflict",
        "unsupported_numeric",
        "unsupported_unit",
    } <= {finding.check for finding in gate.findings}


def test_numeric_only_warning_routes_to_model_instead_of_rejecting() -> None:
    trace = _trace()
    claims = {
        key: list(values) for key, values in trace.result.requirement_claims.items()
    }
    claims["r1"][0] = "甲公司2024年风险事项1金额999亿元"
    trace.result.requirement_claims = claims
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": ["r1", "r2", "r3", "r4"],
                    "findings": [],
                },
            )
        ]
    )

    reviewed = EvidenceVerifierAgent(
        model,
        optimizer_model=model,
        known_companies=["甲公司"],
        require_support_proof=False,
    ).review(trace)

    assert reviewed.result.outcome == "answer"
    assert reviewed.claim_risk_gate is not None
    assert reviewed.claim_risk_gate.status == "review"
    assert {
        "unsupported_numeric",
        "unsupported_unit",
    } <= {
        finding.check for finding in reviewed.claim_risk_gate.findings
    }
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.routed is True
    assert reviewed.evidence_verification.request_count == 1
    assert len(model.calls) == 1


def test_open_language_risk_routes_a_simple_unseen_semantic_fault() -> None:
    trace = _trace(requirement_count=1)
    claims = {
        key: list(values) for key, values in trace.result.requirement_claims.items()
    }
    claims["r1"][0] = (
        claims["r1"][0] + "，因此必然导致下一年度利润下降。"
    )
    trace.result.requirement_claims = claims
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": [],
                    "findings": [
                        {
                            "requirement_id": "r1",
                            "verdict": "insufficient_evidence",
                            "feedback": "引用没有支持新增的因果结论",
                            "evidence_chunk_ids": ["chunk-audit"],
                            "missing_supported_details": [],
                        }
                    ],
                },
            )
        ]
    )

    reviewed = EvidenceVerifierAgent(
        model,
        optimizer_model=model,
        known_companies=["甲公司"],
    ).review(trace)

    assert reviewed.result.outcome == "abstain"
    assert reviewed.claim_risk_gate is not None
    assert reviewed.claim_risk_gate.status == "review"
    assert any(
        finding.check == "low_evidence_language_coverage"
        for finding in reviewed.claim_risk_gate.findings
    )
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.routed is True
    assert reviewed.evidence_verification.final_decision == "abstain"
    assert len(model.calls) == 1


def test_disabling_open_language_risk_preserves_the_p4b_fast_path() -> None:
    trace = _trace(requirement_count=1)
    claims = {
        key: list(values) for key, values in trace.result.requirement_claims.items()
    }
    claims["r1"][0] = (
        claims["r1"][0] + "，因此必然导致下一年度利润下降。"
    )
    trace.result.requirement_claims = claims
    model = ScriptedModel([])

    reviewed = EvidenceVerifierAgent(
        model,
        optimizer_model=model,
        enable_open_language_risk=False,
        enable_requirement_contract_risk=False,
    ).review(trace)

    assert reviewed.result.outcome == "answer"
    assert reviewed.claim_risk_gate is not None
    assert reviewed.claim_risk_gate.status == "pass"
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.routed is False
    assert reviewed.evidence_verification.final_decision == "not_routed"
    assert model.calls == []


def test_requirement_contract_divergence_routes_a_high_overlap_label_swap() -> None:
    trace = _trace(requirement_count=1)
    claims = {
        key: list(values) for key, values in trace.result.requirement_claims.items()
    }
    claims["r1"][0] = claims["r1"][0].replace("风险事项1", "风险问题1")
    trace.result.requirement_claims = claims
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": [],
                    "findings": [
                        {
                            "requirement_id": "r1",
                            "verdict": "contradicted",
                            "feedback": "claim 将 requirement 的标签换成另一事项",
                            "evidence_chunk_ids": ["chunk-audit"],
                            "missing_supported_details": [],
                        }
                    ],
                },
            )
        ]
    )

    reviewed = EvidenceVerifierAgent(model, optimizer_model=model).review(trace)

    assert reviewed.result.outcome == "abstain"
    assert reviewed.claim_risk_gate is not None
    assert reviewed.claim_risk_gate.status == "review"
    assert any(
        finding.check == "requirement_claim_divergence"
        for finding in reviewed.claim_risk_gate.findings
    )
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.routed is True
    assert len(model.calls) == 1


def test_disabling_contract_risk_keeps_a_high_overlap_swap_on_fast_path() -> None:
    trace = _trace(requirement_count=1)
    claims = {
        key: list(values) for key, values in trace.result.requirement_claims.items()
    }
    claims["r1"][0] = claims["r1"][0].replace("风险事项1", "风险问题1")
    trace.result.requirement_claims = claims
    model = ScriptedModel([])

    reviewed = EvidenceVerifierAgent(
        model,
        optimizer_model=model,
        enable_open_language_risk=False,
        enable_requirement_contract_risk=False,
    ).review(trace)

    assert reviewed.result.outcome == "answer"
    assert reviewed.claim_risk_gate is not None
    assert reviewed.claim_risk_gate.status == "pass"
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.routed is False
    assert model.calls == []


def test_accounting_parenthesis_sign_conflict_is_rejected_locally() -> None:
    trace = _trace(requirement_count=1)
    requirement = trace.plan.fact_requirements[0]
    requirement.description = "风险事项1金额100元"
    trace.evidence_memory.items[0].excerpt = "风险事项1金额100元。"
    claims = {
        key: list(values) for key, values in trace.result.requirement_claims.items()
    }
    claims["r1"][0] = "甲公司2024年风险事项1金额(100)元"
    trace.result.requirement_claims = claims
    model = ScriptedModel([])

    reviewed = EvidenceVerifierAgent(model, optimizer_model=model).review(trace)

    assert reviewed.result.outcome == "abstain"
    assert reviewed.stop_reason == "claim_risk_gate_rejected"
    assert reviewed.claim_risk_gate is not None
    assert reviewed.claim_risk_gate.status == "reject"
    assert any(
        finding.check == "accounting_sign_conflict"
        for finding in reviewed.claim_risk_gate.findings
    )
    assert model.calls == []


def test_missing_requirement_number_creates_a_model_review_finding() -> None:
    trace = _trace(requirement_count=1)
    trace.plan.fact_requirements[0].description = "风险事项1金额100元"
    trace.evidence_memory.items[0].excerpt = "风险事项1金额100元。"

    gate = evaluate_claim_risk_gate(trace, known_companies=["甲公司"])

    assert gate.status == "review"
    assert any(
        finding.check == "missing_requirement_numeric"
        for finding in gate.findings
    )


def test_always_route_policy_verifies_a_supported_simple_answer() -> None:
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": [],
                    "support_proofs": _support_proofs(["r1"]),
                    "findings": [],
                },
            )
        ]
    )

    reviewed = EvidenceVerifierAgent(
        model,
        optimizer_model=model,
        route_policy="always",
    ).review(_trace(requirement_count=1))

    assert reviewed.result.outcome == "answer"
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.routed is True
    assert reviewed.evidence_verification.final_decision == "accept_original"
    assert len(model.calls) == 1


def test_local_gate_rejects_a_citation_outside_document_scope() -> None:
    trace = _trace()
    trace.evidence_memory.items[0].company_name = "乙公司"

    gate = evaluate_claim_risk_gate(
        trace,
        known_companies=["甲公司", "乙公司"],
    )

    assert gate.status == "reject"
    assert any(
        finding.check == "citation_scope_conflict"
        for finding in gate.findings
    )


def test_verifier_accepts_fully_supported_answer_without_repair() -> None:
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": ["r2", "r3", "r4"],
                    "support_proofs": _support_proofs(["r1"]),
                    "findings": [],
                },
            )
        ]
    )

    reviewed = EvidenceVerifierAgent(model, optimizer_model=model).review(
        _non_audit_trace()
    )

    assert reviewed.result.outcome == "answer"
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.final_decision == "accept_original"
    assert reviewed.evidence_verification.request_count == 1


def test_supported_findings_are_normalized_without_weakening_other_validation() -> None:
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": ["r1", "r2"],
                    "findings": [
                        {
                            "requirement_id": "r3",
                            "verdict": "supported",
                            "feedback": "证据直接支持",
                            "evidence_chunk_ids": ["chunk-audit"],
                            "missing_supported_details": [],
                        },
                        {
                            "requirement_id": "r4",
                            "verdict": "supported",
                            "feedback": "证据直接支持",
                            "evidence_chunk_ids": ["chunk-audit"],
                            "missing_supported_details": [],
                        },
                    ],
                },
            )
        ]
    )

    reviewed = EvidenceVerifierAgent(
        model,
        optimizer_model=model,
        require_support_proof=False,
    ).review(_trace())

    assert reviewed.result.outcome == "answer"
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.final_decision == "accept_original"
    assert reviewed.evidence_verification.turns[0].supported_requirement_ids == [
        "r1",
        "r2",
        "r3",
        "r4",
    ]


def test_verifier_repairs_once_and_requires_a_second_acceptance() -> None:
    repaired_facts = [
        {
            "text": (
                "甲公司2024年风险事项1包括模型验证"
                if index == 1
                else f"甲公司2024年风险事项{index}"
            ),
            "evidence_chunk_ids": ["chunk-audit"],
            "requirement_ids": [f"r{index}"],
        }
        for index in range(1, 5)
    ]
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": ["r2", "r3", "r4"],
                    "support_proofs": [],
                    "findings": [
                        {
                            "requirement_id": "r1",
                            "verdict": "incomplete",
                            "feedback": "漏掉证据中的模型验证",
                            "evidence_chunk_ids": ["chunk-audit"],
                            "missing_supported_details": ["包括模型验证"],
                        }
                    ],
                },
            ),
            _response(
                "submit_extraction",
                {
                    "status": "answer",
                    "message": "已修复",
                    "facts": repaired_facts,
                    "gaps": [],
                },
            ),
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": ["r2", "r3", "r4"],
                    "support_proofs": _support_proofs(
                        ["r1"],
                        claim_overrides={
                            "r1": "甲公司2024年风险事项1包括模型验证"
                        },
                    ),
                    "findings": [],
                },
            ),
        ]
    )

    reviewed = EvidenceVerifierAgent(
        model,
        optimizer_model=model,
        require_support_proof=False,
    ).review(_trace())

    assert reviewed.result.outcome == "answer"
    assert "模型验证" in reviewed.result.answer.answer
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.final_decision == "accept_repaired"
    assert reviewed.evidence_verification.repair_attempted is True
    assert [turn.role for turn in reviewed.evidence_verification.turns] == [
        "verifier",
        "optimizer",
        "verifier",
    ]


def test_contradicted_claim_is_rejected_without_repair() -> None:
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": ["r2", "r3", "r4"],
                    "support_proofs": [],
                    "findings": [
                        {
                            "requirement_id": "r1",
                            "verdict": "contradicted",
                            "feedback": "claim与证据冲突",
                            "evidence_chunk_ids": ["chunk-audit"],
                            "missing_supported_details": [],
                        }
                    ],
                },
            )
        ]
    )

    reviewed = EvidenceVerifierAgent(model, optimizer_model=model).review(
        _non_audit_trace()
    )

    assert reviewed.result.outcome == "abstain"
    assert reviewed.stop_reason == "evidence_verifier_rejected"
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.final_decision == "abstain"
    assert reviewed.evidence_verification.repair_attempted is False


def test_invalid_verifier_partition_fails_closed() -> None:
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {"supported_requirement_ids": ["r1"], "findings": []},
            )
        ]
    )

    reviewed = EvidenceVerifierAgent(
        model,
        optimizer_model=model,
        require_support_proof=False,
    ).review(_trace())

    assert reviewed.result.outcome == "abstain"
    assert reviewed.status == "failed"
    assert reviewed.stop_reason == "evidence_verifier_error"
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.final_decision == "error"
    assert "classify every fact requirement" in (
        reviewed.evidence_verification.turns[0].validation_errors[0]
    )


def test_supported_ids_without_proofs_are_escalated_for_manual_review() -> None:
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": ["r1", "r2", "r3", "r4"],
                    "findings": [],
                },
            )
        ]
    )

    reviewed = EvidenceVerifierAgent(model, optimizer_model=model).review(
        _non_audit_trace()
    )

    assert reviewed.result.outcome == "abstain"
    assert reviewed.stop_reason == "evidence_verifier_manual_review"
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.final_decision == "manual_review"
    assert reviewed.evidence_verification.human_review_required is True
    assert reviewed.evidence_verification.human_review_reasons
    assert reviewed.evidence_verification.candidate_result is not None
    assert reviewed.evidence_verification.candidate_result.outcome == "answer"


def test_long_audit_review_uses_compact_legacy_partition() -> None:
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": ["r1", "r2", "r3", "r4"],
                    "findings": [],
                },
            )
        ]
    )

    reviewed = EvidenceVerifierAgent(model, optimizer_model=model).review(_trace())

    assert reviewed.result.outcome == "answer"
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.final_decision == "accept_original"
    assert "support-proof-off" in reviewed.evidence_verification.prompt_revision


def test_incomplete_proof_partition_gets_one_bounded_retry() -> None:
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": ["r1"],
                    "support_proofs": [],
                    "findings": [],
                },
            ),
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": ["r2", "r3", "r4"],
                    "support_proofs": _support_proofs(["r1"]),
                    "findings": [],
                },
            ),
        ]
    )

    reviewed = EvidenceVerifierAgent(model, optimizer_model=model).review(
        _non_audit_trace()
    )

    assert reviewed.result.outcome == "answer"
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.verification_retry_attempted is True
    assert [turn.stage for turn in reviewed.evidence_verification.turns] == [
        "initial_verification",
        "support_proof_retry",
    ]


def test_relation_swap_with_verbatim_contrary_quote_is_challenged() -> None:
    trace = _trace(requirement_count=1)
    trace.query = "甲公司2024年不良贷款率及同比变化是什么？"
    requirement = trace.plan.fact_requirements[0]
    requirement.description = "甲公司2024年不良贷款率为1.36%"
    requirement.subject = "不良贷款率"
    trace.evidence_memory.items[0].excerpt = (
        "不良贷款率同比下降2BP至1.36%。"
    )
    trace.result.requirement_claims = {
        "r1": ["甲公司2024年不良贷款率同比上升2BP至1.36%。"]
    }
    model = ScriptedModel(
        [
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": [],
                    "support_proofs": [
                        {
                            "requirement_id": "r1",
                            "claim": "甲公司2024年不良贷款率同比上升2BP至1.36%。",
                            "evidence_quotes": [
                                {
                                    "evidence_chunk_id": "chunk-audit",
                                    "quote": "不良贷款率同比下降2BP至1.36%",
                                }
                            ],
                        }
                    ],
                    "findings": [],
                },
            ),
            _response(
                "submit_evidence_verification",
                {
                    "supported_requirement_ids": [],
                    "findings": [
                        {
                            "requirement_id": "r1",
                            "verdict": "contradicted",
                            "feedback": "claim称上升，但quote明确为下降",
                            "evidence_chunk_ids": ["chunk-audit"],
                            "missing_supported_details": [],
                        }
                    ],
                },
            ),
        ]
    )

    reviewed = EvidenceVerifierAgent(
        model,
        optimizer_model=model,
        route_policy="always",
    ).review(trace)

    assert reviewed.result.outcome == "abstain"
    assert reviewed.stop_reason == "evidence_verifier_rejected"
    assert reviewed.evidence_verification is not None
    assert reviewed.evidence_verification.final_decision == "abstain"
    assert [
        turn.stage for turn in reviewed.evidence_verification.turns
    ] == ["initial_verification", "support_proof_challenge"]
