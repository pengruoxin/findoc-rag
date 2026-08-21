import json
from types import SimpleNamespace

import httpx
import pytest

from findoc_rag.agent_tasks import (
    AgentEvidence,
    AgentTaskPlan,
    AgentTaskRequest,
    AtomicFactRequirement,
    EvidenceMemory,
    plan_compare_task,
)
from findoc_rag.deepseek_agent import (
    DeepSeekCalculateAgent,
    DeepSeekCompareAgent,
    DeepSeekExtractAgent,
    DeepSeekToolCallingModel,
    DeepSeekVisualGraphAgent,
    ModelToolCall,
    SubmitComparisonArguments,
    SubmitExtractionArguments,
    SubmittedClaim,
    SubmittedExtractFact,
    ToolModelResponse,
    _answer_contract,
    _candidate_scope_compatible,
    _complete_required_metric_requirements,
    _normalize_requirement_subject_scopes,
    _parse_extraction_submission,
    _parse_submission,
    _rank_authoritative_hits,
    _repair_extract_authoritative_citations,
    _repair_submitted_fact_scope_labels,
    _tool_definitions,
    _validate_fact_requirements,
)
from findoc_rag.documents.models import BoundingBox, DocumentChunk, ElementReference
from findoc_rag.indexing import SearchHit
from findoc_rag.service import SearchRequest, SearchResponse
from findoc_rag.visual_inspection import (
    PageLayoutReconstruction,
    PageRegionInspection,
    ReconstructedTextColumn,
    VisualRelationshipRow,
    VisualTextNode,
)


def _hit(company: str, chunk_id: str, value: int) -> SearchHit:
    text = f"{company}2024年营业收入为{value}元"
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"document-{company}",
        chunk_index=0,
        text=text,
        section_path=["主要会计数据"],
        page_start=1,
        page_end=1,
        element_references=[
            ElementReference(
                element_id=f"element-{chunk_id}",
                page_number=1,
                bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
            )
        ],
        character_count=len(text),
        estimated_token_count=len(text),
        company_name=company,
        report_year=2024,
    )
    return SearchHit(rank=1, chunk=chunk, score=1.0)


def _agent_evidence(
    chunk_id: str,
    *,
    company: str,
    report_year: int,
    target_ids: list[str],
) -> AgentEvidence:
    return AgentEvidence(
        chunk_id=chunk_id,
        content_sha256="0" * 64,
        target_ids=target_ids,
        document_id=f"document-{company}-{report_year}",
        company_name=company,
        report_year=report_year,
        page_start=1,
        page_end=1,
        section_path=["主要会计数据"],
        excerpt=f"{company}{report_year}年营业收入",
    )


def test_authority_ranking_prefers_key_indicator_source_over_appendix_analysis() -> None:
    low = _hit("中国平安", "chunk-low", 100)
    low.chunk.section_path = ["内含价值分析"]
    low.chunk.text = "内含价值—可比口径 960,608 830,974 15.6"
    high = _hit("中国平安", "chunk-high", 100)
    high.rank = 6
    high.chunk.section_path = ["业绩综述", "寿险及健康险业务"]
    high.chunk.text = (
        "寿险及健康险业务关键指标 新业务价值—可比口径 40,024 31,080 28.8 "
        "内含价值—可比口径 960,608 830,974 15.6"
    )

    ranked = _rank_authoritative_hits(
        "按可比口径核验新业务价值和内含价值",
        [low, high],
        limit=1,
    )

    assert [hit.chunk.chunk_id for hit in ranked] == ["chunk-high"]


def test_extract_authority_repair_adds_supported_financial_summary_citation() -> None:
    plan = AgentTaskPlan(
        task_type="extract",
        fact_requirements=[
            AtomicFactRequirement(
                requirement_id="r1",
                description="2023年归母净利润",
                subject="中国平安",
                subject_scope="group",
                fact_period="2023年",
                evidence_type="table_value",
                candidate_evidence_chunk_ids=["chunk-low"],
            )
        ],
    )
    memory = EvidenceMemory(
        index_id="index-test",
        items=[
            AgentEvidence(
                chunk_id="chunk-low",
                content_sha256="0" * 64,
                target_ids=["task:extract"],
                document_id="pingan-2023",
                document_key="pingan-2023",
                company_name="中国平安",
                report_year=2023,
                page_start=14,
                page_end=14,
                section_path=["业绩综述"],
                excerpt="归属于母公司股东的净利润85,665百万元",
            ),
            AgentEvidence(
                chunk_id="chunk-summary",
                content_sha256="1" * 64,
                target_ids=["task:extract"],
                document_id="pingan-2023",
                document_key="pingan-2023",
                company_name="中国平安",
                report_year=2023,
                page_start=12,
                page_end=12,
                section_path=["财务摘要"],
                excerpt="集团合并 归属于母公司股东的净利润85,665百万元",
            ),
        ],
    )
    submission = SubmitExtractionArguments(
        status="answer",
        message="完成",
        facts=[
            SubmittedExtractFact(
                text="2023年归母净利润为85,665百万元",
                evidence_chunk_ids=["chunk-low"],
                requirement_ids=["r1"],
            )
        ],
        gaps=[],
    )

    repaired = _repair_extract_authoritative_citations(
        "中国平安2023年归母净利润是多少？",
        submission,
        plan,
        memory,
    )

    assert repaired.facts[0].evidence_chunk_ids == [
        "chunk-low",
        "chunk-summary",
    ]


def test_requirement_scope_and_fact_label_repair_preserve_segment_subject() -> None:
    requirement = AtomicFactRequirement(
        requirement_id="r1",
        description="寿险及健康险业务内含价值为830,974百万元",
        subject="中国平安",
        subject_scope="group",
        fact_period="2023年",
        evidence_type="table_value",
    )
    _normalize_requirement_subject_scopes([requirement])
    plan = AgentTaskPlan(task_type="extract", fact_requirements=[requirement])
    submission = SubmitExtractionArguments(
        status="answer",
        message="完成",
        facts=[
            SubmittedExtractFact(
                text="内含价值为830,974百万元",
                evidence_chunk_ids=["chunk-ev"],
                requirement_ids=["r1"],
            )
        ],
        gaps=[],
    )

    repaired = _repair_submitted_fact_scope_labels(submission, plan)

    assert requirement.subject_scope == "business_segment"
    assert requirement.subject == "寿险及健康险业务"
    assert repaired.facts[0].text.startswith("寿险及健康险业务：")


def test_extraction_submission_parser_drops_redundant_singular_requirement_id() -> None:
    parsed = _parse_extraction_submission(
        json.dumps(
            {
                "status": "answer",
                "message": "完成",
                "facts": [
                    {
                        "text": "净利润为100亿元",
                        "evidence_chunk_ids": ["chunk-1"],
                        "requirement_id": "r1",
                        "requirement_ids": ["r1"],
                    }
                ],
                "gaps": [],
            },
            ensure_ascii=False,
        )
    )

    assert parsed.facts[0].requirement_ids == ["r1"]


class FakeSearchBackend:
    def __init__(self) -> None:
        self.manifest = SimpleNamespace(index_id="index-deepseek-agent-test")
        self.calls: list[SearchRequest] = []

    def search(self, request: SearchRequest, request_id: str) -> SearchResponse:
        self.calls.append(request)
        company = request.filters.company_names[0]
        hit = _hit(company, "chunk-a" if company == "甲公司" else "chunk-b", 100)
        return SearchResponse(
            request_id=request_id,
            trace_id=f"trace-{len(self.calls)}",
            index_id=self.manifest.index_id,
            mode=request.mode or "lexical",
            reranked=False,
            base_candidate_k=10,
            effective_candidate_k=10,
            candidate_budget_reason="test",
            took_ms=1.0,
            hits=[hit],
        )


class MissingTargetSearchBackend(FakeSearchBackend):
    def search(self, request: SearchRequest, request_id: str) -> SearchResponse:
        if request.filters.company_names[0] != "乙公司":
            return super().search(request, request_id)
        self.calls.append(request)
        return SearchResponse(
            request_id=request_id,
            trace_id=f"trace-{len(self.calls)}",
            index_id=self.manifest.index_id,
            mode=request.mode or "lexical",
            reranked=False,
            base_candidate_k=10,
            effective_candidate_k=10,
            candidate_budget_reason="test",
            took_ms=1.0,
            hits=[],
        )


class ScriptedToolModel:
    provider = "deepseek"
    model = "deepseek-test"
    endpoint = "https://example.invalid/chat/completions"
    available = True

    def __init__(self, responses: list[ToolModelResponse]) -> None:
        self.responses = responses
        self.messages: list[list[dict]] = []
        self.tool_names: list[set[str]] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> ToolModelResponse:
        names = {tool["function"]["name"] for tool in tools}
        assert "submit_comparison" in names
        self.tool_names.append(names)
        self.messages.append(messages.copy())
        return self.responses.pop(0)


def _response(*calls: ModelToolCall) -> ToolModelResponse:
    return ToolModelResponse(
        tool_calls=list(calls),
        finish_reason="tool_calls",
        input_tokens=100,
        output_tokens=30,
        elapsed_ms=5.0,
    )


def _search(call_id: str, target_id: str) -> ModelToolCall:
    return ModelToolCall(
        call_id=call_id,
        name="search_evidence",
        arguments=json.dumps(
            {"target_id": target_id, "query": "2024年营业收入", "top_k": 1},
            ensure_ascii=False,
        ),
    )


def _submit(*, chunk_b: str = "chunk-b", verdict: str = "") -> ModelToolCall:
    targets = ["company:甲公司:year:2024", "company:乙公司:year:2024"]
    return ModelToolCall(
        call_id="submit-1",
        name="submit_comparison",
        arguments=json.dumps(
            {
                "status": "answer",
                "message": "比较完成",
                "covered_target_ids": targets,
                "claims": [
                    {
                        "target_id": targets[0],
                        "text": "营业收入为100元",
                        "evidence_chunk_ids": ["chunk-a"],
                    },
                    {
                        "target_id": targets[1],
                        "text": f"营业收入为100元{verdict}",
                        "evidence_chunk_ids": [chunk_b],
                    },
                ],
                "gaps": [],
            },
            ensure_ascii=False,
        ),
    )


def test_deepseek_agent_executes_model_selected_searches_and_grounded_submission() -> None:
    model = ScriptedToolModel(
        [
            _response(
                _search("search-a", "company:甲公司:year:2024"),
                _search("search-b", "company:乙公司:year:2024"),
            ),
            _response(_submit()),
        ]
    )
    backend = FakeSearchBackend()

    trace = DeepSeekCompareAgent(
        backend,
        model,
        available_companies=["甲公司", "乙公司"],
    ).run(AgentTaskRequest(query="比较甲公司和乙公司2024年营业收入", top_k=1))

    assert trace.runtime == "deepseek_tool_calling"
    assert trace.result.outcome == "answer"
    assert trace.result.answer.grounded is True
    assert [call.filters.company_names for call in backend.calls] == [
        ["甲公司"],
        ["乙公司"],
    ]
    assert all("近三年主要会计数据" in call.query for call in backend.calls)
    assert [citation.chunk_id for citation in trace.result.answer.citations] == [
        "chunk-a",
        "chunk-b",
    ]
    assert trace.model_trace is not None
    assert trace.model_trace.model == "deepseek-test"
    assert trace.model_trace.request_count == 2
    assert trace.model_trace.input_tokens == 200
    assert model.tool_names == [
        {"search_evidence", "submit_comparison"},
        {"submit_comparison"},
    ]
    assert [message["role"] for message in model.messages[1]] == ["system", "user"]
    assert "最终提交器" in model.messages[1][0]["content"]
    final_payload = json.loads(model.messages[1][1]["content"])
    assert final_payload["answer_contract"]["requires_direction_for_each_metric"]


def test_verification_submission_retries_until_explicit_verdict() -> None:
    model = ScriptedToolModel(
        [
            _response(
                _search("search-a", "company:甲公司:year:2024"),
                _search("search-b", "company:乙公司:year:2024"),
            ),
            _response(_submit()),
            _response(_submit(verdict="，说法成立")),
        ]
    )

    trace = DeepSeekCompareAgent(
        FakeSearchBackend(),
        model,
        available_companies=["甲公司", "乙公司"],
    ).run(
        AgentTaskRequest(
            query="核验说法：甲公司和乙公司2024年营业收入相同。",
            top_k=1,
        )
    )

    assert trace.result.outcome == "answer"
    assert "说法成立" in trace.result.answer.answer
    assert trace.model_trace is not None
    assert trace.model_trace.request_count == 3
    assert "explicit final verdict" in trace.model_trace.turns[1].validation_errors[0]


def test_answer_contract_marks_multifact_verification_requirements() -> None:
    contract = _answer_contract("核验说法：比较两年的收入变化和毛利率。")

    assert contract["requires_explicit_verdict"] is True
    assert contract["requires_direction_for_each_metric"] is True
    assert contract["requires_concise_direction_summary"] is True
    assert contract["render_decline_with_signed_value"] is True
    assert "逐项覆盖" in contract["coverage_rule"]


def test_deepseek_agent_rejects_hallucinated_citation_and_safely_abstains() -> None:
    model = ScriptedToolModel(
        [
            _response(
                _search("search-a", "company:甲公司:year:2024"),
                _search("search-b", "company:乙公司:year:2024"),
            ),
            _response(_submit(chunk_b="invented-chunk")),
        ]
    )

    trace = DeepSeekCompareAgent(
        FakeSearchBackend(),
        model,
        available_companies=["甲公司", "乙公司"],
    ).run(
        AgentTaskRequest(
            query="比较甲公司和乙公司2024年营业收入",
            top_k=1,
            max_rounds=2,
        )
    )

    assert trace.result.outcome == "abstain"
    assert trace.stop_reason == "model_budget_exhausted"
    assert trace.result.answer.grounded is False
    assert "invented-chunk" not in trace.result.answer.answer
    assert "unknown evidence chunk" in trace.model_trace.turns[-1].validation_errors[0]


def test_deepseek_agent_locally_abstains_after_zero_evidence_gap() -> None:
    model = ScriptedToolModel(
        [
            _response(
                _search("search-a", "company:甲公司:year:2024"),
                _search("search-b", "company:乙公司:year:2024"),
            )
        ]
    )

    trace = DeepSeekCompareAgent(
        MissingTargetSearchBackend(),
        model,
        available_companies=["甲公司", "乙公司"],
    ).run(
        AgentTaskRequest(
            query="比较甲公司和乙公司2024年营业收入",
            top_k=1,
            max_rounds=2,
        )
    )

    assert trace.result.outcome == "abstain"
    assert trace.stop_reason == "no_new_evidence"
    assert trace.result.answer.provider == "agent-local-evidence-gate"
    assert "乙公司" in trace.result.answer.answer
    assert trace.model_trace.request_count == 1


def test_restatement_versions_share_one_comparative_table_binding() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(
            query=("海尔智家2024年报对2023年营业收入给出了调整前和调整后两个数，比较并计算调整额")
        ),
        available_companies=["海尔智家"],
    )

    assert DeepSeekCompareAgent._evidence_binding_targets(plan.targets[0].target_id, plan) == [
        target.target_id for target in plan.targets
    ]


def test_statement_scopes_share_cross_target_calculation_evidence() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(query="比较海尔智家2024年合并口径与母公司口径营业收入"),
        available_companies=["海尔智家"],
    )

    assert DeepSeekCompareAgent._evidence_binding_targets(plan.targets[0].target_id, plan) == [
        target.target_id for target in plan.targets
    ]


def test_abstention_submission_allows_omitted_empty_answer_fields() -> None:
    submission = SubmitComparisonArguments.model_validate(
        {"status": "abstain", "message": "2025年实际数未披露"}
    )

    assert submission.claims == []
    assert submission.covered_target_ids == []
    assert submission.gaps == []


def test_submit_tool_still_requests_full_answer_schema_from_model() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(query="比较甲公司和乙公司2024年营业收入"),
        available_companies=["甲公司", "乙公司"],
    )
    submit_tool = _tool_definitions(plan)[-1]

    assert set(submit_tool["function"]["parameters"]["required"]) == {
        "status",
        "message",
        "covered_target_ids",
        "claims",
        "gaps",
    }


def test_provenance_repair_adds_retrieved_required_document_vintage() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(
            query=("长江电力2024年营业收入同比增长8.12%的比较基数是2023年调整前还是调整后？")
        ),
        available_companies=["长江电力"],
    )
    target_ids = [target.target_id for target in plan.targets]
    memory = EvidenceMemory(
        index_id="index-test",
        items=[
            _agent_evidence(
                "chunk-2024",
                company="长江电力",
                report_year=2024,
                target_ids=target_ids,
            ),
            _agent_evidence(
                "chunk-2023",
                company="长江电力",
                report_year=2023,
                target_ids=target_ids,
            ),
        ],
    )
    submission = SubmitComparisonArguments(
        status="answer",
        message="完成",
        covered_target_ids=target_ids,
        claims=[
            SubmittedClaim(
                target_id=target_id,
                text="营业收入",
                evidence_chunk_ids=["chunk-2024"],
            )
            for target_id in target_ids
        ],
        gaps=[],
    )

    repaired = DeepSeekCompareAgent._repair_provenance_citations(submission, plan, memory)

    assert repaired.claims[0].evidence_chunk_ids == ["chunk-2024"]
    assert repaired.claims[1].evidence_chunk_ids == ["chunk-2024", "chunk-2023"]


def test_cross_company_support_requires_company_name_and_comparison_cue() -> None:
    evidence = _agent_evidence(
        "chunk-haier",
        company="海尔智家",
        report_year=2024,
        target_ids=["company:海尔智家:year:2024"],
    )

    assert DeepSeekCompareAgent._is_cross_company_comparison_support(
        SubmittedClaim(
            target_id="company:长江电力:year:2024",
            text="长江电力增幅8.12%高于海尔智家4.29%",
            evidence_chunk_ids=["chunk-haier"],
        ),
        evidence,
    )
    assert not DeepSeekCompareAgent._is_cross_company_comparison_support(
        SubmittedClaim(
            target_id="company:长江电力:year:2024",
            text="长江电力营业收入为844亿元",
            evidence_chunk_ids=["chunk-haier"],
        ),
        evidence,
    )


def test_cross_year_comparison_claim_accepts_same_company_year_evidence() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(query="比较工商银行2023年和2024年不良贷款率"),
        available_companies=["工商银行"],
    )
    targets = {target.target_id: target for target in plan.targets}
    claim = SubmittedClaim(
        target_id="company:工商银行:year:2024",
        text="2024年不良贷款率1.34%，较2023年1.36%下降0.02个百分点",
        evidence_chunk_ids=["chunk-2023"],
    )
    evidence = _agent_evidence(
        "chunk-2023",
        company="工商银行",
        report_year=2023,
        target_ids=["company:工商银行:year:2023"],
    )

    assert DeepSeekCompareAgent._is_cross_target_comparison_support(
        claim, evidence, targets
    )


def test_claim_target_repair_moves_explicit_year_fact_to_bound_target() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(query="比较工商银行2023年和2024年净利润"),
        available_companies=["工商银行"],
    )
    target_ids = [target.target_id for target in plan.targets]
    memory = EvidenceMemory(
        index_id="index-test",
        items=[
            _agent_evidence(
                "chunk-2023",
                company="工商银行",
                report_year=2023,
                target_ids=[target_ids[0]],
            )
        ],
    )
    submission = SubmitComparisonArguments(
        status="answer",
        message="完成",
        covered_target_ids=target_ids,
        claims=[
            SubmittedClaim(
                target_id=target_ids[1],
                text="2023年净利润为365,116百万元",
                evidence_chunk_ids=["chunk-2023"],
            )
        ],
        gaps=[],
    )

    repaired = DeepSeekCompareAgent._repair_claim_target_bindings(
        submission, plan, memory
    )

    assert repaired.claims[0].target_id == target_ids[0]


def test_claim_target_repair_does_not_guess_without_explicit_year() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(query="比较工商银行2023年和2024年净利润"),
        available_companies=["工商银行"],
    )
    target_ids = [target.target_id for target in plan.targets]
    memory = EvidenceMemory(
        index_id="index-test",
        items=[
            _agent_evidence(
                "chunk-2023",
                company="工商银行",
                report_year=2023,
                target_ids=[target_ids[0]],
            )
        ],
    )
    submission = SubmitComparisonArguments(
        status="answer",
        message="完成",
        covered_target_ids=target_ids,
        claims=[
            SubmittedClaim(
                target_id=target_ids[1],
                text="净利润为365,116百万元",
                evidence_chunk_ids=["chunk-2023"],
            )
        ],
        gaps=[],
    )

    repaired = DeepSeekCompareAgent._repair_claim_target_bindings(
        submission, plan, memory
    )

    assert repaired.claims[0].target_id == target_ids[1]


def test_cross_year_support_rejects_unmentioned_evidence_year() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(query="比较工商银行2023年和2024年不良贷款率"),
        available_companies=["工商银行"],
    )
    targets = {target.target_id: target for target in plan.targets}
    claim = SubmittedClaim(
        target_id="company:工商银行:year:2024",
        text="2024年不良贷款率为1.34%",
        evidence_chunk_ids=["chunk-2023"],
    )
    evidence = _agent_evidence(
        "chunk-2023",
        company="工商银行",
        report_year=2023,
        target_ids=["company:工商银行:year:2023"],
    )

    assert not DeepSeekCompareAgent._is_cross_target_comparison_support(
        claim, evidence, targets
    )


def test_nonanswer_parser_recovers_only_safe_outcomes() -> None:
    recovered = _parse_submission(
        json.dumps(
            {
                "status": "abstain",
                "message": "2025年实际数未披露",
                "claims": "not-an-array",
                "unexpected": True,
            },
            ensure_ascii=False,
        )
    )

    assert recovered.status == "abstain"
    assert recovered.claims == []
    with pytest.raises(ValueError):
        _parse_submission(
            json.dumps(
                {
                    "status": "answer",
                    "message": "unsupported answer",
                    "claims": "not-an-array",
                }
            )
        )


def test_deepseek_model_is_unavailable_without_an_endpoint_bound_key() -> None:
    model = DeepSeekToolCallingModel(
        endpoint="https://api.deepseek.com/chat/completions",
        api_key="",
    )

    assert model.available is False


class CapturingHttpClient:
    def __init__(self) -> None:
        self.payload: dict | None = None

    def post(self, url: str, **kwargs) -> httpx.Response:
        self.payload = kwargs["json"]
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "submit-test",
                                    "function": {
                                        "name": "submit_comparison",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )


def test_deepseek_model_forces_named_submit_when_search_is_closed() -> None:
    client = CapturingHttpClient()
    model = DeepSeekToolCallingModel(api_key="test-key", client=client)

    model.complete(
        [{"role": "user", "content": "submit now"}],
        [
            {
                "type": "function",
                "function": {
                    "name": "submit_comparison",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert client.payload is not None
    assert client.payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_comparison"},
    }


class ExtractBackend:
    def __init__(self) -> None:
        self.manifest = SimpleNamespace(index_id="index-extract-test")
        self.search_requests: list[SearchRequest] = []
        self.window_anchors: list[str] = []
        self.original = self._chunk(
            "chunk-original",
            "一、账面原值 5.期末余额 合计 9,158,716,606.73",
            ["使用权资产", "一、账面原值"],
            185,
            185,
        )
        self.depreciation = self._chunk(
            "chunk-depreciation",
            "二、累计折旧 5.期末余额 合计 3,316,847,042.37",
            ["使用权资产", "二、累计折旧"],
            185,
            186,
        )
        self.carrying = self._chunk(
            "chunk-carrying",
            "四、账面价值 1.期末账面价值 合计 5,841,869,564.36",
            ["使用权资产", "四、账面价值"],
            186,
            186,
        )

    @staticmethod
    def _chunk(
        chunk_id: str,
        text: str,
        section_path: list[str],
        page_start: int,
        page_end: int,
    ) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id="haier-2024",
            document_key="haier-2024",
            chunk_index=1 if "original" in chunk_id else 2,
            text=text,
            section_path=section_path,
            page_start=page_start,
            page_end=page_end,
            element_references=[
                ElementReference(
                    element_id=f"element-{chunk_id}",
                    page_number=page_start,
                    bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                )
            ],
            character_count=len(text),
            estimated_token_count=len(text),
            company_name="海尔智家",
            report_year=2024,
        )

    def search(self, request: SearchRequest, request_id: str) -> SearchResponse:
        self.search_requests.append(request)
        return SearchResponse(
            request_id=request_id,
            trace_id="trace-extract",
            index_id=self.manifest.index_id,
            mode=request.mode or "lexical",
            reranked=False,
            base_candidate_k=10,
            effective_candidate_k=10,
            candidate_budget_reason="test",
            took_ms=1.0,
            hits=[SearchHit(rank=1, chunk=self.original, score=1.0)],
        )

    def page_window(self, anchor_chunk_id: str, **_: int) -> list[DocumentChunk]:
        self.window_anchors.append(anchor_chunk_id)
        return [self.original, self.depreciation, self.carrying]


class ExtractScriptedModel:
    provider = "deepseek"
    model = "deepseek-test"
    endpoint = "https://example.invalid/chat/completions"
    available = True

    def __init__(self, responses: list[ToolModelResponse]) -> None:
        self.responses = responses
        self.tool_names: list[list[str]] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> ToolModelResponse:
        del messages
        self.tool_names.append([tool["function"]["name"] for tool in tools])
        return self.responses.pop(0)


def test_extract_agent_locally_abstains_when_actual_year_exceeds_corpus() -> None:
    model = ExtractScriptedModel([])
    backend = ExtractBackend()

    trace = DeepSeekExtractAgent(
        backend,
        model,
        available_companies=["海尔智家"],
        available_report_years_by_company={"海尔智家": [2023, 2024]},
    ).run(
        AgentTaskRequest(
            task_type="extract",
            query="海尔智家2025年全年实际营业收入是多少？",
        )
    )

    assert trace.status == "completed"
    assert trace.stop_reason == "no_new_evidence"
    assert trace.result.outcome == "abstain"
    assert trace.result.answer.grounded is False
    assert trace.result.answer.provider == "agent-local-corpus-coverage-gate"
    assert trace.rounds_completed == 0
    assert trace.tool_calls == []
    assert backend.search_requests == []
    assert model.tool_names == []


def test_extract_agent_uses_page_window_to_disambiguate_repeated_total_rows() -> None:
    model = ExtractScriptedModel(
        [
            _response(
                ModelToolCall(
                    call_id="search-1",
                    name="search_evidence",
                    arguments=json.dumps(
                        {"query": "使用权资产 累计折旧 期末余额 合计", "top_k": 1},
                        ensure_ascii=False,
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="window-1",
                    name="get_page_window",
                    arguments=json.dumps(
                        {
                            "anchor_chunk_id": "chunk-original",
                            "before_pages": 1,
                            "after_pages": 1,
                        }
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="requirements-1",
                    name="submit_fact_requirements",
                    arguments=json.dumps(
                        {
                            "requirements": [
                                {
                                    "requirement_id": "r1",
                                    "description": "累计折旧的期末余额合计",
                                    "subject": "使用权资产",
                                    "subject_scope": "document",
                                    "fact_period": "2024年末",
                                    "evidence_type": "table_value",
                                    "candidate_evidence_chunk_ids": [
                                        "chunk-depreciation"
                                    ],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="submit-1",
                    name="submit_extraction",
                    arguments=json.dumps(
                        {
                            "status": "answer",
                            "message": "抽取完成",
                            "facts": [
                                {
                                    "text": "累计折旧期末余额合计为3,316,847,042.37元",
                                    "evidence_chunk_ids": ["chunk-depreciation"],
                                    "requirement_ids": ["r1"],
                                }
                            ],
                            "gaps": [],
                        },
                        ensure_ascii=False,
                    ),
                )
            ),
        ]
    )
    backend = ExtractBackend()
    trace = DeepSeekExtractAgent(
        backend,
        model,
        available_companies=["海尔智家"],
    ).run(
        AgentTaskRequest(
            task_type="extract",
            query=(
                "海尔智家2024年使用权资产表中，累计折旧的期末余额合计是多少？"
                "不要误取账面原值的同名行。"
            ),
            top_k=1,
            max_rounds=4,
        )
    )

    assert trace.task_type == "extract"
    assert trace.plan.targets == []
    assert [
        requirement.requirement_id
        for requirement in trace.plan.fact_requirements
    ] == ["r1"]
    assert trace.result.requirement_evidence == {
        "r1": ["chunk-depreciation"]
    }
    assert trace.result.outcome == "answer"
    assert "3,316,847,042.37" in trace.result.answer.answer
    assert "9,158,716,606.73" not in trace.result.answer.answer
    assert [citation.chunk_id for citation in trace.result.answer.citations] == [
        "chunk-depreciation"
    ]
    assert backend.window_anchors == ["chunk-original"]
    assert model.tool_names == [
        ["search_evidence"],
        ["get_page_window"],
        ["submit_fact_requirements"],
        ["submit_extraction"],
    ]


class _LayoutInspector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def reconstruct_two_column_page(
        self,
        document_key: str,
        page_number: int,
    ) -> PageLayoutReconstruction:
        self.calls.append((document_key, page_number))
        return PageLayoutReconstruction(
            document_key=document_key,
            page_number=page_number,
            source_sha256="a" * 64,
            columns=[
                ReconstructedTextColumn(
                    label="左栏",
                    text="关键审计事项",
                    average_confidence=0.99,
                ),
                ReconstructedTextColumn(
                    label="右栏",
                    text="审计应对：测试相关内部控制",
                    average_confidence=0.99,
                ),
            ],
        )


def test_extract_agent_adds_rendered_two_column_evidence_for_audit_task() -> None:
    inspector = _LayoutInspector()
    expected = inspector.reconstruct_two_column_page("haier-2024", 185)
    inspector.calls.clear()
    model = ExtractScriptedModel(
        [
            _response(
                ModelToolCall(
                    call_id="search-1",
                    name="search_evidence",
                    arguments=json.dumps(
                        {"query": "关键审计事项 审计应对", "top_k": 1},
                        ensure_ascii=False,
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="window-1",
                    name="get_page_window",
                    arguments=json.dumps(
                        {
                            "anchor_chunk_id": "chunk-original",
                            "before_pages": 1,
                            "after_pages": 1,
                        }
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="requirements-1",
                    name="submit_fact_requirements",
                    arguments=json.dumps(
                        {
                            "requirements": [
                                {
                                    "requirement_id": "r1",
                                    "description": "审计应对：测试相关内部控制",
                                    "subject": "关键审计事项",
                                    "subject_scope": "document",
                                    "fact_period": "2024年",
                                    "evidence_type": "audit_response",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="submit-1",
                    name="submit_extraction",
                    arguments=json.dumps(
                        {
                            "status": "answer",
                            "message": "抽取完成",
                            "facts": [
                                {
                                    "text": "审计应对包括测试相关内部控制",
                                    "evidence_chunk_ids": [
                                        expected.evidence_chunk_id
                                    ],
                                    "requirement_ids": ["r1"],
                                }
                            ],
                            "gaps": [],
                        },
                        ensure_ascii=False,
                    ),
                )
            ),
        ]
    )

    trace = DeepSeekExtractAgent(
        ExtractBackend(),
        model,
        available_companies=["海尔智家"],
        layout_inspector=inspector,
    ).run(
        AgentTaskRequest(
            task_type="extract",
            query="海尔智家2024年关键审计事项的审计应对是什么？",
            top_k=1,
            max_rounds=4,
        )
    )

    assert inspector.calls == [("haier-2024", 185), ("haier-2024", 186)]
    assert [call.tool for call in trace.tool_calls] == [
        "search_evidence",
        "get_page_window",
        "reconstruct_page_layout",
    ]
    assert trace.result.requirement_evidence == {
        "r1": [expected.evidence_chunk_id]
    }
    assert any(
        item.section_path[-1] == "PDF页级双栏重建"
        for item in trace.evidence_memory.items
    )


def test_extract_agent_searches_only_requirements_without_candidate_evidence() -> None:
    class RequirementGapBackend(ExtractBackend):
        def __init__(self) -> None:
            super().__init__()
            self.group_profit = self._chunk(
                "chunk-group-profit",
                "2024年归属于母公司股东的净利润为120亿元",
                ["集团主要经营业绩"],
                10,
                10,
            )

        def search(self, request: SearchRequest, request_id: str) -> SearchResponse:
            if "集团" not in request.query:
                return super().search(request, request_id)
            self.search_requests.append(request)
            return SearchResponse(
                request_id=request_id,
                trace_id="trace-requirement-gap",
                index_id=self.manifest.index_id,
                mode=request.mode or "lexical",
                reranked=False,
                base_candidate_k=10,
                effective_candidate_k=10,
                candidate_budget_reason="test",
                took_ms=1.0,
                hits=[SearchHit(rank=1, chunk=self.group_profit, score=1.0)],
            )

    model = ExtractScriptedModel(
        [
            _response(
                ModelToolCall(
                    call_id="search-1",
                    name="search_evidence",
                    arguments=json.dumps(
                        {"query": "归母净利润", "top_k": 1},
                        ensure_ascii=False,
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="window-1",
                    name="get_page_window",
                    arguments=json.dumps(
                        {
                            "anchor_chunk_id": "chunk-original",
                            "before_pages": 1,
                            "after_pages": 1,
                        }
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="requirements-1",
                    name="submit_fact_requirements",
                    arguments=json.dumps(
                        {
                            "requirements": [
                                {
                                    "requirement_id": "r1",
                                    "description": "2024年归母净利润",
                                    "subject": "海尔智家",
                                    "subject_scope": "group",
                                    "fact_period": "2024年",
                                    "evidence_type": "table_value",
                                    "candidate_evidence_chunk_ids": [],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="submit-1",
                    name="submit_extraction",
                    arguments=json.dumps(
                        {
                            "status": "answer",
                            "message": "抽取完成",
                            "facts": [
                                {
                                    "text": (
                                        "海尔智家2024年归属于母公司股东的"
                                        "净利润为120亿元"
                                    ),
                                    "evidence_chunk_ids": ["chunk-group-profit"],
                                    "requirement_ids": ["r1"],
                                }
                            ],
                            "gaps": [],
                        },
                        ensure_ascii=False,
                    ),
                )
            ),
        ]
    )
    backend = RequirementGapBackend()

    trace = DeepSeekExtractAgent(
        backend,
        model,
        available_companies=["海尔智家"],
    ).run(
        AgentTaskRequest(
            task_type="extract",
            query="海尔智家2024年归母净利润是多少？",
            top_k=1,
            max_rounds=4,
        )
    )

    assert trace.result.outcome == "answer"
    assert trace.plan.fact_requirements[0].candidate_evidence_chunk_ids == [
        "chunk-group-profit"
    ]
    assert len(backend.search_requests) == 2
    assert "集团" in backend.search_requests[1].query
    assert trace.result.requirement_evidence == {"r1": ["chunk-group-profit"]}


def test_extract_result_rejects_segment_fact_for_group_requirement() -> None:
    plan = AgentTaskPlan(
        task_type="extract",
        fact_requirements=[
            AtomicFactRequirement(
                requirement_id="r1",
                description="2024年归母净利润",
                subject="中国平安",
                subject_scope="group",
                fact_period="2024年",
                evidence_type="table_value",
                candidate_evidence_chunk_ids=["chunk-segment"],
            )
        ],
    )
    memory = EvidenceMemory(
        index_id="index-test",
        items=[
            AgentEvidence(
                chunk_id="chunk-segment",
                content_sha256="0" * 64,
                target_ids=["task:extract"],
                document_id="document-pingan",
                company_name="中国平安",
                report_year=2024,
                page_start=39,
                page_end=39,
                section_path=["寿险及健康险业务"],
                excerpt="寿险及健康险业务净利润92,097百万元",
            )
        ],
    )
    submission = SubmitExtractionArguments(
        status="answer",
        message="抽取完成",
        facts=[
            SubmittedExtractFact(
                text="寿险及健康险业务净利润为92,097百万元",
                evidence_chunk_ids=["chunk-segment"],
                requirement_ids=["r1"],
            )
        ],
        gaps=[],
    )

    result, errors = DeepSeekExtractAgent._result(submission, plan, memory)

    assert result is None
    assert any("business-segment fact" in error for error in errors)
    assert _candidate_scope_compatible(
        plan.fact_requirements[0], memory.items[0]
    ) is False


def test_requirement_validation_does_not_treat_risk_discount_rate_as_audit() -> None:
    plan = AgentTaskPlan(task_type="extract", required_metrics=["新业务价值"])
    requirements = [
        AtomicFactRequirement(
            requirement_id="r1",
            description="按最新投资回报率和风险贴现率假设计算的新业务价值",
            subject="寿险及健康险",
            subject_scope="business_segment",
            fact_period="2023年",
            evidence_type="table_value",
        )
    ]

    errors = _validate_fact_requirements(
        plan,
        requirements,
        EvidenceMemory(index_id="index-test"),
        "按最新投资回报率和风险贴现率假设计算的新业务价值是多少？",
    )

    assert not any("audit_risk" in error for error in errors)


def test_requirement_validation_allows_parent_profit_for_named_segments() -> None:
    plan = AgentTaskPlan(task_type="extract")
    requirements = [
        AtomicFactRequirement(
            requirement_id=f"r{index}",
            description=f"{subject}归属于母公司股东的营运利润",
            subject=subject,
            subject_scope="business_segment",
            fact_period="2023年",
            evidence_type="table_value",
        )
        for index, subject in enumerate(
            ("寿险及健康险", "财产保险", "银行"),
            start=1,
        )
    ]

    errors = _validate_fact_requirements(
        plan,
        requirements,
        EvidenceMemory(index_id="index-test"),
        "三项业务归属于母公司股东的营运利润分别是多少？",
    )

    assert errors == []


def test_requirement_completion_adds_only_missing_deterministic_metric() -> None:
    plan = AgentTaskPlan(
        task_type="extract",
        fact_periods=[2024],
        required_metrics=["个人客户数", "客均合同数", "留存率"],
    )
    requirements = [
        AtomicFactRequirement(
            requirement_id="r1",
            description="2024年个人客户数",
            evidence_type="table_value",
        ),
        AtomicFactRequirement(
            requirement_id="r2",
            description="持有四个及以上合同的客户留存率",
            evidence_type="table_value",
        ),
    ]

    completed = _complete_required_metric_requirements(plan, requirements)

    assert [item.requirement_id for item in completed] == ["r1", "r2", "r3"]
    assert completed[-1].description == "2024年 客均合同数"
    assert completed[-1].candidate_evidence_chunk_ids == []


def test_segment_candidate_uses_subject_anchor_without_metric_suffix() -> None:
    requirement = AtomicFactRequirement(
        requirement_id="r1",
        description="寿险及健康险业务归属于母公司股东的营运利润",
        subject="寿险及健康险业务归母营运利润",
        subject_scope="business_segment",
        fact_period="2023年",
        evidence_type="table_value",
    )
    evidence = AgentEvidence(
        chunk_id="chunk-segment",
        content_sha256="0" * 64,
        target_ids=["task:extract"],
        document_id="document-pingan",
        company_name="中国平安",
        report_year=2023,
        page_start=14,
        page_end=14,
        section_path=["寿险及健康险业务"],
        excerpt="寿险及健康险业务营运利润为105,070百万元",
    )

    assert _candidate_scope_compatible(requirement, evidence) is True


def test_calculate_agent_uses_only_cited_decimal_operands() -> None:
    model = ExtractScriptedModel(
        [
            _response(
                ModelToolCall(
                    call_id="search-1",
                    name="search_evidence",
                    arguments=json.dumps(
                        {"query": "使用权资产期末余额勾稽", "top_k": 1},
                        ensure_ascii=False,
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="window-1",
                    name="get_page_window",
                    arguments=json.dumps(
                        {
                            "anchor_chunk_id": "chunk-original",
                            "before_pages": 1,
                            "after_pages": 1,
                        }
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="calculate-1",
                    name="reconcile_subtraction",
                    arguments=json.dumps(
                        {
                            "left": {
                                "label": "账面原值期末余额合计",
                                "value": "9,158,716,606.73",
                                "evidence_chunk_id": "chunk-original",
                            },
                            "right": {
                                "label": "累计折旧期末余额合计",
                                "value": "3,316,847,042.37",
                                "evidence_chunk_id": "chunk-depreciation",
                            },
                            "expected": {
                                "label": "期末账面价值合计",
                                "value": "5,841,869,564.36",
                                "evidence_chunk_id": "chunk-carrying",
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            ),
        ]
    )
    trace = DeepSeekCalculateAgent(
        ExtractBackend(),
        model,
        available_companies=["海尔智家"],
    ).run(
        AgentTaskRequest(
            task_type="calculate",
            query=(
                "核对海尔智家2024年使用权资产：账面原值期末余额合计减累计折旧"
                "期末余额合计，是否等于期末账面价值合计？列出三个原值并给出差额。"
            ),
            top_k=1,
        )
    )

    assert trace.result.outcome == "answer"
    assert "9,158,716,606.73" in trace.result.answer.answer
    assert "3,316,847,042.37" in trace.result.answer.answer
    assert "5,841,869,564.36" in trace.result.answer.answer
    assert "差额为0.00元" in trace.result.answer.answer
    assert trace.tool_calls[-1].tool == "calculate"
    assert len(trace.result.answer.citations) == 3


class GeneralCalculateBackend:
    def __init__(self) -> None:
        self.manifest = SimpleNamespace(index_id="index-general-calculate-test")
        self.search_requests: list[SearchRequest] = []
        self.chunks = {
            year: self._chunk(year, value)
            for year, value in ((2023, "365,116"), (2024, "366,946"))
        }

    @staticmethod
    def _chunk(year: int, value: str) -> DocumentChunk:
        text = f"工商银行{year}年净利润为{value}百万元"
        if year == 2024:
            text += "，比较表列示2023年净利润为365,116百万元"
        return DocumentChunk(
            chunk_id=f"chunk-profit-{year}",
            document_id=f"icbc-{year}",
            document_key=f"cninfo:601398:annual:{year}",
            chunk_index=1,
            text=text,
            section_path=["财务摘要"],
            page_start=14,
            page_end=14,
            element_references=[
                ElementReference(
                    element_id=f"element-profit-{year}",
                    page_number=14,
                    bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                )
            ],
            character_count=len(text),
            estimated_token_count=len(text),
            company_name="工商银行",
            report_year=year,
        )

    def search(self, request: SearchRequest, request_id: str) -> SearchResponse:
        self.search_requests.append(request)
        year = request.filters.report_years[0]
        return SearchResponse(
            request_id=request_id,
            trace_id=f"trace-{year}",
            index_id=self.manifest.index_id,
            mode=request.mode or "lexical",
            reranked=False,
            base_candidate_k=10,
            effective_candidate_k=10,
            candidate_budget_reason="test",
            took_ms=1.0,
            hits=[SearchHit(rank=1, chunk=self.chunks[year], score=1.0)],
        )

    def page_window(self, anchor_chunk_id: str, **_: int) -> list[DocumentChunk]:
        year = int(anchor_chunk_id.rsplit("-", 1)[1])
        return [self.chunks[year]]


def test_general_calculate_agent_executes_grounded_growth_locally() -> None:
    model = ExtractScriptedModel(
        [
            _response(
                ModelToolCall(
                    call_id="calculate-growth",
                    name="execute_grounded_calculation",
                    arguments=json.dumps(
                        {
                            "status": "answer",
                            "message": "操作数齐全",
                            "operands": [
                                {
                                    "name": "profit_2024",
                                    "label": "2024年净利润",
                                    "value": "366,946",
                                    "evidence_chunk_id": "chunk-profit-2024",
                                },
                                {
                                    "name": "profit_2023",
                                    "label": "2023年净利润",
                                    "value": "365,116",
                                    "evidence_chunk_id": "chunk-profit-2024",
                                },
                            ],
                            "steps": [
                                {
                                    "name": "growth",
                                    "label": "2024年相对2023年净利润增长率",
                                    "operation": "growth_percent",
                                    "inputs": ["profit_2024", "profit_2023"],
                                    "decimals": 2,
                                    "unit": "%",
                                }
                            ],
                            "comparisons": [],
                        },
                        ensure_ascii=False,
                    ),
                )
            )
        ]
    )
    backend = GeneralCalculateBackend()
    trace = DeepSeekCalculateAgent(
        backend,
        model,
        available_companies=["工商银行"],
        available_report_years_by_company={"工商银行": [2023, 2024]},
    ).run(
        AgentTaskRequest(
            task_type="calculate",
            query="用年报披露的净利润计算工商银行2024年相对2023年的增长率，保留两位小数。",
            top_k=1,
        )
    )

    assert trace.result.outcome == "answer"
    assert "366,946" in trace.result.answer.answer
    assert "365,116" in trace.result.answer.answer
    assert "0.50%" in trace.result.answer.answer
    assert trace.plan.targets == []
    assert trace.tool_calls[-1].tool == "calculate"
    assert len(trace.result.answer.citations) == 2
    assert len(backend.search_requests) == 2
    assert model.tool_names == [["execute_grounded_calculation"]]


class VisualBackend:
    def __init__(self) -> None:
        self.manifest = SimpleNamespace(index_id="index-visual-test")
        text = (
            "权益结构图 H股股东 17.00% 国家能源投资集团有限责任公司 69.52% "
            "其他A股股东 13.48%"
        )
        self.graph_chunk = DocumentChunk(
            chunk_id="chunk-graph",
            document_id="shenhua-2023",
            document_key="cninfo:601088:annual:2023",
            chunk_index=511,
            text=text,
            section_path=["权益结构图"],
            page_start=306,
            page_end=306,
            element_references=[
                ElementReference(
                    element_id="element-graph",
                    page_number=306,
                    bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
                )
            ],
            character_count=len(text),
            estimated_token_count=len(text),
            company_name="中国神华",
            report_year=2023,
        )

    def search(self, request: SearchRequest, request_id: str) -> SearchResponse:
        return SearchResponse(
            request_id=request_id,
            trace_id="trace-visual",
            index_id=self.manifest.index_id,
            mode=request.mode or "lexical",
            reranked=False,
            base_candidate_k=10,
            effective_candidate_k=10,
            candidate_budget_reason="test",
            took_ms=1.0,
            hits=[SearchHit(rank=1, chunk=self.graph_chunk, score=1.0)],
        )


class VisualInspector:
    def inspect_relationship_rows(
        self, document_key: str, page_number: int
    ) -> PageRegionInspection:
        rows = []
        for index, (label, value) in enumerate(
            [
                ("H股股东", "17.00%"),
                ("国家能源投资集团有限责任公司", "69.52%"),
                ("其他A股股东", "13.48%"),
            ]
        ):
            y_position = 100 + index * 80
            rows.append(
                VisualRelationshipRow(
                    label=VisualTextNode(
                        node_id=f"label-{index}",
                        text=label,
                        bbox=BoundingBox(
                            x0=40, y0=y_position, x1=100, y1=y_position + 10
                        ),
                    ),
                    value=VisualTextNode(
                        node_id=f"value-{index}",
                        text=value,
                        bbox=BoundingBox(
                            x0=120, y0=y_position, x1=150, y1=y_position + 10
                        ),
                    ),
                    vertical_center_delta=0,
                    connector_present=True,
                )
            )
        return PageRegionInspection(
            document_key=document_key,
            page_number=page_number,
            page_width=595,
            page_height=842,
            source_sha256="0" * 64,
            relationship_rows=rows,
            drawing_count=3,
        )


def test_visual_graph_agent_validates_geometry_rows_and_sums_locally() -> None:
    model = ExtractScriptedModel(
        [
            _response(
                ModelToolCall(
                    call_id="search-1",
                    name="search_evidence",
                    arguments=json.dumps(
                        {"query": "权益结构图 持股比例", "top_k": 1},
                        ensure_ascii=False,
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="inspect-1",
                    name="inspect_page_region",
                    arguments=json.dumps(
                        {"anchor_chunk_id": "chunk-graph", "page_number": 306}
                    ),
                )
            ),
            _response(
                ModelToolCall(
                    call_id="sum-1",
                    name="sum_visual_percentages",
                    arguments=json.dumps(
                        {
                            "relationships": [
                                {
                                    "label": "H股股东",
                                    "value": "17.00%",
                                    "evidence_chunk_id": "chunk-graph",
                                },
                                {
                                    "label": "国家能源投资集团有限责任公司",
                                    "value": "69.52%",
                                    "evidence_chunk_id": "chunk-graph",
                                },
                                {
                                    "label": "其他A股股东",
                                    "value": "13.48%",
                                    "evidence_chunk_id": "chunk-graph",
                                },
                            ]
                        },
                        ensure_ascii=False,
                    ),
                )
            ),
        ]
    )
    trace = DeepSeekVisualGraphAgent(
        VisualBackend(),
        model,
        available_companies=["中国神华"],
        region_inspector=VisualInspector(),
    ).run(
        AgentTaskRequest(
            task_type="calculate",
            query=(
                "根据中国神华2023年报权益结构图，H股股东、国家能源投资集团有限"
                "责任公司、其他A股股东的持股比例分别是多少？三者合计是否为100%？"
            ),
            top_k=1,
        )
    )

    assert trace.result.outcome == "answer"
    assert "17.00%" in trace.result.answer.answer
    assert "69.52%" in trace.result.answer.answer
    assert "13.48%" in trace.result.answer.answer
    assert "100.00%" in trace.result.answer.answer
    assert [call.tool for call in trace.tool_calls] == [
        "search_evidence",
        "inspect_page_region",
        "calculate",
    ]
