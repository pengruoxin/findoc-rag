from pathlib import Path
from types import SimpleNamespace

import pytest

from findoc_rag.agent_tasks import (
    AgentEvidence,
    AgentTaskRequest,
    AgentTaskStore,
    CompareTaskController,
    EvidenceMemory,
    extract_metric_requirements,
    judge_sufficiency,
    plan_compare_task,
    plan_document_task,
)
from findoc_rag.answer_generation import Citation, ClaimCitation, GeneratedAnswer
from findoc_rag.documents.models import BoundingBox, DocumentChunk, ElementReference
from findoc_rag.indexing import SearchHit
from findoc_rag.service import SearchRequest, SearchResponse


def _hit(company: str, chunk_id: str) -> SearchHit:
    text = f"{company} 2024年营业收入为100元"
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


class FakeSearchBackend:
    def __init__(self, results: dict[str, list[SearchHit]]) -> None:
        self.results = results
        self.calls: list[SearchRequest] = []
        self.manifest = SimpleNamespace(index_id="index-agent-test")

    def search(self, request: SearchRequest, request_id: str) -> SearchResponse:
        self.calls.append(request)
        company = request.filters.company_names[0]
        hits = self.results.get(company, [])
        return SearchResponse(
            request_id=request_id,
            trace_id=f"retrieval-{len(self.calls)}",
            index_id=self.manifest.index_id,
            mode=request.mode or "lexical",
            reranked=False,
            base_candidate_k=10,
            effective_candidate_k=10,
            candidate_budget_reason="test",
            took_ms=1.0,
            hits=hits,
        )


class EvidenceOnlyGenerator:
    def generate(self, query: str, hits: list[SearchHit]) -> GeneratedAnswer:
        assert query
        assert hits
        return GeneratedAnswer(
            answer="已找到比较证据。",
            citations=[],
            provider="evidence-only",
            grounded=False,
        )


class TargetAnswerGenerator:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def generate(self, query: str, hits: list[SearchHit]) -> GeneratedAnswer:
        self.queries.append(query)
        company = hits[0].chunk.company_name
        assert company is not None
        other = "乙公司" if company == "甲公司" else "甲公司"
        assert company in query
        assert other not in query
        claim = f"{company}营业收入为100元"
        return GeneratedAnswer(
            answer=f"{claim}[1]",
            citations=[
                Citation(
                    ordinal=1,
                    chunk_id=hits[0].chunk.chunk_id,
                    page_start=1,
                    page_end=1,
                    section_path=["主要会计数据"],
                    excerpt=hits[0].chunk.text,
                )
            ],
            provider="test-grounded",
            grounded=True,
            claim_citations=[ClaimCitation(claim=claim, citation_ordinals=[1])],
        )


class MixedCoverageGenerator(TargetAnswerGenerator):
    def generate(self, query: str, hits: list[SearchHit]) -> GeneratedAnswer:
        if hits[0].chunk.company_name == "甲公司":
            return GeneratedAnswer(
                answer="已找到相关证据。",
                citations=[
                    Citation(
                        ordinal=1,
                        chunk_id=hits[0].chunk.chunk_id,
                        page_start=1,
                        page_end=1,
                        section_path=["主要会计数据"],
                        excerpt=hits[0].chunk.text,
                    )
                ],
                provider="evidence-only",
                grounded=False,
            )
        return super().generate(query, hits)


def _controller(
    results: dict[str, list[SearchHit]],
) -> tuple[CompareTaskController, FakeSearchBackend]:
    backend = FakeSearchBackend(results)
    return (
        CompareTaskController(
            backend,
            EvidenceOnlyGenerator(),
            available_companies=["甲公司", "乙公司"],
        ),
        backend,
    )


def test_compare_task_decomposes_targets_and_stops_when_sufficient() -> None:
    controller, backend = _controller(
        {"甲公司": [_hit("甲公司", "chunk-a")], "乙公司": [_hit("乙公司", "chunk-b")]}
    )

    trace = controller.run(AgentTaskRequest(query="比较甲公司和乙公司2024年营业收入", top_k=1))

    assert trace.status == "completed"
    assert trace.stop_reason == "sufficient_evidence"
    assert trace.rounds_completed == 1
    assert trace.sufficiency.status == "sufficient"
    assert trace.result.outcome == "evidence_only"
    assert len(trace.evidence_memory.items) == 2
    assert [call.filters.company_names for call in backend.calls] == [
        ["甲公司"],
        ["乙公司"],
    ]


def test_compare_task_retries_only_the_gap_and_stops_on_no_new_evidence() -> None:
    controller, backend = _controller({"甲公司": [_hit("甲公司", "chunk-a")]})

    trace = controller.run(AgentTaskRequest(query="比较甲公司和乙公司2024年营业收入", top_k=1))

    assert trace.status == "completed"
    assert trace.stop_reason == "no_new_evidence"
    assert trace.rounds_completed == 2
    assert trace.sufficiency.status == "incomplete"
    assert trace.result.outcome == "abstain"
    assert [call.filters.company_names for call in backend.calls] == [
        ["甲公司"],
        ["乙公司"],
        ["乙公司"],
    ]


def test_compare_task_answers_each_target_and_rewrites_citation_ordinals() -> None:
    backend = FakeSearchBackend(
        {"甲公司": [_hit("甲公司", "chunk-a")], "乙公司": [_hit("乙公司", "chunk-b")]}
    )
    generator = TargetAnswerGenerator()
    controller = CompareTaskController(
        backend,
        generator,
        available_companies=["甲公司", "乙公司"],
    )

    trace = controller.run(AgentTaskRequest(query="比较甲公司和乙公司2024年营业收入", top_k=1))

    assert trace.result.outcome == "answer"
    assert trace.result.answer.provider == "agent-comparison"
    assert "甲公司：甲公司营业收入为100元[1]" in trace.result.answer.answer
    assert "乙公司：乙公司营业收入为100元[2]" in trace.result.answer.answer
    assert [citation.chunk_id for citation in trace.result.answer.citations] == [
        "chunk-a",
        "chunk-b",
    ]
    assert len(trace.result.target_answers) == 2


def test_compare_task_never_claims_complete_answer_for_partial_target_coverage() -> None:
    backend = FakeSearchBackend(
        {"甲公司": [_hit("甲公司", "chunk-a")], "乙公司": [_hit("乙公司", "chunk-b")]}
    )
    controller = CompareTaskController(
        backend,
        MixedCoverageGenerator(),
        available_companies=["甲公司", "乙公司"],
    )

    trace = controller.run(AgentTaskRequest(query="比较甲公司和乙公司2024年营业收入", top_k=1))

    assert trace.sufficiency.status == "sufficient"
    assert trace.result.outcome == "evidence_only"
    assert trace.result.answer.grounded is False
    assert trace.result.answer.provider == "agent-comparison-evidence-only"


def test_compare_task_requires_two_explicit_targets() -> None:
    controller, backend = _controller({})

    trace = controller.run(AgentTaskRequest(query="甲公司2024年营业收入是多少"))

    assert trace.status == "needs_clarification"
    assert trace.stop_reason == "needs_clarification"
    assert trace.result.outcome == "clarify"
    assert backend.calls == []


def test_extract_plan_keeps_gold_targets_empty_and_scopes_one_report() -> None:
    plan = plan_document_task(
        AgentTaskRequest(
            task_type="extract",
            query=(
                "海尔智家2024年使用权资产表中，累计折旧的期末余额合计是多少？"
                "不要误取账面原值的同名行。"
            ),
        ),
        available_companies=["海尔智家"],
    )

    assert plan.task_type == "extract"
    assert plan.targets == []
    assert plan.document_scope is not None
    assert plan.document_scope.company_names == ["海尔智家"]
    assert plan.document_scope.report_years == [2024]
    assert plan.document_year == 2024
    assert plan.fact_periods == [2024]
    assert plan.document_scope_resolution == "mentioned_available_year"
    assert "二、累计折旧" in plan.retrieval_hint


def test_extract_plan_separates_explicit_document_year_from_future_fact() -> None:
    plan = plan_document_task(
        AgentTaskRequest(
            task_type="extract",
            query=(
                "只依据美的集团2023年年报，回答2024年实际营业收入；"
                "若未披露必须拒答。"
            ),
        ),
        available_companies=["美的集团"],
        available_report_years_by_company={"美的集团": [2023, 2024]},
    )

    assert plan.clarification is None
    assert plan.document_scope is not None
    assert plan.document_scope.report_years == [2023]
    assert plan.document_year == 2023
    assert plan.fact_periods == [2024]
    assert plan.document_scope_resolution == "explicit_document_year"


def test_extract_plan_records_all_query_metrics_before_model_planning() -> None:
    plan = plan_document_task(
        AgentTaskRequest(
            task_type="extract",
            query=(
                "中国平安2024年归母营运利润、归母净利润，以及寿险及健康险"
                "新业务价值和内含价值分别是多少？"
            ),
        ),
        available_companies=["中国平安"],
    )

    assert plan.required_metrics == [
        "归母营运利润",
        "归母净利润",
        "新业务价值",
        "内含价值",
    ]


def test_extract_plan_uses_unique_index_document_for_yearless_policy() -> None:
    plan = plan_document_task(
        AgentTaskRequest(
            task_type="extract",
            query="伊利股份存货采用什么计价、盘存和低值易耗品摊销政策？",
        ),
        available_companies=["伊利股份"],
        available_report_years_by_company={"伊利股份": [2024]},
    )

    assert plan.clarification is None
    assert plan.document_scope is not None
    assert plan.document_scope.report_years == [2024]
    assert plan.document_year == 2024
    assert plan.fact_periods == []
    assert plan.document_scope_resolution == "unique_index_document"


def test_extract_plan_keeps_yearless_multi_document_company_ambiguous() -> None:
    plan = plan_document_task(
        AgentTaskRequest(
            task_type="extract",
            query="美的集团存货采用什么计价政策？",
        ),
        available_companies=["美的集团"],
        available_report_years_by_company={"美的集团": [2023, 2024]},
    )

    assert plan.document_scope is None
    assert plan.clarification is not None


def test_extract_plan_marks_future_actual_as_unavailable_in_local_corpus() -> None:
    plan = plan_document_task(
        AgentTaskRequest(
            task_type="extract",
            query="工商银行2025年全年实际营业收入和净利润分别是多少？",
        ),
        available_companies=["工商银行"],
        available_report_years_by_company={"工商银行": [2023, 2024]},
    )

    assert plan.clarification is None
    assert plan.document_scope is None
    assert plan.fact_periods == [2025]
    assert plan.corpus_unavailable_reason is not None
    assert "2025年实际事实" in plan.corpus_unavailable_reason


def test_extract_plan_uses_latest_index_document_for_future_forecast() -> None:
    plan = plan_document_task(
        AgentTaskRequest(
            task_type="extract",
            query="工商银行预计2025年净利润目标是多少？",
        ),
        available_companies=["工商银行"],
        available_report_years_by_company={"工商银行": [2023, 2024]},
    )

    assert plan.corpus_unavailable_reason is None
    assert plan.clarification is None
    assert plan.document_scope is not None
    assert plan.document_scope.report_years == [2024]
    assert plan.document_year == 2024
    assert plan.fact_periods == [2025]
    assert plan.document_scope_resolution == "latest_index_document_for_forecast"


def test_extract_plan_does_not_overreject_missing_historical_report_year() -> None:
    plan = plan_document_task(
        AgentTaskRequest(
            task_type="extract",
            query="工商银行2022年净利润是多少？",
        ),
        available_companies=["工商银行"],
        available_report_years_by_company={"工商银行": [2023, 2024]},
    )

    assert plan.corpus_unavailable_reason is None


def test_compare_plan_expands_company_year_pairs_within_budget() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(query="比较甲公司和乙公司2023与2024年营业收入"),
        available_companies=["甲公司", "乙公司"],
    )

    assert len(plan.targets) == 4
    assert {target.filters.report_years[0] for target in plan.targets} == {2023, 2024}


def test_compare_plan_extracts_all_metrics_into_retrieval_hint() -> None:
    query = (
        "比较工商银行2023年和2024年的净利润、资产总额、"
        "客户贷款及垫款总额和客户存款，分别判断增减。"
    )
    plan = plan_compare_task(
        AgentTaskRequest(query=query),
        available_companies=["工商银行"],
    )

    assert plan.required_metrics == [
        "净利润",
        "资产总额",
        "客户贷款及垫款总额",
        "客户存款",
    ]
    assert all(
        all(metric in target.retrieval_hint for metric in plan.required_metrics)
        for target in plan.targets
    )


def test_metric_extraction_prefers_specific_overlapping_terms() -> None:
    assert extract_metric_requirements(
        "比较归母净利润、研发投入金额和研发投入强度"
    ) == ["归母净利润", "研发投入金额", "研发投入强度"]


def test_sufficiency_reports_metric_gaps_after_target_has_evidence() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(query="比较工商银行2023年和2024年净利润与资产总额"),
        available_companies=["工商银行"],
    )
    memory = EvidenceMemory(
        index_id="index-test",
        items=[
            AgentEvidence(
                chunk_id=f"chunk-{target.fact_year}",
                content_sha256="0" * 64,
                target_ids=[target.target_id],
                document_id=f"document-{target.fact_year}",
                company_name="工商银行",
                report_year=target.fact_year,
                page_start=1,
                page_end=1,
                section_path=["主要会计数据"],
                excerpt=f"{target.fact_year}年净利润为100亿元",
            )
            for target in plan.targets
        ],
    )

    decision = judge_sufficiency(plan, memory, minimum_per_target=1)

    assert decision.status == "incomplete"
    assert set(decision.gaps) == {target.target_id for target in plan.targets}
    assert all(
        metrics == ["资产总额"]
        for metrics in decision.metric_gaps_by_target.values()
    )


def test_compare_plan_represents_consolidated_and_parent_statement_scopes() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(query="比较海尔智家2024年合并口径营业收入与母公司口径营业收入"),
        available_companies=["海尔智家"],
    )

    assert [target.target_id for target in plan.targets] == [
        "scope:海尔智家:consolidated:year:2024",
        "scope:海尔智家:parent:year:2024",
    ]
    assert [target.statement_scope for target in plan.targets] == [
        "consolidated",
        "parent",
    ]
    assert all(target.filters.report_years == [2024] for target in plan.targets)
    assert "近三年主要会计数据" in plan.targets[0].retrieval_hint
    assert "母公司财务报表主要项目注释" in plan.targets[1].retrieval_hint


def test_restatement_plan_separates_fact_year_from_document_vintage() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(
            query=("海尔智家2024年报对2023年营业收入给出了调整前和调整后两个数，比较并计算调整额")
        ),
        available_companies=["海尔智家"],
    )

    assert [target.target_id for target in plan.targets] == [
        "fact:海尔智家:revenue:year:2023:as_reported",
        "fact:海尔智家:revenue:year:2023:restated_in_2024",
    ]
    assert all(target.fact_year == 2023 for target in plan.targets)
    assert all(target.document_year == 2024 for target in plan.targets)
    assert all(target.filters.report_years == [2024] for target in plan.targets)


def test_adjusted_four_target_plan_searches_comparatives_in_latest_report() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(
            query=(
                "比较海尔智家和长江电力2023年、2024年营业收入；"
                "2023年统一采用各自2024年报披露的调整后比较数"
            )
        ),
        available_companies=["海尔智家", "长江电力"],
    )

    assert len(plan.targets) == 4
    assert all(target.filters.report_years == [2024] for target in plan.targets)
    assert {target.fact_year: target.value_version for target in plan.targets[:2]} == {
        2023: "restated",
        2024: "current",
    }
    assert all("调整后 调整前" in target.retrieval_hint for target in plan.targets)


def test_growth_basis_plan_preserves_both_document_vintages() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(
            query=("长江电力2024年营业收入同比增长8.12%的比较基数是2023年调整前还是调整后？")
        ),
        available_companies=["长江电力"],
    )

    assert [target.filters.report_years for target in plan.targets] == [[2024], [2023]]
    assert [target.document_year for target in plan.targets] == [2024, 2023]
    assert len({target.cross_target_group for target in plan.targets}) == 1
    assert plan.targets[0].cross_target_group is not None


def test_future_actual_targets_never_share_available_year_evidence() -> None:
    plan = plan_compare_task(
        AgentTaskRequest(
            query=(
                "只依据海尔智家2024年报，比较其2024年与2025年实际营业收入。"
                "若2025年实际数未披露，必须明确拒答。"
            )
        ),
        available_companies=["海尔智家"],
    )

    assert [target.cross_target_group for target in plan.targets] == [None, None]


def test_agent_task_store_round_trips_and_rejects_path_traversal(
    tmp_path: Path,
) -> None:
    controller, _ = _controller(
        {"甲公司": [_hit("甲公司", "chunk-a")], "乙公司": [_hit("乙公司", "chunk-b")]}
    )
    trace = controller.run(AgentTaskRequest(query="比较甲公司和乙公司2024年营业收入", top_k=1))
    store = AgentTaskStore(tmp_path)

    path = store.save(trace)

    assert path.is_file()
    assert store.load(trace.task_id) == trace
    with pytest.raises(ValueError, match="Invalid task ID"):
        store.load("../outside")
