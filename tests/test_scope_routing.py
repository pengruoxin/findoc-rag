from findoc_rag.documents.models import (
    BoundingBox,
    DocumentChunk,
    ElementReference,
    StructuredTable,
    StructuredTableCell,
)
from findoc_rag.indexing import SearchHit
from findoc_rag.scope_routing import (
    infer_query_scope,
    plan_candidate_budget,
    route_by_scope,
    route_structured_evidence,
)


def hit(rank: int, chunk_id: str, section: str) -> SearchHit:
    text = "营业收入 100"
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="document",
        chunk_index=rank,
        text=text,
        section_path=[section],
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
    )
    return SearchHit(rank=rank, chunk=chunk, score=1 / rank)


def test_scope_inference_prefers_explicit_period_and_scope_cues() -> None:
    quarterly = infer_query_scope("甲公司2024年分季度营业收入")
    segment = infer_query_scope("甲公司主营业务分产品收入")
    annual = infer_query_scope("甲公司2024年营业收入是多少")

    assert (quarterly.name, quarterly.confidence) == ("quarterly", "explicit")
    assert (segment.name, segment.confidence) == ("segment", "explicit")
    assert (annual.name, annual.confidence) == ("annual_summary", "default")


def test_scope_routing_promotes_matching_section_without_dropping_candidates() -> None:
    candidates = [
        hit(1, "audit", "财务报告 > 关键审计事项 > 收入确认"),
        hit(2, "segment", "管理层讨论与分析 > 主营业务分行业、分产品"),
        hit(3, "annual", "公司简介和主要财务指标 > 主要会计数据"),
    ]

    scope, ranked = route_by_scope("甲公司2024年营业收入是多少", candidates, top_k=3)

    assert scope.name == "annual_summary"
    assert [item.chunk.chunk_id for item in ranked] == ["annual", "audit", "segment"]
    assert ranked[0].retrieval_rank == 3
    assert ranked[0].scope_rank_delta == 2
    assert {item.chunk.chunk_id for item in ranked} == {"audit", "segment", "annual"}


def test_adaptive_candidate_budget_is_bounded_and_explainable() -> None:
    _, annual = plan_candidate_budget(
        "甲公司2024年营业收入是多少", 20, maximum_candidate_k=80
    )
    _, quarterly = plan_candidate_budget(
        "甲公司2024年分季度营业收入", 20, maximum_candidate_k=100
    )
    _, disabled = plan_candidate_budget(
        "甲公司2024年营业收入是多少", 20, maximum_candidate_k=100, enabled=False
    )

    assert annual.effective_candidate_k == 80
    assert annual.expanded is True
    assert annual.inferred_scope == "annual_summary"
    assert quarterly.effective_candidate_k == 40
    assert disabled.effective_candidate_k == 20
    assert disabled.expanded is False


def test_structured_evidence_router_promotes_verified_table_without_gold_metadata() -> None:
    plain = hit(1, "plain", "行业经营情况")
    quarterly = hit(6, "quarterly", "分季度主要财务数据")
    table = StructuredTable(
        table_id="quarterly:table",
        chunk_id="quarterly",
        chunk_sha256="0" * 64,
        table_type="quarterly",
        page_start=1,
        page_end=1,
        source="coordinate",
        cells=[StructuredTableCell(row="营业收入", column="第一季度", value="1")],
    )
    quarterly = quarterly.model_copy(
        update={"chunk": quarterly.chunk.model_copy(update={"structured_tables": [table]})}
    )

    routed = route_structured_evidence(
        "2024年各季度营业收入是多少", [plain, quarterly], top_k=1
    )

    assert routed[0].chunk.chunk_id == "quarterly"
    assert routed[0].retrieval_rank == 6
    assert routed[0].scope_score == 20
