import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import fitz
import pytest
from typer.testing import CliRunner

from findoc_rag.agent_tasks import (
    AgentEvidence,
    AgentTaskPlan,
    AgentTaskResult,
    AgentTaskStore,
    AgentTaskTrace,
    AtomicFactRequirement,
    EvidenceMemory,
    EvidenceVerificationTrace,
    SufficiencyDecision,
)
from findoc_rag.answer_generation import Citation, GeneratedAnswer
from findoc_rag.cli import app
from findoc_rag.documents.models import (
    BoundingBox,
    DocumentChunk,
    ElementReference,
    StructuredTable,
    StructuredTableCell,
)
from findoc_rag.human_review import HumanReviewStore, agent_task_trace_sha256
from findoc_rag.structured_tables import chunk_payload_sha256
from findoc_rag.table_cell_proof import build_table_cell_proofs
from findoc_rag.visual_inspection import PdfRegionInspector


def _manual_review_trace() -> AgentTaskTrace:
    now = datetime.now(UTC)
    source_chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="document-1",
        chunk_index=0,
        text="甲公司2024年营业收入为100亿元。",
        section_path=["主要会计数据"],
        page_start=10,
        page_end=10,
        element_references=[
            ElementReference(
                element_id="element-1",
                page_number=10,
                bbox=BoundingBox(x0=40, y0=100, x1=300, y1=120),
            )
        ],
        character_count=20,
        estimated_token_count=20,
        document_key="甲公司:2024",
        company_name="甲公司",
        report_year=2024,
    )
    source_chunk.structured_tables = [
        StructuredTable(
            table_id="chunk-1:annual_data",
            chunk_id="chunk-1",
            chunk_sha256=chunk_payload_sha256(source_chunk),
            table_type="annual_data",
            page_start=10,
            page_end=10,
            unit="亿元",
            source="coordinate",
            cells=[
                StructuredTableCell(
                    row="营业收入",
                    row_index=1,
                    column="2024年",
                    column_index=1,
                    value="100",
                    page_number=10,
                    value_bbox=BoundingBox(x0=200, y0=100, x1=250, y1=110),
                    coordinate_space="pymupdf_unrotated_page",
                )
            ],
        )
    ]
    evidence = AgentEvidence(
        chunk_id="chunk-1",
        content_sha256=chunk_payload_sha256(source_chunk),
        target_ids=["task:extract"],
        document_id="document-1",
        document_key="甲公司:2024",
        company_name="甲公司",
        report_year=2024,
        page_start=10,
        page_end=10,
        section_path=["主要会计数据"],
        excerpt=source_chunk.text,
        table_cell_proofs=build_table_cell_proofs(source_chunk),
    )
    candidate = AgentTaskResult(
        outcome="answer",
        answer=GeneratedAnswer(
            answer="甲公司2024年营业收入为100亿元。[1]",
            citations=[
                Citation(
                    ordinal=1,
                    chunk_id="chunk-1",
                    page_start=10,
                    page_end=10,
                    section_path=["主要会计数据"],
                    excerpt=evidence.excerpt,
                )
            ],
            provider="deepseek",
        ),
        target_evidence={"task:extract": ["chunk-1"]},
        requirement_claims={"r1": ["甲公司2024年营业收入为100亿元"]},
        requirement_evidence={"r1": ["chunk-1"]},
        requirement_scope_validated={"r1": True},
    )
    verification = EvidenceVerificationTrace(
        prompt_revision="test-review",
        prompt_sha256="c" * 64,
        routed=True,
        route_reason="support proof invalid",
        final_decision="manual_review",
        human_review_required=True,
        human_review_reasons=["support proof quote does not match cited evidence"],
        candidate_result=candidate,
    )
    return AgentTaskTrace(
        task_id="a" * 32,
        task_type="extract",
        runtime="deepseek_tool_calling",
        status="completed",
        stop_reason="evidence_verifier_manual_review",
        query="甲公司2024年营业收入是多少？",
        index_id="index-test",
        created_at=now,
        completed_at=now,
        rounds_completed=1,
        plan=AgentTaskPlan(
            task_type="extract",
            fact_requirements=[
                AtomicFactRequirement(
                    requirement_id="r1",
                    description="甲公司2024年营业收入",
                    subject="甲公司",
                    fact_period="2024",
                    evidence_type="table_value",
                )
            ],
        ),
        tool_calls=[],
        evidence_memory=EvidenceMemory(index_id="index-test", items=[evidence]),
        sufficiency=SufficiencyDecision(
            status="incomplete",
            evidence_count_by_target={"task:extract": 1},
            gaps=["verifiable evidence support proof"],
            requirement_gaps=["r1"],
        ),
        result=AgentTaskResult(
            outcome="abstain",
            answer=GeneratedAnswer(
                answer="系统已暂停自动回答并升级人工复核。",
                citations=[],
                provider="evidence-support-proof-gate",
                grounded=False,
            ),
            target_evidence={},
        ),
        evidence_verification=verification,
    )


def test_review_queue_preserves_candidate_and_resolves_approve_once(
    tmp_path: Path,
) -> None:
    trace = _manual_review_trace()
    task_store = AgentTaskStore(tmp_path / "tasks")
    task_store.save(trace)
    review_store = HumanReviewStore(tmp_path / "reviews")

    packet, packet_path = review_store.create(trace)
    duplicate, duplicate_path = review_store.create(trace)

    assert packet_path.is_file()
    assert duplicate == packet
    assert duplicate_path == packet_path
    assert packet.candidate_result is not None
    assert packet.candidate_result.answer.answer.startswith("甲公司2024年营业收入")
    assert packet.requirements[0].evidence[0].page_start == 10
    assert packet.requirements[0].table_cell_proofs[0].geometry_status == "coordinate"
    assert [item.status for item in review_store.list(status="pending")] == ["pending"]

    original_sha256 = agent_task_trace_sha256(task_store.load(trace.task_id))
    resolution, resolution_path = review_store.resolve(
        packet.review_id,
        task_store=task_store,
        decision="approve",
        reviewer="reviewer-a",
        comment="已核对第10页表格。",
    )

    assert resolution_path.is_file()
    assert resolution.final_outcome == "answer"
    assert resolution.evidence_chunk_ids == ["chunk-1"]
    assert review_store.inspect(packet.review_id).status == "resolved"
    assert agent_task_trace_sha256(task_store.load(trace.task_id)) == original_sha256
    with pytest.raises(ValueError, match="already resolved"):
        review_store.resolve(
            packet.review_id,
            task_store=task_store,
            decision="reject",
            reviewer="reviewer-b",
        )


def test_review_packet_can_include_bounded_pdf_region_proof(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "annual.pdf"
    document = fitz.open()
    for _ in range(10):
        document.new_page(width=400, height=500)
    document[9].insert_text((205, 108), "100", fontsize=10)
    document.save(pdf_path)
    document.close()
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_key": "甲公司:2024",
                        "local_file": "annual.pdf",
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    packet, _ = HumanReviewStore(tmp_path / "reviews").create(
        _manual_review_trace(),
        region_inspector=PdfRegionInspector(
            manifest_path,
            workspace=tmp_path,
        ),
        region_directory=tmp_path / "reviews" / "regions",
    )

    [region] = packet.requirements[0].region_proofs
    assert region.page_number == 10
    assert region.rendered_area_ratio <= 0.2
    assert (tmp_path / region.image_path).is_file()
    assert packet.requirements[0].region_proof_errors == []


def test_review_resolution_rejects_a_stale_source_trace(tmp_path: Path) -> None:
    trace = _manual_review_trace()
    task_store = AgentTaskStore(tmp_path / "tasks")
    task_store.save(trace)
    review_store = HumanReviewStore(tmp_path / "reviews")
    packet, _ = review_store.create(trace)
    task_store.save(trace.model_copy(update={"query": "已被修改的问题"}))

    with pytest.raises(ValueError, match="Task trace changed"):
        review_store.resolve(
            packet.review_id,
            task_store=task_store,
            decision="approve",
            reviewer="reviewer-a",
        )


def test_corrected_answer_must_use_evidence_from_the_review_packet(
    tmp_path: Path,
) -> None:
    trace = _manual_review_trace()
    task_store = AgentTaskStore(tmp_path / "tasks")
    task_store.save(trace)
    review_store = HumanReviewStore(tmp_path / "reviews")
    packet, _ = review_store.create(trace)

    with pytest.raises(ValueError, match="outside the immutable packet"):
        review_store.resolve(
            packet.review_id,
            task_store=task_store,
            decision="correct",
            reviewer="reviewer-a",
            corrected_answer="修正后的答案",
            evidence_chunk_ids=["invented-chunk"],
        )
    with pytest.raises(ValueError, match="at least one"):
        review_store.resolve(
            packet.review_id,
            task_store=task_store,
            decision="correct",
            reviewer="reviewer-a",
            corrected_answer="修正后的答案",
        )

    resolution, _ = review_store.resolve(
        packet.review_id,
        task_store=task_store,
        decision="correct",
        reviewer="reviewer-a",
        corrected_answer="甲公司2024年营业收入为100亿元。",
        evidence_chunk_ids=["chunk-1"],
    )

    assert resolution.final_answer == "甲公司2024年营业收入为100亿元。"
    assert resolution.evidence_chunk_ids == ["chunk-1"]


def test_approve_requires_every_candidate_citation_in_the_packet(
    tmp_path: Path,
) -> None:
    trace = _manual_review_trace()
    assert trace.evidence_verification is not None
    assert trace.evidence_verification.candidate_result is not None
    trace.evidence_verification.candidate_result.answer.citations[
        0
    ].chunk_id = "outside-candidate-evidence"
    task_store = AgentTaskStore(tmp_path / "tasks")
    task_store.save(trace)
    review_store = HumanReviewStore(tmp_path / "reviews")
    packet, _ = review_store.create(trace)

    with pytest.raises(ValueError, match="outside the immutable packet"):
        review_store.resolve(
            packet.review_id,
            task_store=task_store,
            decision="approve",
            reviewer="reviewer-a",
        )


def test_review_cli_lists_inspects_and_rejects_a_packet(tmp_path: Path) -> None:
    trace = _manual_review_trace()
    task_dir = tmp_path / "tasks"
    review_dir = tmp_path / "reviews"
    AgentTaskStore(task_dir).save(trace)
    packet, _ = HumanReviewStore(review_dir).create(trace)
    runner = CliRunner()

    listed = runner.invoke(
        app,
        ["agent", "review", "list", "--review-dir", str(review_dir)],
    )
    inspected = runner.invoke(
        app,
        [
            "agent",
            "review",
            "inspect",
            packet.review_id,
            "--review-dir",
            str(review_dir),
        ],
    )
    resolved = runner.invoke(
        app,
        [
            "agent",
            "review",
            "resolve",
            packet.review_id,
            "reject",
            "--reviewer",
            "reviewer-a",
            "--task-dir",
            str(task_dir),
            "--review-dir",
            str(review_dir),
        ],
    )

    assert listed.exit_code == 0, listed.output
    assert packet.review_id in listed.stdout
    assert inspected.exit_code == 0, inspected.output
    assert "Candidate answer" in inspected.stdout
    assert "pages 10-10" in inspected.stdout
    assert "Cell proof SHA-256" in inspected.stdout
    assert "row[1]=营业收入" in inspected.stdout
    assert resolved.exit_code == 0, resolved.output
    assert "Review resolved: reject" in resolved.stdout


def test_review_ids_reject_path_traversal(tmp_path: Path) -> None:
    store = HumanReviewStore(tmp_path)

    with pytest.raises(ValueError, match="Invalid review ID"):
        store.inspect("../outside")


def test_task_store_loads_legacy_verifier_computed_fields(tmp_path: Path) -> None:
    trace = _manual_review_trace()
    store = AgentTaskStore(tmp_path)
    path = store.save(trace)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence_verification"].update(
        {"request_count": 0, "input_tokens": None, "output_tokens": None}
    )
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    loaded = store.load(trace.task_id)

    assert loaded.evidence_verification is not None
    assert loaded.evidence_verification.request_count == 0
