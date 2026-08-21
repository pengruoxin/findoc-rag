"""Evaluate the P4-F human-review lifecycle on one stored real Agent trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from findoc_rag.agent_tasks import (
    AgentTaskResult,
    AgentTaskStore,
    AgentTaskTrace,
    EvidenceVerificationTrace,
    SufficiencyDecision,
)
from findoc_rag.answer_generation import GeneratedAnswer
from findoc_rag.human_review import HumanReviewStore, agent_task_trace_sha256

DEFAULT_SOURCE = Path(
    "reports/agent/"
    "agent-hard-v3-dev-deepseek-p3b2-authority-ranking-extract-posthoc-v3.json"
)
DEFAULT_OUTPUT = Path("reports/agent/agent-p4f-human-review-workflow-v1.json")
SOURCE_CASE_ID = "v3_601318_y23_segments"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source_trace(path: Path) -> AgentTaskTrace:
    report = json.loads(path.read_text(encoding="utf-8"))
    payload = next(
        item["trace"] for item in report["items"] if item["case_id"] == SOURCE_CASE_ID
    )
    trace = AgentTaskTrace.model_validate(payload)
    if trace.result.outcome != "answer":
        raise ValueError("Source case must contain an answered Agent trace")
    return trace


def _manual_trace(source: AgentTaskTrace, label: str) -> AgentTaskTrace:
    task_id = hashlib.sha256(f"{source.task_id}:{label}".encode()).hexdigest()[:32]
    candidate = source.result.model_copy(deep=True)
    verification = EvidenceVerificationTrace(
        prompt_revision="p4f-workflow-evaluation",
        prompt_sha256=hashlib.sha256(b"p4f-workflow-evaluation").hexdigest(),
        routed=True,
        route_reason="stored-trace workflow evaluation",
        final_decision="manual_review",
        human_review_required=True,
        human_review_reasons=["support proof requires human source confirmation"],
        candidate_result=candidate,
    )
    return source.model_copy(
        deep=True,
        update={
            "task_id": task_id,
            "stop_reason": "evidence_verifier_manual_review",
            "completed_at": datetime.now(UTC),
            "sufficiency": SufficiencyDecision(
                status="incomplete",
                evidence_count_by_target={"task:extract": len(source.evidence_memory.items)},
                gaps=["verifiable evidence support proof"],
                requirement_gaps=[
                    requirement.requirement_id
                    for requirement in source.plan.fact_requirements
                ],
            ),
            "result": AgentTaskResult(
                outcome="abstain",
                answer=GeneratedAnswer(
                    answer="系统已暂停自动回答并升级人工复核。",
                    citations=[],
                    provider="evidence-support-proof-gate",
                    grounded=False,
                ),
                target_evidence={},
            ),
            "evidence_verification": verification,
        },
    )


def _run_contract(source: AgentTaskTrace) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    def record(check: str, passed: bool, detail: str) -> None:
        results.append({"check": check, "passed": passed, "detail": detail})
        if not passed:
            raise AssertionError(f"{check}: {detail}")

    with TemporaryDirectory(prefix="findoc-p4f-") as temporary:
        root = Path(temporary)

        approve_trace = _manual_trace(source, "approve")
        approve_tasks = AgentTaskStore(root / "approve" / "tasks")
        approve_reviews = HumanReviewStore(root / "approve" / "reviews")
        approve_tasks.save(approve_trace)
        packet, _ = approve_reviews.create(approve_trace)
        record(
            "candidate_preserved",
            packet.candidate_result == source.result,
            "withheld DeepSeek result is present in the immutable packet",
        )
        record(
            "claim_page_evidence_visible",
            any(requirement.claims and requirement.evidence for requirement in packet.requirements),
            "at least one atomic claim exposes chunk hash, page, section, and excerpt",
        )
        record(
            "durable_pending_queue",
            len(approve_reviews.list(status="pending")) == 1,
            "new packet is discoverable as pending",
        )
        source_hash_before = agent_task_trace_sha256(approve_tasks.load(approve_trace.task_id))
        approved, _ = approve_reviews.resolve(
            packet.review_id,
            task_store=approve_tasks,
            decision="approve",
            reviewer="evaluation-reviewer",
        )
        record(
            "approve_lifecycle",
            approved.final_outcome == "answer"
            and approved.final_answer == source.result.answer.answer,
            "approval restores the exact withheld answer",
        )
        record(
            "source_trace_immutable",
            agent_task_trace_sha256(approve_tasks.load(approve_trace.task_id))
            == source_hash_before,
            "resolution does not overwrite the original model trace",
        )
        try:
            approve_reviews.resolve(
                packet.review_id,
                task_store=approve_tasks,
                decision="reject",
                reviewer="evaluation-reviewer-2",
            )
        except ValueError as exc:
            duplicate_blocked = "already resolved" in str(exc)
        else:
            duplicate_blocked = False
        record(
            "one_shot_resolution",
            duplicate_blocked,
            "a second decision cannot replace the first audit record",
        )

        correct_trace = _manual_trace(source, "correct")
        correct_tasks = AgentTaskStore(root / "correct" / "tasks")
        correct_reviews = HumanReviewStore(root / "correct" / "reviews")
        correct_tasks.save(correct_trace)
        correct_packet, _ = correct_reviews.create(correct_trace)
        evidence_id = next(
            evidence.chunk_id
            for requirement in correct_packet.requirements
            for evidence in requirement.evidence
        )
        corrected, _ = correct_reviews.resolve(
            correct_packet.review_id,
            task_store=correct_tasks,
            decision="correct",
            reviewer="evaluation-reviewer",
            corrected_answer="人工修正后的证据绑定答案。",
            evidence_chunk_ids=[evidence_id],
        )
        record(
            "correct_lifecycle",
            corrected.final_outcome == "answer"
            and corrected.evidence_chunk_ids == [evidence_id],
            "corrected answer remains bound to packet evidence",
        )

        reject_trace = _manual_trace(source, "reject")
        reject_tasks = AgentTaskStore(root / "reject" / "tasks")
        reject_reviews = HumanReviewStore(root / "reject" / "reviews")
        reject_tasks.save(reject_trace)
        reject_packet, _ = reject_reviews.create(reject_trace)
        rejected, _ = reject_reviews.resolve(
            reject_packet.review_id,
            task_store=reject_tasks,
            decision="reject",
            reviewer="evaluation-reviewer",
        )
        record(
            "reject_lifecycle",
            rejected.final_outcome == "abstain" and not rejected.evidence_chunk_ids,
            "rejection keeps the task safely abstained",
        )

        stale_trace = _manual_trace(source, "stale")
        stale_tasks = AgentTaskStore(root / "stale" / "tasks")
        stale_reviews = HumanReviewStore(root / "stale" / "reviews")
        stale_tasks.save(stale_trace)
        stale_packet, _ = stale_reviews.create(stale_trace)
        stale_tasks.save(stale_trace.model_copy(update={"query": "changed after enqueue"}))
        try:
            stale_reviews.resolve(
                stale_packet.review_id,
                task_store=stale_tasks,
                decision="approve",
                reviewer="evaluation-reviewer",
            )
        except ValueError as exc:
            stale_blocked = "Task trace changed" in str(exc)
        else:
            stale_blocked = False
        record(
            "stale_trace_guard",
            stale_blocked,
            "approval is rejected after any source-trace change",
        )

        outside_trace = _manual_trace(source, "outside-evidence")
        outside_tasks = AgentTaskStore(root / "outside" / "tasks")
        outside_reviews = HumanReviewStore(root / "outside" / "reviews")
        outside_tasks.save(outside_trace)
        outside_packet, _ = outside_reviews.create(outside_trace)
        try:
            outside_reviews.resolve(
                outside_packet.review_id,
                task_store=outside_tasks,
                decision="correct",
                reviewer="evaluation-reviewer",
                corrected_answer="试图引用审核包外证据。",
                evidence_chunk_ids=["invented-evidence"],
            )
        except ValueError as exc:
            outside_blocked = "outside the immutable packet" in str(exc)
        else:
            outside_blocked = False
        record(
            "outside_evidence_guard",
            outside_blocked,
            "reviewer cannot inject an unseen chunk ID",
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source.resolve()
    trace = _load_source_trace(source)
    checks = _run_contract(trace)
    passed = sum(bool(check["passed"]) for check in checks)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1",
        "status": "complete" if passed == len(checks) else "failed",
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "evaluation": "P4-F append-only human-review workflow contract",
        "source": {
            "path": str(source.relative_to(Path.cwd().resolve())).replace("\\", "/"),
            "sha256": _sha256(source),
            "case_id": SOURCE_CASE_ID,
            "provider": trace.model_trace.provider if trace.model_trace else "unknown",
            "model": trace.model_trace.model if trace.model_trace else "unknown",
        },
        "baseline": {
            "stage": "P4-E",
            "manual_review_marker": True,
            "operational_workflow_checks_passed": 0,
            "operational_workflow_checks_total": len(checks),
        },
        "p4f": {
            "checks_passed": passed,
            "checks_total": len(checks),
            "pass_rate": passed / len(checks),
            "checks": checks,
        },
        "runtime_cost": {
            "model_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "note": "Stored real Agent trace; workflow evaluation makes no provider call.",
        },
        "frozen_test_opened": False,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"P4-F workflow: {passed}/{len(checks)} checks passed")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
