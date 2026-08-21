"""Run trajectory checks for the real DeepSeek compare agent or an explicit baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from findoc_rag.agent_tasks import AgentTaskRequest, CompareTaskController
from findoc_rag.answer_generation import GroundedAnswerGenerator
from findoc_rag.config import ObservabilitySettings, RerankerSettings, load_settings
from findoc_rag.deepseek_agent import DeepSeekCompareAgent, DeepSeekToolCallingModel
from findoc_rag.service import RetrievalService


class CompareEvalCase(BaseModel):
    case_id: str
    query: str
    expected_target_ids: list[str]
    expected_status: Literal["completed", "needs_clarification"]


class CompareEvalDataset(BaseModel):
    schema_version: Literal["1"] = "1"
    dataset_id: str
    index_scope: str
    cases: list[CompareEvalCase]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/agent-compare-v1.json"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/indexes/benchmark-v3/frozen_test"),
    )
    parser.add_argument(
        "--runtime",
        choices=("deepseek", "deterministic-baseline"),
        default="deepseek",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--require-remote", action="store_true")
    return parser.parse_args()


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _default_output(runtime: str, *, available: bool) -> Path:
    if runtime == "deterministic-baseline":
        suffix = "deterministic-baseline"
    else:
        suffix = "deepseek" if available else "deepseek-not-run"
    return Path(f"reports/agent/agent-compare-v1-{suffix}.json")


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    dataset_bytes = args.dataset.read_bytes()
    dataset = CompareEvalDataset.model_validate_json(dataset_bytes)
    model = (
        DeepSeekToolCallingModel(model=args.model, endpoint=args.endpoint)
        if args.runtime == "deepseek"
        else None
    )
    available = bool(model and model.available)
    output = args.output or _default_output(args.runtime, available=available)
    base_report = {
        "schema_version": "2",
        "dataset_id": dataset.dataset_id,
        "dataset_path": args.dataset.resolve().as_posix(),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "index_scope": dataset.index_scope,
        "index_path": args.index_dir.resolve().as_posix(),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "runtime": args.runtime,
    }
    if args.runtime == "deepseek" and not available:
        report = {
            **base_report,
            "status": "not_run",
            "reason": "missing_provider_api_key",
            "provider": {
                "name": "deepseek",
                "model": model.model if model else None,
                "endpoint": model.endpoint if model else None,
            },
            "metrics": None,
            "hard_gates_pass": None,
            "items": [],
        }
        _write(output, report)
        print("status=not_run")
        print("reason=missing_provider_api_key")
        print(f"report={output.resolve()}")
        if args.require_remote:
            raise SystemExit(2)
        return

    settings = load_settings()
    retrieval = RetrievalService(
        settings.retrieval.model_copy(update={"index_dir": args.index_dir.resolve()}),
        ObservabilitySettings(enabled=False),
        RerankerSettings(enabled=False),
        scope_settings=settings.scope_routing,
    )
    available_companies = retrieval.index.list_company_names()
    if model is not None:
        controller = DeepSeekCompareAgent(
            retrieval,
            model,
            available_companies=available_companies,
        )
        provider = {
            "name": model.provider,
            "model": model.model,
            "endpoint": model.endpoint,
        }
    else:
        controller = CompareTaskController(
            retrieval,
            GroundedAnswerGenerator(enabled=False),
            available_companies=available_companies,
        )
        provider = {"name": None, "model": None, "endpoint": None}

    rows: list[dict] = []
    for case in dataset.cases:
        trace = controller.run(AgentTaskRequest(query=case.query, top_k=args.top_k))
        actual_targets = [target.target_id for target in trace.plan.targets]
        plan_exact = actual_targets == case.expected_target_ids
        status_exact = trace.status == case.expected_status
        bounded = trace.rounds_completed <= 3 and len(trace.tool_calls) <= 8
        tools_valid = all(call.tool == "search_evidence" for call in trace.tool_calls)
        target_filters = {
            target.target_id: target.filters for target in trace.plan.targets
        }
        filters_valid = all(
            call.target_id in target_filters
            and call.filters == target_filters[call.target_id]
            for call in trace.tool_calls
        )
        unsafe_partial_answer = trace.result.outcome == "answer" and not all(
            answer.grounded for answer in trace.result.target_answers.values()
        )
        target_count = len(trace.plan.targets)
        evidenced_targets = sum(
            count > 0 for count in trace.sufficiency.evidence_count_by_target.values()
        )
        grounded_targets = sum(
            answer.grounded for answer in trace.result.target_answers.values()
        )
        rows.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "plan_target_exact": plan_exact,
                "status_exact": status_exact,
                "bounded": bounded,
                "tools_valid": tools_valid,
                "filters_valid": filters_valid,
                "unsafe_partial_answer": unsafe_partial_answer,
                "target_count": target_count,
                "evidenced_target_count": evidenced_targets,
                "grounded_target_count": grounded_targets,
                "trace": trace.model_dump(mode="json"),
            }
        )

    executable_rows = [row for row in rows if row["target_count"] > 0]
    target_total = sum(row["target_count"] for row in executable_rows)
    metrics = {
        "case_count": len(rows),
        "plan_target_exact_rate": _rate(
            [row["plan_target_exact"] for row in rows]
        ),
        "status_exact_rate": _rate([row["status_exact"] for row in rows]),
        "bounded_run_rate": _rate([row["bounded"] for row in rows]),
        "tool_selection_valid_rate": _rate(
            [row["tools_valid"] for row in rows]
        ),
        "target_filter_valid_rate": _rate(
            [row["filters_valid"] for row in rows]
        ),
        "evidence_target_coverage": (
            sum(row["evidenced_target_count"] for row in executable_rows)
            / target_total
            if target_total
            else 0.0
        ),
        "grounded_target_coverage": (
            sum(row["grounded_target_count"] for row in executable_rows)
            / target_total
            if target_total
            else 0.0
        ),
        "unsafe_partial_answer_rate": _rate(
            [row["unsafe_partial_answer"] for row in rows]
        ),
    }
    hard_gates_pass = all(
        (
            metrics["plan_target_exact_rate"] == 1.0,
            metrics["status_exact_rate"] == 1.0,
            metrics["bounded_run_rate"] == 1.0,
            metrics["tool_selection_valid_rate"] == 1.0,
            metrics["target_filter_valid_rate"] == 1.0,
            metrics["unsafe_partial_answer_rate"] == 0.0,
        )
    )
    report = {
        **base_report,
        "status": "complete",
        "reason": None,
        "provider": provider,
        "index_id": retrieval.manifest.index_id,
        "metrics": metrics,
        "hard_gates_pass": hard_gates_pass,
        "items": rows,
    }
    _write(output, report)
    print("status=complete")
    print(f"hard_gates_pass={hard_gates_pass}")
    print(f"plan_target_exact_rate={metrics['plan_target_exact_rate']:.4f}")
    print(f"evidence_target_coverage={metrics['evidence_target_coverage']:.4f}")
    print(f"grounded_target_coverage={metrics['grounded_target_coverage']:.4f}")
    print(f"report={output.resolve()}")


if __name__ == "__main__":
    main()
