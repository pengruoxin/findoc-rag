"""Run the adversarial PDF agent benchmark with local, model-free scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from findoc_rag.agent_evaluation import (
    AgentHardDataset,
    diagnose_agent_requirements,
    score_agent_hard_case,
    validate_agent_hard_sources,
)
from findoc_rag.agent_tasks import (
    AgentTaskRequest,
    AgentTaskTrace,
    CompareTaskController,
)
from findoc_rag.answer_generation import GroundedAnswerGenerator
from findoc_rag.config import ObservabilitySettings, RerankerSettings, load_settings
from findoc_rag.deepseek_agent import (
    DeepSeekCalculateAgent,
    DeepSeekCompareAgent,
    DeepSeekExtractAgent,
    DeepSeekToolCallingModel,
    DeepSeekVisualGraphAgent,
)
from findoc_rag.evidence_verifier import EvidenceVerifierAgent
from findoc_rag.service import RetrievalService
from findoc_rag.visual_inspection import PdfRegionInspector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/agent-hard-v1.json"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/indexes/benchmark-v3"),
    )
    parser.add_argument(
        "--runtime",
        choices=("deepseek", "deterministic-baseline"),
        default="deepseek",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--verifier-model", default="")
    parser.add_argument("--verifier-endpoint", default="")
    parser.add_argument("--disable-support-proof", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--require-remote", action="store_true")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--rescore-from",
        type=Path,
        help="Recompute local scores from stored traces without another model call.",
    )
    source_group.add_argument(
        "--verify-from",
        type=Path,
        help=(
            "Apply the independent evidence verifier to stored traces, then "
            "recompute local scores without rerunning retrieval."
        ),
    )
    return parser.parse_args()


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _default_output(runtime: str, *, remote_available: bool) -> Path:
    suffix = (
        "deepseek"
        if runtime == "deepseek" and remote_available
        else "deepseek-not-run"
        if runtime == "deepseek"
        else "deterministic-baseline"
    )
    return Path(f"reports/agent/agent-hard-v1-{suffix}.json")


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _compute_metrics(rows: list[dict], dataset: AgentHardDataset) -> dict:
    executed = [row for row in rows if row["status"] == "executed"]
    unsupported = [row for row in rows if row["status"] != "executed"]
    executed_fact_count = sum(row["expected_fact_count"] for row in executed)
    executed_fact_matches = sum(
        sum(fact["matched"] for fact in row["score"]["fact_scores"])
        for row in executed
    )
    all_fact_count = sum(row["expected_fact_count"] for row in rows)
    abstention_rows = [
        row
        for row, case in zip(rows, dataset.cases, strict=True)
        if row["status"] == "executed" and case.expected_behavior == "abstain"
    ]
    clarification_rows = [
        row
        for row, case in zip(rows, dataset.cases, strict=True)
        if row["status"] == "executed" and case.expected_behavior == "clarify"
    ]
    requirement_rows = [
        row["requirement_diagnostics"]
        for row in executed
        if row.get("requirement_diagnostics", {}).get("applicable")
    ]
    planned_requirements = sum(
        row["planned_requirement_count"] for row in requirement_rows
    )
    scoped_requirements = sum(
        row["scoped_requirement_count"] for row in requirement_rows
    )
    return {
        "case_count": len(rows),
        "executed_case_count": len(executed),
        "unsupported_case_count": len(unsupported),
        "task_coverage_rate": len(executed) / len(rows),
        "plan_target_exact_rate": _rate(
            [row["score"]["plan_target_exact"] for row in executed]
        ),
        "behavior_accuracy": _rate(
            [row["score"]["behavior_correct"] for row in executed]
        ),
        "supported_fact_accuracy": (
            executed_fact_matches / executed_fact_count if executed_fact_count else 0.0
        ),
        "end_to_end_fact_accuracy": (
            executed_fact_matches / all_fact_count if all_fact_count else 0.0
        ),
        "supported_case_pass_rate": _rate(
            [row["score"]["case_pass"] for row in executed]
        ),
        "end_to_end_case_pass_rate": (
            sum(row["score"]["case_pass"] for row in executed) / len(rows)
        ),
        "safe_abstention_accuracy": _rate(
            [row["score"]["behavior_correct"] for row in abstention_rows]
        ),
        "clarification_accuracy": _rate(
            [row["score"]["behavior_correct"] for row in clarification_rows]
        ),
        "atomic_requirement_case_count": len(requirement_rows),
        "task_requirement_coverage": (
            sum(row["covered_requirement_count"] for row in requirement_rows)
            / planned_requirements
            if planned_requirements
            else None
        ),
        "requirement_evidence_coverage": (
            sum(
                row["evidence_bound_requirement_count"]
                for row in requirement_rows
            )
            / planned_requirements
            if planned_requirements
            else None
        ),
        "scope_validation_rate": (
            sum(
                row["scope_validated_requirement_count"]
                for row in requirement_rows
            )
            / scoped_requirements
            if scoped_requirements
            else None
        ),
        "claim_citation_completeness": (
            sum(row["claim_citation_completeness"] for row in requirement_rows)
            / len(requirement_rows)
            if requirement_rows
            else None
        ),
    }


def _rescore_rows(dataset: AgentHardDataset, source_report: dict) -> list[dict]:
    source_by_case = {item["case_id"]: item for item in source_report["items"]}
    rows: list[dict] = []
    for case in dataset.cases:
        source = source_by_case.get(case.case_id)
        if source is None:
            raise SystemExit(f"rescore report is missing case {case.case_id}")
        if source["status"] != "executed":
            rows.append(source)
            continue
        trace = AgentTaskTrace.model_validate(source["trace"])
        score = score_agent_hard_case(case, trace)
        requirement_diagnostics = diagnose_agent_requirements(trace)
        rows.append(
            {
                **source,
                "score": score.model_dump(mode="json"),
                "requirement_diagnostics": requirement_diagnostics.model_dump(
                    mode="json"
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    dataset_bytes = args.dataset.read_bytes()
    dataset = AgentHardDataset.model_validate_json(dataset_bytes)
    source_validation = validate_agent_hard_sources(
        dataset, workspace=Path.cwd()
    )
    if not source_validation.valid:
        raise SystemExit(
            "hard-dataset source validation failed: "
            + "; ".join(source_validation.errors)
        )
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    base_report = {
        "schema_version": "1",
        "dataset_id": dataset.dataset_id,
        "dataset_path": args.dataset.resolve().as_posix(),
        "dataset_sha256": dataset_sha256,
        "index_scope": dataset.index_scope,
        "index_path": args.index_dir.resolve().as_posix(),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "runtime": args.runtime,
        "scoring": "local_exact_gold_and_citation_overlap",
        "source_validation": source_validation.model_dump(mode="json"),
    }
    if args.rescore_from is not None:
        source_report = json.loads(args.rescore_from.read_text(encoding="utf-8"))
        if source_report.get("dataset_sha256") != dataset_sha256:
            raise SystemExit("rescore report dataset SHA-256 does not match")
        rows = _rescore_rows(dataset, source_report)
        metrics = _compute_metrics(rows, dataset)
        output = args.output or args.rescore_from.with_name(
            f"{args.rescore_from.stem}-rescored.json"
        )
        report = {
            **source_report,
            **base_report,
            "rescored_at": datetime.now(UTC).isoformat(),
            "rescore_source_path": args.rescore_from.resolve().as_posix(),
            "metrics": metrics,
            "items": rows,
        }
        _write_report(output, report)
        print("status=rescored")
        print(f"end_to_end_case_pass_rate={metrics['end_to_end_case_pass_rate']:.4f}")
        print(f"safe_abstention_accuracy={metrics['safe_abstention_accuracy']:.4f}")
        print(f"report={output.resolve()}")
        return

    if args.verify_from is not None:
        source_report = json.loads(args.verify_from.read_text(encoding="utf-8"))
        if source_report.get("dataset_sha256") != dataset_sha256:
            raise SystemExit("verification source dataset SHA-256 does not match")
        source_by_case = {
            item["case_id"]: item for item in source_report.get("items", [])
        }
        verifier_model = DeepSeekToolCallingModel(
            model=args.verifier_model or args.model,
            endpoint=args.verifier_endpoint or args.endpoint,
        )
        optimizer_model = DeepSeekToolCallingModel(
            model=args.model,
            endpoint=args.endpoint,
        )
        output = args.output or args.verify_from.with_name(
            f"{args.verify_from.stem}-evidence-verifier.json"
        )
        if not verifier_model.available or not optimizer_model.available:
            report = {
                **base_report,
                "status": "not_run",
                "reason": "missing_provider_api_key",
                "verification_source_path": args.verify_from.resolve().as_posix(),
                "provider": {
                    "name": verifier_model.provider,
                    "model": verifier_model.model,
                    "endpoint": verifier_model.endpoint,
                },
                "metrics": None,
                "items": [],
            }
            _write_report(output, report)
            print("status=not_run")
            print("reason=missing_provider_api_key")
            print(f"report={output.resolve()}")
            if args.require_remote:
                raise SystemExit(2)
            return
        reviewer = EvidenceVerifierAgent(
            verifier_model,
            optimizer_model=optimizer_model,
            known_companies=sorted(
                {
                    item["company_name"]
                    for source in source_report.get("items", [])
                    if source.get("trace")
                    for item in source["trace"]["evidence_memory"]["items"]
                    if item.get("company_name")
                }
            ),
            require_support_proof=not args.disable_support_proof,
        )
        rows: list[dict] = []
        for case in dataset.cases:
            source = source_by_case.get(case.case_id)
            if source is None:
                raise SystemExit(
                    f"verification source report is missing case {case.case_id}"
                )
            if source["status"] != "executed" or case.task_type != "extract":
                rows.append(source)
                continue
            trace = AgentTaskTrace.model_validate(source["trace"])
            reviewed_trace = reviewer.review(trace)
            score = score_agent_hard_case(case, reviewed_trace)
            requirement_diagnostics = diagnose_agent_requirements(reviewed_trace)
            rows.append(
                {
                    **source,
                    "score": score.model_dump(mode="json"),
                    "requirement_diagnostics": requirement_diagnostics.model_dump(
                        mode="json"
                    ),
                    "trace": reviewed_trace.model_dump(mode="json"),
                }
            )
        metrics = _compute_metrics(rows, dataset)
        report = {
            **source_report,
            **base_report,
            "evaluated_at": datetime.now(UTC).isoformat(),
            "status": "complete",
            "reason": None,
            "verification_source_path": args.verify_from.resolve().as_posix(),
            "provider": {
                "name": verifier_model.provider,
                "model": verifier_model.model,
                "endpoint": verifier_model.endpoint,
                "optimizer_model": optimizer_model.model,
                "optimizer_endpoint": optimizer_model.endpoint,
            },
            "support_proof_required": not args.disable_support_proof,
            "metrics": metrics,
            "items": rows,
        }
        _write_report(output, report)
        print("status=complete")
        print(f"supported_fact_accuracy={metrics['supported_fact_accuracy']:.4f}")
        print(
            "end_to_end_case_pass_rate="
            f"{metrics['end_to_end_case_pass_rate']:.4f}"
        )
        print(f"report={output.resolve()}")
        return

    model = (
        DeepSeekToolCallingModel(model=args.model, endpoint=args.endpoint)
        if args.runtime == "deepseek"
        else None
    )
    remote_available = bool(model and model.available)
    output = args.output or _default_output(
        args.runtime, remote_available=remote_available
    )
    if args.runtime == "deepseek" and not remote_available:
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
            "items": [],
        }
        _write_report(output, report)
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
    available_report_years_by_company = retrieval.index.list_company_report_years()
    if model is not None:
        pdf_inspector = PdfRegionInspector(
            (Path.cwd() / dataset.source_manifest).resolve(),
            workspace=Path.cwd(),
        )
        compare_controller = DeepSeekCompareAgent(
            retrieval,
            model,
            available_companies=available_companies,
        )
        extract_controller = DeepSeekExtractAgent(
            retrieval,
            model,
            available_companies=available_companies,
            available_report_years_by_company=available_report_years_by_company,
            layout_inspector=pdf_inspector,
        )
        calculate_controller = DeepSeekCalculateAgent(
            retrieval,
            model,
            available_companies=available_companies,
            available_report_years_by_company=available_report_years_by_company,
        )
        visual_controller = DeepSeekVisualGraphAgent(
            retrieval,
            model,
            available_companies=available_companies,
            available_report_years_by_company=available_report_years_by_company,
            region_inspector=pdf_inspector,
        )
        provider = {
            "name": model.provider,
            "model": model.model,
            "endpoint": model.endpoint,
        }
    else:
        compare_controller = CompareTaskController(
            retrieval,
            GroundedAnswerGenerator(enabled=False),
            available_companies=available_companies,
        )
        extract_controller = None
        calculate_controller = None
        visual_controller = None
        provider = {"name": None, "model": None, "endpoint": None}

    rows: list[dict] = []
    for case in dataset.cases:
        if case.task_type == "compare":
            controller = compare_controller
        elif case.task_type == "extract" and extract_controller is not None:
            controller = extract_controller
        elif (
            case.task_type == "calculate"
            and visual_controller is not None
            and visual_controller.supports(case.query)
        ):
            controller = visual_controller
        elif (
            case.task_type == "calculate"
            and calculate_controller is not None
            and calculate_controller.supports(case.query)
        ):
            controller = calculate_controller
        else:
            rows.append(
                {
                    "case_id": case.case_id,
                    "task_type": case.task_type,
                    "challenge_types": case.challenge_types,
                    "status": "unsupported_task_type",
                    "expected_fact_count": len(case.expected_facts),
                    "score": None,
                    "requirement_diagnostics": None,
                    "trace": None,
                }
            )
            continue
        trace = controller.run(
            AgentTaskRequest(
                task_type=case.task_type,
                query=case.query,
                top_k=args.top_k,
                max_rounds=4 if case.task_type == "extract" else 3,
                max_tool_calls=8,
            )
        )
        score = score_agent_hard_case(case, trace)
        requirement_diagnostics = diagnose_agent_requirements(trace)
        rows.append(
            {
                "case_id": case.case_id,
                "task_type": case.task_type,
                "challenge_types": case.challenge_types,
                "status": "executed",
                "expected_fact_count": len(case.expected_facts),
                "score": score.model_dump(mode="json"),
                "requirement_diagnostics": requirement_diagnostics.model_dump(
                    mode="json"
                ),
                "trace": trace.model_dump(mode="json"),
            }
        )

    metrics = _compute_metrics(rows, dataset)
    report = {
        **base_report,
        "status": "complete",
        "reason": None,
        "provider": provider,
        "index_id": retrieval.manifest.index_id,
        "metrics": metrics,
        "items": rows,
    }
    _write_report(output, report)
    print("status=complete")
    print(f"task_coverage_rate={metrics['task_coverage_rate']:.4f}")
    print(f"plan_target_exact_rate={metrics['plan_target_exact_rate']:.4f}")
    print(f"supported_fact_accuracy={metrics['supported_fact_accuracy']:.4f}")
    print(f"end_to_end_case_pass_rate={metrics['end_to_end_case_pass_rate']:.4f}")
    print(f"report={output.resolve()}")


if __name__ == "__main__":
    main()
