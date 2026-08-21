import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from findoc_rag import __version__
from findoc_rag.agent_tasks import (
    AgentTaskRequest,
    AgentTaskStore,
    CompareTaskController,
)
from findoc_rag.answer_generation import GroundedAnswerGenerator
from findoc_rag.chunking import ChunkingConfig, build_chunking_report, chunk_document
from findoc_rag.config import load_settings
from findoc_rag.corpus import build_active_corpus_index, resolve_current_index
from findoc_rag.datasets.financebench import convert_financebench, write_jsonl
from findoc_rag.deepseek_agent import (
    DeepSeekCalculateAgent,
    DeepSeekCompareAgent,
    DeepSeekExtractAgent,
    DeepSeekToolCallingModel,
    DeepSeekVisualGraphAgent,
)
from findoc_rag.diagnostics import (
    DiagnosticDataset,
    DiagnosticEvaluation,
    DocumentProfile,
    analyze_recall_failures,
    evaluate_diagnostic_dataset,
    generate_diagnostic_dataset,
)
from findoc_rag.documents.models import DocumentChunk
from findoc_rag.documents.pdf import PdfExtractionConfig, parse_pdf
from findoc_rag.evaluation.retrieval import evaluate_retriever
from findoc_rag.evidence_verifier import (
    EvidenceVerifiedExtractAgent,
    EvidenceVerifierAgent,
)
from findoc_rag.holdout import (
    generate_holdout_review_pack,
    holdout_eval_to_diagnostics,
    render_review_markdown,
)
from findoc_rag.human_review import HumanReviewStore
from findoc_rag.indexing import DEFAULT_DENSE_MODEL, PersistentIndex, SearchFilters
from findoc_rag.ingestion import ingest_pdf
from findoc_rag.io import read_jsonl, write_dict_jsonl, write_json
from findoc_rag.registry import DocumentRegistry
from findoc_rag.reranking import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from findoc_rag.retrieval.bm25 import BM25Retriever
from findoc_rag.retrieval.dense import DEFAULT_MODEL, DenseRetriever
from findoc_rag.schemas import BenchmarkQuestion, CorpusDocument
from findoc_rag.scope_routing import plan_candidate_budget, route_by_scope
from findoc_rag.service import RetrievalService
from findoc_rag.sources.cninfo import (
    CninfoClient,
    select_chinese_annual_report,
    write_artifact_manifest,
)
from findoc_rag.visual_inspection import PdfRegionInspector

app = typer.Typer(help="Verifiable RAG for complex Chinese listed-company documents.")
agent_app = typer.Typer(help="Run and inspect bounded, auditable document-agent tasks.")
agent_review_app = typer.Typer(help="Inspect and resolve evidence-verifier review tasks.")
app.add_typer(agent_app, name="agent")
agent_app.add_typer(agent_review_app, name="review")
console = Console()


@app.callback()
def main() -> None:
    """Run FinDocRAG dataset, retrieval, and evaluation commands."""


@agent_app.command("run")
def agent_run_command(
    query: Annotated[str, typer.Argument(help="Natural-language task request.")],
    task_type: Annotated[
        str, typer.Option("--task", help="Task type: compare, extract, or calculate.")
    ] = ("compare"),
    index_dir: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Index or corpus-index root."),
    ] = None,
    source_manifest: Annotated[
        Path | None,
        typer.Option(
            "--source-manifest",
            exists=True,
            dir_okay=False,
            help=(
                "SHA-256-bound PDF source manifest used by extract layout "
                "reconstruction and visual calculation."
            ),
        ),
    ] = None,
    task_dir: Annotated[Path, typer.Option(help="Durable task-trace directory.")] = Path(
        "data/agent/tasks"
    ),
    review_dir: Annotated[
        Path | None,
        typer.Option(
            help=("Human-review queue directory. Defaults to a reviews directory next to task-dir.")
        ),
    ] = None,
    mode: Annotated[str, typer.Option(help="lexical, dense, or hybrid.")] = "lexical",
    top_k: Annotated[int, typer.Option(min=1, max=10)] = 3,
    max_rounds: Annotated[int, typer.Option(min=1, max=4)] = 4,
    max_tool_calls: Annotated[int, typer.Option(min=1, max=8)] = 8,
    runtime: Annotated[
        str,
        typer.Option(help="deepseek or deterministic-baseline."),
    ] = "deepseek",
    agent_model: Annotated[
        str, typer.Option(help="DeepSeek agent model; defaults to environment/current default.")
    ] = "",
    agent_endpoint: Annotated[str, typer.Option(help="DeepSeek Chat Completions endpoint.")] = "",
    evidence_verifier: Annotated[
        bool,
        typer.Option(
            "--evidence-verifier",
            help="Run a separate-context verifier and at most one repair for complex extract tasks.",
        ),
    ] = False,
    verifier_policy: Annotated[
        str,
        typer.Option(
            "--verifier-policy",
            help=(
                "Evidence verifier routing: auto for high-risk/open-risk extracts, "
                "off to disable, or always for every answered extract."
            ),
        ),
    ] = "auto",
    verifier_model: Annotated[
        str,
        typer.Option(help="Optional DeepSeek model override for the evidence verifier."),
    ] = "",
    verifier_endpoint: Annotated[
        str,
        typer.Option(help="Optional endpoint override for the evidence verifier."),
    ] = "",
    verifier_support_proof: Annotated[
        bool,
        typer.Option(
            "--verifier-support-proof/--no-verifier-support-proof",
            help=(
                "Require verbatim requirement-to-claim evidence proofs; invalid "
                "proofs pause the answer for manual review."
            ),
        ),
    ] = True,
) -> None:
    """Run a bounded agent task and persist its complete trace."""
    if task_type not in {"compare", "extract", "calculate"}:
        raise typer.BadParameter("task must be compare, extract, or calculate")
    if mode not in {"lexical", "dense", "hybrid"}:
        raise typer.BadParameter("mode must be lexical, dense, or hybrid")
    if runtime not in {"deepseek", "deterministic-baseline"}:
        raise typer.BadParameter("runtime must be deepseek or deterministic-baseline")
    if verifier_policy not in {"auto", "off", "always"}:
        raise typer.BadParameter("verifier-policy must be auto, off, or always")
    if runtime == "deterministic-baseline" and task_type != "compare":
        raise typer.BadParameter("deterministic-baseline only supports task=compare")
    if evidence_verifier and task_type != "extract":
        raise typer.BadParameter("evidence-verifier only supports task=extract")

    settings = load_settings()
    resolved_index = index_dir.resolve() if index_dir else settings.retrieval.index_dir
    if not resolved_index.is_dir():
        raise typer.BadParameter(f"Index directory does not exist: {resolved_index}")
    retrieval_settings = settings.retrieval.model_copy(update={"index_dir": resolved_index})
    retrieval = RetrievalService(
        retrieval_settings,
        settings.observability,
        settings.reranker,
        scope_settings=settings.scope_routing,
    )
    task_request = AgentTaskRequest(
        task_type=task_type,
        query=query,
        mode=mode,
        top_k=top_k,
        max_rounds=(max_rounds if task_type == "extract" else min(max_rounds, 3)),
        max_tool_calls=max_tool_calls,
    )
    available_companies = retrieval.index.list_company_names()
    available_report_years_by_company = retrieval.index.list_company_report_years()
    source_inspector = (
        PdfRegionInspector(source_manifest.resolve(), workspace=Path.cwd())
        if source_manifest is not None
        else None
    )
    if runtime == "deepseek":
        model = DeepSeekToolCallingModel(
            model=agent_model,
            endpoint=agent_endpoint,
        )
        if not model.available:
            raise typer.BadParameter(
                "DeepSeek tool-calling runtime requires DEEPSEEK_API_KEY "
                "or an endpoint-bound provider key"
            )
        if task_type == "calculate" and DeepSeekVisualGraphAgent.supports(query):
            visual_inspector = source_inspector or PdfRegionInspector(
                Path("data/evaluation/benchmark-v3-source-manifest.json"),
                workspace=Path.cwd(),
            )
            trace = DeepSeekVisualGraphAgent(
                retrieval,
                model,
                available_companies=available_companies,
                available_report_years_by_company=available_report_years_by_company,
                region_inspector=visual_inspector,
            ).run(task_request)
        elif task_type == "extract":
            base_extract_agent = DeepSeekExtractAgent(
                retrieval,
                model,
                available_companies=available_companies,
                available_report_years_by_company=(available_report_years_by_company),
                layout_inspector=source_inspector,
            )
            if evidence_verifier or verifier_policy != "off":
                verifier_runtime = DeepSeekToolCallingModel(
                    model=verifier_model,
                    endpoint=verifier_endpoint,
                )
                trace = EvidenceVerifiedExtractAgent(
                    base_extract_agent,
                    EvidenceVerifierAgent(
                        verifier_runtime,
                        optimizer_model=model,
                        known_companies=available_companies,
                        route_policy=("always" if verifier_policy == "always" else "auto"),
                        require_support_proof=verifier_support_proof,
                    ),
                ).run(task_request)
            else:
                trace = base_extract_agent.run(task_request)
        else:
            agent_class = {
                "compare": DeepSeekCompareAgent,
                "calculate": DeepSeekCalculateAgent,
            }[task_type]
            trace = agent_class(
                retrieval,
                model,
                available_companies=available_companies,
                **(
                    {"available_report_years_by_company": (available_report_years_by_company)}
                    if task_type == "calculate"
                    else {}
                ),
            ).run(task_request)
    else:
        answer_generator = GroundedAnswerGenerator(
            settings.answer_generation.model,
            settings.answer_generation.endpoint,
            settings.answer_generation.enabled,
        )
        trace = CompareTaskController(
            retrieval,
            answer_generator,
            available_companies=available_companies,
        ).run(task_request)
    trace_path = AgentTaskStore(task_dir).save(trace)
    review_packet = None
    review_path = None
    if trace.stop_reason == "evidence_verifier_manual_review":
        resolved_review_dir = review_dir or task_dir.resolve().parent / "reviews"
        review_packet, review_path = HumanReviewStore(resolved_review_dir).create(
            trace,
            region_inspector=source_inspector,
            region_directory=resolved_review_dir / "regions",
        )

    console.print(f"[green]Agent task {trace.status}.[/green]")
    console.print(f"Task ID: {trace.task_id}")
    console.print(f"Runtime: {trace.runtime}")
    console.print(f"Stop reason: {trace.stop_reason}")
    console.print(f"Rounds/tool calls: {trace.rounds_completed}/{len(trace.tool_calls)}")
    console.print(f"Evidence items: {len(trace.evidence_memory.items)}")
    console.print(f"Outcome: {trace.result.outcome}")
    console.print(trace.result.answer.answer)
    if (
        trace.evidence_verification is not None
        and trace.evidence_verification.human_review_required
    ):
        console.print("Human review required: yes")
        candidate = trace.evidence_verification.candidate_result
        if candidate is not None:
            console.print("Withheld candidate answer:")
            console.print(candidate.answer.answer)
    console.print(f"Trace: {trace_path}")
    if review_packet is not None:
        console.print(f"Human review ID: {review_packet.review_id}")
        console.print(f"Review packet: {review_path}")


@agent_app.command("inspect")
def agent_inspect_command(
    task_id: Annotated[str, typer.Argument(help="Task ID returned by agent run.")],
    task_dir: Annotated[Path, typer.Option(help="Durable task-trace directory.")] = Path(
        "data/agent/tasks"
    ),
    as_json: Annotated[bool, typer.Option("--json", help="Print the full trace JSON.")] = (False),
) -> None:
    """Inspect one persisted agent task without executing it again."""
    try:
        trace = AgentTaskStore(task_dir).load(task_id)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if as_json:
        console.print_json(trace.model_dump_json(indent=2))
        return
    console.print(f"Task ID: {trace.task_id}")
    console.print(f"Type/status: {trace.task_type}/{trace.status}")
    console.print(f"Runtime: {trace.runtime}")
    console.print(f"Stop reason: {trace.stop_reason}")
    console.print(f"Index ID: {trace.index_id}")
    console.print(f"Targets: {', '.join(target.label for target in trace.plan.targets)}")
    console.print(f"Rounds/tool calls: {trace.rounds_completed}/{len(trace.tool_calls)}")
    console.print(f"Evidence items: {len(trace.evidence_memory.items)}")
    console.print(f"Outcome: {trace.result.outcome}")
    console.print(trace.result.answer.answer)
    if (
        trace.evidence_verification is not None
        and trace.evidence_verification.human_review_required
    ):
        console.print("Human review required: yes")
        candidate = trace.evidence_verification.candidate_result
        if candidate is not None:
            console.print("Withheld candidate answer:")
            console.print(candidate.answer.answer)


@agent_review_app.command("enqueue")
def agent_review_enqueue_command(
    task_id: Annotated[str, typer.Argument(help="Manual-review task ID.")],
    task_dir: Annotated[Path, typer.Option(help="Durable task-trace directory.")] = Path(
        "data/agent/tasks"
    ),
    review_dir: Annotated[Path, typer.Option(help="Human-review queue directory.")] = Path(
        "data/agent/reviews"
    ),
    source_manifest: Annotated[
        Path | None,
        typer.Option(help="Optional manifest for bounded PDF region proofs."),
    ] = None,
) -> None:
    """Create an idempotent review packet for an existing paused task."""
    try:
        trace = AgentTaskStore(task_dir).load(task_id)
        region_inspector = (
            PdfRegionInspector(source_manifest.resolve(), workspace=Path.cwd())
            if source_manifest is not None
            else None
        )
        packet, path = HumanReviewStore(review_dir).create(
            trace,
            region_inspector=region_inspector,
            region_directory=review_dir / "regions",
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Review ID: {packet.review_id}")
    console.print("Status: pending")
    console.print(f"Packet: {path}")


@agent_review_app.command("list")
def agent_review_list_command(
    review_dir: Annotated[Path, typer.Option(help="Human-review queue directory.")] = Path(
        "data/agent/reviews"
    ),
    status: Annotated[
        str, typer.Option(help="Queue status: pending, resolved, or all.")
    ] = "pending",
) -> None:
    """List queued human-review tasks without running a model."""
    try:
        items = HumanReviewStore(review_dir).list(status=status)  # type: ignore[arg-type]
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not items:
        console.print("No review tasks.")
        return
    for item in items:
        packet = item.packet
        console.print(
            f"{packet.review_id}  {item.status}  task={packet.task_id}  "
            f"created={packet.created_at.isoformat()}"
        )
        console.print(f"  {packet.query}")


@agent_review_app.command("inspect")
def agent_review_inspect_command(
    review_id: Annotated[str, typer.Argument(help="Review ID.")],
    review_dir: Annotated[Path, typer.Option(help="Human-review queue directory.")] = Path(
        "data/agent/reviews"
    ),
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full review packet and resolution.")
    ] = False,
) -> None:
    """Show the candidate claims, pages, evidence, and current decision."""
    try:
        item = HumanReviewStore(review_dir).inspect(review_id)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if as_json:
        console.print_json(item.model_dump_json(indent=2))
        return
    packet = item.packet
    console.print(f"Review ID: {packet.review_id}")
    console.print(f"Task ID: {packet.task_id}")
    console.print(f"Status: {item.status}")
    console.print(f"Trace SHA-256: {packet.task_trace_sha256}")
    console.print(f"Query: {packet.query}")
    console.print("Reasons:")
    for reason in packet.reasons:
        console.print(f"  - {reason}")
    if packet.candidate_result is not None:
        console.print("Candidate answer:")
        console.print(packet.candidate_result.answer.answer)
    else:
        console.print("Candidate answer: unavailable in this legacy trace")
    for requirement in packet.requirements:
        console.print(f"[{requirement.requirement_id}] {requirement.description}")
        for claim in requirement.claims:
            console.print(f"  Claim: {claim}")
        for evidence in requirement.evidence:
            console.print(
                f"  Evidence: {evidence.chunk_id} pages {evidence.page_start}-{evidence.page_end}"
            )
            console.print(f"  {evidence.excerpt}")
        for proof in requirement.table_cell_proofs:
            location = (
                f"page {proof.page_number}, bbox={proof.value_bbox.model_dump()}"
                if proof.geometry_status == "coordinate" and proof.value_bbox is not None
                else "text-only; no verified cell bbox"
            )
            console.print(
                f"  Cell: table={proof.table_id} "
                f"row[{proof.row_index}]={proof.row} "
                f"column[{proof.column_index}]={proof.column} "
                f"value={proof.value} {proof.unit}"
            )
            console.print(f"  Geometry: {location}")
            console.print(f"  Cell proof SHA-256: {proof.binding_sha256}")
        for proof in requirement.region_proofs:
            console.print(
                f"  Region: page {proof.page_number}, "
                f"area={proof.rendered_area_ratio:.2%}, image={proof.image_path}"
            )
            console.print(f"  Region image SHA-256: {proof.image_sha256}")
            console.print(f"  Region proof SHA-256: {proof.binding_sha256}")
        for error in requirement.region_proof_errors:
            console.print(f"  Region proof unavailable: {error}")
    if item.resolution is not None:
        console.print(f"Resolution: {item.resolution.decision} by {item.resolution.reviewer}")
        console.print(f"Final outcome: {item.resolution.final_outcome}")
        console.print(item.resolution.final_answer)


@agent_review_app.command("resolve")
def agent_review_resolve_command(
    review_id: Annotated[str, typer.Argument(help="Review ID.")],
    decision: Annotated[str, typer.Argument(help="Decision: approve, correct, or reject.")],
    reviewer: Annotated[str, typer.Option("--reviewer", help="Human reviewer identifier.")],
    task_dir: Annotated[Path, typer.Option(help="Durable task-trace directory.")] = Path(
        "data/agent/tasks"
    ),
    review_dir: Annotated[Path, typer.Option(help="Human-review queue directory.")] = Path(
        "data/agent/reviews"
    ),
    comment: Annotated[str, typer.Option(help="Optional review note.")] = "",
    corrected_answer: Annotated[
        str,
        typer.Option(help="Replacement answer; required only when decision is correct."),
    ] = "",
    evidence_chunk_id: Annotated[
        list[str] | None,
        typer.Option(
            "--evidence-chunk-id",
            help="Approved packet evidence ID; repeat for a corrected answer.",
        ),
    ] = None,
) -> None:
    """Resolve one review exactly once, bound to the unchanged source trace."""
    if decision not in {"approve", "correct", "reject"}:
        raise typer.BadParameter("decision must be approve, correct, or reject")
    try:
        resolution, path = HumanReviewStore(review_dir).resolve(
            review_id,
            task_store=AgentTaskStore(task_dir),
            decision=decision,  # type: ignore[arg-type]
            reviewer=reviewer,
            comment=comment,
            corrected_answer=corrected_answer or None,
            evidence_chunk_ids=evidence_chunk_id,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Review resolved: {resolution.decision}")
    console.print(f"Final outcome: {resolution.final_outcome}")
    console.print(resolution.final_answer)
    console.print(f"Resolution: {path}")


@app.command()
def doctor() -> None:
    """Check that the local project environment is ready."""
    root = Path(__file__).resolve().parents[2]
    console.print(f"[green]FinDocRAG {__version__} is ready.[/green]")
    console.print(f"Project root: {root}")
    console.print("Current milestone: validate retrieval before importing Chinese filings.")


@app.command("fetch-annual-report")
def fetch_annual_report(
    company: Annotated[str, typer.Option(help="Exact Chinese listed-company name.")],
    year: Annotated[int, typer.Option(min=1990, max=2100, help="Report year.")],
    output_dir: Annotated[Path, typer.Option(help="Local filing artifact directory.")] = Path(
        "data/artifacts/cninfo"
    ),
) -> None:
    """Fetch an exact Chinese annual report from the official CNInfo source."""
    client = CninfoClient()
    announcements = client.search_annual_reports(company, year)
    selected = select_chinese_annual_report(announcements, company, year)
    stem = f"{selected.security_code}_{year}_{selected.announcement_id}"
    artifact = client.download(selected, output_dir / f"{stem}.pdf")
    manifest_path = output_dir / f"{stem}.manifest.json"
    write_artifact_manifest(artifact, manifest_path)

    console.print(f"[green]Downloaded {selected.title}[/green]")
    console.print(f"Security: {selected.security_code} {selected.security_name}")
    console.print(f"PDF: {artifact.local_file}")
    console.print(f"SHA-256: {artifact.sha256}")
    console.print(f"Manifest: {manifest_path}")


@app.command("parse-pdf")
def parse_pdf_command(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option(help="Parsed Document IR JSON path.")] = None,
    ocr_mode: Annotated[
        str, typer.Option(help="OCR policy: disabled, auto, or force.")
    ] = "disabled",
    ocr_backend: Annotated[str, typer.Option(help="Optional OCR backend name.")] = "rapidocr",
    ocr_dpi: Annotated[int, typer.Option(min=72, max=600)] = 180,
) -> None:
    """Parse any local PDF into the coordinate-preserving Document IR."""
    extraction_config = PdfExtractionConfig(mode=ocr_mode, ocr_backend=ocr_backend, ocr_dpi=ocr_dpi)
    document = parse_pdf(source, extraction_config)
    target = output or Path("data/processed/documents") / f"{document.content_sha256}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")

    ocr_pages = [page.page_number for page in document.pages if page.needs_ocr]
    console.print(f"[green]Parsed {document.filename}: {document.page_count} pages.[/green]")
    console.print(f"Document ID: {document.document_id}")
    console.print(f"Pages requiring OCR fallback: {len(ocr_pages)}")
    if ocr_pages:
        console.print(f"OCR page numbers: {ocr_pages[:20]}")
    console.print(f"Output: {target.resolve()}")


@app.command("chunk-pdf")
def chunk_pdf_command(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option(help="Chunk JSONL output path.")] = None,
    target_tokens: Annotated[int, typer.Option(min=50)] = 450,
    max_tokens: Annotated[int, typer.Option(min=100)] = 650,
    min_tokens: Annotated[int, typer.Option(min=1)] = 100,
    overlap_tokens: Annotated[int, typer.Option(min=0)] = 60,
    ocr_mode: Annotated[
        str, typer.Option(help="OCR policy: disabled, auto, or force.")
    ] = "disabled",
    ocr_backend: Annotated[str, typer.Option(help="Optional OCR backend name.")] = "rapidocr",
    ocr_dpi: Annotated[int, typer.Option(min=72, max=600)] = 180,
) -> None:
    """Parse and structurally chunk any local PDF with source provenance."""
    document = parse_pdf(
        source,
        PdfExtractionConfig(mode=ocr_mode, ocr_backend=ocr_backend, ocr_dpi=ocr_dpi),
    )
    config = ChunkingConfig(
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_tokens=overlap_tokens,
    )
    chunks = chunk_document(document, config)
    report = build_chunking_report(document, chunks, config)
    target = output or Path("data/processed/chunks") / f"{document.content_sha256}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8")
    report_path = target.with_suffix(".summary.json")
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    console.print(f"[green]Created {len(chunks)} chunks from {document.page_count} pages.[/green]")
    console.print(
        "Tokens min/p50/p95/max: "
        f"{report.token_min}/{report.token_p50}/{report.token_p95}/{report.token_max}"
    )
    console.print(
        f"Source text-element coverage: {report.source_element_coverage:.2%} "
        f"({report.referenced_text_element_count} referenced, "
        f"{report.repeated_margin_element_count} repeated margins removed)"
    )
    console.print(
        f"Chunks with section context: {report.section_context_chunk_count}/{len(chunks)}"
    )
    console.print(f"Cross-page chunks: {report.cross_page_chunk_count}/{len(chunks)}")
    console.print(f"Output: {target.resolve()}")
    console.print(f"Quality report: {report_path.resolve()}")


@app.command("ingest-document")
def ingest_document_command(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    document_key: Annotated[str, typer.Option(help="Stable logical document key.")],
    registry_path: Annotated[Path, typer.Option(help="Document registry SQLite path.")] = Path(
        "data/catalog/registry.sqlite3"
    ),
    storage_dir: Annotated[Path, typer.Option(help="Versioned artifact directory.")] = Path(
        "data/catalog/versions"
    ),
    ocr_mode: Annotated[
        str, typer.Option(help="OCR policy: disabled, auto, or force.")
    ] = "disabled",
    ocr_backend: Annotated[str, typer.Option(help="Optional OCR backend name.")] = "rapidocr",
    ocr_dpi: Annotated[int, typer.Option(min=72, max=600)] = 180,
) -> None:
    """Ingest a PDF into a transactional, versioned document registry."""
    result = ingest_pdf(
        source,
        document_key,
        DocumentRegistry(registry_path),
        storage_dir,
        pdf_extraction_config=PdfExtractionConfig(
            mode=ocr_mode, ocr_backend=ocr_backend, ocr_dpi=ocr_dpi
        ),
    )
    console.print(f"[green]Ingestion action: {result.action}[/green]")
    console.print(f"Document key: {result.version.document_key}")
    console.print(f"Version: {result.version.version_id}")
    console.print(f"Status: {result.version.status}")
    console.print(f"Chunks: {result.version.chunk_count}")


@app.command("list-active-documents")
def list_active_documents_command(
    registry_path: Annotated[Path, typer.Option(help="Document registry SQLite path.")] = Path(
        "data/catalog/registry.sqlite3"
    ),
) -> None:
    """List the exact document versions visible to the active corpus."""
    versions = DocumentRegistry(registry_path).active_versions()
    if not versions:
        console.print("No active documents.")
        return
    for version in versions:
        console.print(
            f"{version.document_key} | {version.version_id} | "
            f"chunks={version.chunk_count} | sha256={version.content_sha256[:12]}"
        )


@app.command("delete-document")
def delete_document_command(
    document_key: Annotated[str, typer.Argument()],
    registry_path: Annotated[Path, typer.Option(help="Document registry SQLite path.")] = Path(
        "data/catalog/registry.sqlite3"
    ),
    yes: Annotated[bool, typer.Option("--yes", help="Confirm the soft deletion.")] = False,
) -> None:
    """Soft-delete a logical document from the active corpus."""
    if not yes:
        raise typer.BadParameter("Pass --yes to confirm soft deletion")
    DocumentRegistry(registry_path).soft_delete(document_key)
    console.print(f"[yellow]Soft-deleted {document_key} from the active corpus.[/yellow]")


@app.command("build-corpus-index")
def build_corpus_index_command(
    registry_path: Annotated[Path, typer.Option(help="Document registry SQLite path.")] = Path(
        "data/catalog/registry.sqlite3"
    ),
    index_root: Annotated[Path, typer.Option(help="Versioned corpus-index root.")] = Path(
        "data/indexes/corpus"
    ),
    dense: Annotated[bool, typer.Option(help="Build/reuse dense embeddings.")] = False,
    dense_model: Annotated[str, typer.Option()] = DEFAULT_DENSE_MODEL,
) -> None:
    """Build and atomically activate an index over all active document versions."""
    result = build_active_corpus_index(
        DocumentRegistry(registry_path),
        index_root,
        dense_model=dense_model if dense else None,
    )
    console.print(f"[green]Corpus index action: {result.action}[/green]")
    console.print(f"Index ID: {result.pointer.index_id}")
    console.print(f"Active document versions: {len(result.pointer.active_version_ids)}")
    console.print(f"Reused embeddings: {result.manifest['reused_embedding_count']}")
    console.print(f"Encoded embeddings: {result.manifest['encoded_embedding_count']}")


@app.command("build-index")
def build_index_command(
    chunks_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option(help="New persistent index directory.")],
    dense: Annotated[bool, typer.Option(help="Build persistent dense embeddings.")] = False,
    dense_model: Annotated[str, typer.Option(help="Sentence Transformers model name.")] = (
        DEFAULT_DENSE_MODEL
    ),
) -> None:
    """Build a versioned persistent lexical and optional dense chunk index."""
    chunks = read_jsonl(chunks_path, DocumentChunk)
    index = PersistentIndex.build(
        output_dir,
        chunks,
        source_chunk_path=chunks_path,
        dense_model=dense_model if dense else None,
    )
    console.print(f"[green]Built index {index.manifest.index_id}.[/green]")
    console.print(f"Chunks: {index.manifest.chunk_count}")
    console.print(f"Documents: {len(index.manifest.document_ids)}")
    console.print(f"Dense model: {index.manifest.dense_model or 'disabled'}")
    console.print(f"Output: {index.directory}")


@app.command("search-index")
def search_index_command(
    index_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    query: Annotated[str, typer.Argument(help="Search query.")],
    mode: Annotated[str, typer.Option(help="lexical, dense, or hybrid.")] = "lexical",
    top_k: Annotated[int, typer.Option(min=1)] = 5,
    candidate_k: Annotated[int, typer.Option(min=1)] = 50,
    output: Annotated[Path | None, typer.Option(help="Optional full result JSON.")] = None,
    rerank: Annotated[bool, typer.Option(help="Cross-encode and rerank retrieved candidates.")] = (
        False
    ),
    reranker_model: Annotated[str, typer.Option(help="CrossEncoder model name.")] = (
        DEFAULT_RERANKER_MODEL
    ),
    reranker_batch_size: Annotated[int, typer.Option(min=1, max=256)] = 16,
    company: Annotated[str | None, typer.Option(help="Exact company metadata filter.")] = None,
    report_year: Annotated[int | None, typer.Option(help="Exact report-year filter.")] = None,
    document_key: Annotated[str | None, typer.Option(help="Exact document-key filter.")] = None,
    document_type: Annotated[str | None, typer.Option(help="Exact document-type filter.")] = None,
    scope_routing: Annotated[bool, typer.Option(help="Rerank candidates by query scope.")] = False,
    adaptive_candidate_budget: Annotated[
        bool, typer.Option(help="Expand candidate budget based on inferred scope.")
    ] = False,
    max_candidate_k: Annotated[int, typer.Option(min=1, max=1000)] = 100,
) -> None:
    """Search a persistent index and return chunks with source provenance."""
    if mode not in {"lexical", "dense", "hybrid"}:
        raise typer.BadParameter("mode must be lexical, dense, or hybrid")
    index = (
        resolve_current_index(index_dir)
        if (index_dir / "current.json").is_file()
        else PersistentIndex(index_dir)
    )
    _, budget = plan_candidate_budget(
        query, candidate_k, maximum_candidate_k=max_candidate_k, enabled=adaptive_candidate_budget
    )
    effective_candidate_k = budget.effective_candidate_k
    retrieval_k = max(effective_candidate_k, top_k) if rerank or scope_routing else top_k
    hits = index.search(
        query,
        top_k=retrieval_k,
        mode=mode,
        candidate_k=max(effective_candidate_k, top_k),
        filters=SearchFilters(
            company_names=[company] if company else [],
            report_years=[report_year] if report_year else [],
            document_keys=[document_key] if document_key else [],
            document_types=[document_type] if document_type else [],
        ),
    )
    if scope_routing:
        scope, hits = route_by_scope(
            query, hits, max(effective_candidate_k, top_k) if rerank else top_k
        )
        console.print(
            f"Scope: {scope.name} ({scope.confidence}); cues={', '.join(scope.matched_cues)}"
        )
    if adaptive_candidate_budget:
        console.print(
            f"Candidate budget: {candidate_k} -> {effective_candidate_k} ({budget.reason})"
        )
    if rerank:
        hits = CrossEncoderReranker(reranker_model, reranker_batch_size).rerank(query, hits, top_k)
    for hit in hits:
        section = " > ".join(hit.chunk.section_path) or "(no section)"
        excerpt = hit.chunk.text.replace("\n", " ")[:180]
        console.print(f"[bold]#{hit.rank} score={hit.score:.6f}[/bold]")
        if hit.rerank_score is not None:
            console.print(
                f"rerank_score={hit.rerank_score:.6f} | original_rank={hit.original_rank} "
                f"| rank_delta={hit.rank_delta:+d}"
            )
        console.print(f"pages={hit.chunk.page_start}-{hit.chunk.page_end} | {section}")
        console.print(excerpt)
        console.print()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps([hit.model_dump(mode="json") for hit in hits], ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        console.print(f"Full results: {output.resolve()}")


@app.command("generate-ranking-diagnostics")
def generate_ranking_diagnostics_command(
    profiles_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    registry_path: Annotated[Path, typer.Option()] = Path("data/catalog/registry.sqlite3"),
    index_root: Annotated[Path, typer.Option()] = Path("data/indexes/corpus"),
    output: Annotated[Path, typer.Option()] = Path("data/diagnostics/ranking-diagnostics-v1.json"),
    candidate_k: Annotated[int, typer.Option(min=5, max=100)] = 20,
) -> None:
    """Generate structure-anchored Chinese ranking judgments for review."""
    profiles = [
        DocumentProfile.model_validate(item)
        for item in json.loads(profiles_path.read_text(encoding="utf-8"))
    ]
    index = (
        resolve_current_index(index_root)
        if (index_root / "current.json").is_file()
        else PersistentIndex(index_root)
    )
    dataset = generate_diagnostic_dataset(
        DocumentRegistry(registry_path), index, profiles, candidate_k
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dataset.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Generated {dataset.query_count} diagnostic queries.[/green]")
    console.print(f"Accepted: {dataset.accepted_count}")
    console.print(f"Needs review: {dataset.needs_review_count}")
    console.print(f"Dataset ID: {dataset.dataset_id}")
    console.print(f"Output: {output.resolve()}")


@app.command("apply-document-profiles")
def apply_document_profiles_command(
    profiles_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    registry_path: Annotated[Path, typer.Option()] = Path("data/catalog/registry.sqlite3"),
) -> None:
    """Attach reviewed company/year/type metadata to active registry versions."""
    profiles = [
        DocumentProfile.model_validate(item)
        for item in json.loads(profiles_path.read_text(encoding="utf-8"))
    ]
    registry = DocumentRegistry(registry_path)
    for profile in profiles:
        version = registry.update_metadata(
            profile.document_key,
            {
                "company_name": profile.company,
                "report_year": profile.year,
                "document_type": "annual",
            },
        )
        console.print(f"[green]Updated {version.document_key} ({version.version_id}).[/green]")


@app.command("evaluate-ranking-diagnostics")
def evaluate_ranking_diagnostics_command(
    dataset_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    index_root: Annotated[Path, typer.Option()] = Path("data/indexes/corpus"),
    output: Annotated[Path, typer.Option()] = Path("reports/ranking/hybrid-v1.json"),
    mode: Annotated[str, typer.Option()] = "lexical",
    top_k: Annotated[int, typer.Option(min=1, max=100)] = 5,
    candidate_k: Annotated[int, typer.Option(min=1, max=1000)] = 20,
    rerank: Annotated[bool, typer.Option()] = False,
    reranker_model: Annotated[str, typer.Option()] = DEFAULT_RERANKER_MODEL,
    metadata_filters: Annotated[bool, typer.Option()] = False,
    scope_routing: Annotated[bool, typer.Option()] = False,
    adaptive_candidate_budget: Annotated[bool, typer.Option()] = False,
    max_candidate_k: Annotated[int, typer.Option(min=1, max=1000)] = 100,
) -> None:
    """Evaluate only accepted, structure-anchored ranking judgments."""
    if mode not in {"lexical", "dense", "hybrid"}:
        raise typer.BadParameter("mode must be lexical, dense, or hybrid")
    dataset = DiagnosticDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    index = (
        resolve_current_index(index_root)
        if (index_root / "current.json").is_file()
        else PersistentIndex(index_root)
    )
    reranker = CrossEncoderReranker(reranker_model) if rerank else None
    evaluation = evaluate_diagnostic_dataset(
        dataset,
        index,
        mode=mode,
        top_k=top_k,
        candidate_k=max(candidate_k, top_k),
        reranker=reranker,
        use_metadata_filters=metadata_filters,
        use_scope_routing=scope_routing,
        adaptive_candidate_budget=adaptive_candidate_budget,
        max_candidate_k=max_candidate_k,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(evaluation.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Evaluated {evaluation.evaluated_query_count} queries.[/green]")
    console.print(f"Hit@{top_k}: {evaluation.hit_at_k:.4f}")
    console.print(f"MRR@{top_k}: {evaluation.mrr_at_k:.4f}")
    console.print(f"Average effective candidate_k: {evaluation.average_effective_candidate_k:.1f}")
    console.print(f"Candidate recall: {evaluation.candidate_recall_rate:.4f}")
    console.print(f"Output: {output.resolve()}")


@app.command("evaluate-holdout")
def evaluate_holdout_command(
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)] = Path(
        "data/diagnostics/holdout-eval-v2.json"
    ),
    index_root: Annotated[Path, typer.Option()] = Path("data/indexes/corpus"),
    output: Annotated[Path, typer.Option()] = Path("reports/ranking/holdout-eval-v2-runtime.json"),
    mode: Annotated[str, typer.Option()] = "lexical",
    top_k: Annotated[int, typer.Option(min=1, max=100)] = 5,
    candidate_k: Annotated[int, typer.Option(min=1, max=1000)] = 20,
    metadata_filters: Annotated[bool, typer.Option()] = False,
    scope_routing: Annotated[bool, typer.Option()] = False,
    adaptive_candidate_budget: Annotated[bool, typer.Option()] = True,
    max_candidate_k: Annotated[int, typer.Option(min=1, max=1000)] = 100,
    rerank: Annotated[bool, typer.Option()] = False,
    reranker_model: Annotated[str, typer.Option()] = DEFAULT_RERANKER_MODEL,
) -> None:
    """Run the existing retrieval evaluator on the reviewed holdout manifest."""
    if mode not in {"lexical", "dense", "hybrid"}:
        raise typer.BadParameter("mode must be lexical, dense, or hybrid")
    index = (
        resolve_current_index(index_root)
        if (index_root / "current.json").is_file()
        else PersistentIndex(index_root)
    )
    dataset = holdout_eval_to_diagnostics(manifest_path, index)
    reranker = CrossEncoderReranker(reranker_model) if rerank else None
    evaluation = evaluate_diagnostic_dataset(
        dataset,
        index,
        mode=mode,
        top_k=top_k,
        candidate_k=max(candidate_k, top_k),
        reranker=reranker,
        use_metadata_filters=metadata_filters,
        use_scope_routing=scope_routing,
        adaptive_candidate_budget=adaptive_candidate_budget,
        max_candidate_k=max_candidate_k,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(evaluation.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Evaluated {evaluation.evaluated_query_count} holdout queries.[/green]")
    console.print(f"Hit@{top_k}: {evaluation.hit_at_k:.4f} | MRR: {evaluation.mrr_at_k:.4f}")


@app.command("analyze-ranking-failures")
def analyze_ranking_failures_command(
    dataset_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    evaluation_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    index_root: Annotated[Path, typer.Option()] = Path("data/indexes/corpus"),
    output: Annotated[Path, typer.Option()] = Path("reports/ranking/recall-failures.json"),
) -> None:
    """Explain candidate-recall failures using full component rankings."""
    dataset = DiagnosticDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    evaluation = DiagnosticEvaluation.model_validate_json(
        evaluation_path.read_text(encoding="utf-8")
    )
    index = (
        resolve_current_index(index_root)
        if (index_root / "current.json").is_file()
        else PersistentIndex(index_root)
    )
    report = analyze_recall_failures(dataset, evaluation, index)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Analyzed {report.failure_count} recall failures.[/green]")
    for failure_type, count in report.failures_by_type.items():
        console.print(f"{failure_type}: {count}")
    console.print(f"Output: {output.resolve()}")


@app.command("analyze-holdout-failures")
def analyze_holdout_failures_command(
    evaluation_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("reports/ranking/holdout-eval-v2-failures.json"),
) -> None:
    """Classify misses from a reviewed holdout evaluation without mixing datasets."""
    evaluation = DiagnosticEvaluation.model_validate_json(
        evaluation_path.read_text(encoding="utf-8")
    )
    failures = []
    for result in evaluation.results:
        if result.hit_at_k:
            continue
        if not result.candidate_recall:
            failure_type = "candidate_recall"
        elif result.first_relevant_rank is None:
            failure_type = "downstream_ranking"
        else:
            failure_type = "unresolved"
        failures.append(
            {
                "query_id": result.query_id,
                "failure_type": failure_type,
                "first_relevant_rank": result.first_relevant_rank,
                "candidate_first_rank": result.candidate_first_rank,
                "effective_candidate_k": result.effective_candidate_k,
            }
        )
    report = {
        "schema_version": 1,
        "evaluation_path": str(evaluation_path),
        "failure_count": len(failures),
        "failures": failures,
        "failures_by_type": {
            kind: sum(item["failure_type"] == kind for item in failures)
            for kind in {item["failure_type"] for item in failures}
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Analyzed {len(failures)} holdout failures.[/green]")


@app.command("generate-holdout-review")
def generate_holdout_review_command(
    profiles_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    dataset_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)] = Path(
        "data/diagnostics/ranking-diagnostics-v1.json"
    ),
    registry_path: Annotated[Path, typer.Option()] = Path("data/catalog/registry.sqlite3"),
    index_root: Annotated[Path, typer.Option()] = Path("data/indexes/corpus"),
    output: Annotated[Path, typer.Option()] = Path("data/diagnostics/holdout-review-v1.json"),
    markdown: Annotated[Path, typer.Option()] = Path("reports/ranking/holdout-review-v1.md"),
) -> None:
    """Generate new human-reviewable holdout questions without freezing gold labels."""
    profiles = [
        DocumentProfile.model_validate(item)
        for item in json.loads(profiles_path.read_text(encoding="utf-8"))
    ]
    dataset = DiagnosticDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    index = (
        resolve_current_index(index_root)
        if (index_root / "current.json").is_file()
        else PersistentIndex(index_root)
    )
    pack = generate_holdout_review_pack(
        DocumentRegistry(registry_path), profiles, index.manifest.index_id, dataset
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(pack.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_review_markdown(pack), encoding="utf-8")
    console.print(f"[green]Generated {pack.item_count} review candidates.[/green]")
    console.print(f"Excluded existing queries: {pack.excluded_query_count}")
    console.print(f"JSON: {output.resolve()}")
    console.print(f"Review sheet: {markdown.resolve()}")


@app.command("serve")
def serve_command(
    config: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="TOML service configuration."),
    ] = None,
) -> None:
    """Run the long-lived retrieval API with one validated index instance."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("API dependencies are missing. Run: uv sync --extra api") from exc

    from findoc_rag.api import create_app

    settings = load_settings(config)
    uvicorn.run(
        create_app(settings),
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.server.log_level,
    )


@app.command("import-financebench")
def import_financebench(
    source: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Public FinanceBench JSONL file."),
    ] = Path("data/raw/financebench_open_source.jsonl"),
    output_dir: Annotated[Path, typer.Option(help="Output directory.")] = Path("data/processed"),
) -> None:
    """Convert FinanceBench into corpus and question JSONL files."""
    corpus, questions = convert_financebench(source)
    write_jsonl(corpus, output_dir / "financebench_corpus.jsonl")
    write_jsonl(questions, output_dir / "financebench_questions.jsonl")
    console.print(f"[green]Imported {len(questions)} questions.[/green]")
    console.print(f"Created {len(corpus)} unique evidence-page documents.")
    console.print(
        "[yellow]Scope: bootstrap evidence-page corpus, not full 10-K retrieval.[/yellow]"
    )


@app.command("evaluate-bm25")
def evaluate_bm25(
    corpus_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/processed/financebench_corpus.jsonl"
    ),
    questions_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/processed/financebench_questions.jsonl"
    ),
    report_dir: Annotated[Path, typer.Option()] = Path("reports/bm25"),
    top_k: Annotated[int, typer.Option(min=10)] = 10,
) -> None:
    """Evaluate the BM25 baseline against gold evidence pages."""
    corpus = read_jsonl(corpus_path, CorpusDocument)
    questions = read_jsonl(questions_path, BenchmarkQuestion)
    summary, results = evaluate_retriever(BM25Retriever(corpus), questions, top_k=top_k)
    write_json(summary, report_dir / "summary.json")
    write_dict_jsonl(results, report_dir / "per_question.jsonl")

    console.print(f"[green]Evaluated {summary['question_count']} questions.[/green]")
    for name, value in summary["metrics"].items():
        console.print(f"{name}: {value:.4f}")
    console.print(f"Reports: {report_dir.resolve()}")


@app.command("evaluate-dense")
def evaluate_dense(
    corpus_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/processed/financebench_corpus.jsonl"
    ),
    questions_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/processed/financebench_questions.jsonl"
    ),
    report_dir: Annotated[Path, typer.Option()] = Path("reports/dense"),
    model_name: Annotated[str, typer.Option(help="Sentence Transformers model name.")] = (
        DEFAULT_MODEL
    ),
    top_k: Annotated[int, typer.Option(min=10)] = 10,
) -> None:
    """Evaluate a dense embedding baseline against gold evidence pages."""
    corpus = read_jsonl(corpus_path, CorpusDocument)
    questions = read_jsonl(questions_path, BenchmarkQuestion)
    retriever = DenseRetriever(corpus, model_name=model_name)
    summary, results = evaluate_retriever(retriever, questions, top_k=top_k)
    summary["model"] = model_name
    write_json(summary, report_dir / "summary.json")
    write_dict_jsonl(results, report_dir / "per_question.jsonl")

    console.print(f"[green]Evaluated {summary['question_count']} questions.[/green]")
    console.print(f"model: {model_name}")
    for name, value in summary["metrics"].items():
        console.print(f"{name}: {value:.4f}")
    console.print(f"Reports: {report_dir.resolve()}")


if __name__ == "__main__":
    app()
