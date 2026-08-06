import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from findoc_rag import __version__
from findoc_rag.chunking import ChunkingConfig, build_chunking_report, chunk_document
from findoc_rag.config import load_settings
from findoc_rag.corpus import build_active_corpus_index, resolve_current_index
from findoc_rag.datasets.financebench import convert_financebench, write_jsonl
from findoc_rag.diagnostics import (
    DiagnosticDataset,
    DiagnosticEvaluation,
    DocumentProfile,
    analyze_recall_failures,
    evaluate_diagnostic_dataset,
    generate_diagnostic_dataset,
)
from findoc_rag.documents.models import DocumentChunk
from findoc_rag.documents.pdf import parse_pdf
from findoc_rag.evaluation.retrieval import evaluate_retriever
from findoc_rag.holdout import (
    generate_holdout_review_pack,
    holdout_eval_to_diagnostics,
    render_review_markdown,
)
from findoc_rag.indexing import DEFAULT_DENSE_MODEL, PersistentIndex, SearchFilters
from findoc_rag.ingestion import ingest_pdf
from findoc_rag.io import read_jsonl, write_dict_jsonl, write_json
from findoc_rag.registry import DocumentRegistry
from findoc_rag.reranking import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from findoc_rag.retrieval.bm25 import BM25Retriever
from findoc_rag.retrieval.dense import DEFAULT_MODEL, DenseRetriever
from findoc_rag.schemas import BenchmarkQuestion, CorpusDocument
from findoc_rag.scope_routing import plan_candidate_budget, route_by_scope
from findoc_rag.sources.cninfo import (
    CninfoClient,
    select_chinese_annual_report,
    write_artifact_manifest,
)

app = typer.Typer(help="Verifiable RAG for complex Chinese listed-company documents.")
console = Console()


@app.callback()
def main() -> None:
    """Run FinDocRAG dataset, retrieval, and evaluation commands."""


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
) -> None:
    """Parse any local PDF into the coordinate-preserving Document IR."""
    document = parse_pdf(source)
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
) -> None:
    """Parse and structurally chunk any local PDF with source provenance."""
    document = parse_pdf(source)
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
    target.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8"
    )
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
) -> None:
    """Ingest a PDF into a transactional, versioned document registry."""
    result = ingest_pdf(
        source,
        document_key,
        DocumentRegistry(registry_path),
        storage_dir,
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
    mode: Annotated[str, typer.Option(help="lexical, dense, or hybrid.")] = "hybrid",
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
    retrieval_k = (
        max(effective_candidate_k, top_k) if rerank or scope_routing else top_k
    )
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
        hits = CrossEncoderReranker(reranker_model, reranker_batch_size).rerank(
            query, hits, top_k
        )
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
    output: Annotated[Path, typer.Option()] = Path(
        "data/diagnostics/ranking-diagnostics-v1.json"
    ),
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
    mode: Annotated[str, typer.Option()] = "hybrid",
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
    console.print(
        f"Average effective candidate_k: {evaluation.average_effective_candidate_k:.1f}"
    )
    console.print(f"Candidate recall: {evaluation.candidate_recall_rate:.4f}")
    console.print(f"Output: {output.resolve()}")


@app.command("evaluate-holdout")
def evaluate_holdout_command(
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)] = Path("data/diagnostics/holdout-eval-v2.json"),
    index_root: Annotated[Path, typer.Option()] = Path("data/indexes/corpus"),
    output: Annotated[Path, typer.Option()] = Path("reports/ranking/holdout-eval-v2-runtime.json"),
    mode: Annotated[str, typer.Option()] = "hybrid",
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
    index = resolve_current_index(index_root) if (index_root / "current.json").is_file() else PersistentIndex(index_root)
    dataset = holdout_eval_to_diagnostics(manifest_path, index)
    reranker = CrossEncoderReranker(reranker_model) if rerank else None
    evaluation = evaluate_diagnostic_dataset(
        dataset, index, mode=mode, top_k=top_k, candidate_k=max(candidate_k, top_k),
        reranker=reranker, use_metadata_filters=metadata_filters,
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
    output: Annotated[Path, typer.Option()] = Path(
        "reports/ranking/recall-failures.json"
    ),
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
    evaluation = DiagnosticEvaluation.model_validate_json(evaluation_path.read_text(encoding="utf-8"))
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
        failures.append({
            "query_id": result.query_id,
            "failure_type": failure_type,
            "first_relevant_rank": result.first_relevant_rank,
            "candidate_first_rank": result.candidate_first_rank,
            "effective_candidate_k": result.effective_candidate_k,
        })
    report = {
        "schema_version": 1, "evaluation_path": str(evaluation_path),
        "failure_count": len(failures), "failures": failures,
        "failures_by_type": {kind: sum(item["failure_type"] == kind for item in failures) for kind in {item["failure_type"] for item in failures}},
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
    output: Annotated[Path, typer.Option()] = Path(
        "data/diagnostics/holdout-review-v1.json"
    ),
    markdown: Annotated[Path, typer.Option()] = Path(
        "reports/ranking/holdout-review-v1.md"
    ),
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
    console.print("[yellow]Scope: bootstrap evidence-page corpus, not full 10-K retrieval.[/yellow]")


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
