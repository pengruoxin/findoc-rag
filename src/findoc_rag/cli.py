from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from findoc_rag import __version__
from findoc_rag.datasets.financebench import convert_financebench, write_jsonl
from findoc_rag.evaluation.retrieval import evaluate_retriever
from findoc_rag.io import read_jsonl, write_dict_jsonl, write_json
from findoc_rag.retrieval.bm25 import BM25Retriever
from findoc_rag.retrieval.dense import DEFAULT_MODEL, DenseRetriever
from findoc_rag.schemas import BenchmarkQuestion, CorpusDocument
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
