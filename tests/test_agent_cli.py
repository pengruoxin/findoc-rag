from pathlib import Path

from typer.testing import CliRunner

from findoc_rag.agent_tasks import AgentTaskStore
from findoc_rag.cli import app
from findoc_rag.documents.models import BoundingBox, DocumentChunk, ElementReference
from findoc_rag.indexing import PersistentIndex


def _chunk(company: str, chunk_id: str, value: int) -> DocumentChunk:
    text = f"{company}2024年营业收入为{value}元"
    return DocumentChunk(
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


def test_agent_run_and_inspect_commands_persist_a_replayable_trace(
    tmp_path: Path,
) -> None:
    chunks = [_chunk("甲公司", "chunk-a", 100), _chunk("乙公司", "chunk-b", 200)]
    source = tmp_path / "chunks.jsonl"
    source.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    PersistentIndex.build(index_dir, chunks, source)
    task_dir = tmp_path / "tasks"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "比较甲公司和乙公司2024年营业收入",
            "--index-dir",
            str(index_dir),
            "--task-dir",
            str(task_dir),
            "--top-k",
            "1",
            "--runtime",
            "deterministic-baseline",
        ],
        env={"FINDOC_RAG_TRACING_ENABLED": "false"},
    )

    assert result.exit_code == 0, result.output
    assert "Agent task completed" in result.stdout
    traces = list(task_dir.glob("*.json"))
    assert len(traces) == 1
    task_id = traces[0].stem
    trace = AgentTaskStore(task_dir).load(task_id)
    assert trace.stop_reason == "sufficient_evidence"
    assert len(trace.tool_calls) == 2

    inspected = runner.invoke(
        app,
        ["agent", "inspect", task_id, "--task-dir", str(task_dir)],
    )

    assert inspected.exit_code == 0, inspected.output
    assert f"Task ID: {task_id}" in inspected.stdout
    assert "Targets: 甲公司, 乙公司" in inspected.stdout


def test_agent_run_requires_a_provider_key_for_the_default_deepseek_runtime(
    tmp_path: Path,
) -> None:
    chunks = [_chunk("甲公司", "chunk-a", 100), _chunk("乙公司", "chunk-b", 200)]
    source = tmp_path / "chunks.jsonl"
    source.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    PersistentIndex.build(index_dir, chunks, source)

    result = CliRunner().invoke(
        app,
        [
            "agent",
            "run",
            "比较甲公司和乙公司2024年营业收入",
            "--index-dir",
            str(index_dir),
        ],
        env={
            "DEEPSEEK_API_KEY": "",
            "FINDOC_RAG_PROVIDER_API_KEY": "",
            "FINDOC_RAG_TRACING_ENABLED": "false",
        },
    )

    assert result.exit_code != 0
    assert "DeepSeek tool-calling runtime requires" in result.output


def test_evidence_verifier_is_restricted_to_extract_tasks() -> None:
    result = CliRunner().invoke(
        app,
        [
            "agent",
            "run",
            "比较甲公司和乙公司的营业收入",
            "--task",
            "compare",
            "--evidence-verifier",
        ],
    )

    assert result.exit_code != 0
    assert "evidence-verifier only supports task=extract" in result.output


def test_verifier_policy_rejects_an_unknown_mode() -> None:
    result = CliRunner().invoke(
        app,
        [
            "agent",
            "run",
            "列出甲公司营业收入",
            "--task",
            "extract",
            "--verifier-policy",
            "unknown",
        ],
    )

    assert result.exit_code != 0
    assert "verifier-policy must be auto, off, or always" in result.output
