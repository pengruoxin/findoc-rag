from pathlib import Path

import pymupdf

from findoc_rag.chunking import ChunkingConfig
from findoc_rag.corpus import build_active_corpus_index, resolve_current_index
from findoc_rag.documents.models import DocumentChunk
from findoc_rag.ingestion import file_sha256, ingest_pdf
from findoc_rag.io import read_jsonl
from findoc_rag.registry import DocumentRegistry


def write_pdf(path: Path, text: str) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def test_ingestion_skips_identical_content_and_activates_new_version(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, "Annual revenue was 100 million in 2023. " * 10)
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    storage = tmp_path / "versions"
    config = ChunkingConfig(target_tokens=100, max_tokens=150, min_tokens=20, overlap_tokens=10)

    first = ingest_pdf(source, "company:annual-report", registry, storage, config)
    unchanged = ingest_pdf(source, "company:annual-report", registry, storage, config)
    first_version_id = first.version.version_id

    source.unlink()
    write_pdf(source, "Annual revenue was 120 million in 2024. " * 10)
    second = ingest_pdf(source, "company:annual-report", registry, storage, config)
    versions = registry.versions("company:annual-report")

    assert first.action == "ingested"
    assert unchanged.action == "unchanged"
    assert second.action == "ingested"
    assert second.version.version_id != first_version_id
    assert [version.status for version in versions] == ["superseded", "active"]
    assert registry.active_versions()[0].version_id == second.version.version_id
    assert Path(second.version.document_ir_path).is_file()
    assert Path(second.version.chunks_path).is_file()


def test_failed_staging_version_does_not_replace_active_version(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, "Revenue was 100 million. " * 10)
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    active = ingest_pdf(source, "company:report", registry, tmp_path / "versions")

    decision = registry.begin_ingestion(
        "company:report",
        "f" * 64,
        source,
    )
    registry.mark_failed(decision.version.version_id, "parser failed")

    assert registry.active_versions()[0].version_id == active.version.version_id
    versions = registry.versions("company:report")
    assert versions[-1].status == "failed"
    assert versions[-1].error_message == "parser failed"


def test_soft_delete_removes_document_from_active_corpus(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, "Revenue was 100 million. " * 10)
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    ingest_pdf(source, "company:report", registry, tmp_path / "versions")

    registry.soft_delete("company:report")

    assert registry.active_versions() == []
    assert registry.versions("company:report")[-1].status == "deleted"


def test_file_hash_changes_with_pdf_content(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, "Version one")
    first_hash = file_sha256(source)
    source.unlink()
    write_pdf(source, "Version two")

    assert file_sha256(source) != first_hash


def test_active_corpus_index_generation_is_atomic_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, "Revenue was 100 million. " * 10)
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    ingest_pdf(source, "company:report", registry, tmp_path / "versions")
    index_root = tmp_path / "corpus-index"

    first = build_active_corpus_index(registry, index_root)
    second = build_active_corpus_index(registry, index_root)
    current = resolve_current_index(index_root)

    assert first.action == "built"
    assert second.action == "unchanged"
    assert current.manifest.index_id == first.pointer.index_id
    assert (index_root / "current.json").is_file()


def test_reingesting_historical_content_reuses_version_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, "Revenue was 100 million in 2023. " * 10)
    original_bytes = source.read_bytes()
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    storage = tmp_path / "versions"

    first = ingest_pdf(source, "company:report", registry, storage)
    source.unlink()
    write_pdf(source, "Revenue was 120 million in 2024. " * 10)
    second = ingest_pdf(source, "company:report", registry, storage)
    source.write_bytes(original_bytes)
    restored = ingest_pdf(source, "company:report", registry, storage)

    assert first.version.version_id != second.version.version_id
    assert restored.action == "reused"
    assert restored.version.version_id == first.version.version_id
    assert registry.active_versions()[0].version_id == first.version.version_id
    assert [version.status for version in registry.versions("company:report")] == [
        "active",
        "superseded",
    ]


def test_deleted_document_is_excluded_from_next_generation(tmp_path: Path) -> None:
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    storage = tmp_path / "versions"
    first_source = tmp_path / "first.pdf"
    second_source = tmp_path / "second.pdf"
    write_pdf(first_source, "Alpha revenue was 100 million. " * 10)
    write_pdf(second_source, "Beta revenue was 200 million. " * 10)
    first = ingest_pdf(first_source, "alpha:report", registry, storage)
    ingest_pdf(second_source, "beta:report", registry, storage)
    index_root = tmp_path / "corpus-index"

    before = build_active_corpus_index(registry, index_root)
    registry.soft_delete("alpha:report")
    after = build_active_corpus_index(registry, index_root)
    current = resolve_current_index(index_root)
    deleted_chunk_ids = {
        chunk.chunk_id for chunk in read_jsonl(Path(first.version.chunks_path), DocumentChunk)
    }

    assert before.manifest["chunk_count"] > after.manifest["chunk_count"]
    assert first.version.version_id not in after.pointer.active_version_ids
    assert current._load_chunks(list(deleted_chunk_ids)) == {}
