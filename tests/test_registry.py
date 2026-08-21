import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pymupdf
import pytest

from findoc_rag.chunking import ChunkingConfig
from findoc_rag.corpus import build_active_corpus_index, resolve_current_index
from findoc_rag.documents.models import DocumentChunk
from findoc_rag.documents.quality import PdfQualityConfig, PdfQualityError
from findoc_rag.ingestion import build_processing_fingerprint, file_sha256, ingest_pdf
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


def test_chunking_config_change_creates_artifact_revision_with_stable_content_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, "Annual revenue was 100 million in 2023. " * 10)
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    storage = tmp_path / "versions"
    first_config = ChunkingConfig(repeated_margin_ratio=0.12)
    second_config = ChunkingConfig(repeated_margin_ratio=0.13)

    first = ingest_pdf(source, "company:report", registry, storage, first_config)
    second = ingest_pdf(source, "company:report", registry, storage, second_config)
    first_chunks = read_jsonl(Path(first.version.chunks_path), DocumentChunk)
    second_chunks = read_jsonl(Path(second.version.chunks_path), DocumentChunk)

    assert second.action == "ingested"
    assert second.version.version_id != first.version.version_id
    assert second.version.content_version_id == first.version.content_version_id
    assert second.version.processing_fingerprint != first.version.processing_fingerprint
    assert [chunk.chunk_id for chunk in second_chunks] == [
        chunk.chunk_id for chunk in first_chunks
    ]


def test_historical_processing_config_is_reused_on_rollback(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, "Annual revenue was 100 million in 2023. " * 10)
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    storage = tmp_path / "versions"
    original_config = ChunkingConfig(repeated_margin_ratio=0.12)
    revised_config = ChunkingConfig(repeated_margin_ratio=0.13)

    original = ingest_pdf(source, "company:report", registry, storage, original_config)
    revised = ingest_pdf(source, "company:report", registry, storage, revised_config)
    restored = ingest_pdf(source, "company:report", registry, storage, original_config)

    assert revised.version.version_id != original.version.version_id
    assert restored.action == "reused"
    assert restored.version.version_id == original.version.version_id
    assert registry.active_versions()[0].version_id == original.version.version_id


def test_processing_fingerprint_covers_parser_ir_and_chunking_components() -> None:
    config = ChunkingConfig()
    baseline, components = build_processing_fingerprint(config)
    parser_changed, _ = build_processing_fingerprint(config, parser_version="next")
    schema_changed, _ = build_processing_fingerprint(config, ir_schema_version="next")
    chunking_changed, _ = build_processing_fingerprint(
        ChunkingConfig(repeated_margin_ratio=0.13)
    )

    assert len({baseline, parser_changed, schema_changed, chunking_changed}) == 4
    assert components["ocr_config"] == {"mode": "disabled"}
    assert components["quality_config"] == PdfQualityConfig().model_dump(mode="json")


def test_ingestion_manifest_records_processing_provenance(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, "Annual revenue was 100 million in 2023. " * 10)
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    result = ingest_pdf(source, "company:report", registry, tmp_path / "versions")
    manifest_path = Path(result.version.document_ir_path).with_name("ingestion-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["content_version_id"] == result.version.content_version_id
    assert manifest["processing_fingerprint"] == result.version.processing_fingerprint
    assert manifest["processing_components"]["chunking_config"] == ChunkingConfig().model_dump(
        mode="json"
    )
    assert manifest["processing_components"]["quality_config"]["policy"] == "warning"
    quality_path = Path(manifest["pdf_quality_report_path"])
    assert quality_path.is_file()
    assert json.loads(quality_path.read_text(encoding="utf-8"))["policy"] == "warning"


def test_strict_pdf_quality_failure_does_not_activate_document(tmp_path: Path) -> None:
    source = tmp_path / "image-only.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2, 2), False)
    page.insert_image(pymupdf.Rect(0, 0, 100, 100), pixmap=pixmap)
    pdf.save(source)
    pdf.close()
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")

    with pytest.raises(PdfQualityError, match="unresolved OCR"):
        ingest_pdf(
            source,
            "company:image-report",
            registry,
            tmp_path / "versions",
            pdf_quality_config=PdfQualityConfig(policy="strict"),
        )

    assert registry.active_versions() == []
    assert registry.versions("company:image-report")[-1].status == "failed"


def test_registry_migrates_legacy_content_unique_schema_without_data_loss(
    tmp_path: Path,
) -> None:
    database = tmp_path / "registry.sqlite3"
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                document_key TEXT PRIMARY KEY, current_version_id TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                deleted_at TEXT, metadata_json TEXT NOT NULL
            );
            CREATE TABLE document_versions (
                version_id TEXT PRIMARY KEY, document_key TEXT NOT NULL,
                content_sha256 TEXT NOT NULL, source_path TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL, activated_at TEXT,
                document_ir_path TEXT, chunks_path TEXT, chunk_count INTEGER,
                error_message TEXT, metadata_json TEXT NOT NULL,
                UNIQUE(document_key, content_sha256)
            );
            CREATE INDEX version_document_status_idx
                ON document_versions(document_key, status);
            CREATE INDEX version_content_idx ON document_versions(content_sha256);
            """
        )
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, NULL, ?)",
            ("company:report", "historical-id", now, now, '{"company":"甲"}'),
        )
        connection.execute(
            """
            INSERT INTO document_versions VALUES (
                ?, ?, ?, ?, 'active', ?, ?, ?, ?, 3, NULL, ?
            )
            """,
            (
                "historical-id",
                "company:report",
                "a" * 64,
                "/source.pdf",
                now,
                now,
                "/document.json",
                "/chunks.jsonl",
                '{"company":"甲"}',
            ),
        )

    registry = DocumentRegistry(database)
    migrated = registry.active_versions()[0]
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    next_revision = registry.begin_ingestion(
        "company:report", "a" * 64, source, processing_fingerprint="current"
    )

    assert migrated.version_id == "historical-id"
    assert migrated.processing_fingerprint == "legacy"
    assert migrated.chunk_count == 3
    assert migrated.metadata == {"company": "甲"}
    assert next_revision.action == "new"
    assert next_revision.version.content_version_id == migrated.content_version_id


def test_metadata_stays_aligned_for_unchanged_and_staged_revisions(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, "Annual revenue was 100 million in 2023. " * 10)
    database = tmp_path / "registry.sqlite3"
    registry = DocumentRegistry(database)
    storage = tmp_path / "versions"
    config = ChunkingConfig(repeated_margin_ratio=0.12)

    first = ingest_pdf(
        source, "company:report", registry, storage, config, metadata={"label": "first"}
    )
    unchanged = ingest_pdf(
        source, "company:report", registry, storage, config, metadata={"label": "updated"}
    )
    pending_fingerprint, _ = build_processing_fingerprint(
        ChunkingConfig(repeated_margin_ratio=0.13)
    )
    registry.begin_ingestion(
        "company:report",
        first.version.content_sha256,
        source,
        metadata={"label": "pending"},
        processing_fingerprint=pending_fingerprint,
    )
    with sqlite3.connect(database) as connection:
        document_metadata = json.loads(
            connection.execute(
                "SELECT metadata_json FROM documents WHERE document_key = ?",
                ("company:report",),
            ).fetchone()[0]
        )

    assert unchanged.version.metadata == {"label": "updated"}
    assert registry.active_versions()[0].metadata == {"label": "updated"}
    assert document_metadata == {"label": "updated"}


def test_processing_upgrade_inherits_active_metadata_when_not_resupplied(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, "Annual revenue was 100 million in 2023. " * 10)
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    storage = tmp_path / "versions"
    first = ingest_pdf(
        source,
        "company:report",
        registry,
        storage,
        ChunkingConfig(repeated_margin_ratio=0.12),
        metadata={"company_name": "甲公司", "report_year": 2023},
    )

    upgraded = ingest_pdf(
        source,
        "company:report",
        registry,
        storage,
        ChunkingConfig(repeated_margin_ratio=0.13),
    )

    assert upgraded.version.version_id != first.version.version_id
    assert upgraded.version.metadata == {
        "company_name": "甲公司",
        "report_year": 2023,
    }
    assert registry.active_versions()[0].metadata == upgraded.version.metadata


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


def test_benchmark_index_excludes_unselected_splits(tmp_path: Path) -> None:
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    storage = tmp_path / "versions"
    dev_source = tmp_path / "dev.pdf"
    frozen_source = tmp_path / "frozen.pdf"
    write_pdf(dev_source, "Development revenue was 100 million. " * 10)
    write_pdf(frozen_source, "Frozen revenue was 200 million. " * 10)
    dev = ingest_pdf(
        dev_source,
        "company:dev:2024",
        registry,
        storage,
        metadata={"benchmark_split": "dev"},
    )
    frozen = ingest_pdf(
        frozen_source,
        "company:frozen:2024",
        registry,
        storage,
        metadata={"benchmark_split": "frozen_test"},
    )

    result = build_active_corpus_index(
        registry,
        tmp_path / "development-index",
        benchmark_splits={"dev"},
    )

    assert result.pointer.active_version_ids == [dev.version.version_id]
    assert frozen.version.version_id not in result.pointer.active_version_ids


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
