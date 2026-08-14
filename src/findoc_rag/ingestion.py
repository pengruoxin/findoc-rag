import hashlib
import json
from pathlib import Path

import pymupdf
from pydantic import BaseModel

from findoc_rag.chunking import ChunkingConfig, build_chunking_report, chunk_document
from findoc_rag.documents.pdf import parse_pdf
from findoc_rag.documents.quality import PdfQualityConfig, evaluate_pdf_quality
from findoc_rag.registry import DocumentRegistry, DocumentVersion

DOCUMENT_IR_SCHEMA_VERSION = "2"
PDF_PARSER_NAME = "pymupdf"
OCR_CONFIG = {"mode": "disabled"}
QUALITY_CONFIG = PdfQualityConfig().model_dump(mode="json")


class IngestionResult(BaseModel):
    action: str
    version: DocumentVersion


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build_processing_fingerprint(
    chunking_config: ChunkingConfig,
    *,
    ir_schema_version: str = DOCUMENT_IR_SCHEMA_VERSION,
    parser_name: str = PDF_PARSER_NAME,
    parser_version: str = pymupdf.VersionBind,
    ocr_config: dict | None = None,
    quality_config: dict | None = None,
) -> tuple[str, dict]:
    """Return a canonical fingerprint and its auditable processing components."""
    components = {
        "ir_schema_version": ir_schema_version,
        "parser": {"name": parser_name, "version": parser_version},
        "chunking_config": chunking_config.model_dump(mode="json"),
        "ocr_config": OCR_CONFIG if ocr_config is None else ocr_config,
        "quality_config": QUALITY_CONFIG if quality_config is None else quality_config,
    }
    canonical = json.dumps(
        components,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest(), components


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def ingest_pdf(
    source: Path,
    document_key: str,
    registry: DocumentRegistry,
    storage_directory: Path,
    chunking_config: ChunkingConfig | None = None,
    metadata: dict | None = None,
    pdf_quality_config: PdfQualityConfig | None = None,
) -> IngestionResult:
    resolved_source = source.resolve(strict=True)
    content_hash = file_sha256(resolved_source)
    if metadata is None:
        metadata = next(
            (
                version.metadata
                for version in registry.active_versions()
                if version.document_key == document_key
            ),
            None,
        )
    config = chunking_config or ChunkingConfig()
    quality_config = pdf_quality_config or PdfQualityConfig()
    processing_fingerprint, processing_components = build_processing_fingerprint(
        config,
        quality_config=quality_config.model_dump(mode="json"),
    )
    decision = registry.begin_ingestion(
        document_key,
        content_hash,
        resolved_source,
        metadata=metadata,
        processing_fingerprint=processing_fingerprint,
    )
    if decision.action == "unchanged":
        return IngestionResult(action="unchanged", version=decision.version)
    if decision.action == "reuse":
        version = decision.version
        if not version.document_ir_path or not version.chunks_path or version.chunk_count is None:
            raise ValueError("Reusable version is missing required artifact metadata")
        activated = registry.activate(
            version.version_id,
            Path(version.document_ir_path),
            Path(version.chunks_path),
            version.chunk_count,
        )
        return IngestionResult(action="reused", version=activated)

    version_directory = storage_directory.resolve() / decision.version.version_id
    document_path = version_directory / "document.json"
    chunks_path = version_directory / "chunks.jsonl"
    report_path = version_directory / "chunking-report.json"
    quality_report_path = version_directory / "pdf-quality-report.json"
    manifest_path = version_directory / "ingestion-manifest.json"

    try:
        document = parse_pdf(resolved_source)
        if document.content_sha256 != content_hash:
            raise ValueError("Source PDF changed while ingestion was running")
        quality_report = evaluate_pdf_quality(document, quality_config)
        chunks = chunk_document(document, config)
        if not chunks:
            raise ValueError("Document produced no chunks")
        report = build_chunking_report(document, chunks, config)

        _atomic_write_text(document_path, document.model_dump_json(indent=2) + "\n")
        _atomic_write_text(
            chunks_path,
            "".join(chunk.model_dump_json() + "\n" for chunk in chunks),
        )
        _atomic_write_text(report_path, report.model_dump_json(indent=2) + "\n")
        _atomic_write_text(
            quality_report_path,
            quality_report.model_dump_json(indent=2) + "\n",
        )
        _atomic_write_text(
            manifest_path,
            json.dumps(
                {
                    "version_id": decision.version.version_id,
                    "document_key": decision.version.document_key,
                    "source_path": resolved_source.as_posix(),
                    "content_sha256": content_hash,
                    "content_version_id": decision.version.content_version_id,
                    "processing_fingerprint": processing_fingerprint,
                    "processing_components": processing_components,
                    "document_ir_path": document_path.as_posix(),
                    "chunks_path": chunks_path.as_posix(),
                    "chunking_report_path": report_path.as_posix(),
                    "pdf_quality_report_path": quality_report_path.as_posix(),
                    "chunk_count": len(chunks),
                    "chunking_config": config.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        activated = registry.activate(
            decision.version.version_id,
            document_path,
            chunks_path,
            len(chunks),
        )
        return IngestionResult(action="ingested", version=activated)
    except Exception as exc:
        registry.mark_failed(decision.version.version_id, str(exc))
        raise
