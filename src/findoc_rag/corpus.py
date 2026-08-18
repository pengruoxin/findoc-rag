import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from findoc_rag.documents.models import DocumentChunk, ParsedDocument
from findoc_rag.indexing import PersistentIndex
from findoc_rag.io import read_jsonl, write_text_lf
from findoc_rag.registry import DocumentRegistry
from findoc_rag.structured_tables import (
    STRUCTURED_TABLE_GENERATOR,
    STRUCTURED_TABLE_SCHEMA_VERSION,
    build_structured_tables,
)


class CurrentIndexPointer(BaseModel):
    index_id: str
    generation_path: str
    activated_at: datetime
    active_version_ids: list[str]


class CorpusIndexResult(BaseModel):
    action: str
    pointer: CurrentIndexPointer
    manifest: dict


def collect_active_chunks(registry: DocumentRegistry) -> tuple[list[DocumentChunk], list[str]]:
    chunks: list[DocumentChunk] = []
    version_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    for version in registry.active_versions():
        if not version.chunks_path:
            raise ValueError(f"Active version has no chunk artifact: {version.version_id}")
        path = Path(version.chunks_path)
        if not path.is_file():
            raise FileNotFoundError(f"Active chunk artifact is missing: {path}")
        version_chunks = read_jsonl(path, DocumentChunk)
        key_parts = version.document_key.split(":")
        inferred_type = key_parts[-2] if len(key_parts) >= 2 else None
        inferred_year = int(key_parts[-1]) if key_parts[-1].isdigit() else None
        version_chunks = [
            chunk.model_copy(
                update={
                    "document_key": version.document_key,
                    "company_name": version.metadata.get("company_name"),
                    "report_year": version.metadata.get("report_year", inferred_year),
                    "document_type": version.metadata.get("document_type", inferred_type),
                }
            )
            for chunk in version_chunks
        ]
        for chunk in version_chunks:
            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate active chunk ID: {chunk.chunk_id}")
            seen_chunk_ids.add(chunk.chunk_id)
        chunks.extend(version_chunks)
        version_ids.append(version.version_id)
    if not chunks:
        raise ValueError("The active corpus contains no chunks")
    return chunks, version_ids


def collect_active_documents(
    registry: DocumentRegistry,
) -> dict[str, ParsedDocument]:
    """Load active persisted IR for sidecar construction, keyed by document ID."""
    documents: dict[str, ParsedDocument] = {}
    for version in registry.active_versions():
        if not version.document_ir_path:
            raise ValueError(f"Active version has no document IR: {version.version_id}")
        path = Path(version.document_ir_path)
        if not path.is_file():
            raise FileNotFoundError(f"Active document IR is missing: {path}")
        document = ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        existing = documents.get(document.document_id)
        if existing is not None and existing.content_sha256 != document.content_sha256:
            raise ValueError(f"Conflicting active document IR: {document.document_id}")
        documents[document.document_id] = document
    return documents


def _write_snapshot(chunks: list[DocumentChunk], snapshots_directory: Path) -> Path:
    content = "".join(chunk.model_dump_json() + "\n" for chunk in chunks)
    digest = hashlib.sha256(content.encode()).hexdigest()
    path = snapshots_directory / f"{digest}.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".jsonl.part")
        write_text_lf(temporary, content)
        temporary.replace(path)
    return path


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _load_pointer(index_root: Path) -> CurrentIndexPointer | None:
    path = index_root / "current.json"
    return CurrentIndexPointer.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None


def resolve_current_index(index_root: Path) -> PersistentIndex:
    root = index_root.resolve()
    pointer = _load_pointer(root)
    if pointer is None:
        raise FileNotFoundError(f"No active index pointer exists under {root}")
    generation_path = (root / pointer.generation_path).resolve()
    if not generation_path.is_relative_to(root):
        raise ValueError("Active index pointer escapes the configured index root")
    index = PersistentIndex(generation_path)
    if pointer.index_id != index.manifest.index_id:
        raise ValueError(
            "Active index pointer ID does not match the referenced index manifest: "
            f"{pointer.index_id!r} != {index.manifest.index_id!r}"
        )
    return index


def build_active_corpus_index(
    registry: DocumentRegistry,
    index_root: Path,
    dense_model: str | None = None,
) -> CorpusIndexResult:
    root = index_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    chunks, version_ids = collect_active_chunks(registry)
    documents = collect_active_documents(registry)
    structured_tables = build_structured_tables(chunks, documents)
    snapshot_path = _write_snapshot(chunks, root / "snapshots")
    source_digest = _file_sha256(snapshot_path)
    current_pointer = _load_pointer(root)
    previous_index = None
    if current_pointer is not None:
        try:
            previous_index = PersistentIndex(root / current_pointer.generation_path)
        except (FileNotFoundError, ValueError):
            previous_index = None
        if (
            previous_index is not None
            and previous_index.manifest.source_chunk_sha256 == source_digest
            and previous_index.manifest.dense_model == dense_model
            and previous_index.manifest.structured_table_schema_version
            == STRUCTURED_TABLE_SCHEMA_VERSION
            and previous_index.manifest.structured_table_generator
            == STRUCTURED_TABLE_GENERATOR
        ):
            return CorpusIndexResult(
                action="unchanged",
                pointer=current_pointer,
                manifest=previous_index.manifest.model_dump(mode="json"),
            )

    generation_name = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    )
    generation_path = root / "generations" / generation_name
    index = PersistentIndex.build(
        generation_path,
        chunks,
        source_chunk_path=snapshot_path,
        dense_model=dense_model,
        reuse_dense_from=previous_index,
        structured_tables=structured_tables,
    )
    pointer = CurrentIndexPointer(
        index_id=index.manifest.index_id,
        generation_path=generation_path.relative_to(root).as_posix(),
        activated_at=datetime.now(UTC),
        active_version_ids=version_ids,
    )
    pointer_path = root / "current.json"
    temporary_pointer = root / "current.json.part"
    write_text_lf(temporary_pointer, pointer.model_dump_json(indent=2) + "\n")
    temporary_pointer.replace(pointer_path)
    return CorpusIndexResult(
        action="built",
        pointer=pointer,
        manifest=index.manifest.model_dump(mode="json"),
    )
