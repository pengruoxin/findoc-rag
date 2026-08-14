"""Auditable migration binding for a frozen benchmark and a replacement index.

The canonical benchmark stays immutable.  A migration manifest proves that the
judged evidence has the same semantic core in a new corpus snapshot and binds
evaluation to the exact replacement-index artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from findoc_rag.benchmark_assets import file_sha256

MIGRATION_SCHEMA_VERSION = 1
SEMANTIC_CORE_FIELDS = (
    "chunk_id",
    "document_id",
    "chunk_index",
    "text",
    "section_path",
    "page_start",
    "page_end",
    "character_count",
    "estimated_token_count",
    "is_continuation",
)
TARGET_ARTIFACTS = (
    "manifest.json",
    "dense_embeddings.npy",
    "dense_chunk_ids.json",
    "lexical.sqlite3",
    "structured_tables.jsonl",
    "structured_tables.manifest.json",
)


@dataclass
class MigrationValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semantic_core(chunk: dict) -> dict:
    return {field_name: chunk.get(field_name) for field_name in SEMANTIC_CORE_FIELDS}


def load_jsonl_by_chunk_id(path: Path) -> dict[str, dict]:
    chunks: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        chunk_id = chunk["chunk_id"]
        if chunk_id in chunks:
            raise ValueError(f"Duplicate chunk_id in {path}: {chunk_id}")
        chunks[chunk_id] = chunk
    return chunks


def judged_chunk_usage(view: dict) -> dict[str, dict[str, list[str]]]:
    usage: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"gold_for": [], "hard_negative_for": []}
    )
    for item in view.get("items") or []:
        query_id = item["query_id"]
        for chunk_id in item.get("gold_chunk_ids") or []:
            usage[chunk_id]["gold_for"].append(query_id)
        for negative in item.get("hard_negatives") or []:
            usage[negative["chunk_id"]]["hard_negative_for"].append(query_id)
    return {
        chunk_id: {
            "gold_for": sorted(set(roles["gold_for"])),
            "hard_negative_for": sorted(set(roles["hard_negative_for"])),
        }
        for chunk_id, roles in sorted(usage.items())
    }


def resolve_index_generation(index_root: Path) -> tuple[dict, Path, dict]:
    current_path = index_root / "current.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    generation = index_root / current["generation_path"]
    index_manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    return current, generation, index_manifest


def build_migration_manifest(
    *,
    view_path: Path,
    source_evidence_path: Path,
    target_index_root: Path,
    migration_id: str,
    dense_model_revision: str,
    dense_model_artifact_sha256: str,
) -> dict:
    view = json.loads(view_path.read_text(encoding="utf-8"))
    source = load_jsonl_by_chunk_id(source_evidence_path)
    current, generation, index_manifest = resolve_index_generation(target_index_root)
    snapshot_sha = index_manifest["source_chunk_sha256"]
    snapshot_path = target_index_root / "snapshots" / f"{snapshot_sha}.jsonl"
    target = load_jsonl_by_chunk_id(snapshot_path)
    usage = judged_chunk_usage(view)

    missing_source = sorted(set(usage) - set(source))
    missing_target = sorted(set(usage) - set(target))
    if missing_source or missing_target:
        raise ValueError(
            f"Judged chunks missing; source={missing_source}, target={missing_target}"
        )

    mappings = []
    for chunk_id, roles in usage.items():
        source_chunk = source[chunk_id]
        target_chunk = target[chunk_id]
        source_core = semantic_core(source_chunk)
        target_core = semantic_core(target_chunk)
        mappings.append(
            {
                "chunk_id": chunk_id,
                **roles,
                "source_payload_sha256": canonical_json_sha256(source_chunk),
                "target_payload_sha256": canonical_json_sha256(target_chunk),
                "semantic_core_sha256": canonical_json_sha256(source_core),
                "semantic_core_equal": source_core == target_core,
                "changed_fields": sorted(
                    key
                    for key in set(source_chunk) | set(target_chunk)
                    if source_chunk.get(key) != target_chunk.get(key)
                ),
            }
        )

    artifacts = {}
    for filename in TARGET_ARTIFACTS:
        path = generation / filename
        if path.is_file():
            artifacts[filename] = {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }

    gold_occurrences = sum(
        len(item.get("gold_chunk_ids") or []) for item in view.get("items") or []
    )
    negative_occurrences = sum(
        len(item.get("hard_negatives") or []) for item in view.get("items") or []
    )
    return {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "migration_id": migration_id,
        "policy": {
            "canonical_benchmark_immutable": True,
            "questions_answers_metrics_changed": False,
            "comparison_mode": "paired_by_query_id",
            "promotion_requires_validation": True,
        },
        "benchmark": {
            "dataset_id": view["dataset_id"],
            "view_path": "data/evaluation/benchmark-v2-retrieval-view.json",
            "view_sha256": file_sha256(view_path),
            "source_index_id": view["corpus_index_id"],
            "chunk_schema_version": view["chunk_schema_version"],
        },
        "source_evidence": {
            "path": "data/evaluation/benchmark-evidence-v1.jsonl",
            "sha256": file_sha256(source_evidence_path),
            "chunk_count": len(source),
        },
        "target_index": {
            "index_id": index_manifest["index_id"],
            "index_format_version": index_manifest["index_format_version"],
            "source_snapshot_sha256": snapshot_sha,
            "source_snapshot_size_bytes": snapshot_path.stat().st_size,
            "chunk_count": index_manifest["chunk_count"],
            "dense_model": index_manifest["dense_model"],
            "dense_model_revision": dense_model_revision,
            "dense_model_artifact_sha256": dense_model_artifact_sha256,
            "embedding_dimension": index_manifest["embedding_dimension"],
            "generation_artifacts": artifacts,
            "active_version_ids": current.get("active_version_ids") or [],
        },
        "evidence_contract": {
            "semantic_core_fields": list(SEMANTIC_CORE_FIELDS),
            "mapping_count": len(mappings),
            "gold_occurrence_count": gold_occurrences,
            "unique_gold_chunk_count": sum(bool(x["gold_for"]) for x in mappings),
            "hard_negative_occurrence_count": negative_occurrences,
            "unique_hard_negative_chunk_count": sum(
                bool(x["hard_negative_for"]) for x in mappings
            ),
            "all_semantic_cores_equal": all(x["semantic_core_equal"] for x in mappings),
            "mappings": mappings,
        },
    }


def validate_migration_manifest(
    manifest: dict,
    *,
    view_path: Path,
    source_evidence_path: Path,
    target_index_root: Path,
    dense_model_artifact_path: Path | None = None,
) -> MigrationValidationResult:
    errors: list[str] = []
    if manifest.get("schema_version") != MIGRATION_SCHEMA_VERSION:
        errors.append(f"Unsupported migration schema: {manifest.get('schema_version')!r}")
        return MigrationValidationResult(ok=False, errors=errors)

    view = json.loads(view_path.read_text(encoding="utf-8"))
    benchmark = manifest.get("benchmark") or {}
    if file_sha256(view_path) != benchmark.get("view_sha256"):
        errors.append("Canonical retrieval view SHA does not match migration manifest")
    if view.get("corpus_index_id") != benchmark.get("source_index_id"):
        errors.append("Canonical retrieval view source index does not match migration manifest")
    policy = manifest.get("policy") or {}
    if policy.get("canonical_benchmark_immutable") is not True:
        errors.append("Migration policy must keep the canonical benchmark immutable")
    if policy.get("questions_answers_metrics_changed") is not False:
        errors.append("Migration must not change questions, answers, or metrics")

    evidence_info = manifest.get("source_evidence") or {}
    if file_sha256(source_evidence_path) != evidence_info.get("sha256"):
        errors.append("Source evidence SHA does not match migration manifest")

    try:
        current, generation, index_manifest = resolve_index_generation(target_index_root)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"Target index cannot be resolved: {exc}")
        return MigrationValidationResult(ok=False, errors=errors)

    target_info = manifest.get("target_index") or {}
    if current.get("index_id") != target_info.get("index_id"):
        errors.append("Active target index ID does not match migration manifest")
    for field_name in (
        "index_id",
        "index_format_version",
        "source_snapshot_sha256",
        "chunk_count",
        "dense_model",
        "embedding_dimension",
    ):
        index_field = "source_chunk_sha256" if field_name == "source_snapshot_sha256" else field_name
        if index_manifest.get(index_field) != target_info.get(field_name):
            errors.append(f"Target index manifest field mismatch: {field_name}")

    snapshot_sha = target_info.get("source_snapshot_sha256")
    snapshot_path = target_index_root / "snapshots" / f"{snapshot_sha}.jsonl"
    if not snapshot_path.is_file() or file_sha256(snapshot_path) != snapshot_sha:
        errors.append("Target source snapshot is missing or its SHA is invalid")

    for filename, expected in (target_info.get("generation_artifacts") or {}).items():
        path = generation / filename
        if not path.is_file():
            errors.append(f"Target generation artifact missing: {filename}")
            continue
        if file_sha256(path) != expected.get("sha256"):
            errors.append(f"Target generation artifact SHA mismatch: {filename}")
        if path.stat().st_size != expected.get("size_bytes"):
            errors.append(f"Target generation artifact size mismatch: {filename}")

    if (
        dense_model_artifact_path is not None
        and file_sha256(dense_model_artifact_path)
        != target_info.get("dense_model_artifact_sha256")
    ):
        errors.append("Dense model artifact SHA does not match migration manifest")

    if errors or not snapshot_path.is_file():
        return MigrationValidationResult(ok=False, errors=errors)

    source = load_jsonl_by_chunk_id(source_evidence_path)
    target = load_jsonl_by_chunk_id(snapshot_path)
    usage = judged_chunk_usage(view)
    contract = manifest.get("evidence_contract") or {}
    mappings = contract.get("mappings") or []
    if contract.get("semantic_core_fields") != list(SEMANTIC_CORE_FIELDS):
        errors.append("Semantic-core field contract changed")
    if contract.get("mapping_count") != len(mappings) or len(mappings) != len(usage):
        errors.append("Evidence mapping count is inconsistent")
    mapping_by_id = {mapping.get("chunk_id"): mapping for mapping in mappings}
    if len(mapping_by_id) != len(mappings):
        errors.append("Evidence mappings contain duplicate chunk IDs")

    for chunk_id, roles in usage.items():
        mapping = mapping_by_id.get(chunk_id)
        if mapping is None:
            errors.append(f"Judged chunk has no migration mapping: {chunk_id}")
            continue
        if chunk_id not in source or chunk_id not in target:
            errors.append(f"Mapped chunk missing from source or target: {chunk_id}")
            continue
        source_chunk = source[chunk_id]
        target_chunk = target[chunk_id]
        source_core = semantic_core(source_chunk)
        target_core = semantic_core(target_chunk)
        expected = {
            "source_payload_sha256": canonical_json_sha256(source_chunk),
            "target_payload_sha256": canonical_json_sha256(target_chunk),
            "semantic_core_sha256": canonical_json_sha256(source_core),
        }
        for key, value in expected.items():
            if mapping.get(key) != value:
                errors.append(f"{chunk_id}: mapping {key} mismatch")
        if mapping.get("gold_for") != roles["gold_for"]:
            errors.append(f"{chunk_id}: gold usage mismatch")
        if mapping.get("hard_negative_for") != roles["hard_negative_for"]:
            errors.append(f"{chunk_id}: hard-negative usage mismatch")
        if source_core != target_core or mapping.get("semantic_core_equal") is not True:
            errors.append(f"{chunk_id}: semantic core changed")
        changed_fields = sorted(
            key
            for key in set(source_chunk) | set(target_chunk)
            if source_chunk.get(key) != target_chunk.get(key)
        )
        if mapping.get("changed_fields") != changed_fields:
            errors.append(f"{chunk_id}: changed_fields mismatch")

    if contract.get("all_semantic_cores_equal") is not True:
        errors.append("Manifest does not assert semantic-core equality")
    return MigrationValidationResult(ok=not errors, errors=errors)


def resolve_evaluation_index_id(
    *,
    view: dict,
    index_id: str,
    migration_manifest: dict | None,
) -> str:
    """Enforce either the canonical binding or an explicitly validated migration."""
    if migration_manifest is None:
        if index_id != view["corpus_index_id"]:
            raise ValueError(
                "Retrieval evaluation requires the dataset-bound index: "
                f"{index_id!r} != {view['corpus_index_id']!r}"
            )
        return index_id
    benchmark = migration_manifest.get("benchmark") or {}
    target = migration_manifest.get("target_index") or {}
    if benchmark.get("source_index_id") != view.get("corpus_index_id"):
        raise ValueError("Migration manifest is not bound to this benchmark source index")
    if target.get("index_id") != index_id:
        raise ValueError("Migration manifest is not bound to the active target index")
    return index_id
