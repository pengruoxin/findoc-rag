"""Tests for explicit, fail-closed benchmark migration contracts."""

from __future__ import annotations

import json
from pathlib import Path

from findoc_rag.benchmark_migration import (
    build_migration_manifest,
    canonical_json_sha256,
    resolve_evaluation_index_id,
    validate_migration_manifest,
)
from findoc_rag.io import write_text_lf


def _write_json(path: Path, payload: dict) -> None:
    write_text_lf(path, json.dumps(payload) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    write_text_lf(path, "".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_chunk = {
        "chunk_id": "c1",
        "document_id": "d1",
        "chunk_index": 0,
        "text": "营业收入 100 元",
        "section_path": ["主要财务数据"],
        "page_start": 1,
        "page_end": 1,
        "character_count": 10,
        "estimated_token_count": 8,
        "is_continuation": False,
        "element_references": [{"element_id": "old"}],
    }
    target_chunk = {
        **source_chunk,
        "element_references": [{"element_id": "new"}],
        "company_name": "测试公司",
    }
    view = {
        "dataset_id": "benchmark-view",
        "corpus_index_id": "old-index",
        "chunk_schema_version": 3,
        "items": [
            {
                "query_id": "q1",
                "gold_chunk_ids": ["c1"],
                "hard_negatives": [],
            }
        ],
    }
    view_path = tmp_path / "view.json"
    evidence_path = tmp_path / "evidence.jsonl"
    index_root = tmp_path / "index"
    snapshot_payload = json.dumps(target_chunk) + "\n"
    import hashlib

    snapshot_sha = hashlib.sha256(snapshot_payload.encode()).hexdigest()
    snapshot_path = index_root / "snapshots" / f"{snapshot_sha}.jsonl"
    generation = index_root / "generations/g1"
    _write_json(view_path, view)
    _write_jsonl(evidence_path, [source_chunk])
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(snapshot_path, snapshot_payload)
    index_manifest = {
        "index_id": "new-index",
        "index_format_version": 3,
        "source_chunk_sha256": snapshot_sha,
        "chunk_count": 1,
        "dense_model": "test/e5",
        "embedding_dimension": 3,
    }
    _write_json(generation / "manifest.json", index_manifest)
    write_text_lf(generation / "dense_chunk_ids.json", '["c1"]\n')
    _write_json(
        index_root / "current.json",
        {
            "index_id": "new-index",
            "generation_path": "generations/g1",
            "active_version_ids": ["v1"],
        },
    )
    return view_path, evidence_path, index_root


def test_valid_migration_allows_metadata_only_payload_change(tmp_path: Path) -> None:
    view_path, evidence_path, index_root = _fixture(tmp_path)
    manifest = build_migration_manifest(
        view_path=view_path,
        source_evidence_path=evidence_path,
        target_index_root=index_root,
        migration_id="migration-v1",
        dense_model_revision="revision-1",
        dense_model_artifact_sha256="a" * 64,
    )

    result = validate_migration_manifest(
        manifest,
        view_path=view_path,
        source_evidence_path=evidence_path,
        target_index_root=index_root,
    )

    assert result.ok, result.errors
    mapping = manifest["evidence_contract"]["mappings"][0]
    assert mapping["semantic_core_equal"] is True
    assert mapping["changed_fields"] == ["company_name", "element_references"]


def test_migration_fails_when_judged_text_changes(tmp_path: Path) -> None:
    view_path, evidence_path, index_root = _fixture(tmp_path)
    manifest = build_migration_manifest(
        view_path=view_path,
        source_evidence_path=evidence_path,
        target_index_root=index_root,
        migration_id="migration-v1",
        dense_model_revision="revision-1",
        dense_model_artifact_sha256="a" * 64,
    )
    snapshot_sha = manifest["target_index"]["source_snapshot_sha256"]
    snapshot_path = index_root / "snapshots" / f"{snapshot_sha}.jsonl"
    target = json.loads(snapshot_path.read_text(encoding="utf-8"))
    target["text"] = "被篡改"
    write_text_lf(snapshot_path, json.dumps(target) + "\n")

    result = validate_migration_manifest(
        manifest,
        view_path=view_path,
        source_evidence_path=evidence_path,
        target_index_root=index_root,
    )

    assert not result.ok
    assert any("snapshot" in error.lower() for error in result.errors)


def test_binding_rejects_unrelated_target() -> None:
    view = {"corpus_index_id": "old-index"}
    migration = {
        "benchmark": {"source_index_id": "old-index"},
        "target_index": {"index_id": "new-index"},
    }
    assert resolve_evaluation_index_id(
        view=view,
        index_id="new-index",
        migration_manifest=migration,
    ) == "new-index"
    try:
        resolve_evaluation_index_id(
            view=view,
            index_id="other-index",
            migration_manifest=migration,
        )
    except ValueError as exc:
        assert "active target" in str(exc)
    else:
        raise AssertionError("unrelated index was accepted")


def test_canonical_json_hash_ignores_key_order() -> None:
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256(
        {"b": 2, "a": 1}
    )
