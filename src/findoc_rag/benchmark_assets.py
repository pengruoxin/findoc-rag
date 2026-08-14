"""Committed bindings and evidence for the canonical benchmark.

The full local catalog is intentionally ignored because it is regenerated from
the source PDFs.  Evaluation integrity, however, must also work in a clean
clone, so the benchmark carries a compact, source-derived evidence catalog and
an external lock file.  Neither file changes the benchmark questions or metric
definitions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

LOCK_RELATIVE_PATH = Path("data/evaluation/benchmark-lock-v1.json")
EVIDENCE_RELATIVE_PATH = Path("data/evaluation/benchmark-evidence-v1.jsonl")


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_benchmark_lock(root: Path) -> dict:
    path = root / LOCK_RELATIVE_PATH
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError(f"Unsupported benchmark lock schema: {lock.get('schema_version')!r}")
    return lock


def validate_benchmark_lock(root: Path, lock: dict | None = None) -> None:
    """Fail if a canonical benchmark asset differs from its external lock."""
    lock = lock or load_benchmark_lock(root)
    for relative_path, expected_sha256 in lock["files"].items():
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"Locked benchmark asset is missing: {relative_path}")
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Locked benchmark asset changed: {relative_path} "
                f"({actual_sha256} != {expected_sha256})"
            )


def benchmark_chunk_paths(root: Path) -> list[Path]:
    """Return full local chunks when available plus committed benchmark evidence."""
    paths = sorted((root / "data/catalog/versions").glob("*/chunks.jsonl"))
    evidence_path = root / EVIDENCE_RELATIVE_PATH
    if evidence_path.is_file():
        paths.append(evidence_path)
    return paths
