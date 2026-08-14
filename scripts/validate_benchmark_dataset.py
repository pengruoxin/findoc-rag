#!/usr/bin/env python3
"""CLI for the benchmark-v2 dataset-level integrity gate (fail closed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.benchmark_assets import (
    benchmark_chunk_paths,
    load_benchmark_lock,
    validate_benchmark_lock,
)
from findoc_rag.benchmark_integrity import (
    EXPECTED_ITEM_COUNT,
    load_chunks,
    load_corpus_binding,
    validate_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data/evaluation/benchmark-v2.json"
DEFAULT_INDEX_ROOT = ROOT / "data/indexes/corpus"
DEFAULT_CHUNK_DIR = ROOT / "data/catalog/versions"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument(
        "--chunks",
        type=Path,
        nargs="*",
        default=benchmark_chunk_paths(ROOT),
        help="chunks.jsonl paths; defaults to local catalog plus committed evidence catalog",
    )
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    canonical = args.benchmark.resolve() == DEFAULT_BENCHMARK.resolve()
    lock = load_benchmark_lock(ROOT) if canonical else None
    if canonical:
        validate_benchmark_lock(ROOT, lock)
    index_id, format_version = load_corpus_binding(args.index_root)
    if canonical and index_id is None:
        index_id = lock["corpus_index_id"]
        format_version = lock["chunk_schema_version"]
    chunks = load_chunks(args.chunks)
    result = validate_benchmark(
        benchmark,
        corpus_index_id=index_id,
        chunk_schema_version=format_version,
        chunks=chunks,
        expected_item_count=EXPECTED_ITEM_COUNT,
    )

    for warning in result.warnings:
        print(f"WARNING: {warning}")
    if not result.ok:
        print("INVALID")
        for error in result.errors:
            print(f"- {error}")
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
