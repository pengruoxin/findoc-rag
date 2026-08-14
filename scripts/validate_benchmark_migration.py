#!/usr/bin/env python3
"""Validate a benchmark migration against the immutable source and target index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.benchmark_migration import validate_migration_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--view",
        type=Path,
        default=ROOT / "data/evaluation/benchmark-v2-retrieval-view.json",
    )
    parser.add_argument(
        "--source-evidence",
        type=Path,
        default=ROOT / "data/evaluation/benchmark-evidence-v1.jsonl",
    )
    parser.add_argument("--target-index-root", type=Path, required=True)
    parser.add_argument("--dense-model-artifact", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = validate_migration_manifest(
        manifest,
        view_path=args.view,
        source_evidence_path=args.source_evidence,
        target_index_root=args.target_index_root,
        dense_model_artifact_path=args.dense_model_artifact,
    )
    if not result.ok:
        print("INVALID")
        for error in result.errors:
            print(f"- {error}")
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
