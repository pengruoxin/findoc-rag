#!/usr/bin/env python3
"""Build a benchmark-to-index migration manifest without changing the benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.benchmark_migration import build_migration_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--migration-id", required=True)
    parser.add_argument("--dense-model-revision", required=True)
    parser.add_argument("--dense-model-artifact-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite migration manifest: {args.output}")
    manifest = build_migration_manifest(
        view_path=args.view,
        source_evidence_path=args.source_evidence,
        target_index_root=args.target_index_root,
        migration_id=args.migration_id,
        dense_model_revision=args.dense_model_revision,
        dense_model_artifact_sha256=args.dense_model_artifact_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
