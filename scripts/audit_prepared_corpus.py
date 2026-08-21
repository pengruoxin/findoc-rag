"""Verify that an acquired benchmark corpus preserves its sealed split boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.evaluation.governance import BenchmarkCorpusPlan, audit_prepared_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-corpus-plan.json"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-source-manifest.json"),
    )
    parser.add_argument(
        "--version-manifest",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-version-manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/validation/benchmark-v3-prepared-corpus.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = BenchmarkCorpusPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    version_manifest = json.loads(args.version_manifest.read_text(encoding="utf-8"))
    report = audit_prepared_corpus(plan, source_manifest, version_manifest)
    rendered = report.model_dump_json(indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report.ready_for_annotation:
        raise SystemExit("Prepared corpus failed its isolation audit")


if __name__ == "__main__":
    main()
