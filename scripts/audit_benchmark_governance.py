"""Audit whether a benchmark is ready to support external quality claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.evaluation.governance import GovernancePolicy, audit_benchmark_governance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/evaluation/benchmark-v2.json")
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("configs/evaluation-governance-p0.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-external-claims-ready", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = json.loads(args.dataset.read_text(encoding="utf-8"))
    policy = GovernancePolicy.model_validate_json(args.policy.read_text(encoding="utf-8"))
    report = audit_benchmark_governance(benchmark, policy)
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.require_external_claims_ready and not report.ready_for_external_claims:
        raise SystemExit("Benchmark is not ready for external quality claims")


if __name__ == "__main__":
    main()
