from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.evaluation.governance import (
    BenchmarkCorpusPlan,
    GovernancePolicy,
    audit_corpus_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = BenchmarkCorpusPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    policy_payload = json.loads(args.policy.read_text(encoding="utf-8"))
    policy_payload["split_strategy"] = plan.split_strategy
    policy = GovernancePolicy.model_validate(policy_payload)
    report = audit_corpus_plan(plan, policy)
    rendered = report.model_dump_json(indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not report.ready_for_acquisition:
        raise SystemExit("Corpus plan is not ready for acquisition")


if __name__ == "__main__":
    main()
