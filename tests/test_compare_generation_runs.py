"""Controlled-variable guardrails for generation-run comparisons."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_compare_module():
    path = ROOT / "scripts" / "compare_generation_runs.py"
    spec = importlib.util.spec_from_file_location("compare_generation_runs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = _load_compare_module()


def _write_run(
    directory: Path,
    run_id: str,
    *,
    code_revision: str | None = "abc123",
    remote: bool = True,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset_id": "benchmark-v2",
                "lane": "retrieved_context",
                "remote_generation": remote,
                "code_revision": code_revision,
                "strict_success_rate": 0.5,
                "expected_behavior_accuracy": 0.8,
                "run_error_rate": 0.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (directory / "deterministic-scores.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "query_id": "q1",
                    "expected_behavior_correct": True,
                    "strict_success": True,
                    "strict_success_eligible": True,
                }
            )
            for _ in range(1)
        )
        + "\n",
        encoding="utf-8",
    )


def test_same_revision_comparison_is_allowed(tmp_path: Path) -> None:
    baseline = tmp_path / "base"
    candidate = tmp_path / "cand"
    _write_run(baseline, "base-run")
    _write_run(candidate, "cand-run")
    report = compare.compare_runs(baseline, candidate)
    assert report["code_revision_match"] is True
    assert report["controlled_change"] is None


def test_different_revision_requires_declared_change(tmp_path: Path) -> None:
    baseline = tmp_path / "base"
    candidate = tmp_path / "cand"
    _write_run(baseline, "base-run", code_revision="abc123")
    _write_run(candidate, "cand-run", code_revision="def456")
    try:
        compare.compare_runs(baseline, candidate)
    except ValueError as exc:
        assert "controlled comparison" in str(exc)
    else:
        raise AssertionError("Expected ValueError for undeclared code drift")

    report = compare.compare_runs(
        baseline, candidate, change="single variable: abstention detection"
    )
    assert report["code_revision_match"] is False
    assert report["controlled_change"] == "single variable: abstention detection"


def test_remote_flag_mismatch_is_rejected(tmp_path: Path) -> None:
    baseline = tmp_path / "base"
    candidate = tmp_path / "cand"
    _write_run(baseline, "base-run", remote=True)
    _write_run(candidate, "cand-run", remote=False)
    try:
        compare.compare_runs(baseline, candidate, change="any change")
    except ValueError as exc:
        assert "remote_generation" in str(exc)
    else:
        raise AssertionError("Expected ValueError for remote flag mismatch")
