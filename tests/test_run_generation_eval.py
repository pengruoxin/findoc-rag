"""Trust-boundary tests for the end-to-end generation evaluation runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _load_runner_module():
    path = ROOT / "scripts" / "run_generation_eval.py"
    spec = importlib.util.spec_from_file_location("run_generation_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner_module()


class _RecordingIndex:
    def __init__(self) -> None:
        self.query = None
        self.filters = None

    def search(self, query, **kwargs):
        self.query = query
        self.filters = kwargs["filters"]
        return []


def test_retrieved_lane_defaults_to_query_derived_metadata() -> None:
    item = SimpleNamespace(company_names=["错误公司"], report_years=[1999])
    index = _RecordingIndex()

    _, source = runner.retrieved_hits(
        item,
        "扩写后的检索文本",
        index,
        routing_query="伊利2024年营业收入是多少？",
    )

    assert source == "query_router"
    assert index.query == "扩写后的检索文本"
    assert index.filters.company_names == ["伊利股份"]
    assert index.filters.report_years == [2024]


def test_retrieved_lane_does_not_filter_forecast_target_as_report_year() -> None:
    item = SimpleNamespace(company_names=["错误公司"], report_years=[1999])
    index = _RecordingIndex()

    _, source = runner.retrieved_hits(
        item,
        "伊利股份是否保证2025年实现1190亿元营业收入目标",
        index,
    )

    assert source == "query_router"
    assert index.filters.company_names == ["伊利股份"]
    assert index.filters.report_years == []


def test_retrieved_lane_handles_benchmark_forecast_guarantee_wording() -> None:
    item = SimpleNamespace(company_names=["错误公司"], report_years=[1999])
    index = _RecordingIndex()

    _, source = runner.retrieved_hits(
        item,
        "伊利股份是否保证2025年一定实现1,190亿元营业总收入？",
        index,
    )

    assert source == "query_router"
    assert index.filters.company_names == ["伊利股份"]
    assert index.filters.report_years == []


def test_retrieved_lane_uses_no_filter_when_query_has_no_routing_signal() -> None:
    item = SimpleNamespace(company_names=["gold company"], report_years=[2024])
    index = _RecordingIndex()

    _, source = runner.retrieved_hits(item, "关键审计事项有哪些？", index)

    assert source == "none"
    assert index.filters is None


def test_oracle_metadata_filter_is_explicit_diagnostic_mode() -> None:
    item = SimpleNamespace(company_names=["gold company"], report_years=[2024])
    index = _RecordingIndex()

    _, source = runner.retrieved_hits(
        item,
        "查询没有这些元数据",
        index,
        oracle_metadata=True,
    )

    assert source == "oracle_metadata"
    assert index.filters.company_names == ["gold company"]
    assert index.filters.report_years == [2024]


def test_runner_source_uses_dataset_or_validated_migration_bound_index_gate() -> None:
    source = (ROOT / "scripts/run_generation_eval.py").read_text(encoding="utf-8")
    assert "validate_migration_manifest(" in source
    assert "resolve_evaluation_index_id(" in source
    assert 'args.lane == "retrieved_context" or args.migration_manifest is not None' in source
    assert "index if migration is not None else None" in source


def test_required_remote_run_fails_closed_after_retaining_error_artifacts() -> None:
    source = (ROOT / "scripts/run_generation_eval.py").read_text(encoding="utf-8")
    assert '"remote_configured": api_key_set' in source
    assert '"remote_generation": remote_success_count > 0' in source
    assert "if args.require_remote and run_error_count:" in source
    assert "artifacts were retained for audit" in source
