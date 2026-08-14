"""Guardrails for retrieval hyperparameter tuning."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_sweep_module():
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "run_retrieval_fusion_sweep.py"
        spec = importlib.util.spec_from_file_location("run_retrieval_fusion_sweep", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


sweep = _load_sweep_module()


def test_fusion_sweep_rejects_frozen_test_split() -> None:
    view = {
        "items": [
            {"split": "frozen_test", "retrieval_judgment": "positive_gold"},
        ]
    }

    with pytest.raises(ValueError, match="restricted"):
        sweep.select_tuning_items(view, "frozen_test")


def test_fusion_sweep_selects_only_requested_non_frozen_split() -> None:
    dev = {"query_id": "dev", "split": "dev", "retrieval_judgment": "positive_gold"}
    view = {
        "items": [
            dev,
            {
                "query_id": "calibration",
                "split": "calibration",
                "retrieval_judgment": "positive_gold",
            },
            {
                "query_id": "frozen",
                "split": "frozen_test",
                "retrieval_judgment": "positive_gold",
            },
        ]
    }

    assert sweep.select_tuning_items(view, "dev") == [dev]


def test_retrieval_filter_parser_does_not_treat_forecast_target_as_report_year() -> None:
    resolved, companies, years = sweep.parse_for_filter(
        "伊利股份是否保证2025年实现1190亿元营业收入目标",
        None,
    )

    assert resolved == "伊利股份是否保证2025年实现1190亿元营业收入目标"
    assert companies == ["伊利股份"]
    assert years == []


def test_retrieval_filter_parser_recognizes_short_company_alias() -> None:
    _, companies, years = sweep.parse_for_filter("伊利去年营业收入", "2025-04-30")

    assert companies == ["伊利股份"]
    assert years == [2024]


def test_retrieval_scripts_enforce_corpus_or_validated_migration_binding() -> None:
    for script in ("run_retrieval_variant_eval.py", "run_retrieval_fusion_sweep.py"):
        source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert "validate_migration_manifest(" in source
        assert "resolve_evaluation_index_id(" in source
