from pathlib import Path

from scripts.summarize_pdf_hard_v2_improvements import build_summary


def test_stage7_summary_preserves_honest_formal_boundary() -> None:
    report = build_summary(Path("reports/pdf-extraction"))

    stages = {stage["stage"]: stage for stage in report["stages"]}
    assert report["formal_quota_count"] == 0
    assert stages["S2_rotation_coordinate_contract"]["strict_probe_recall_delta"] > 0
    assert stages["S4_hierarchical_header"]["structured_cell_recall_delta"] > 0
    assert stages["S5_wrapped_row_label"]["structured_cell_recall_delta"] > 0
    assert stages["S6_deepseek_high_risk_only"]["unsafe_auto_accept_rate"] == 0
    assert stages["S6_deepseek_high_risk_only"]["fallback_candidate_value_accuracy"] == 0
    assert stages["S7_adaptive_red_channel_ocr"]["structured_cell_recall"] == 1.0
    assert stages["S7_adaptive_red_channel_ocr"]["retry_page_rate"] == 0.25
    assert len(report["fixed_benchmark_sha256"]) == 64
    assert (
        report["whole_page_dpi_sweep"]["240"]["structured_cell_recall"]
        == report["whole_page_dpi_sweep"]["180"]["structured_cell_recall"]
    )
    assert report["cross_page_regression"]["numeric_recall"] == 1.0
