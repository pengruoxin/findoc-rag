"""Create a machine-readable stage summary from fixed scan evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.pdf_evaluation import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/pdf-extraction"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pdf-extraction/pdf-hard-v2-stage7-summary.json"),
    )
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(reports_dir: Path) -> dict:
    paths = {
        "rapidocr_pre_rotation": reports_dir
        / "pdf-hard-v2-genuine-scan-pre-rotation-fix-v1.json",
        "post_rotation": reports_dir
        / "pdf-hard-v2-genuine-scan-post-rotation-fix-v1.json",
        "pre_hierarchy": reports_dir
        / "pdf-hard-v2-genuine-scan-pre-hierarchical-header-v1.json",
        "pre_wrapped": reports_dir
        / "pdf-hard-v2-genuine-scan-pre-wrapped-label-v1.json",
        "current": reports_dir / "pdf-hard-v2-genuine-scan-baseline-v1.json",
        "deepseek": reports_dir
        / "pdf-hard-v2-deepseek-high-risk-fallback-v1.json",
        "adaptive_ocr": reports_dir
        / "pdf-hard-v2-adaptive-red-channel-ocr-v1.json",
        "dpi_240": reports_dir / "pdf-hard-v2-genuine-scan-dpi240-v1.json",
        "dpi_300": reports_dir / "pdf-hard-v2-genuine-scan-dpi300-v1.json",
        "cross_page": reports_dir
        / "pdf-cross-page-v1-post-structure-regression.json",
    }
    loaded = {name: _load(path.resolve(strict=True)) for name, path in paths.items()}
    pdf_hashes = {
        loaded[name]["pdf_sha256"]
        for name in (
            "rapidocr_pre_rotation",
            "post_rotation",
            "pre_hierarchy",
            "pre_wrapped",
            "current",
            "deepseek",
            "adaptive_ocr",
            "dpi_240",
            "dpi_300",
        )
    }
    if len(pdf_hashes) != 1:
        raise ValueError("Scan stage reports do not use the same fixed PDF")
    benchmark_hashes = {
        loaded[name]["benchmark_sha256"]
        for name in (
            "rapidocr_pre_rotation",
            "post_rotation",
            "pre_hierarchy",
            "pre_wrapped",
            "current",
            "deepseek",
            "adaptive_ocr",
            "dpi_240",
            "dpi_300",
        )
    }
    if len(benchmark_hashes) != 1:
        raise ValueError("Scan stage reports do not use the same corrected benchmark")

    rapid = loaded["rapidocr_pre_rotation"]["lanes"]
    post_rotation = loaded["post_rotation"]["lanes"]["hybrid"]["metrics"]
    pre_hierarchy = loaded["pre_hierarchy"]["lanes"]["hybrid"]["metrics"]
    pre_wrapped = loaded["pre_wrapped"]["lanes"]["hybrid"]["metrics"]
    current = loaded["current"]["lanes"]["hybrid"]["metrics"]
    deepseek = loaded["deepseek"]["metrics"]
    adaptive = loaded["adaptive_ocr"]
    cross_page = loaded["cross_page"]["lanes"]["hybrid"]

    stages = [
        {
            "stage": "S0_native_text_only",
            "strict_probe_recall": rapid["native"]["metrics"]["strict_probe_recall"],
            "same_row_association_recall": rapid["native"]["metrics"][
                "same_row_association_recall"
            ],
            "notes": "零文本层扫描页无法从原生 PDF 文本恢复。",
        },
        {
            "stage": "S1_rapidocr_baseline",
            "strict_probe_recall": rapid["hybrid"]["metrics"]["strict_probe_recall"],
            "same_row_association_recall": rapid["hybrid"]["metrics"][
                "same_row_association_recall"
            ],
            "ocr_coordinate_bounds_rate": rapid["hybrid"]["metrics"][
                "ocr_coordinate_bounds_rate"
            ],
            "notes": "数值召回已达100%，旋转页坐标和同行关联失败。",
        },
        {
            "stage": "S2_rotation_coordinate_contract",
            "strict_probe_recall": post_rotation["strict_probe_recall"],
            "same_row_association_recall": post_rotation[
                "same_row_association_recall"
            ],
            "ocr_coordinate_bounds_rate": post_rotation[
                "ocr_coordinate_bounds_rate"
            ],
            "strict_probe_recall_delta": post_rotation["strict_probe_recall"]
            - rapid["hybrid"]["metrics"]["strict_probe_recall"],
            "notes": "页面宽高与元素坐标统一为未旋转空间，结构分析转回显示空间。",
        },
        {
            "stage": "S3_direct_row_column_structure",
            "structured_cell_recall": pre_hierarchy["structured_cell_recall"],
            "notes": "只接受单行表头与直接行列对齐。",
        },
        {
            "stage": "S4_hierarchical_header",
            "structured_cell_recall": pre_wrapped["structured_cell_recall"],
            "structured_cell_recall_delta": pre_wrapped["structured_cell_recall"]
            - pre_hierarchy["structured_cell_recall"],
            "notes": "合并同列垂直表头片段，不跨列拼接。",
        },
        {
            "stage": "S5_wrapped_row_label",
            "structured_cell_recall": current["structured_cell_recall"],
            "structured_cell_recall_delta": current["structured_cell_recall"]
            - pre_wrapped["structured_cell_recall"],
            "notes": "仅合并数值行附近、左侧且属于目标行名的换行片段。",
        },
        {
            "stage": "S6_deepseek_high_risk_only",
            "fallback_page_rate": deepseek["fallback_page_rate"],
            "fallback_cell_rate": deepseek["fallback_cell_rate"],
            "fallback_candidate_value_accuracy": deepseek[
                "fallback_candidate_value_accuracy"
            ],
            "candidate_coverage_if_human_confirms": deepseek[
                "candidate_coverage_if_human_confirms"
            ],
            "unsafe_auto_accept_rate": deepseek["unsafe_auto_accept_rate"],
            "input_tokens": deepseek["total_input_tokens"],
            "output_tokens": deepseek["total_output_tokens"],
            "notes": "只审计本地无法证明的1格；模型证据不足时拒答，不替代行列证据。",
        },
        {
            "stage": "S7_adaptive_red_channel_ocr",
            "structured_cell_recall": adaptive["final_metrics"][
                "structured_cell_recall"
            ],
            "structured_cell_recall_delta": adaptive["deltas"][
                "structured_cell_recall"
            ],
            "retry_page_rate": adaptive["retry_page_rate"],
            "retry_page_numbers": adaptive["retry_page_numbers"],
            "notes": "只对失败页做红色印章抑制和240 DPI重识别，补齐最后1格。",
        },
    ]
    return {
        "schema_version": "1",
        "dataset_id": "pdf-hard-v2-stage7-summary",
        "evaluation_status": "provisional_development_not_formal_gold",
        "fixed_scan_pdf_sha256": next(iter(pdf_hashes)),
        "fixed_benchmark_sha256": next(iter(benchmark_hashes)),
        "probe_count": current["probe_count"],
        "cell_probe_count": deepseek["cell_probe_count"],
        "formal_quota_count": 0,
        "stages": stages,
        "cross_page_regression": {
            "numeric_recall": cross_page["overall"]["mean_numeric_recall"],
            "table_recoverable_rate": cross_page["table_structure"]["overall"][
                "recoverable_rate"
            ],
            "route_accuracy": cross_page["overall"]["route_accuracy"],
        },
        "whole_page_dpi_sweep": {
            str(dpi): {
                "structured_cell_recall": loaded[name]["lanes"]["hybrid"][
                    "metrics"
                ]["structured_cell_recall"],
                "elapsed_ms": loaded[name]["lanes"]["hybrid"]["metrics"][
                    "elapsed_ms"
                ],
            }
            for dpi, name in ((180, "current"), (240, "dpi_240"), (300, "dpi_300"))
        },
        "source_reports": {
            name: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }


def main() -> None:
    args = parse_args()
    report = build_summary(args.reports_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"stages={len(report['stages'])}")
    print(f"probe_count={report['probe_count']}")
    print(f"formal_quota_count={report['formal_quota_count']}")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
