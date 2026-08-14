"""Coverage accounting for external RAGAS judge outputs."""

from __future__ import annotations


def summarize_metric_coverage(
    rows: list[dict], metric_names: list[str]
) -> tuple[dict, dict]:
    """Return observed means plus explicit missing-judge coverage."""
    row_count = len(rows)
    metrics: dict[str, float | None] = {}
    coverage: dict[str, dict[str, float | int]] = {}
    for name in metric_names:
        values = [row.get(name) for row in rows if row.get(name) is not None]
        valid_count = len(values)
        metrics[name] = float(sum(values) / valid_count) if valid_count else None
        coverage[name] = {
            "eligible_count": row_count,
            "valid_count": valid_count,
            "failure_count": row_count - valid_count,
            "coverage": valid_count / row_count if row_count else 0.0,
        }
    return metrics, coverage


def count_complete_metric_rows(rows: list[dict], metric_names: list[str]) -> int:
    """Count rows where every requested judge metric produced a value."""
    return sum(
        all(row.get(name) is not None for name in metric_names) for row in rows
    )
