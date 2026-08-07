"""Deterministic relative-time resolution anchored to an explicit as_of_date.

The benchmark freezes ``as_of_date`` per relative-time query variant so that
expressions like "去年" have exactly one target year regardless of when the
evaluation is executed. Production code must pass an explicit anchor instead
of relying on the system clock; ``datetime.now()`` is forbidden in evaluation
paths.
"""

from __future__ import annotations

import re
from datetime import date

RELATIVE_TIME_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"去年|上一年|上年"), -1),
    (re.compile(r"前年"), -2),
    (re.compile(r"今年|本年|本年度"), 0),
)


def resolve_relative_time(
    query: str,
    as_of_date: date | None,
) -> tuple[str, list[str]]:
    """Replace relative time expressions with explicit years.

    Returns ``(resolved_query, matched_cues)``. Raises ``ValueError`` when the
    query contains relative time expressions but no anchor is provided, because
    the resolution would otherwise depend on the runtime clock.
    """
    if as_of_date is None:
        if any(pattern.search(query) for pattern, _ in RELATIVE_TIME_PATTERNS):
            raise ValueError(
                "Relative-time query requires an as_of_date anchor; "
                "evaluation must never use the system clock"
            )
        return query, []

    resolved = query
    cues: list[str] = []
    for pattern, delta in RELATIVE_TIME_PATTERNS:
        year = as_of_date.year + delta
        if pattern.search(resolved):
            cues.append(pattern.pattern)
            resolved = pattern.sub(f"{year}年", resolved)
    return resolved, cues


def parse_as_of_date(value: str | None) -> date | None:
    """Parse an ISO date string from benchmark variant metadata."""
    if not value:
        return None
    return date.fromisoformat(value)
