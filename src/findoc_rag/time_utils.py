"""Deterministic relative-time resolution with separate query/document clocks.

User-deictic expressions such as ``去年`` are anchored to ``as_of_date``. When
the expression follows an explicit annual-report reference in the same clause,
it is instead anchored to that report year. This prevents a 2022 annual report's
``去年`` from silently becoming 2025 merely because the query runs in 2026.
"""

from __future__ import annotations

import re
from datetime import date

RELATIVE_TIME_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"去年|上一年|上年"), -1),
    (re.compile(r"前年"), -2),
    (re.compile(r"今年|本年|本年度"), 0),
)
DOCUMENT_TIME_ANCHOR_PATTERN = re.compile(
    r"(?P<year>20\d{2})\s*年?\s*(?:年报|年度报告|报告期|报告年度)"
)
CLAUSE_BOUNDARY_PATTERN = re.compile(r"[。！？!?；;]")
MAX_DOCUMENT_ANCHOR_DISTANCE = 64


def _preceding_document_year(query: str, cue_start: int) -> int | None:
    """Return a nearby preceding report year for a document-internal cue."""

    anchors = list(DOCUMENT_TIME_ANCHOR_PATTERN.finditer(query, 0, cue_start))
    if not anchors:
        return None
    anchor = anchors[-1]
    between = query[anchor.end() : cue_start]
    if (
        len(between) > MAX_DOCUMENT_ANCHOR_DISTANCE
        or CLAUSE_BOUNDARY_PATTERN.search(between)
    ):
        return None
    return int(anchor.group("year"))


def _relative_time_replacer(
    query: str,
    delta: int,
    as_of_date: date | None,
):
    def replace(match: re.Match[str]) -> str:
        document_year = _preceding_document_year(query, match.start())
        if document_year is not None:
            return f"{document_year + delta}年"
        if as_of_date is None:
            raise ValueError(
                "Relative-time query requires an as_of_date anchor; "
                "evaluation must never use the system clock"
            )
        return f"{as_of_date.year + delta}年"

    return replace


def resolve_relative_time(
    query: str,
    as_of_date: date | None,
) -> tuple[str, list[str]]:
    """Replace relative time expressions with explicit years.

    Returns ``(resolved_query, matched_cues)``. A nearby explicit annual-report
    year takes precedence for cues that follow it. Raises ``ValueError`` only
    when a user-deictic cue remains and no ``as_of_date`` is provided.
    """
    resolved = query
    cues: list[str] = []
    for pattern, delta in RELATIVE_TIME_PATTERNS:
        if not pattern.search(resolved):
            continue
        cues.append(pattern.pattern)

        resolved = pattern.sub(
            _relative_time_replacer(resolved, delta, as_of_date), resolved
        )
    return resolved, cues


def parse_as_of_date(value: str | None) -> date | None:
    """Parse an ISO date string from benchmark variant metadata."""
    if not value:
        return None
    return date.fromisoformat(value)
