"""Dataset-level integrity checks for the canonical benchmark (fail closed).

Hard failures must abort evaluation: the benchmark must never silently run on
missing gold chunks, unverifiable quotes, or unanchored relative-time variants.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

EXPECTED_ITEM_COUNT = 48
EXPECTED_VARIANTS_PER_ITEM = 2

KNOWN_TICKERS: dict[str, str] = {
    "600519": "贵州茅台",
    "600887": "伊利股份",
}
KNOWN_COMPANIES = tuple(KNOWN_TICKERS.values())
WHITESPACE = re.compile(r"\s+")
YEAR_PATTERN = re.compile(r"20\d{2}")


@dataclass
class IntegrityResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Canonical whitespace normalization for quote matching (fixed rule)."""
    return WHITESPACE.sub("", text)


def load_chunks(chunk_paths: list[Path]) -> dict[str, dict]:
    chunks: dict[str, dict] = {}
    for path in chunk_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunks[chunk["chunk_id"]] = chunk
    return chunks


def load_corpus_binding(index_root: Path) -> tuple[str | None, int | None]:
    """Return (active_index_id, index_format_version) from current.json."""
    current = index_root / "current.json"
    if not current.is_file():
        return None, None
    payload = json.loads(current.read_text(encoding="utf-8"))
    index_id = payload.get("index_id")
    generation_path = Path(str(payload.get("generation_path", "")))
    manifest_path = index_root / generation_path / "manifest.json"
    format_version = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        format_version = manifest.get("index_format_version")
    return index_id, format_version


def validate_benchmark(
    benchmark: dict,
    *,
    corpus_index_id: str | None,
    chunk_schema_version: int | None,
    chunks: dict[str, dict],
    expected_item_count: int | None = None,
) -> IntegrityResult:
    result = IntegrityResult(ok=True)
    items = benchmark.get("items") or []
    errors = result.errors

    if benchmark.get("item_count") != len(items):
        errors.append(
            f"item_count {benchmark.get('item_count')} != actual items {len(items)}"
        )
    if expected_item_count is not None and len(items) != expected_item_count:
        errors.append(f"expected {expected_item_count} items, got {len(items)}")
    query_ids = [item.get("query_id") for item in items]
    duplicates = sorted(
        query_id for query_id, count in Counter(query_ids).items() if count > 1
    )
    if duplicates:
        errors.append(f"duplicate query_id: {duplicates}")

    if benchmark.get("corpus_index_id") != corpus_index_id:
        errors.append(
            f"corpus_index_id {benchmark.get('corpus_index_id')!r} "
            f"does not match active corpus {corpus_index_id!r}"
        )
    if benchmark.get("chunk_schema_version") != chunk_schema_version:
        errors.append(
            f"chunk_schema_version {benchmark.get('chunk_schema_version')!r} "
            f"does not match index format {chunk_schema_version!r}"
        )

    for item in items:
        for chunk_id in item.get("gold_chunk_ids") or []:
            if chunk_id not in chunks:
                errors.append(
                    f"gold chunk missing from corpus: {chunk_id} ({item.get('query_id')})"
                )
        for evidence in item.get("gold_evidence") or []:
            chunk = chunks.get(evidence.get("chunk_id") or "")
            if chunk is None:
                continue
            quote = evidence.get("verbatim_quote") or ""
            if not quote:
                errors.append(f"empty quote: {evidence.get('evidence_id')}")
                continue
            if normalize_text(quote) not in normalize_text(chunk.get("text") or ""):
                errors.append(
                    f"quote not found in chunk after normalization: "
                    f"{evidence.get('evidence_id')}"
                )

    for item in items:
        variants = item.get("query_variants") or []
        if len(variants) != EXPECTED_VARIANTS_PER_ITEM:
            errors.append(
                f"{item.get('query_id')}: expected {EXPECTED_VARIANTS_PER_ITEM} "
                f"variants, got {len(variants)}"
            )
        company_names = item.get("company_names") or []
        report_years = item.get("report_years") or []
        for variant in variants:
            query = variant.get("query") or ""
            missing = [
                key
                for key in ("variant_id", "query", "variant_types", "query_regime")
                if not variant.get(key)
            ]
            if missing:
                errors.append(f"{item.get('query_id')}: variant missing {missing}")
            variant_types = variant.get("variant_types") or []
            if "relative_time" in variant_types and not variant.get("as_of_date"):
                errors.append(
                    f"{item.get('query_id')}:{variant.get('variant_id')} "
                    "relative_time variant requires as_of_date"
                )
            as_of_date = variant.get("as_of_date")
            if as_of_date:
                try:
                    date.fromisoformat(str(as_of_date))
                except ValueError:
                    errors.append(
                        f"{item.get('query_id')}:{variant.get('variant_id')} "
                        f"invalid as_of_date {as_of_date!r}"
                    )
            allowed_years = set(report_years)
            allowed_years.update(
                year + delta
                for year in report_years
                for delta in (-1, 1)
            )
            for raw_year in YEAR_PATTERN.findall(query):
                if int(raw_year) not in allowed_years:
                    errors.append(
                        f"{item.get('query_id')}:{variant.get('variant_id')} "
                        f"explicit year {raw_year} outside report_years {report_years} "
                        "and adjacent years"
                    )
            for ticker, company in KNOWN_TICKERS.items():
                if ticker in query and company not in company_names:
                    errors.append(
                        f"{item.get('query_id')}:{variant.get('variant_id')} "
                        f"ticker {ticker} does not match item companies {company_names}"
                    )
            for company in KNOWN_COMPANIES:
                if company in query and company not in company_names:
                    errors.append(
                        f"{item.get('query_id')}:{variant.get('variant_id')} "
                        f"company {company} not in item companies {company_names}"
                    )

    result.ok = not errors
    result.warnings.append(
        f"{len(items) * EXPECTED_VARIANTS_PER_ITEM} variants passed structural checks; "
        "semantic fact preservation still requires human review"
    )
    return result
