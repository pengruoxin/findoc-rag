"""Guards against platform newline translation breaking hash-bound artifacts.

Several integrity checks hash a string in memory and later re-hash the same
payload read back from disk. ``Path.write_text`` defaults to ``newline=None``,
which rewrites ``\\n`` as ``\\r\\n`` on Windows, so those two digests can never
agree and every fail-closed check turns into a false positive. The bug is
invisible on Linux, which is what CI used to run exclusively.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from findoc_rag.benchmark_assets import validate_benchmark_lock
from findoc_rag.io import write_dict_jsonl, write_json, write_text_lf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Modules that persist a payload whose SHA-256 is compared against the file on
# disk. These must never call Path.write_text directly.
HASH_BOUND_MODULES = (
    "src/findoc_rag/indexing.py",
    "src/findoc_rag/corpus.py",
    "src/findoc_rag/ingestion.py",
    "src/findoc_rag/upload_jobs.py",
    "src/findoc_rag/query_rewriting.py",
)


def test_write_text_lf_keeps_newlines_verbatim(tmp_path: Path) -> None:
    content = '{"a":1}\n{"b":2}\n'
    path = tmp_path / "payload.jsonl"
    write_text_lf(path, content)

    assert path.read_bytes() == content.encode("utf-8")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        hashlib.sha256(content.encode("utf-8")).hexdigest()
    )


def test_write_json_and_jsonl_stay_lf(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "a.json"
    jsonl_path = tmp_path / "nested" / "b.jsonl"
    write_json({"营业收入": 100}, json_path)
    write_dict_jsonl([{"a": 1}, {"b": 2}], jsonl_path)

    for path in (json_path, jsonl_path):
        assert b"\r\n" not in path.read_bytes(), path


@pytest.mark.parametrize("relative_path", HASH_BOUND_MODULES)
def test_hash_bound_modules_do_not_use_raw_write_text(relative_path: str) -> None:
    source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    offenders = re.findall(r"\.write_text\(", source)
    assert not offenders, (
        f"{relative_path} calls Path.write_text directly; use write_text_lf so the "
        "in-memory digest matches the bytes on disk on every platform"
    )


def test_committed_benchmark_assets_match_their_lock() -> None:
    """A clean clone must verify the benchmark lock on any platform.

    Without an ``.gitattributes`` eol=lf pin, ``core.autocrlf=true`` checks these
    files out as CRLF and every locked digest mismatches.
    """
    validate_benchmark_lock(REPOSITORY_ROOT)


def test_gitattributes_pins_hashed_payloads_to_lf() -> None:
    attributes = (REPOSITORY_ROOT / ".gitattributes").read_text(encoding="utf-8")
    rules = {
        parts[0]: parts[1:]
        for line in attributes.splitlines()
        if (parts := line.split()) and not line.startswith("#")
    }
    for pattern in ("*.json", "*.jsonl"):
        assert "eol=lf" in rules.get(pattern, []), pattern
