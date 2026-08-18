#!/usr/bin/env python
"""Build a queryable demo index from the committed benchmark evidence chunks.

The real corpus (full annual-report PDFs, parsed IR, dense embeddings) is
gitignored, so a clean clone has nothing to serve. The 38 minimal source
evidence chunks that back the frozen benchmark *are* committed and SHA-locked,
which is enough to bring up a lexical index and let a newcomer run real
queries against real filing text.

This is a demo lane, not an evaluation lane: 38 chunks cannot reproduce any
published metric. Reported baselines require rebuilding the full corpus index
per README.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from findoc_rag.benchmark_assets import validate_benchmark_lock
from findoc_rag.corpus import CurrentIndexPointer
from findoc_rag.documents.models import DocumentChunk
from findoc_rag.indexing import PersistentIndex
from findoc_rag.io import read_jsonl, write_text_lf
from findoc_rag.structured_tables import build_structured_tables

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = Path("data/evaluation/benchmark-evidence-v1.jsonl")
BENCHMARK_PATH = Path("data/evaluation/benchmark-v2.json")


def derive_document_metadata(root: Path) -> dict[str, dict[str, object]]:
    """Recover company/year per document from the committed benchmark.

    The evidence file carries chunk text only -- no company_name or report_year --
    and ``/v1/query`` routes on exactly those fields, so without them every query
    filters its own candidates away and the service can only abstain. The mapping
    is derivable from committed data: a chunk_id is prefixed with its document's
    short hash, and single-company benchmark items name the company that each
    gold chunk belongs to. Multi-company items are skipped because they would
    attribute one chunk to both filings.
    """
    benchmark = json.loads((root / BENCHMARK_PATH).read_text(encoding="utf-8"))
    votes: dict[str, Counter] = defaultdict(Counter)
    for item in benchmark["items"]:
        names = item.get("company_names") or []
        codes = item.get("company_ids") or []
        years = item.get("report_years") or []
        if len(names) != 1 or len(codes) != 1 or len(years) != 1:
            continue
        for chunk_id in item.get("gold_chunk_ids") or []:
            votes[chunk_id.split(":")[0]][(names[0], codes[0], years[0])] += 1

    metadata: dict[str, dict[str, object]] = {}
    for prefix, counted in votes.items():
        (name, code, year), _ = counted.most_common(1)[0]
        metadata[prefix] = {
            "document_key": f"cninfo:{code}:annual:{year}",
            "company_name": name,
            "report_year": year,
            "document_type": "annual",
        }
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path("data/indexes/demo"),
        help="Versioned index root to create.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the index root already exists.",
    )
    parser.add_argument(
        "--skip-lock-check",
        action="store_true",
        help="Skip external SHA lock verification (not recommended).",
    )
    arguments = parser.parse_args()

    if not arguments.skip_lock_check:
        validate_benchmark_lock(REPOSITORY_ROOT)
        print("Benchmark lock: VALID")

    root: Path = arguments.index_root
    if root.exists():
        if not arguments.force:
            print(f"Index root already exists, nothing to do: {root}")
            return 0
        shutil.rmtree(root)

    chunks = read_jsonl(REPOSITORY_ROOT / EVIDENCE_PATH, DocumentChunk)
    if not chunks:
        raise SystemExit(f"No evidence chunks found in {EVIDENCE_PATH}")

    metadata = derive_document_metadata(REPOSITORY_ROOT)
    unattributed = sorted(
        {c.chunk_id.split(":")[0] for c in chunks} - set(metadata)
    )
    if unattributed:
        raise SystemExit(
            "Cannot attribute these documents to a company from the benchmark: "
            f"{unattributed}"
        )
    chunks = [
        chunk.model_copy(update=metadata[chunk.chunk_id.split(":")[0]])
        for chunk in chunks
    ]

    # The evidence file carries chunk text only; there is no parsed IR to supply
    # geometry, so structured tables fall back to the conservative text path.
    structured_tables = build_structured_tables(chunks)

    snapshot_directory = root / "snapshots"
    content = "".join(chunk.model_dump_json() + "\n" for chunk in chunks)
    snapshot_path = snapshot_directory / "demo-evidence-v1.jsonl"
    write_text_lf(snapshot_path, content)

    generation_name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-demo"
    generation_path = root / "generations" / generation_name
    index = PersistentIndex.build(
        generation_path,
        chunks,
        source_chunk_path=snapshot_path,
        dense_model=None,
        structured_tables=structured_tables,
    )
    pointer = CurrentIndexPointer(
        index_id=index.manifest.index_id,
        generation_path=generation_path.relative_to(root).as_posix(),
        activated_at=datetime.now(UTC),
        active_version_ids=[],
    )
    write_text_lf(root / "current.json", pointer.model_dump_json(indent=2) + "\n")

    print(f"Demo index built at {root}")
    print(f"  index_id              {index.manifest.index_id}")
    print(f"  documents             {len(metadata)}")
    for prefix, fields in sorted(metadata.items()):
        print(f"    {prefix}  {fields['company_name']}  {fields['document_key']}")
    print(f"  chunks                {index.manifest.chunk_count}")
    print(f"  structured tables     {index.manifest.structured_table_count}")
    print(f"  structured cells      {index.manifest.structured_table_cell_count}")
    print("  mode                  lexical only (no dense embeddings)")
    print("Demo lane: 38 evidence chunks cannot reproduce published metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
