# FinDocRAG

FinDocRAG is a verifiable RAG system for complex Chinese listed-company documents,
including annual reports, quarterly reports, announcements, prospectuses, and
exchange inquiry responses.

The project focuses on evidence-grounded financial fact retrieval, cross-period and
cross-company comparison, deterministic numerical calculation, page-level citations,
and stage-by-stage RAG failure diagnosis.

## Project status

The current bootstrap stage validates retrieval and evaluation code with the 150
public FinanceBench examples. FinanceBench is an English technical baseline only;
the product dataset and final demonstration will use public Chinese listed-company
filings. Current experiments must not be reported as full-document or Chinese-data
performance.

Implemented:

- Official CNInfo annual-report discovery, exact-edition selection, and download provenance
- Generic PDF parsing into page-level text/image elements with coordinates and OCR flags
- A unified corpus, question, gold-answer, and gold-evidence schema
- BM25 lexical retrieval
- E5 dense retrieval with local embedding cache
- Hit@K, Recall@K, MRR, and per-question rankings
- Reproducible BM25-versus-dense comparison report

Next:

- Reciprocal Rank Fusion hybrid retrieval
- Chinese filing collection and document parsing
- Table-aware evidence extraction and deterministic calculations
- Page-level citations and retrieval failure diagnosis

## Setup

```powershell
uv sync --extra dev
uv run findoc-rag doctor
uv run findoc-rag import-financebench
uv run findoc-rag evaluate-bm25

uv sync --extra dev --extra dense
uv run findoc-rag evaluate-dense
uv run pytest
```

Parse any local PDF into the common Document IR:

```powershell
uv run findoc-rag parse-pdf path/to/document.pdf
```

The parsed record preserves the file digest, page number, page dimensions, element
reading order, bounding boxes, extracted character counts, and pages requiring an
OCR fallback.

Fetch a real Chinese annual report from the official disclosure source:

```powershell
uv run findoc-rag fetch-annual-report --company 贵州茅台 --year 2024
```

Downloaded filings and manifests stay under `data/artifacts/` and are intentionally
excluded from Git. Each manifest records the announcement ID, security code,
publication time, official source URL, SHA-256 digest, and local artifact path.

See [the bootstrap comparison](reports/baseline_comparison.md) for results and
scope limitations.

See [the product scope](docs/product-scope.md) for the target user, business tasks,
and non-toy acceptance criteria.
