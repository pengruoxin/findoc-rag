# FinDocRAG

Production-oriented retrieval infrastructure for complex Chinese listed-company
filings. FinDocRAG turns long PDFs into coordinate-preserving evidence, builds
versioned hybrid indexes, and makes every retrieval stage inspectable.

> The project currently focuses on ingestion, retrieval, ranking, and diagnostics.
> Answer generation is intentionally deferred until evidence quality is measurable.

## Why this project

Chinese annual reports are not plain-text knowledge bases. A single metric may appear
in annual summaries, quarterly tables, segment disclosures, audit notes, consolidated
statements, and parent-company statements. Keyword similarity alone cannot determine
which scope answers a query.

FinDocRAG therefore treats retrieval as a traceable pipeline:

```text
official PDF
  -> coordinate-preserving Document IR
  -> structure-aware chunks with page/bbox provenance
  -> transactional document registry
  -> immutable lexical+dense index generation
  -> metadata filtering
  -> BM25 / multilingual E5 / RRF
  -> explainable query-scope routing
  -> optional CrossEncoder reranking
  -> evidence with stage-level traces
```

## What works today

- Official CNInfo annual-report discovery with edition filtering, source URL,
  announcement ID, and SHA-256 provenance.
- PDF Document IR preserving pages, reading order, element IDs, bounding boxes,
  typography, and OCR fallback flags.
- Structure-aware chunking with repeated-margin removal, heading hierarchy, bounded
  overlap, section paths, and source-element coverage reports.
- SQLite BM25 plus memory-mapped multilingual E5 embeddings and RRF hybrid retrieval.
- Metadata filters for document key, company, reporting year, and document type.
- Versioned document lifecycle and immutable corpus generations with atomic
  `current.json` activation and incremental embedding reuse.
- Long-lived FastAPI service with model caching, structured errors, request IDs,
  privacy-aware SQLite traces, and aggregate metrics.
- Optional multilingual CrossEncoder reranking with original rank, final score, and
  rank-delta diagnostics.
- Explainable annual/quarterly/segment/statement/note/audit scope routing.
- Versioned Chinese ranking diagnostics built from real filing structure rather than
  LLM-generated gold answers.

## Quick start

### Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- Windows PowerShell commands are shown below. The CLI itself is cross-platform.

Clone and install all development, API, and dense-retrieval dependencies:

```powershell
git clone https://github.com/yiwu17/findoc-rag.git
cd findoc-rag
uv sync --extra dev --extra api --extra dense
uv run findoc-rag doctor
uv run pytest -q
```

### Run the smallest local-PDF pipeline

Use any local PDF; no external service is required after dependencies are installed.

```powershell
uv run findoc-rag parse-pdf path/to/report.pdf
uv run findoc-rag chunk-pdf path/to/report.pdf
```

`chunk-pdf` prints the generated JSONL path. Use that path to build an index:

```powershell
uv run findoc-rag build-index `
  data/processed/chunks/<sha256>.jsonl `
  --output-dir data/indexes/my-report `
  --dense

uv run findoc-rag search-index data/indexes/my-report `
  "2024年营业收入是多少" `
  --mode hybrid `
  --top-k 5
```

The first dense command downloads `intfloat/multilingual-e5-small`; later queries
reuse the local model cache.

## End-to-end real filing workflow

### 1. Download from the official disclosure source

```powershell
uv run findoc-rag fetch-annual-report --company 贵州茅台 --year 2024
```

The command prints the selected PDF and manifest paths. Downloads stay under
`data/artifacts/` and are excluded from Git.

### 2. Ingest a versioned document

Use the PDF path printed above and a stable logical key:

```powershell
uv run findoc-rag ingest-document `
  data/artifacts/cninfo/<downloaded-report>.pdf `
  --document-key cninfo:600519:annual:2024

uv run findoc-rag list-active-documents
```

Reimporting identical content is a no-op. A new content hash creates a staging
version and replaces the active version only after parsing and chunking succeed.

### 3. Build and atomically activate the corpus index

```powershell
uv run findoc-rag build-corpus-index --dense
```

The active generation is written under `data/indexes/corpus/generations/`. Readers
follow `data/indexes/corpus/current.json`, so a partial build is never served.

### 4. Search with provenance and constraints

```powershell
uv run findoc-rag search-index data/indexes/corpus `
  "贵州茅台2024年分季度营业收入是多少" `
  --mode hybrid `
  --company 贵州茅台 `
  --report-year 2024 `
  --scope-routing `
  --candidate-k 20 `
  --top-k 5
```

Add `--rerank` to enable `BAAI/bge-reranker-v2-m3`. Its first download is much
larger than the E5 retriever and may require Hugging Face authentication in
rate-limited environments.

## Run the API

Point the service at a concrete index or the corpus root:

```powershell
$env:FINDOC_RAG_INDEX_DIR = (Resolve-Path data/indexes/corpus)
$env:FINDOC_RAG_DEFAULT_MODE = "hybrid"
uv run findoc-rag serve
```

Check readiness and submit a filtered request from another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready

$body = @{
  query = "贵州茅台2024年分季度营业收入是多少"
  mode = "hybrid"
  top_k = 5
  candidate_k = 20
  scope_routing = $true
  filters = @{
    company_names = @("贵州茅台")
    report_years = @(2024)
    document_types = @("annual")
  }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/search `
  -ContentType "application/json" `
  -Body $body
```

Available endpoints:

```text
GET  /health/live
GET  /health/ready
GET  /v1/index
POST /v1/search
GET  /v1/traces/{trace_id}
GET  /v1/metrics
```

See [API configuration and contracts](docs/api.md) and
[retrieval observability](docs/observability.md).

## Reproduce the Chinese ranking diagnostics

The checked-in v1 slice contains 10 structure-anchored queries over the 2024 annual
reports of Kweichow Moutai and Yili. Unreviewed candidates are never treated as
implicit negatives.

After ingesting the same two filings and building the corpus index:

```powershell
uv run findoc-rag apply-document-profiles `
  configs/ranking-diagnostic-profiles.json

uv run findoc-rag build-corpus-index --dense

uv run findoc-rag generate-ranking-diagnostics `
  configs/ranking-diagnostic-profiles.json `
  --candidate-k 20

uv run findoc-rag evaluate-ranking-diagnostics `
  data/diagnostics/ranking-diagnostics-v1.json `
  --mode hybrid `
  --metadata-filters `
  --scope-routing `
  --candidate-k 20 `
  --top-k 5 `
  --output reports/ranking/my-run.json
```

Current controlled result at `candidate_k=20`:

| Pipeline | Hit@5 | MRR@5 |
|---|---:|---:|
| BM25 + metadata | 0.70 | 0.70 |
| BM25 + metadata + scope | 0.80 | 0.80 |
| Hybrid + metadata | 0.50 | 0.2783 |
| Hybrid + metadata + scope | 0.80 | 0.80 |

These are development diagnostics, not independent accuracy claims: dataset gold and
scope routing both use filing section cues. See the
[scope-routing experiment](reports/ranking/scope-routing-v1.md) for candidate-depth
analysis and explicit claim limits.

## Repository map

```text
src/findoc_rag/
  documents/        PDF IR and parsing
  chunking.py       structure-aware chunking
  registry.py       transactional document lifecycle
  corpus.py         immutable corpus generations
  indexing.py       BM25, dense search, RRF, metadata filters
  scope_routing.py  explainable intent/scope ranking
  reranking.py      optional CrossEncoder reranker
  service.py        long-lived retrieval boundary
  api.py            FastAPI application
  observability.py  traces and metrics
  diagnostics.py    Chinese ranking dataset and evaluation

configs/            service and reviewed document profiles
docs/               design and operational documentation
reports/            reproducible experiment summaries and snapshots
tests/              unit and integration tests
```

## Design documents

- [Product scope and non-toy criteria](docs/product-scope.md)
- [Persistent indexing](docs/indexing.md)
- [Document versioning and atomic generations](docs/versioning.md)
- [Cross-encoder reranking](docs/reranking.md)
- [Query scope routing](docs/scope-routing.md)
- [Chinese ranking diagnostics](docs/ranking-diagnostics.md)
- [Engineering validation record](reports/engineering_validation.md)

## Current limitations and roadmap

- The Chinese diagnostic set is intentionally small and needs an independently
  reviewed holdout before resume-level accuracy claims are justified.
- Table cells are currently preserved through source elements, but table-structure
  reconstruction and deterministic numerical calculation are not implemented yet.
- Scope routing is an inspectable rule baseline; learned routing and calibrated
  reranking still require broader labels.
- Reranker model acquisition depends on external Hugging Face availability.
- Answer generation, citation composition, and abstention come after the evidence
  layer has stronger independent evaluation.

Next milestones are candidate-recall diagnostics, a reviewed Chinese holdout set,
table-aware evidence construction, and deterministic financial calculations.

## Development

```powershell
uv run pytest -q
uv run ruff check .
git diff --check
```

Generated filings, indexes, traces, registries, and model caches remain local and are
excluded from Git. Versioned diagnostic data and compact experiment reports are kept
in the repository for regression analysis.
