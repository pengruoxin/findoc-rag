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
git clone https://github.com/pengruoxin/findoc-rag.git
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

For repeatable local Dense timings after the model is cached, disable Hub network
checks explicitly:

```powershell
$env:HF_HUB_OFFLINE = "1"
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
- [Independent holdout review](docs/holdout-review.md)
- [Engineering validation record](reports/engineering_validation.md)

## Current limitations and roadmap

- The Chinese diagnostic set is intentionally small and needs an independently
  reviewed holdout before resume-level accuracy claims are justified.
- Table cells are currently preserved through source elements, but table-structure
  reconstruction and deterministic numerical calculation are not implemented yet.
- Scope routing is an inspectable rule baseline; learned routing and calibrated
  reranking still require broader labels.
- Adaptive candidate budgets and component-union recall diagnostics are implemented;
  current results remain development diagnostics pending an independent holdout.
- Reranker model acquisition depends on external Hugging Face availability.
- Answer generation, citation composition, and abstention come after the evidence
  layer has stronger independent evaluation.

Next milestones are candidate-recall diagnostics, a reviewed Chinese holdout set,
table-aware evidence construction, and deterministic financial calculations.

The current pending holdout pack contains 16 new candidate questions. Generate or
review it with the workflow in [holdout-review.md](docs/holdout-review.md); pending
items are never scored as gold.

## Development

```powershell
uv run pytest -q
uv run ruff check .
git diff --check
```

Generated filings, indexes, traces, registries, and model caches remain local and are
excluded from Git. Versioned diagnostic data and compact experiment reports are kept
in the repository for regression analysis.

## Local product walkthrough

Install [uv](https://docs.astral.sh/uv/), then from the repository root:

```powershell
uv sync --extra dev --extra api
uv run pytest -q
uv run ruff check .
```

Start the API with one command:

```powershell
./scripts/start.ps1
```

Startup runs `scripts/validate-artifacts.ps1` first and aborts if reviewed evidence
or experiment metadata is invalid. To run the gate independently:

```powershell
./scripts/validate-artifacts.ps1
```

The runtime evaluation entry point is:

```powershell
./scripts/run-holdout-eval.ps1 -TopK 5
```

It validates the manifest before execution and writes the versioned result to
`reports/ranking/holdout-eval-v2.json`. Scores are omitted until a real local
retrieval runtime is available; no placeholder metrics are reported.

The service exposes `/health/live`, `/health/ready`, `/v1/search`, `/v1/traces/{trace_id}`,
and `/v1/metrics`. The static review and experiment pages can be opened directly in a
browser after pushing the repository:

- `docs/holdout-review.html` — candidate evidence review
- `docs/holdout-eval.html` — assistant-reviewed provisional holdout manifest
- `docs/experiment-dashboard.html` — versioned experiment registry and caveats
- `docs/holdout-failures.html` — runtime retrieval failure taxonomy and examples
- `docs/workspace-v3.html` — current user-facing query workspace
- `reports/processing/2026-08-06-baseline-v1.md` — PDF/IR/chunking processing baseline
- `docs/findoc-rag-interview-guide-zh.md` — FinDocRAG 项目亮点、技术路线与面试问答

The normalized holdout input is `data/diagnostics/holdout-eval-v2.json`. It contains
16 reviewed questions and chunk-level evidence IDs. It is intentionally labelled
`independent_gold: false`; do not report it as an independently annotated benchmark.

## Optional DeepSeek answer generation

Answer generation is disabled by default. The retrieval/evidence layer works without
any model token. When you are ready to enable DeepSeek, set the token in the shell:

```powershell
$env:DEEPSEEK_API_KEY = "your-token"
$env:FINDOC_RAG_ANSWER_ENABLED = "true"
$env:FINDOC_RAG_ANSWER_MODEL = "deepseek-chat"
```

The default endpoint is `https://api.deepseek.com/chat/completions`. The generator
passes only retrieved evidence to the model, requires citation markers, and falls
back to an extractive answer when the token or model is not configured. Never commit
the token to Git.

### 生成评测集与三轨运行

当前冻结回归集包含 48 条问题、120 个原子事实、67 条 gold evidence 和 53 个真实年报 hard negatives。先运行数据门禁：

```powershell
.venv\Scripts\python.exe scripts\validate_generation_eval_dataset.py
```

在同一个 PowerShell 窗口临时设置 Token；关闭窗口后变量自动失效，不会写入用户环境变量：

```powershell
$env:DEEPSEEK_API_KEY = "your-token"
```

分别运行 Oracle、真实检索和干扰上下文三条轨道：

```powershell
.venv\Scripts\python.exe scripts\run_generation_eval.py --lane oracle_context --require-remote
.venv\Scripts\python.exe scripts\run_generation_eval.py --lane retrieved_context --require-remote
.venv\Scripts\python.exe scripts\run_generation_eval.py --lane robustness --require-remote
```

Run 目录不可覆盖，并包含逐题答案、实际上下文、`gold / retrieved / hard_negative:<type>` 标签、确定性分数和汇总。当前 Dataset ID 为 `generation-eval-v1-b7f4d6113c96`。

对三条 lane 的 run 都可以执行 RAGAS；脚本会先校验 sibling `summary.json`、Dataset/Index ID、lane 问题范围、重复或未知 query、run error 和回答结构，再调用 Judge。Oracle/Retrieved 必须覆盖完整数据集，Robustness 必须精确覆盖 29 条 hard-negative 子集。行为与 gold contract 不一致的回答不会被剔除，而会保留在 eligible 分母并写入 `behavior_mismatch_query_ids`，避免错误拒答造成幸存者偏差：

```powershell
.venv\Scripts\python.exe scripts\run_ragas_generation_eval.py `
  reports\generation\runs\oracle_context-generation-eval-v1-b7f4d6113c96-deepseek-chat\items.jsonl `
  --output reports\generation\ragas-oracle-b7f4d6113c96.json
```

输出会保存 `lane`、`eligible_count`、`dataset_answerable_count`、`coverage`、`lane_coverage` 和逐项 query ID。当前 Robustness 的可回答交集为 18 条，因此相对全部 37 条可回答题的 coverage 为 `18/37`，lane 内 coverage 为 `1.0`。

如果回答模型和 judge 都是 `deepseek-chat`，产物会明确记录 `independent_judge=false`；这类同模型自评只能作为语义诊断，不能替代数值、单位、引用和行为门禁。

同一 dataset/lane 的前后版本可做逐题配对比较：

```powershell
.venv\Scripts\python.exe scripts\compare_generation_runs.py BASELINE_RUN_DIR CANDIDATE_RUN_DIR `
  --output-prefix reports\generation\comparisons\experiment-name
```

Every experiment report should record the dataset ID, index ID, retrieval mode,
candidate budget, metric definition, and limitations. Use
`reports/ranking/experiment-registry-v1.json` as the compact comparison registry.
