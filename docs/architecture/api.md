# Retrieval API

## Configuration

Configuration precedence is:

```text
built-in defaults < TOML file < environment variables
```

Supported environment variables:

```text
FINDOC_RAG_HOST
FINDOC_RAG_PORT
FINDOC_RAG_LOG_LEVEL
FINDOC_RAG_INDEX_DIR
FINDOC_RAG_DEFAULT_MODE
FINDOC_RAG_TOP_K
FINDOC_RAG_CANDIDATE_K
FINDOC_RAG_RRF_K
FINDOC_RAG_TRACING_ENABLED
FINDOC_RAG_TRACE_DB
FINDOC_RAG_CAPTURE_QUERY_TEXT
FINDOC_RAG_MAX_RECORDED_HITS
FINDOC_RAG_RERANKER_ENABLED
FINDOC_RAG_RERANKER_MODEL
FINDOC_RAG_RERANKER_BATCH_SIZE
FINDOC_RAG_SCOPE_ROUTING_ENABLED
```

Relative index paths in TOML are resolved from the TOML file's directory. The
configuration rejects invalid ports, modes, limits, and `candidate_k < top_k`
before the server begins accepting traffic.

## Start

```powershell
uv sync --extra api --extra dense
uv run findoc-rag serve --config configs/findoc-rag.example.toml
```

The API opens and validates the persistent index during application lifespan
startup. An invalid or inconsistent index prevents readiness instead of failing on
the first user query.

## Endpoints

### `GET /health/live`

Confirms that the HTTP process is running.

### `GET /health/ready`

Confirms that the index passed validation and returns the active index ID.

### `GET /v1/index`

Returns the complete index manifest, including source digest, format version,
document IDs, tokenizer, BM25 settings, dense model, and embedding dimension.

### `POST /v1/search`

Request:

```json
{
  "query": "2024年营业收入是多少",
  "mode": "hybrid",
  "top_k": 5,
  "candidate_k": 50,
  "rerank": true,
  "scope_routing": true,
  "filters": {
    "company_names": ["贵州茅台"],
    "report_years": [2024],
    "document_types": ["annual"]
  }
}
```

`mode`, `top_k`, `candidate_k`, and `rerank` are optional and inherit validated
service defaults. Reranking must be configured before a request can enable it. A
response includes:

- request ID;
- active index ID;
- effective retrieval mode;
- service-side retrieval latency;
- fused and component scores;
- final reranker score and pre-rerank position when enabled;
- complete chunks, section paths, pages, element IDs, and bounding boxes.

The response also includes a unique `trace_id`. Use `GET /v1/traces/{trace_id}`
to inspect stage timings and ranking snapshots. `GET /v1/metrics` returns persisted
success/error counts and latency distributions.

## Request identity and errors

Clients may send `X-Request-ID` using letters, numbers, `.`, `_`, `:`, or `-`, up
to 128 characters. Invalid values are not reflected; the server generates a random
request ID. Every response returns the effective ID in its header.

Application and validation errors use:

```json
{
  "request_id": "...",
  "error": {
    "code": "invalid_search_request",
    "message": "..."
  }
}
```

## Long-lived resources

The service owns one `PersistentIndex` instance. Dense model weights are loaded on
the first dense/hybrid request and reused thereafter. Dense embeddings are
memory-mapped and the ordered chunk IDs are cached. Model inference is guarded by
a process-local reentrant lock for correctness under concurrent requests.

In the current local validation, the first request includes model-loading time,
while a repeated hot query avoids that cost. Multi-worker deployment would create
one model instance per worker and will require explicit capacity planning later.
