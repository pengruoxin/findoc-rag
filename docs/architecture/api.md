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
- base/effective candidate budget and expansion reason;
- fused and component scores;
- final reranker score and pre-rerank position when enabled;
- complete chunks, section paths, pages, element IDs, and bounding boxes.

The response also includes a unique `trace_id`. Use `GET /v1/traces/{trace_id}`
to inspect lexical, dense/RRF, scope, structured-table, and rerank stage timings and
ranking snapshots. `GET /v1/metrics` returns persisted success/error counts and latency
distributions. The API and generation evaluator share the same structured evidence
router and scope-adaptive candidate-budget planner, so offline scores exercise the
production routing code instead of a duplicated evaluator-only implementation.

### `POST /v1/query`（Agent 主入口）

在既有检索与回答链路上增加稳定的 `contract_version: "1.0"`。响应除答案、引用外，还返回：

- `request_id`、`trace_id`、`index_id`，用于一次 Agent 调用的端到端追踪；
- `original_query`、`resolved_query`、`rewrite_mode`、`rewrite_gate`；
- `applied_filters` 与 `route`，明确公司、报告年份、事实期间、预测目标年份和过滤策略；
- `outcome`：`answer | abstain | clarify | evidence_only`，Agent 不需要从自然语言猜控制流；
- `claim_citations`：每个答案 claim 对应的 citation ordinal，引用越界或无引用回答不能标为 grounded。

无远程模型时，接口明确返回 `evidence_only` 且 `grounded=false`，不会把检索摘录伪装成已验证答案。模型实际读取的证据片段与 citation excerpt 使用同一 1800 字符边界。

### `GET /v1/capabilities`

返回当前运行时实际可用的检索模式、reranker、scope routing、tracing、确定性表格、claim citation 和 evidence resolve 能力，以及 `top_k`、`candidate_k`、生成上下文和 evidence resolve 的上下限。声明来自已加载索引和配置，不是静态产品文案；例如无 dense embeddings 的索引只声明 `lexical`。`structured_table_artifacts` 与 `deterministic_tables` 分开：前者表示当前索引确实包含经过摘要和 source-chunk SHA 校验的结构化表格 sidecar，后者仅表示回答层的确定性表格能力没有被运行时开关关闭。

结构化表格不写入原始 chunk payload，因此不会改写 benchmark 绑定的 chunk SHA 或 index identity。索引启动会交叉校验 sidecar schema、生成器、index ID、source snapshot digest、内容 digest、表/单元格计数，并逐表复核 `chunk_id` 与 persisted chunk SHA；检索命中后才把 cells 作为 runtime enrichment 注入。旧索引没有 sidecar 时仍可读取，回答层保留线性文本 fallback。

### `POST /v1/evidence:resolve`

Agent 用 `index_id` 和 1–50 个 `chunk_ids` 解析完整证据。接口保留请求顺序和重复项，每个 chunk 同时返回稳定的 SHA-256。请求索引与当前索引不符返回 `409 index_id_mismatch`，缺失 chunk 返回 `404 evidence_not_found`；这保证计划阶段拿到的引用不会在索引切换后被静默解释成另一份证据。

### PDF ingestion jobs

文档写入能力默认关闭，只有显式设置 `ingestion.enabled=true`（或 `FINDOC_RAG_INGESTION_ENABLED=true`）后才开放：

- `POST /v1/uploads` 只持久化 PDF，状态为 `uploaded`，不会静默修改 active corpus；
- `POST /v1/uploads/{job_id}:process` 要求 Agent 明确提交稳定 `document_key` 和可选 metadata，原子 claim 后进入 `validating → ingesting → indexing → ready | failed`；
- `GET /v1/uploads/{job_id}` 跨服务重启读取落盘状态，ready 时返回 `document_version_id` 和 `index_id`；
- 处理使用 strict PDF 质量门禁，空文件、伪 PDF、未解决 OCR 或解析/索引失败都进入 `failed`，保留可读错误；服务重启时在途任务明确标为 `process_restarted`，不会永久伪装成处理中；
- 同一 job 只能启动一次；索引构建仍走不可变 generation + 原子 current pointer，成功后服务切换到新 index-bound retrieval 实例。

`/v1/capabilities.features.ingestion_jobs` 反映当前运行时是否授权该能力。生产部署还应把写接口放在认证/审计边界之后；当前 API 不把“能查询”自动等价为“能改语料”。

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
