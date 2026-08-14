# Query scope routing

Scope routing is an optional, explainable ranking stage between candidate retrieval
and cross-encoder reranking. It recognizes explicit query cues for quarterly,
segment, consolidated-statement, parent-company, note, and audit scopes. Queries that
contain a common financial metric but no explicit scope default to the annual-summary
intent; unrelated queries remain unspecified and receive no scope adjustment.

The router does not hard-filter sections. It scores positive and conflicting section
cues, then stably reorders the candidate pool. Every hit retains its retrieval rank,
scope score, and rank delta. Trace output records a separate `scope` stage and the
inferred scope/confidence.

```json
{
  "query": "贵州茅台2024年分季度营业收入是多少",
  "scope_routing": true,
  "filters": {
    "company_names": ["贵州茅台"],
    "report_years": [2024]
  }
}
```

The rule tables are intentionally small and inspectable. They are a routing baseline,
not a substitute for an independently labeled intent classifier. Unknown queries are
left unchanged rather than forced into a financial scope.

Scope routing can also drive an adaptive candidate budget. With the default base of 20,
the bounded policy uses 100 candidates for annual-summary, 40 for quarterly/segment,
60 for consolidated/parent/note/audit, and 20 for unspecified queries. Every request
records the base and effective budget plus the expansion reason. The policy is an
experiment control; its value comes from the frozen end-to-end run, not from assuming
that a larger candidate pool is always better.

When an index contains a validated structured-table sidecar, a second schema-aware
`structured` stage promotes evidence whose table family matches the query: quarterly,
segment, note-cost, annual-data, or concentration. This stage only reranks the bounded
candidate pool and never invents cells. The sidecar has already been checked against the
active index, source snapshot, content digest, chunk ID, and source-chunk SHA. The API
service and generation evaluator call the same `route_structured_evidence` function,
preventing an offline/online routing fork.

For hybrid retrieval, scope routing operates on the union of BM25 and Dense top-k
candidates. RRF scores are retained, but fusion does not truncate a component's valid
candidate before the scope stage can inspect it. The final response and reranker input
remain bounded downstream. Traces expose lexical, dense/RRF when enabled, scope,
structured, and rerank stages separately so an Agent or evaluator can distinguish
retrieval failure from schema-routing failure.
