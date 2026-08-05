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
