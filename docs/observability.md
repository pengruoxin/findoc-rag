# Retrieval observability

## Trace identity

Client-provided request IDs are correlation values and may be reused. Each search
therefore receives a separate random `trace_id`, which is the primary key for one
retrieval execution.

## Recorded stages

Lexical searches record one stage, dense searches one stage, and hybrid searches
record:

```text
lexical retrieval
dense retrieval
RRF fusion
```

Every stage contains its duration, candidate count, and a bounded ranking snapshot
with chunk IDs, ranks, scores, and available component ranks. The maximum snapshot
size is configured independently from retrieval `candidate_k`.

## Trace payload

A persisted trace contains:

- trace and request IDs;
- index ID;
- start and completion timestamps;
- query SHA-256;
- optional query text;
- effective mode, top-k, and candidate-k;
- total and per-stage latency;
- result count and ranking snapshots;
- success/error status;
- error type and message.

## Query privacy

`capture_query_text` defaults to `false`. The system stores a deterministic SHA-256
for grouping identical queries but does not store their original content. Explicitly
enabling text capture should be treated as a data-governance decision, not a debug
convenience.

## Failure behavior

Validation and backend failures are traced before returning. Client errors use HTTP
400; unexpected retrieval/model failures use HTTP 500. Both return a trace ID.
Request-body schema failures occur before retrieval begins and return a structured
422 response with a request ID but no retrieval trace.

Trace persistence is fail-open for retrieval availability: a SQLite trace-write
failure is logged and does not discard an otherwise valid search response.

## Metrics

The persisted metrics endpoint currently reports:

- request, success, and error counts;
- error rate;
- latency P50, P95, and maximum;
- requests grouped by mode;
- errors grouped by exception type.

The local reference implementation computes these over all retained traces. Time
windows, retention, histograms, and Prometheus/OpenTelemetry export belong to the
next scale-oriented observability iteration.

## Cold versus hot validation

On the current single-document multilingual hybrid index, a cold query records model
loading inside the dense stage. A repeated query on the same service instance uses
the cached model and memory-mapped embeddings, so the trace clearly separates startup
cost from steady-state retrieval latency.
