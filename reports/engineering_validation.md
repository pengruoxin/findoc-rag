# Engineering validation record

This report preserves verified engineering results for later regression analysis and
resume evidence. The machine-readable snapshot is
[`validation/2026-08-05.json`](validation/2026-08-05.json).

## Validation scope

Recorded on 2026-08-05 using Python 3.12.13 on Windows. The code was based on commit
`722512f72e749425e8396b2abf6dacb33ba0ba32` plus local, uncommitted changes. This
distinction matters: the snapshot should be tied to the next commit when the user
chooses to publish the work.

Quality gates:

| Check | Result |
|---|---:|
| Unit/integration tests | 30 passed, 0 failed |
| Pytest-reported duration | 1.76 s |
| Ruff | passed |
| `git diff --check` | passed |

The tests cover structure-aware chunking, persistent lexical/dense indexes,
incremental embedding reuse, API contracts, trace persistence, configuration,
document lifecycle, historical artifact reuse, soft deletion, atomic generation
activation, and corpus-root resolution.

## Real Chinese filing run

These are pipeline measurements on official annual-report PDFs, not synthetic test
fixtures.

| Document | Pages | Text elements | Chunks | Provenance coverage | Repeated margins removed | Cross-page chunks | Token min / p50 / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 贵州茅台 2024 年报 | 143 | 3,336 | 349 | 100% | 286 | 95 | 97 / 236 / 506 / 640 |
| 伊利股份 2024 年报 | 270 | 8,898 | 609 | 100% | 537 | 225 | 39 / 226 / 475 / 644 |

The active hybrid generation contains 2 documents and 958 chunks using
`intfloat/multilingual-e5-small` (384 dimensions). When the second document was
added, all 349 embeddings from the parent generation were reused and only 609 new
chunks were encoded. Rebuilding the same active corpus returned `unchanged`.

## Retrieval runtime observation

One hybrid query was executed twice against the same 349-chunk generation in one
long-lived process (`candidate_k=30`, `top_k=3`). This isolates model initialization
from warm request behavior.

| Run | Total | Lexical | Dense | RRF |
|---|---:|---:|---:|---:|
| Cold | 17,358.21 ms | 4.55 ms | 17,353.43 ms | 0.11 ms |
| Warm | 31.14 ms | 5.47 ms | 25.38 ms | 0.12 ms |

This is a diagnostic trace, not a concurrency test or latency percentile benchmark.
It supports the narrower claim that the service caches the dense model across
requests and that model initialization dominates the observed cold request.

## Safe resume claims

The current evidence supports statements such as:

- Built a coordinate-preserving ingestion and structure-aware chunking pipeline for
  413 pages of real Chinese listed-company annual reports, producing 958 traceable
  chunks with 100% retained-element provenance coverage.
- Designed immutable corpus generations with atomic activation and incremental dense
  indexing; a two-document update reused 349 existing embeddings and encoded 609 new
  embeddings.
- Implemented a long-lived hybrid retrieval API with persistent stage-level tracing;
  one controlled cold/warm observation reduced total latency from 17.36 s to 31.14 ms
  after model initialization.

Do not yet claim Chinese retrieval accuracy, answer correctness, production QPS, or
semantic chunking quality. Those require a versioned Chinese evaluation set and
repeatable benchmark protocol, which will be added as the project matures.
