# Adaptive candidate budget experiment v1

The evaluator now supports batched dense query encoding and records the effective
candidate budget per query. The policy expands the annual-summary scope to 100
candidates, quarterly/segment scopes to 40, and keeps unspecified queries at the
base budget of 20, bounded by `max_candidate_k=100`.

## Completed result

On the 10-query diagnostic slice, lexical + metadata + scope + adaptive budget used
an average of 76 candidates and achieved:

```text
Hit@5 = 1.0000
MRR@5  = 1.0000
```

The 76 average is lower than a fixed k=100 policy while preserving this slice's
lexical result. This is a routing-cost observation, not a general quality claim.

## Hybrid result and environment note

With the local Hugging Face cache forced offline (`HF_HUB_OFFLINE=1`), the batched
Hybrid run completed in 10.23 seconds:

```text
Hit@5 = 1.0000
MRR@5  = 1.0000
```

Index opening took 0.30 seconds and first dense-model loading took 8.15 seconds.
Earlier online-mode attempts exceeded 120 seconds while model acquisition/checking
was in progress; those were environment timeouts, not retrieval results. Offline
evaluation is therefore the reproducible protocol for this local cache.

Failure analysis initially found one `fusion_displacement`: the Moutai annual-revenue
gold ranked 79 in BM25, 120 in Dense, and 104 after RRF, just outside the adaptive
budget of 100. Scope routing could not recover evidence discarded by early fusion.

The pipeline now preserves the union of each retriever's top-k candidates when scope
routing is enabled, applies RRF scores without truncating that union, and lets the
scope stage perform the controlled reduction. Candidate recall and final Hit@5/MRR@5
then reached 1.00 on this slice without increasing the per-retriever average budget
above 76. The failure analyzer reports zero remaining misses.

Each evaluation JSON emits `candidate_first_rank`, `candidate_recall`, and
`candidate_pool_size`. This is a small, partly circular diagnostic set and not an
independent production quality claim.
