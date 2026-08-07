# Document versioning and corpus generations

FinDocRAG separates document ingestion from searchable-index activation. This keeps
an existing corpus available while new PDFs are parsed, chunked, and indexed.

## Stable document identity

Callers provide a stable `document_key`, such as
`cninfo:600519:annual-report:2024`. The registry stores each distinct PDF content
hash as an immutable version. Reimporting identical active content is a no-op;
reimporting historical content reactivates its existing artifacts without parsing
the PDF again.

Versions move through these states:

```text
staging -> active -> superseded
       \-> failed
active  -> deleted
```

Parsing or chunking failures are recorded on the staging version and never replace
the active version. Deletion is soft: it removes the document from future corpus
generations while retaining its provenance and artifacts.

## Immutable generations

`build-corpus-index` takes a deterministic snapshot of all active chunks and builds
a new immutable directory under `generations/`. Only after index integrity checks
succeed is `current.json` atomically replaced. Readers therefore see either the old
complete generation or the new complete generation, never a partial index.

```text
corpus/
  current.json
  snapshots/<sha256>.jsonl
  generations/<timestamp>-<id>/
    manifest.json
    lexical.sqlite3
    dense_embeddings.npy
    dense_chunk_ids.json
```

When the active snapshot and dense model are unchanged, rebuilding is a no-op. For
a changed corpus, dense vectors for unchanged chunk IDs and content are copied from
the previous generation; only new or changed chunks are encoded. The index manifest
records the parent index ID and encoded/reused vector counts.

The CLI and long-lived retrieval service both accept either a concrete generation
directory or a corpus root containing `current.json`. A running service pins the
generation selected at startup; restart or an explicit future reload operation is
required to adopt a newer pointer.
