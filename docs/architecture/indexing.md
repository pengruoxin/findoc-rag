# Persistent indexing and retrieval

## Goals

The local index is designed to make retrieval behavior inspectable before adding
an external search service. It stores full chunk provenance, persists lexical and
dense indexes, and keeps retrieval scores separate until rank fusion.

## Index layout

```text
index-directory/
├── manifest.json
├── lexical.sqlite3
├── dense_embeddings.npy  # only when dense indexing is enabled
└── dense_chunk_ids.json
```

`manifest.json` records the index-format version, index ID, creation time, source
chunk-file SHA-256, document IDs, chunk count, average lexical document length,
BM25 parameters, tokenizer version, dense model, and embedding dimensions.

The SQLite database stores the complete validated `DocumentChunk` payload and a
real inverted index consisting of term document frequencies and per-chunk term
frequencies. Search does not rebuild BM25 from chunk text.

Index format v3 also persists `document_key`, `company_name`, `report_year`, and
`document_type` as indexed columns. Lexical and dense retrieval apply the same exact
filters before returning candidates, so constraints survive RRF and reranking.
Metadata comes from reviewed registry profiles rather than being guessed from text.

## Mixed-language lexical tokenization

The version-one tokenizer emits:

- lowercase Latin and numeric tokens;
- overlapping bigrams for contiguous CJK text;
- single CJK characters when a run contains only one character.

Section paths are indexed twice before the body text, giving structural context a
controlled lexical boost without modifying the original chunk.

This tokenizer is deterministic and dictionary independent. A learned or
domain-specific tokenizer can be benchmarked later under a new tokenizer version.

## Dense retrieval

The default dense model is `intfloat/multilingual-e5-small`. Passages and queries
use the model's required `passage:` and `query:` prefixes and are L2 normalized.
Document embeddings are persisted as an uncompressed NumPy array so a long-lived
process can memory-map them instead of decompressing the entire matrix per query.
Ordered chunk IDs are stored separately in `dense_chunk_ids.json`. Only the query
is encoded at search time, and both the mmap and ID ordering are cached by the
opened index instance.

## Hybrid retrieval

BM25 scores and cosine similarities are not directly comparable. Hybrid mode uses
Reciprocal Rank Fusion:

```text
RRF(document) = sum(1 / (k + rank))
```

The returned result keeps its lexical rank/score, dense rank/score, fused score,
full chunk, section path, pages, and element bounding boxes.

## Integrity and failure behavior

- A new index is built in a sibling staging directory.
- The staging directory is renamed into place only after all files and the manifest
  have been written.
- An existing target directory is never overwritten implicitly.
- Opening an index runs SQLite's integrity check.
- Manifest and database chunk counts must match.
- Dense matrix shape and unique chunk IDs must match the manifest.
- An index with missing or unexpected dense files fails fast.

## Known next-layer concerns

- A long-running API process should retain the embedding model in memory; repeated
  one-shot CLI calls reload it.
- Temporal granularity such as annual versus quarterly belongs in query analysis or
  reranking, not in generic RRF.
- External vector/search backends should implement the same retrieval contract so
  results remain comparable with the local reference implementation.
