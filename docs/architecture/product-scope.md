# FinDocRAG product scope

## Target user

The initial user is a research assistant who must extract and verify public facts
from Chinese listed-company filings. FinDocRAG supports evidence retrieval and
review; it does not provide investment advice or price predictions.

## Initial business tasks

1. Locate an exact financial or operating metric in one filing.
2. Compare the same metric across reporting periods or companies.
3. Retrieve management's documented explanation for a change.
4. Return every material fact with its source document, version, page, and quote.
5. Refuse or mark a result incomplete when required evidence is missing.

## Product data

- Annual and quarterly reports
- Performance announcements
- Prospectuses
- Exchange inquiry letters and company responses

The first vertical slice uses annual reports published through CNInfo. Raw filings
are downloaded locally with provenance manifests and are not redistributed in Git.

## Non-toy acceptance criteria

- Official-source ingestion is reproducible and records a SHA-256 content digest.
- Chinese PDF parsing preserves page numbers, document identity, and section context.
- Tables are represented as structured cells as well as searchable text.
- Retrieval is evaluated separately from answer generation.
- Material claims link to page-level evidence and can be reviewed in the source PDF.
- Numerical comparisons use deterministic code after evidence extraction.
- Document updates and corrected filings do not silently mix versions.
- Quality, latency, and failure-stage metrics are recorded for every pipeline version.

## Explicit non-goals for the first release

- A generic no-code workflow builder
- Stock recommendations or price forecasts
- A chat UI without evidence verification
- Claiming production quality from synthetic or evidence-only benchmark corpora
