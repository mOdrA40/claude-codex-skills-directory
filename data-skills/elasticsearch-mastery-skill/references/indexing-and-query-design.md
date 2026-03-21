# Indexing and Query Design (Elasticsearch)

## Indexing Rules

- Use explicit mappings for important fields.
- Choose analyzers based on language and retrieval needs.
- Separate full-text and exact-match fields intentionally.
- Avoid oversharding and giant mappings.

## Query Rules

- Distinguish filter context from scoring queries.
- Benchmark representative searches, not happy-path demos.
- Keep aggregation cost visible on large datasets.

## Principal Review Lens

- Are mappings stable and intentional?
- Which searches dominate cost?
- Is relevance tuned against real user queries?
