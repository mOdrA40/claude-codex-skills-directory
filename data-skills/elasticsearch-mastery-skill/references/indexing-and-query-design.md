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

## Design Heuristics

### Index design and query design should be reviewed together

Mappings, analyzers, shard layout, and query structure form one system. Tuning queries without revisiting index design often only treats the symptom.

### Query behavior should reflect workload class

User-facing search, support investigation, analytical exploration, and batch reprocessing can all hit Elasticsearch very differently and should not be treated as one generic query pattern.

### Simplicity beats clever search debt

Search stacks become fragile when teams keep layering mapping tricks, query complexity, and aggregation work faster than they improve explainability.

## Common Failure Modes

### Query fix without schema fix

One expensive search is optimized while the underlying mapping or analyzer choices that cause repeated cost remain unchanged.

### One-cluster design ambiguity

The same indices and query patterns are asked to serve too many incompatible workloads without explicit tradeoff ownership.

### Demo-query confidence

The platform looks healthy under curated test searches, but real traffic mixes expose much worse cost and relevance behavior.

## Principal Review Lens

- Are mappings stable and intentional?
- Which searches dominate cost?
- Is relevance tuned against real user queries?
- Which indexing or query shortcut is currently accumulating the most long-term search debt?
