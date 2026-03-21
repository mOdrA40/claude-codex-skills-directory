# Schema Design and MergeTree Strategy (ClickHouse)

## Rules

- Schema design should follow real analytical queries, not normalized OLTP instincts.
- Engine choice matters: MergeTree family variants encode operational and semantic tradeoffs.
- Column types, codecs, and nullable usage have performance and storage consequences.
- Denormalization should be intentional and tied to workload value.

## Design Heuristics

- Optimize for the most expensive recurring queries first.
- Keep high-cardinality and sparse dimensions visible during schema review.
- Use materialized or derived structures when they reduce repeated compute materially.
- Avoid accidental schema choices that make merges, storage, or joins more expensive later.

## Principal Review Lens

- Does the schema reflect actual query shape or inherited relational habits?
- Which column choice will hurt compression or scan cost most?
- Are we selecting an engine based on semantics or popularity?
- What data model decision today becomes hardest to reverse at scale?
