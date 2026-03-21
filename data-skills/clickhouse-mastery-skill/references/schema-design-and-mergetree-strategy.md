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

## Strategy Heuristics

### Engine choice is an operational contract

MergeTree-family selection affects not just query behavior, but merge patterns, deduplication semantics, storage cost, and how teams reason about correctness under ingestion realities.

### Analytical schemas should privilege recurring value

The right schema is usually the one that makes the most valuable repeated workloads boring and cheap, even if it sacrifices some elegance or generality.

### Compression and nullability deserve deliberate review

Column choices that seem harmless early can materially affect scan cost, storage footprint, and merge behavior once the dataset becomes large.

## Common Failure Modes

### Engine choice by popularity

Teams choose a familiar MergeTree variant without proving its semantics actually fit ingestion, deduplication, or query behavior.

### Join debt hidden in schema simplicity

The schema looks normalized or tidy, but repeated analytical joins create unnecessary recurring cost.

### Type choice tax

Column type, nullability, or codec decisions seem minor until they repeatedly inflate storage and scan inefficiency.

## Principal Review Lens

- Does the schema reflect actual query shape or inherited relational habits?
- Which column choice will hurt compression or scan cost most?
- Are we selecting an engine based on semantics or popularity?
- What data model decision today becomes hardest to reverse at scale?
- Which schema choice is most likely to create hidden merge or storage pain later?
