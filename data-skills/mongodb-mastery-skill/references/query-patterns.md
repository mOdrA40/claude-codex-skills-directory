# Query Patterns (MongoDB)

## Rules

- Query design should match available indexes and shard strategy.
- Aggregation pipelines are powerful but can hide expensive stages.
- Optimize hottest filters, sorts, and fan-out lookups first.
- Review query selectivity before reaching for more hardware.

## Principal Review Lens

- Which stage dominates cost in this pipeline?
- Are we filtering late when we should filter early?
- Will this query remain healthy as cardinality grows?
