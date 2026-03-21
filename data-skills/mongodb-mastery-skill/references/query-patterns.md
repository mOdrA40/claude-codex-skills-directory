# Query Patterns (MongoDB)

## Rules

- Query design should match available indexes and shard strategy.
- Aggregation pipelines are powerful but can hide expensive stages.
- Optimize hottest filters, sorts, and fan-out lookups first.
- Review query selectivity before reaching for more hardware.

## Query Heuristics

### Design queries with index and growth in mind

MongoDB queries that feel fine early can become operationally painful when cardinality grows, arrays expand, or shard routing degrades.

### Pipelines need stage-level accountability

Aggregation pipelines should be explainable by stage, especially where filtering, grouping, sorting, unwinding, or lookup expansion creates the main cost.

### Optimize by workload class

Separate:

- hot user-facing reads
- analytical or reporting queries
- background maintenance scans
- aggregation-heavy product features

Each class deserves different cost expectations and guardrails.

## Common Failure Modes

### Early-stage laziness

Filters are applied too late, causing the pipeline to process and move far more data than necessary.

### Pipeline elegance over operability

The query is expressive but too hard for teams to debug quickly when performance regresses.

### Hardware-first thinking

Teams scale infrastructure before proving whether query shape, index strategy, or data model is the real problem.

## Principal Review Lens

- Which stage dominates cost in this pipeline?
- Are we filtering late when we should filter early?
- Will this query remain healthy as cardinality grows?
- What query pattern will age worst if data volume or tenant skew doubles?
