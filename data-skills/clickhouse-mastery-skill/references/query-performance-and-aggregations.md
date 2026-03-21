# Query Performance and Aggregations

## Rules

- Query design should minimize scanned data, expensive joins, and unnecessary high-cardinality work.
- Pre-aggregation and materialized views are useful when they simplify repeated heavy workloads.
- Understand the cost of distincts, joins, windows, and wide scans in your workload.
- Fast dashboards and exploratory analytics may need different optimization strategies.

## Performance Guidance

- Review representative queries with realistic concurrency.
- Watch CPU, memory, disk I/O, and read amplification together.
- Avoid one-size-fits-all optimization advice across product analytics and observability use cases.
- Tune around business-critical queries first, not synthetic benchmarks.

## Aggregation Heuristics

### Precompute for repeated business value, not laziness

Materialized views and pre-aggregations are strongest when they simplify a repeated, expensive access pattern with clear ownership and refresh semantics.

### Query classes matter more than single query heroes

Separate:

- high-frequency dashboard queries
- ad hoc analyst exploration
- heavy backfills or investigative workloads
- operational observability queries

Each class deserves different expectations and optimization tactics.

## Common Failure Modes

### Fast demo queries, weak production shape

Sample queries look excellent, but real concurrent dashboard filters, wider time ranges, or higher-cardinality dimensions behave very differently.

### Aggregation debt

Precomputed structures accumulate because they were easy to add, but no one reevaluates whether they still match current workload and cost.

## Principal Review Lens

- Which query class drives the most infrastructure cost?
- Are we precomputing wisely or merely masking poor design?
- What query still looks innocent but hurts most at scale?
- Can teams explain why one dashboard is fast and another is not?
- What workload should be isolated or re-modeled before adding more hardware?
