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

## Principal Review Lens

- Which query class drives the most infrastructure cost?
- Are we precomputing wisely or merely masking poor design?
- What query still looks innocent but hurts most at scale?
- Can teams explain why one dashboard is fast and another is not?
