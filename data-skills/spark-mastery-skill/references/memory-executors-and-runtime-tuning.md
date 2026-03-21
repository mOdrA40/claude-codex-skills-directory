# Memory, Executors, and Runtime Tuning

## Rules

- Runtime tuning should follow measurement, not folklore.
- Executor sizing, memory overhead, and parallelism all trade off with cluster economics.
- Tuning that helps one job can hurt multi-tenant platform health.
- Runtime settings should be interpreted together with data layout and workload shape.

## Practical Guidance

- Track executor utilization, spill, GC, skew, and task duration distribution.
- Tune around the most expensive and frequent jobs first.
- Avoid excessive per-job snowflake tuning that no platform owner can sustain.
- Benchmark changes using representative scale and data skew.

## Principal Review Lens

- Which setting is compensating for a deeper design flaw?
- Are we optimizing performance at the expense of fleet efficiency?
- What runtime bottleneck actually matters most today?
- Which job should be redesigned instead of tuned further?
