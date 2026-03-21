# Low-Latency Performance and Workload Tuning

## Rules

- Tune around tail latency and hotspot reality, not average success stories.
- ScyllaDB performance depends on workload shape, shard distribution, and maintenance state together.
- Measure with realistic concurrency and failure conditions.
- Avoid optimization that hides poor partition design.

## Practical Guidance

- Track p95/p99 latency, coordinator load, hot shards, and background maintenance interference.
- Benchmark with realistic partition distributions and query mixes.
- Isolate workload classes when necessary.
- Understand how maintenance events change latency behavior.

## Principal Review Lens

- Which workload is driving tail pain most?
- Are we tuning around symptoms instead of redesigning the data model?
- What maintenance event breaks latency SLOs first?
- Which optimization assumption is least evidence-based?
