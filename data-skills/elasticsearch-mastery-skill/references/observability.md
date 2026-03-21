# Observability (Elasticsearch)

## Rules

- Monitor heap, GC, indexing throughput, search latency, thread pools, and shard state.
- Dashboards should separate indexing pain from search pain.
- Slow logs and query visibility must be usable in incidents.
- Alert on user-visible risk, not only noisy internals.

## Principal Review Lens

- Which signal predicts a bad cluster day earliest?
- Can we isolate one broken index or workload quickly?
- Are alerts actionable for on-call engineers?
