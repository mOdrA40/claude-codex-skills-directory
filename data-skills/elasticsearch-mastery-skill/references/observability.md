# Observability (Elasticsearch)

## Rules

- Monitor heap, GC, indexing throughput, search latency, thread pools, and shard state.
- Dashboards should separate indexing pain from search pain.
- Slow logs and query visibility must be usable in incidents.
- Alert on user-visible risk, not only noisy internals.

## Observability Heuristics

### Metrics should map to user pain and recovery risk

The most useful signals are the ones that predict stale indexing, broken relevance, slow search, unsafe shard movement, or fragile heap behavior before users or operators are overwhelmed.

### Separate workload classes explicitly

Observability should make it easy to distinguish:

- indexing distress
- search distress
- shard recovery stress
- one index or workload dominating cluster health

### Slow logs need operational context

A slow query entry is only useful if the team can connect it to the index, workload, business path, and mitigation options quickly.

## Common Failure Modes

### Metric abundance, weak diagnosis

The dashboards are rich, but on-call still cannot tell whether the problem starts in mappings, shards, heap, indexing, or one specific query class.

### Cluster-wide averages hiding local pain

One index or search path is failing badly while overall cluster medians remain deceptively calm.

### Alerting on internals without user translation

The system pages on low-level metrics without helping operators understand the actual user or business consequence.

## Principal Review Lens

- Which signal predicts a bad cluster day earliest?
- Can we isolate one broken index or workload quickly?
- Are alerts actionable for on-call engineers?
- Which missing signal would most improve early incident diagnosis?
