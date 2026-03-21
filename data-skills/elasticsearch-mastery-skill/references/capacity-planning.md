# Capacity Planning (Elasticsearch)

## Rules

- Capacity includes heap, disk, merge pressure, query concurrency, and reindex load.
- Plan headroom for node loss and recovery.
- Growth models must include shards and replicas, not only raw documents.
- Benchmark with realistic indexing and search mixes.

## Capacity Heuristics

### Capacity is defined by failure and recovery posture

An Elasticsearch cluster is only truly well-sized if it can handle indexing, search, shard movement, and node recovery together without collapsing into prolonged instability.

### Heap and disk are not interchangeable limits

Some workloads fail first on heap and GC behavior, others on storage growth, merge pressure, or recovery bandwidth. Planning must separate those failure modes clearly.

### Reindex and incident load belong in the model

A cluster that handles normal traffic comfortably may still be undersized if one schema change, restore, or tenant-driven reindex overwhelms it.

## Common Failure Modes

### Normal-load sizing only

The cluster looks fine in day-to-day behavior, but one recovery or reindex event reveals that true operational headroom was never there.

### Heap-centric blind spot

Teams watch JVM behavior closely while disk, merge, and shard-movement limits become the real bottleneck.

### Shard growth underestimated

Capacity plans model document count and query rate but ignore the long-tail tax of shard count, replicas, and recovery movement.

## Principal Review Lens

- What fails first under 2x indexing or 2x query load?
- Is heap pressure or disk pressure the real limit?
- How much headroom remains during rebalance?
- Which capacity assumption is most likely to break during restore, reindex, or node loss?
