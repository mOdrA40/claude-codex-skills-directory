# Capacity Planning (Elasticsearch)

## Rules

- Capacity includes heap, disk, merge pressure, query concurrency, and reindex load.
- Plan headroom for node loss and recovery.
- Growth models must include shards and replicas, not only raw documents.
- Benchmark with realistic indexing and search mixes.

## Principal Review Lens

- What fails first under 2x indexing or 2x query load?
- Is heap pressure or disk pressure the real limit?
- How much headroom remains during rebalance?
