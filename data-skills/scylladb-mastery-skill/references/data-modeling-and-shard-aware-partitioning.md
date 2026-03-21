# Data Modeling and Shard-Aware Partitioning (ScyllaDB)

## Rules

- Model data for query paths, partition balance, and shard-local efficiency.
- Partition keys still define scale behavior and hotspot risk.
- Shard-aware client and workload assumptions should be validated, not presumed.
- Avoid designs that create oversized partitions or cross-shard pain.

## Design Guidance

- Validate partition cardinality and hottest-key distribution explicitly.
- Align table design with expected read/write concurrency.
- Duplicate data where it simplifies stable high-throughput access.
- Keep tenant and time dimensions visible in load planning.

## Principal Review Lens

- Which partition pattern breaks low-latency assumptions first?
- Are we relying on shard-aware behavior we have not measured?
- What key choice becomes dangerous at 10x traffic?
- Which table should be redesigned before scale magnifies it?
