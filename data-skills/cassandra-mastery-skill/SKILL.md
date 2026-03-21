---
name: cassandra-principal-engineer
description: |
  Principal/Senior-level Cassandra playbook for data modeling, partition strategy, consistency, repair, compaction, and operating large-scale distributed wide-column systems.
  Use when: designing Cassandra schemas, reviewing partition and consistency tradeoffs, operating clusters, or debugging latency and repair issues in production.
---

# Cassandra Mastery (Senior → Principal)

## Operate

- Start from access patterns, partitioning reality, and consistency needs.
- Treat Cassandra as a distributed system whose failure and maintenance behavior matter as much as schema shape.
- Prefer simple, query-driven data models over relational instincts.
- Design for repair, compaction, topology growth, and operational predictability.

## Default Standards

- Data modeling must follow read/write patterns explicitly.
- Partition keys should control hotspot and size risk.
- Consistency choices must map to business semantics.
- Repair, compaction, and tombstone behavior need proactive ownership.
- Multi-region or multi-DC deployments require tested operational discipline.

## References

- Data modeling and partition strategy: [references/data-modeling-and-partition-strategy.md](references/data-modeling-and-partition-strategy.md)
- Consistency, replication, and topology: [references/consistency-replication-and-topology.md](references/consistency-replication-and-topology.md)
- Read/write performance and query patterns: [references/read-write-performance-and-query-patterns.md](references/read-write-performance-and-query-patterns.md)
- Compaction, tombstones, and storage behavior: [references/compaction-tombstones-and-storage-behavior.md](references/compaction-tombstones-and-storage-behavior.md)
- Repair, anti-entropy, and operational safety: [references/repair-anti-entropy-and-operational-safety.md](references/repair-anti-entropy-and-operational-safety.md)
- Multi-tenant governance and capacity planning: [references/multi-tenant-governance-and-capacity-planning.md](references/multi-tenant-governance-and-capacity-planning.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
