---
name: scylladb-principal-engineer
description: |
  Principal/Senior-level ScyllaDB playbook for shard-aware data modeling, consistency, compaction, topology, performance tuning, and operating high-throughput distributed wide-column systems.
  Use when: designing ScyllaDB schemas, tuning low-latency clusters, reviewing topology and repair posture, or operating ScyllaDB in production.
---

# ScyllaDB Mastery (Senior → Principal)

## Operate

- Start from access patterns, shard-aware performance, and failure semantics.
- Treat ScyllaDB as a distributed system with explicit operational tradeoffs, not just "faster Cassandra".
- Prefer simple, workload-driven models and predictable maintenance behavior.
- Optimize for low-latency consistency with operational realism.

## Default Standards

- Partition design must fit query and hotspot reality.
- Shard-aware performance should be understood, not assumed.
- Consistency and topology choices must match business semantics.
- Compaction, repair, and maintenance require proactive ownership.
- Capacity planning must include node loss and maintenance tax.

## References

- Data modeling and shard-aware partitioning: [references/data-modeling-and-shard-aware-partitioning.md](references/data-modeling-and-shard-aware-partitioning.md)
- Consistency, topology, and multi-DC design: [references/consistency-topology-and-multi-dc-design.md](references/consistency-topology-and-multi-dc-design.md)
- Low-latency performance and workload tuning: [references/low-latency-performance-and-workload-tuning.md](references/low-latency-performance-and-workload-tuning.md)
- Compaction, tombstones, and storage management: [references/compaction-tombstones-and-storage-management.md](references/compaction-tombstones-and-storage-management.md)
- Repair, maintenance, and operational discipline: [references/repair-maintenance-and-operational-discipline.md](references/repair-maintenance-and-operational-discipline.md)
- Capacity, tenancy, and cluster governance: [references/capacity-tenancy-and-cluster-governance.md](references/capacity-tenancy-and-cluster-governance.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
