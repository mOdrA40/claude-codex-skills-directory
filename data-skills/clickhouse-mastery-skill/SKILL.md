---
name: clickhouse-principal-engineer
description: |
  Principal/Senior-level ClickHouse playbook for analytical schema design, partitioning, ingestion, query performance, replication, storage strategy, and operating large-scale columnar systems.
  Use when: designing OLAP workloads, reviewing MergeTree layout, tuning analytical queries, building event analytics platforms, or operating ClickHouse in production.
---

# ClickHouse Mastery (Senior → Principal)

## Operate

- Start from workload reality: event analytics, BI, observability, product analytics, or mixed ad-hoc usage.
- Treat partitioning, ordering keys, ingestion shape, and retention as architecture decisions.
- Prefer explicit tradeoffs between freshness, storage cost, and query speed.
- Design for operational boringness around merges, replication, and late-arriving data.

## Default Standards

- Table engine and key design must match query patterns.
- Partitioning should serve retention and query pruning, not fashion.
- Ingestion rate, merge pressure, and background work must stay visible.
- Materialized views should simplify workloads, not hide complexity debt.
- Capacity planning must include disk, CPU, and merge behavior together.

## References

- Schema design and MergeTree strategy: [references/schema-design-and-mergetree-strategy.md](references/schema-design-and-mergetree-strategy.md)
- Partitioning, ordering, and pruning: [references/partitioning-ordering-and-pruning.md](references/partitioning-ordering-and-pruning.md)
- Ingestion pipelines and late data: [references/ingestion-pipelines-and-late-data.md](references/ingestion-pipelines-and-late-data.md)
- Query performance and aggregations: [references/query-performance-and-aggregations.md](references/query-performance-and-aggregations.md)
- Replication, sharding, and reliability: [references/replication-sharding-and-reliability.md](references/replication-sharding-and-reliability.md)
- Storage, retention, and capacity planning: [references/storage-retention-and-capacity-planning.md](references/storage-retention-and-capacity-planning.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
