---
name: mongodb-principal-engineer
description: |
  Principal/Senior-level MongoDB playbook for document modeling, indexing, replication, sharding, query design, observability, and production reliability.
  Use when: designing document schemas, reviewing aggregation/query performance, operating replicas/shards, or hardening MongoDB-backed systems.
---

# MongoDB Mastery (Senior → Principal)

## Operate

- Start from access patterns, not abstract document purity.
- Choose embedding vs referencing based on write/read shape and document growth.
- Treat indexes, replication, and shard keys as architecture, not tuning afterthoughts.
- Model operational failure modes early: elections, lag, chunk imbalance, hot partitions.

## Default Standards

- Schema flexibility is not permission for schema chaos.
- Indexes must serve real query patterns.
- Keep document growth and update amplification visible.
- Design shard keys to avoid hot shards.
- Monitor replication lag and query planner regressions.

## References

- Schema and indexing: [references/schema-and-indexing.md](references/schema-and-indexing.md)
- Operations and reliability: [references/operations-and-reliability.md](references/operations-and-reliability.md)
- Document modeling: [references/document-modeling.md](references/document-modeling.md)
- Query patterns: [references/query-patterns.md](references/query-patterns.md)
- Aggregation and analytics: [references/aggregation-and-analytics.md](references/aggregation-and-analytics.md)
- Transactions and consistency: [references/transactions-and-consistency.md](references/transactions-and-consistency.md)
- Replication and elections: [references/replication-and-elections.md](references/replication-and-elections.md)
- Sharding and shard keys: [references/sharding-and-shard-keys.md](references/sharding-and-shard-keys.md)
- Capacity planning: [references/capacity-planning.md](references/capacity-planning.md)
- Backup, restore, and DR: [references/backup-restore-and-dr.md](references/backup-restore-and-dr.md)
- Observability: [references/observability.md](references/observability.md)
- Security and tenancy: [references/security-and-tenancy.md](references/security-and-tenancy.md)
- Schema evolution: [references/schema-evolution.md](references/schema-evolution.md)
- Multi-tenant MongoDB: [references/multi-tenant-mongodb.md](references/multi-tenant-mongodb.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
