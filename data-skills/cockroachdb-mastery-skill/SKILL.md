---
name: cockroachdb-principal-engineer
description: |
  Principal/Senior-level CockroachDB playbook for distributed SQL design, transactions, multi-region topology, schema evolution, observability, and operational reliability.
  Use when: designing geo-distributed systems, reviewing transaction semantics, hardening multi-region SQL workloads, or operating CockroachDB in production.
---

# CockroachDB Mastery (Senior → Principal)

## Operate

- Confirm whether the real problem is multi-region latency, survivability, or operational simplicity.
- Treat topology, locality, and transaction shape as first-class design inputs.
- Design with contention, retries, and regional failure in mind.
- Prefer simple data models and explicit transaction boundaries.

## Default Standards

- Expect transaction retries and design idempotently.
- Keep hot-row contention visible and minimized.
- Use regional placement intentionally, not by default cargo cult.
- Review schema changes with rollout behavior and backfill cost in mind.
- Observe latency by region, statement, and contention class.

## “Bad vs Good”

```sql
-- ❌ BAD: one global counter row becomes a contention hotspot.
-- ✅ GOOD: shard counters or redesign aggregation semantics.
```

## Validation Commands

- Review statement plans and contention insights.
- Test transaction retry behavior in application code.
- Validate regional latency and failover assumptions with realistic traffic.

## References

- Distributed SQL patterns: [references/distributed-sql-patterns.md](references/distributed-sql-patterns.md)
- Transactions and multi-region operations: [references/transactions-and-multi-region.md](references/transactions-and-multi-region.md)
- Contention and hotspots: [references/contention-and-hotspots.md](references/contention-and-hotspots.md)
- Schema changes and backfills: [references/schema-changes-and-backfills.md](references/schema-changes-and-backfills.md)
- Locality and zone configs: [references/locality-and-zone-configs.md](references/locality-and-zone-configs.md)
- Survivability and failures: [references/survivability-and-failures.md](references/survivability-and-failures.md)
- SQL performance: [references/sql-performance.md](references/sql-performance.md)
- Indexing strategy: [references/indexing-strategy.md](references/indexing-strategy.md)
- Multi-tenant design: [references/multi-tenant-design.md](references/multi-tenant-design.md)
- Observability: [references/observability.md](references/observability.md)
- Security and boundaries: [references/security-and-boundaries.md](references/security-and-boundaries.md)
- Backup, restore, and DR: [references/backup-restore-and-dr.md](references/backup-restore-and-dr.md)
- Capacity planning: [references/capacity-planning.md](references/capacity-planning.md)
- Application patterns: [references/application-patterns.md](references/application-patterns.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
