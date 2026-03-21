---
name: postgresql-principal-engineer
description: |
  Principal/Senior-level PostgreSQL playbook for schema design, transactions, query tuning, indexing, reliability, migrations, observability, and production operations.
  Use when: designing relational schemas, reviewing SQL/query plans, fixing locks and slow queries, hardening migrations, or operating PostgreSQL in production.
---

# PostgreSQL Mastery (Senior → Principal)

## Operate

- Start by confirming workload shape: OLTP, analytical, mixed, multi-tenant, or event-heavy.
- Clarify scale: row counts, write rate, hottest tables, retention, and latency SLO.
- Treat PostgreSQL as a production system, not just a storage library: backups, locks, migrations, observability, and failover matter.
- Prefer boring relational design over clever schema tricks.

## Default Standards

- Model invariants with constraints first, code second.
- Use transactions intentionally; keep them short.
- Index for real query patterns, not theoretical ones.
- Avoid ORMs hiding expensive SQL on hot paths.
- Make migration rollout and rollback strategy explicit.

## “Bad vs Good”

```sql
-- ❌ BAD: filter on an unindexed hot-path column.
SELECT * FROM orders WHERE status = 'pending';

-- ✅ GOOD: index with realistic filter/order pattern.
CREATE INDEX CONCURRENTLY idx_orders_status_created_at
ON orders (status, created_at DESC);
```

```sql
-- ❌ BAD: delete millions of rows in one transaction.
DELETE FROM events WHERE created_at < now() - interval '90 days';

-- ✅ GOOD: batch or partition for controlled retention.
```

## Validation Commands

- Run `psql -f migration.sql` only after review of lock/runtime impact.
- Run `EXPLAIN (ANALYZE, BUFFERS)` for hot queries.
- Run `VACUUM (ANALYZE)` and review autovacuum posture where needed.
- Validate backup and restore workflows before calling a system production-ready.

## References

- Schema and migrations: [references/schema-and-migrations.md](references/schema-and-migrations.md)
- Performance and operations: [references/performance-and-operations.md](references/performance-and-operations.md)
- Query planning: [references/query-planning.md](references/query-planning.md)
- Index design: [references/index-design.md](references/index-design.md)
- Transactions and locking: [references/transactions-and-locking.md](references/transactions-and-locking.md)
- Partitioning and retention: [references/partitioning-and-retention.md](references/partitioning-and-retention.md)
- Replication and failover: [references/replication-and-failover.md](references/replication-and-failover.md)
- Vacuum and bloat: [references/vacuum-and-bloat.md](references/vacuum-and-bloat.md)
- Backup and restore: [references/backup-and-restore.md](references/backup-and-restore.md)
- Connection management: [references/connection-management.md](references/connection-management.md)
- Observability: [references/observability.md](references/observability.md)
- Security and compliance: [references/security-and-compliance.md](references/security-and-compliance.md)
- Multi-tenant PostgreSQL: [references/multi-tenant-postgres.md](references/multi-tenant-postgres.md)
- Zero-downtime migrations: [references/zero-downtime-migrations.md](references/zero-downtime-migrations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
