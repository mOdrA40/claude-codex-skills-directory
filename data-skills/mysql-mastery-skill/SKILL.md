---
name: mysql-principal-engineer
description: |
  Principal/Senior-level MySQL playbook for schema design, indexing, transactions, replication, operational reliability, online migrations, and production workload tuning.
  Use when: designing relational systems, reviewing query/index strategy, operating MySQL fleets, debugging contention or replication lag, or hardening MySQL-backed applications.
---

# MySQL Mastery (Senior → Principal)

## Operate

- Start from workload shape, consistency requirements, and operational model.
- Treat MySQL as a production system: schema, indexes, replication, backups, failover, and migrations all matter.
- Prefer explicit transactional boundaries and predictable query patterns.
- Optimize for boring operations rather than magical ORM assumptions.

## Default Standards

- Model invariants with schema and transactions first.
- Index for real query patterns and sort paths.
- Keep long transactions and lock-heavy flows visible.
- Replication lag and failover behavior must be known, not guessed.
- Online change workflows need review before rollout.

## References

- Schema design and indexing: [references/schema-design-and-indexing.md](references/schema-design-and-indexing.md)
- Query planning and performance: [references/query-planning-and-performance.md](references/query-planning-and-performance.md)
- Transactions, locks, and isolation: [references/transactions-locks-and-isolation.md](references/transactions-locks-and-isolation.md)
- Replication, lag, and failover: [references/replication-lag-and-failover.md](references/replication-lag-and-failover.md)
- Online migrations and operational change: [references/online-migrations-and-operational-change.md](references/online-migrations-and-operational-change.md)
- Backup, restore, and disaster recovery: [references/backup-restore-and-disaster-recovery.md](references/backup-restore-and-disaster-recovery.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
