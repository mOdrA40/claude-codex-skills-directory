# Multi-Tenant Design (CockroachDB)

## Rules

- Tenant isolation is a blast-radius question before it is a schema question.
- Workload controls and placement rules matter for noisy-neighbor safety.
- Tenant-aware operations must remain auditable and supportable.
- Backups and export flows should preserve isolation.

## Principal Review Lens

- Can one tenant create contention or region imbalance for others?
- Which controls enforce isolation during incidents?
- How fast can support isolate one tenant safely?
