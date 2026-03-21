# Zero-Downtime Migrations (PostgreSQL)

## Rules

- Prefer expand-and-contract changes.
- Avoid blocking DDL on hot paths without a mitigation plan.
- Backfill in throttled batches with observability.
- Application rollout order matters as much as SQL correctness.

## Principal Review Lens

- Can old and new app versions run safely together?
- What is the lock and rewrite risk of this migration?
- What is the fallback if backfill overruns the maintenance window?
