# Zero-Downtime Migrations (PostgreSQL)

## Rules

- Prefer expand-and-contract changes.
- Avoid blocking DDL on hot paths without a mitigation plan.
- Backfill in throttled batches with observability.
- Application rollout order matters as much as SQL correctness.

## Migration Heuristics

### Zero downtime is choreography, not wishful naming

The safest migrations usually depend on compatibility windows, phased rollout, backfill observation points, and explicit cleanup timing—not just one clever SQL command.

### Lock risk and rewrite risk must be treated separately

Some changes are dangerous because they block. Others are dangerous because they rewrite large data paths, saturate IO, or create background load that hurts production indirectly.

### Old and new app behavior must both be understood

The migration plan should explain what each application version can read, write, tolerate, and ignore during the transitional state.

## Common Failure Modes

### Online-sounding migration overconfidence

Teams assume a change is safe because the migration pattern sounds standard, while one hot table or one compatibility mistake creates the real incident.

### Backfill success, product pain

The data movement technically works, but user-facing latency, queueing, or replica pressure makes the rollout operationally unsafe.

### Cleanup too early

The team removes compatibility paths or old schema before enough production time has proven the new state is truly stable.

## Principal Review Lens

- Can old and new app versions run safely together?
- What is the lock and rewrite risk of this migration?
- What is the fallback if backfill overruns the maintenance window?
- Which assumption in this migration is least likely to hold under peak production load?
