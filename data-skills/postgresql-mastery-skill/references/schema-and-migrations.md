# Schema and Migrations (PostgreSQL)

## Design Defaults

- Normalize until it hurts, then denormalize intentionally.
- Enforce invariants with `NOT NULL`, `CHECK`, `UNIQUE`, and foreign keys.
- Use additive schema changes first for rolling deploy safety.
- Prefer explicit status/state machines over ad-hoc boolean sprawl.

## Migration Rules

- Assume old and new app versions may coexist.
- Review table rewrite risk and lock strength before applying migrations.
- Backfill in controlled batches.
- Prefer forward-fix for destructive changes.

## Schema Heuristics

### Schema is where invariants should become boring

Good PostgreSQL schema design reduces ambiguity by putting durable truth in constraints, explicit state shape, and reviewable relationships rather than distributing it across fragile application code paths.

### Migration safety comes from choreography

The strongest migration plans create compatibility windows, observation points, and cleanup timing that let teams pause safely before the next irreversible step.

### Destructive changes deserve extra suspicion

Any migration that rewrites data, drops compatibility, or changes widely used semantics should be evaluated as an operational event, not just a schema diff.

## Common Failure Modes

### Constraint avoidance by convenience

The application enforces important invariants informally while the schema remains too weak to protect the data shape during edge cases and future changes.

### Safe SQL, unsafe rollout

The migration statement is technically correct, but the coexistence of app versions, backfill behavior, or rollback path remains under-modeled.

### Forward-fix fantasy for destructive changes

Teams plan to fix problems later even though the migration removes the very compatibility or data state needed for safe recovery.

## Principal Review Lens

- Which invariant belongs in the schema, not app code?
- Can the migration run safely during peak load?
- What happens if deploy is rolled back halfway through?
- Which schema change looks simple in review but carries the highest operational ambiguity in rollout?
