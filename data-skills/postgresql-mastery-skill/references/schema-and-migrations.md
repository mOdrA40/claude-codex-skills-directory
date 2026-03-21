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

## Principal Review Lens

- Which invariant belongs in the schema, not app code?
- Can the migration run safely during peak load?
- What happens if deploy is rolled back halfway through?
