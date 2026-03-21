# Online Migrations and Operational Change

## Rules

- Schema changes on live systems should be staged with lock and rewrite awareness.
- Expand-and-contract patterns are usually safer than one-shot changes.
- Backfills need throttling, visibility, and rollback posture.
- Application and schema rollout order must be coordinated.

## Failure Modes

- Blocking DDL on hot tables without a fallback plan.
- Mixed-version applications incompatible with transitional schema.
- Backfills saturating replicas or overwhelming primary write paths.
- Successful migration commands that still create operational instability.

## Principal Review Lens

- What is the lock risk and data-copy risk of this change?
- Can old and new app versions coexist safely?
- Which backfill step is most dangerous at peak traffic?
- What recovery plan exists if the migration halfway succeeds?
