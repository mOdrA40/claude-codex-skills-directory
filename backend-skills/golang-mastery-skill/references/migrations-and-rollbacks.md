# Migrations and Rollbacks (Go Services)

Schema change discipline is one of the fastest ways to separate mature teams from teams that cause outages during deploys.

## First Principles

- Treat schema changes as distributed systems events, not local SQL edits.
- Prefer backward-compatible migrations during rolling deploys.
- Assume old and new binaries may run at the same time.
- Have a rollback plan before applying a risky migration.

## Safe Migration Sequence

For most rolling deployments:

1. Add new nullable column / new table / additive index.
2. Deploy app that can read old and new shape.
3. Backfill gradually if needed.
4. Switch reads/writes to new path.
5. Remove old column or constraint only after the fleet is stable.

## Dangerous Changes

High-risk examples:

- renaming columns without compatibility layer,
- dropping columns still used by old binaries,
- long-running table rewrites during peak traffic,
- uniqueness constraints added without prior cleanup,
- destructive migrations with no verified restore path.

## Rollback Rules

- App rollback and schema rollback are different decisions.
- Prefer forward-fix for destructive schema changes.
- Roll back the app quickly if schema is still backward-compatible.
- Never assume you can “just revert” a migration after data shape changes.

## Backfills

- Run backfills as controlled jobs, not request-path logic.
- Make them resumable and idempotent.
- Rate-limit them so they do not starve production traffic.
- Emit progress metrics and logs.

## Deployment Checklist

- Migration reviewed for lock behavior and runtime cost.
- Verified compatibility with previous app version.
- Rollback/forward-fix plan documented.
- Metrics and alerts ready for DB saturation, lock waits, and error spikes.
- Backup/restore posture known for critical data.

## Principal Review Lens

Ask:

- Can old and new binaries coexist safely?
- Which migration step is most likely to lock hot tables?
- If deploy fails halfway, what exact state are we left in?
- Is rollback actually safe, or only emotionally comforting?
