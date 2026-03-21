# SQL Performance (CockroachDB)

## Rules

- Start from statements that dominate latency or retries.
- Query shape, scan scope, and index choice still matter in distributed SQL.
- Reduce unnecessary round trips and cross-region reads.
- Review plans with locality and contention in mind.

## Principal Review Lens

- Is this query slow because of SQL shape or topology placement?
- Are we reading far more rows than needed?
- Which index reduces both latency and contention?
