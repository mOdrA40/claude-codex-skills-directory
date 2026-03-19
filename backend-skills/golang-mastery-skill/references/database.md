# Database & Queries (Correctness + Performance)

## Query Hygiene

- Every DB call uses `context.Context` with a deadline (no “infinite” queries).
- Always close rows (`rows.Close()`) and check `rows.Err()`.
- Treat “no rows” as a first-class outcome; define error taxonomy (`not found`, `conflict`, `invalid`, `unavailable`).

## Good vs bad (rows handling)

Bad (leaks rows / misses scan errors):

```go
rows, _ := db.QueryContext(ctx, q, arg)
for rows.Next() {
  // ...
}
```

Good:

```go
rows, err := db.QueryContext(ctx, q, arg)
if err != nil { return err }
defer rows.Close()
for rows.Next() {
  // scan...
}
if err := rows.Err(); err != nil { return err }
```

## Performance Patterns

- Prefer keyset pagination for large datasets (avoid deep `OFFSET`).
- Avoid N+1: batch by IDs, join, or prefetch; measure and add indexes.
- For bulk writes:
  - batch inserts
  - use Postgres `COPY` where appropriate (big ingestion)
- Keep transactions short; don’t mix network calls inside a transaction.

## Concurrency & Isolation

- Plan for retries on transient DB errors:
  - serialization failures
  - deadlocks
  - connection interruptions
- If you retry:
  - make operations idempotent or protect with unique constraints
  - bound retries with exponential backoff + jitter + overall deadline

## Pool Tuning (high-level)

Tune based on the DB’s actual capacity and p99 latency:
- Too many connections can reduce throughput (DB CPU thrash, lock contention).
- Start conservative; measure query latency, queueing time, and pool wait time.

## Migration Safety (expand/contract)

For zero-downtime deploys:
- Expand: add nullable columns / new tables first.
- Deploy app that writes both (or supports both).
- Backfill asynchronously.
- Contract: remove old reads/writes, then drop columns in a later deploy.

## “Best practice” logging for queries

- Log slow queries with a threshold (e.g. >200ms) and include:
  - operation name
  - duration
  - rows affected
  - error class
- Avoid logging full SQL with user parameters if it can contain PII/secrets.
