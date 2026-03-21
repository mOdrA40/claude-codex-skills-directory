# Transactions and Locking (PostgreSQL)

## Rules

- Keep transactions short and scoped to a real invariant.
- Understand row locks, relation locks, and DDL lock impact.
- Avoid holding transactions open across network calls or user think time.
- Deadlocks are design feedback, not random accidents.

## Locking Model

In PostgreSQL, concurrency pain often appears long before total database saturation. The main problem is usually that application flows create lock shapes or transaction lifetimes that do not match real workload concurrency.

High-risk patterns include:

- transactions that touch shared rows in inconsistent order
- business workflows that mix slow external I/O with open transactions
- background jobs and user flows competing on the same hot entities
- DDL changes colliding with steady operational traffic

## Practical Heuristics

### Scope transactions to one real invariant

If a transaction spans multiple unrelated concerns, it is often a sign that application boundaries are weak rather than that stronger atomicity is truly needed.

### Model lock order intentionally

Deadlocks are usually telling you something deterministic about access order, not something random about the database.

### Observe waiting, not just failures

Long waits, blocked sessions, and growing queueing often reveal trouble earlier than deadlock errors alone.

## Common Failure Modes

### Correct in tests, unstable under concurrency

The flow is semantically correct in low-contention testing but becomes fragile when multiple workers or user requests compete on the same rows.

### DDL surprise during normal traffic

Operational changes look routine until lock interactions with production traffic create queueing or availability pain.

### Retry without redesign

Teams add retries around deadlocks or lock timeouts but never fix the transaction shape causing the issue.

## Principal Review Lens

- Which invariant requires atomicity here?
- What lock shape does this query create under contention?
- How will retries behave if this flow deadlocks?
- What transaction boundary should be split or reordered before growth makes it painful?
