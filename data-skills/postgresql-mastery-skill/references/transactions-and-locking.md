# Transactions and Locking (PostgreSQL)

## Rules

- Keep transactions short and scoped to a real invariant.
- Understand row locks, relation locks, and DDL lock impact.
- Avoid holding transactions open across network calls or user think time.
- Deadlocks are design feedback, not random accidents.

## Principal Review Lens

- Which invariant requires atomicity here?
- What lock shape does this query create under contention?
- How will retries behave if this flow deadlocks?
