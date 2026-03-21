# Transactions and Consistency (MongoDB)

## Rules

- Use transactions when invariants truly require them.
- Document databases still need explicit consistency thinking.
- Multi-document transactions add cost and should stay short.
- Retry behavior must be idempotent and observable.

## Consistency Heuristics

### Prefer one-document atomicity when it truly fits

MongoDB is strongest when the data model lets important invariants stay within document boundaries. Multi-document transactions should be a deliberate exception, not a default habit.

### Treat retry and side effects as one design problem

A retried transaction may be safe in the database but still unsafe for downstream notifications, integrations, or user-visible actions if idempotency is weak.

### Make read/write guarantees explicit

Application teams should know when they are trading simplicity for stronger consistency, and what latency or operational cost that choice introduces.

## Common Failure Modes

### Transaction convenience drift

Teams start using multi-document transactions to compensate for weak modeling instead of fixing the underlying ownership and document-boundary problem.

### Hidden consistency assumptions

The application quietly relies on stronger read or write guarantees than the team has actually modeled, tested, or communicated.

### Retry success masking workflow risk

The database recovers well enough, but surrounding business actions remain ambiguous on replay or partial completion.

## Principal Review Lens

- Why is this not modeled as one-document atomicity?
- What happens on retry after partial external side effects?
- Is stronger consistency worth the latency and complexity?
- Which invariant should be re-modeled before transaction usage spreads further?
