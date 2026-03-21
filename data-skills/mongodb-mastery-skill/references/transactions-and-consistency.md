# Transactions and Consistency (MongoDB)

## Rules

- Use transactions when invariants truly require them.
- Document databases still need explicit consistency thinking.
- Multi-document transactions add cost and should stay short.
- Retry behavior must be idempotent and observable.

## Principal Review Lens

- Why is this not modeled as one-document atomicity?
- What happens on retry after partial external side effects?
- Is stronger consistency worth the latency and complexity?
