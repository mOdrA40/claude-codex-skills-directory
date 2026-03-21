# Retry Semantics and Idempotency (CockroachDB)

## Principle

CockroachDB retry behavior is not merely a driver concern. It is part of application correctness, side-effect safety, and user-visible operability.

## Rules

- Treat retryable errors as a normal design input for write paths.
- Keep external side effects outside ambiguous transaction boundaries unless idempotency is explicit.
- Distinguish safe replay of SQL work from unsafe replay of business side effects.
- Make retry ceilings, backoff, and operator visibility explicit.

## Design Heuristics

### SQL retry does not equal workflow safety

A transaction may be safe to retry at the database layer while the overall business workflow is unsafe to replay if it already triggered notifications, payments, or downstream mutations.

### Idempotency keys should reflect business intent

If the operation is externally visible, the idempotency strategy should map to the business action, not merely to one HTTP request instance.

### Bound retries and observe them

Unbounded retries can turn contention, regional latency, or overloaded paths into silent user pain.

## Common Failure Modes

### Duplicate-safe SQL, duplicate-unsafe business flow

The database work can be retried safely, but the surrounding application flow causes multiple externally visible effects.

### Retry storms normalized as success

The system eventually succeeds often enough that teams treat retries as harmless, even while latency, cost, and support burden keep rising.

### Hidden retry ownership

The driver, ORM, or service wrapper retries implicitly, and no one can explain what the full replay surface really is.

## Principal Review Lens

- Which write path is retryable at the SQL layer but unsafe at the business layer?
- What side effect needs idempotency rather than hope?
- Which retry path is currently succeeding while silently hurting latency or cost?
- Can operators tell whether retries are helping resilience or masking design debt?
