# Distributed Systems (Patterns that Prevent Production Pain)

## Non-negotiables

- Timeouts are mandatory.
- Retries are dangerous unless idempotent.
- At-least-once delivery is the default (queues, client retries, network flakes).

## Retry strategy (safe shape)

- Only retry idempotent ops (or ops protected by idempotency keys).
- Use exponential backoff + jitter.
- Bound attempts and total time (deadline wins).
- Retry on *classified* transient failures, not on everything.

## Circuit breakers (when dependencies flap)

Use when:
- downstream is unstable
- retries amplify failure

Rules:
- trip breaker on high error rate/timeouts
- fail fast while open (protect upstream resources)
- half-open probes to recover

## Bulkheads (resource isolation)

- Separate pools/queues per dependency (DB vs external API).
- Avoid one dependency consuming all goroutines/threads/connections.

## Backpressure

- Bounded queues everywhere.
- When saturated: shed load early (503) instead of building an unbounded backlog.

## Consistency patterns

- Avoid dual writes (DB + queue) without outbox; see `idempotency-outbox.md`.
- For sagas:
  - prefer orchestration when you need clear business semantics
  - use compensating actions (and design them carefully)

## “Exactly once” reality check

Aim for:
- exactly-once *effects* in your domain (via unique constraints and idempotency)
- at-least-once delivery in transport (accept duplicates)

