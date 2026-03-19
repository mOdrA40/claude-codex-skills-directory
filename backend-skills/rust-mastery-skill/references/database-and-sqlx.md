# Database and SQLX (Rust)

Database bugs in Rust services are usually not about syntax. They are about transaction boundaries, query ownership, saturation, and state drift.

## Defaults

- Prefer explicit SQL for important flows.
- Keep query shape reviewable.
- Use compile-time checked queries when practical.
- Time out pool acquisition and slow queries.
- Keep transaction boundaries minimal and meaningful.

## Repository Rules

- Repositories should express domain intent, not just CRUD wrappers.
- Avoid hiding important SQL behavior behind generic abstractions.
- Return domain-relevant outcomes where possible.
- Preserve technical error causes for logs/telemetry.

## Transactions

Use a transaction when:

- multiple writes maintain one invariant,
- a side effect must be persisted with an outbox record,
- read-modify-write semantics must be atomic.

Do not use a transaction as a blanket wrapper around whole request handling.

## Reliability Guardrails

- Bound pool size intentionally.
- Watch queueing time for connection acquisition.
- Avoid N+1 on hot paths.
- Prefer idempotency keys or unique constraints where retries exist.
- Review lock behavior for high-contention updates.

## Principal Review Lens

Ask:

- Which invariant depends on this transaction?
- What happens if the request is retried halfway through?
- How do we detect saturation before timeouts pile up?
- Is this abstraction hiding important SQL behavior?
