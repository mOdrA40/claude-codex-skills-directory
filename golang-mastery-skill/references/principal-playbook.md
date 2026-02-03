# Principal Engineer Playbook (Go Services)

This is not a “new framework”. It’s a senior/principal way of thinking about Go services that survive production: crisp boundaries, closed failure modes, complete observability, and low ops cost.

## Quick usage

- If you’re unsure “which architecture”: start with `architecture.md`, then follow the decision tree below.
- If you’re designing risky features (payments, inventory, quotas, billing): start with “Money-moving / side effects”.
- If you’re chasing p95/p99: start with “Latency & capacity”.
- If you’re firefighting incidents: start with “Reliability & operability”.

## Golden questions (answer before coding)

1. What is the SLO/SLA? p95/p99 latency? error budget?
2. Load profile: RPS, payload size, concurrency, burst, growth (3–6 months).
3. Key failure modes: dependency timeouts, retry storms, partial failure, duplicate requests, deploy rollback.
4. Data: source of truth, transactional boundaries, required consistency (strong vs eventual).
5. Security: authz model, data classification (PII), minimal threat model.

## Decision tree (pragmatic)

### 1) Monolith vs microservices

- Default: **modular monolith** (cheaper ops, faster iteration).
- Choose microservices only with a strong reason:
  - truly separate domains with clear team ownership
  - drastically different scaling needs
  - compliance / isolation requirement
  - operational maturity exists: tracing, SLOs, on-call, alerting, rollout discipline

### 2) Sync vs async

- Sync HTTP/gRPC is best when you need an answer “now”.
- Async (queue/event) is best for:
  - heavy / long-running work
  - reducing coupling
  - smoothing bursts (backpressure)
  - at-least-once semantics (with idempotency)

### 3) “Exactly-once” is a myth; design for at-least-once

- Assume requests/events can be **duplicated**.
- Assume handlers can **crash** after partial side effects.
- Assume event publishing can **fail** after DB commit (or vice versa).

See `idempotency-outbox.md`.

## Casebook (real-world cases + guardrails)

### Money-moving / side effects (payments, billing, inventory)

Checklist:
- Require an **idempotency key** at the boundary (HTTP handler / consumer).
- Side effects must be safely repeatable (safe retries) or detectable as duplicates.
- If publishing events to a broker: use an **outbox pattern** (DB tx → outbox row → publisher).
- Audit logging is separate from app logs (immutable + queryable).

References:
- `idempotency-outbox.md`
- `errors.md` (error taxonomy & mapping)
- `reliability.md` (timeouts/retries/backoff)

### Latency & capacity (p95/p99)

Checklist:
- Define an end-to-end timeout budget (including dependency calls).
- Avoid unbounded fan-out; use bounded concurrency.
- Profile before optimizing: `performance.md`.
- Cache only with a clear invalidation story.

References:
- `performance.md`
- `outbound-http.md` (HTTP client hardening + timeouts)
- `concurrency.md` (bounded pools/backpressure)

### High-throughput ingestion (queue consumers, ETL)

Checklist:
- Bounded worker pool + bounded queue (backpressure), not unlimited goroutines.
- Define a “poison message” policy: retry count, DLQ, quarantine.
- Observability is required: lag, throughput, failure rate, retry rate.
- Shutdown must drain: stop fetching → finish in-flight → commit offset/ack.

References:
- `concurrency.md`
- `observability.md`
- `reliability.md`

### Multi-tenant SaaS

Checklist:
- Tenant isolation: schema-per-tenant vs row-level; pick based on compliance & cost.
- Every request carries tenant context; logs/traces include tenant ID.
- Rate limits / quotas per tenant.

References:
- `security.md`
- `log-correlation.md`
- `database.md`

## Guardrails (principal defaults)

### Error taxonomy (avoid “everything is 500”)

Define consistent categories (example):
- `invalid` (400)
- `unauthorized` (401)
- `forbidden` (403)
- `not_found` (404)
- `conflict` (409)
- `rate_limited` (429)
- `unavailable` (503)

References: `errors.md`, `http-api.md`.

### Timeouts & retries (prevent retry storms)

- Timeouts are always explicit (server & client).
- Retry only safe errors (transient) and only for idempotent operations.
- Add jitter/backoff; cap total attempts and total time.
- Avoid retries in multiple layers (client + gateway + job runner) without coordination.

References: `reliability.md`, `outbound-http.md`.

### Minimal required observability

- Logs: structured + request ID/correlation ID.
- Metrics: RED (Rate/Errors/Duration) + saturation (queue depth, goroutines if relevant).
- Traces: especially for multi-boundary requests.

References: `observability.md`, `otel-bootstrap.md`, `log-correlation.md`.

## Expensive anti-patterns

- Shared mutable state without clear ownership (races/leaks).
- Global singletons for dependencies (poor testability, hidden coupling).
- Catch-all retries without idempotency (duplicate side effects).
- “Cache first” without invalidation (stale-data bugs).

References: `anti-patterns.md`.

## What “done” looks like (production-ready)

- Tests cover core use-cases + regressions found.
- Timeouts, retry policy, shutdown, and backpressure are explicit.
- Error mapping is consistent; no double logging.
- Minimum observability is in place for incident response.
