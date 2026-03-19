# Reliability and Operations (Bun Services)

This guide focuses on the operational details that separate a fast prototype from a dependable Bun service.

## Golden Questions

- What is the expected traffic shape: steady load, bursty traffic, cron-style spikes?
- Which dependencies can fail: Postgres, Redis, third-party APIs, queues?
- What is the timeout budget for an end-to-end request?
- Which endpoints are idempotent and safe to retry?

## Request Lifecycle Defaults

- Validate environment on startup and fail fast.
- Validate request payloads at the transport boundary.
- Assign or propagate a request ID for every inbound request.
- Set maximum body size and explicit timeout policies.
- Return stable error envelopes.

## Error Taxonomy

Use a small set of stable error categories:

- `VALIDATION_ERROR`
- `UNAUTHORIZED`
- `FORBIDDEN`
- `NOT_FOUND`
- `CONFLICT`
- `RATE_LIMITED`
- `DEPENDENCY_UNAVAILABLE`
- `INTERNAL_ERROR`

Log internal details, but do not leak stack traces or raw driver errors to clients.

## Outbound IO Rules

- Every outbound HTTP call needs a timeout.
- Retries must be bounded, jittered, and only used for safe/idempotent operations.
- Prefer circuit-breaker or fail-fast behavior for degraded dependencies.
- Avoid retrying in multiple layers without coordination.

## Data Layer Guardrails

- Bound database pool size intentionally.
- Time out pool acquisition and surface saturation explicitly.
- Use transactions for multi-step invariants.
- Use unique constraints and application keys for idempotent writes.

## Graceful Shutdown

At minimum:

1. Stop accepting new requests.
2. Mark readiness as failed.
3. Drain in-flight requests with a bounded timeout.
4. Close database and Redis connections.
5. Flush logs and telemetry if configured.

## Observability Minimum

- Structured JSON logs in production.
- Log fields should include `service`, `env`, `requestId`, `route`, `statusCode`, and `latencyMs`.
- Metrics should cover rate, errors, duration, queue depth, and dependency latency.
- Health endpoints should distinguish liveness from readiness.

## Docker Defaults

- Pin Bun image tags.
- Run as non-root.
- Use multi-stage builds when bundling or compiling assets.
- Keep only production dependencies in the final image.
- Add explicit health checks.

## Checklist Before Release

- `bun install --frozen-lockfile`
- `bun run test`
- `bunx tsc --noEmit`
- `bunx @biomejs/biome check .`
- `bun run build`
- Build and smoke test the Docker image if shipped

## Expensive Anti-Patterns

```typescript
// ❌ BAD: no timeout, no classification, no visibility.
const response = await fetch(url)
```

```typescript
// ✅ GOOD: explicit timeout, classification, and logging.
const controller = new AbortController()
const timer = setTimeout(() => controller.abort(), 3_000)
try {
  const response = await fetch(url, { signal: controller.signal })
  if (!response.ok) {
    throw new AppError("Dependency failed", 503, "DEPENDENCY_UNAVAILABLE")
  }
} finally {
  clearTimeout(timer)
}
```

## Principal Review Lens

Before approving a Bun backend design, ask:

- What is the blast radius if Redis or Postgres slows down?
- How do we prevent duplicate side effects during retries?
- Can on-call isolate one failing request quickly from logs/metrics?
- What happens during deploy shutdown with in-flight writes?
- Does the runtime choice simplify operations, or only benchmarks?
