# Reliability (Rust Services)

Reliability is the set of design choices that keep failure bounded, visible, and recoverable.

## Timeouts and Deadlines

- Every outbound HTTP, DB acquisition, queue poll, and background dependency call needs a timeout.
- Derive child budgets from the parent request deadline when possible.
- Prefer fast failure to slow saturation when dependencies are degraded.

## Retries

- Retry only transient failures and only when the operation is safe or idempotent.
- Add jitter and a total deadline.
- Avoid retries in multiple layers without coordination.
- Retries without idempotency create duplicate side effects.

## Backpressure

- Bound queue sizes, channel capacity, and concurrent fan-out.
- Expose saturation signals so overload is visible before total failure.
- Prefer explicit rejection over unbounded memory growth.

## Graceful Shutdown

At minimum:

1. Stop accepting new work.
2. Fail readiness.
3. Cancel background loops.
4. Drain in-flight requests or jobs with a bounded timeout.
5. Close pools and flush telemetry if relevant.

## Idempotency

- Assume duplicate delivery can happen.
- Use unique keys, state-machine transitions, or deduplication records.
- Keep publish-after-commit behavior explicit when using queues or outbox flows.

## Reliability Review Lens

Ask:

- What fails first under load?
- What is bounded and what is not?
- Which side effects are retry-safe?
- What happens if shutdown occurs in the middle of a multi-step operation?
