# Production Readiness (Rust Services)

This guide closes the gap between “code compiles” and “service survives production”.

## Golden Questions

- What is the SLO: latency, throughput, error budget?
- What are the dependency failure modes: DB saturation, HTTP timeout, queue lag, deployment rollback?
- What is the shutdown contract: how does the service stop accepting work and drain in-flight tasks?
- Which paths may panic, block, deadlock, or retry indefinitely?

## Runtime Defaults

- Prefer explicit Tokio runtime configuration for services with non-trivial load.
- Set request deadlines and outbound timeouts explicitly.
- Bound concurrency for fan-out work; avoid spawning unbounded tasks.
- Close pools and background workers during shutdown.

## Error Taxonomy

Define a stable taxonomy and map it once at the transport boundary:

- `invalid`
- `unauthorized`
- `forbidden`
- `not_found`
- `conflict`
- `rate_limited`
- `unavailable`
- `internal`

Keep internal causes in logs/traces, not in client-facing payloads.

## Async & Concurrency Checklist

- Every `tokio::spawn` has an owner.
- Every long-running loop listens for cancellation.
- Shared state documents the protected invariant.
- Semaphore, channel capacity, or queue depth is bounded.
- Blocking work is isolated with `spawn_blocking` when appropriate.

## Observability Minimum

- Structured logs with `service`, `env`, `trace_id`, `request_id`, and key domain identifiers.
- RED metrics for inbound requests and critical dependencies.
- Tracing across DB, outbound HTTP, and queue boundaries.
- Startup log should include version, git SHA, runtime mode, and bind address.

## Deployment Guardrails

- Run as non-root in containers.
- Use reproducible builds (`--locked`).
- Pin base images and review OpenSSL/native dependencies.
- Add `/health` and `/ready` if the crate is an HTTP service.
- Avoid shipping debug-only endpoints or admin surfaces without auth.

## Data & Persistence

- Prefer transactions for multi-write invariants.
- Use unique constraints to enforce idempotency where retries are possible.
- Make migration order explicit and reversible where feasible.
- Time out connection acquisition so saturation fails fast.

## Release Checklist

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features`
- `cargo audit`
- Smoke test the release artifact or container image
- Verify configuration validation on startup

## Expensive Anti-Patterns

```rust
// ❌ BAD: retry loop with no deadline and no jitter.
loop {
    if do_call().await.is_ok() {
        break;
    }
}
```

```rust
// ✅ GOOD: retries are bounded by policy and request context.
let result = tokio::time::timeout(total_budget, retrying_call()).await;
```

## Principal Review Lens

Before approving a design, ask:

- What happens during partial failure?
- What prevents duplicate side effects?
- How do we debug a single bad request in under 5 minutes?
- Which metric will page us before the customer notices?
- Can the team safely roll this back tonight?
