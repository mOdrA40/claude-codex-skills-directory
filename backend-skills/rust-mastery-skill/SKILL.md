---
name: rust-principal-engineer
description: |
  Principal/Senior-level Rust playbook for architecture, ownership, async systems, error handling, observability, security, testing, and production readiness.
  Use when: designing Rust services or CLIs, reviewing unsafe/concurrent code, debugging panics and performance regressions, hardening APIs, or preparing a codebase for production.
---

# Rust Mastery (Senior → Principal)

## Operate

- Start by confirming: goal, scope, crate type (bin/lib/workspace), Rust/MSRV constraints, target platform, unsafe requirements, latency/throughput goals, and the definition of done.
- Prefer small, reviewable changes with tests and explicit tradeoffs.
- Default to stable Rust, stdlib-first patterns, and boring solutions before adding macros or dependencies.
- Treat production code as an operable system: timeouts, shutdown, observability, and failure modes are part of the feature.

> The target is not “clever Rust”. The target is code that remains correct, observable, and maintainable under production stress.

## Default Rust Standards

- Keep `main.rs` thin; put business logic in testable modules or crates.
- Prefer typed domain errors with `thiserror`; use `anyhow` at application boundaries and CLIs.
- No `unwrap()`/`expect()` on production paths unless the invariant is truly impossible and documented by the code structure.
- Introduce traits at the consumer boundary, not pre-emptively.
- Prefer ownership and borrowing that make invalid states unrepresentable before reaching for `Arc<Mutex<_>>`.
- Every spawned task needs an owner, a cancellation path, and an error handling strategy.
- Keep unsafe code isolated, minimal, and justified with explicit invariants.

## “Bad vs Good” (common production pitfalls)

```rust
// ❌ BAD: panic in a production path with no context.
let user = repo.find(id).await.unwrap();

// ✅ GOOD: propagate context with a typed error.
let user = repo
    .find(id)
    .await
    .map_err(AppError::from)?
    .ok_or(AppError::UserNotFound { id })?;
```

```rust
// ❌ BAD: detached task with no owner and no shutdown path.
tokio::spawn(async move {
    loop {
        run_job().await;
    }
});

// ✅ GOOD: task respects cancellation and reports failures.
tokio::spawn(async move {
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            result = run_job() => {
                if let Err(err) = result {
                    tracing::error!(error = %err, "job failed");
                }
            }
        }
    }
});
```

## Workflow (Feature / Refactor / Bug)

1. Reproduce the behavior or codify it with a failing test.
2. Decide boundaries: transport, orchestration, domain, adapters, persistence.
3. Define failure modes: panics, cancellation, partial writes, retries, timeouts, shutdown.
4. Implement the smallest end-to-end slice.
5. Add tests, benchmarks, or property tests when the risk justifies them.
6. Validate formatting, lints, security, and release behavior.

## Validation Commands

- Run `cargo fmt --all --check`.
- Run `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
- Run `cargo test --workspace --all-features`.
- Run `cargo test -- --nocapture` when debugging test output.
- Run `cargo nextest run --workspace --all-features` if available for faster suites.
- Run `cargo llvm-cov` if coverage matters.
- Run `cargo audit` before release.
- Run `cargo deny check` if the repo uses policy checks for licenses/advisories.

## Architecture & Boundaries

- Prefer a modular monolith before splitting into many crates or services.
- Keep boundary direction explicit: transport -> use-case -> domain -> ports -> adapters.
- Map errors once at the boundary: HTTP/gRPC/CLI should translate domain errors consistently.
- Keep domain types free from transport-specific concerns where practical.

## Async, Concurrency, and Ownership Guardrails

- Avoid “shared mutable state first”; prefer message passing or ownership transfer.
- If you use `Arc<Mutex<_>>`, document the protected invariant and expected contention.
- Bound concurrency for fan-out work; avoid unbounded task spawning.
- Always set timeouts for outbound IO and database acquisition.
- Treat cancellation as part of correctness, not just cleanup.

## Service/API Defaults

- Use structured tracing with stable fields such as `service`, `trace_id`, `request_id`, `tenant_id`, and `status`.
- Expose health/readiness endpoints for services.
- Validate input at the boundary; never trust deserialized payloads blindly.
- Make error taxonomy explicit: invalid, unauthorized, forbidden, not-found, conflict, unavailable.
- Prefer idempotent handlers for side-effecting operations where retries may happen.

## Performance & Safety Defaults

- Measure before optimizing with `criterion`, flamegraphs, or profiler traces.
- Watch clone frequency, allocation churn, lock contention, and serialization hotspots.
- Prefer zero-copy and borrowing only when it improves the real bottleneck and keeps code readable.
- Use `panic = "abort"` only when the operational tradeoff is understood.

## Security Checklist (Minimum)

- No secrets in logs, panic messages, or `Debug` output.
- Validate lengths, counts, recursion depth, and body sizes for untrusted input.
- Use parameterized SQL and least-privilege credentials.
- Prefer allowlists for outbound network and file operations in high-risk systems.
- Keep unsafe blocks isolated and reviewed as security-sensitive code.

## References

- Architecture and dependency direction: [references/architecture.md](references/architecture.md)
- Advanced patterns: [references/advanced-patterns.md](references/advanced-patterns.md)
- Agent instructions for backend tasks: [references/agent-instructions-for-backend-tasks.md](references/agent-instructions-for-backend-tasks.md)
- Backend cost and performance tradeoffs: [references/backend-cost-performance-tradeoffs.md](references/backend-cost-performance-tradeoffs.md)
- Dependency failure decision tree: [references/dependency-failure-decision-tree.md](references/dependency-failure-decision-tree.md)
- Bug prevention: [references/bug-prevention.md](references/bug-prevention.md)
- Debugging and profiling: [references/debugging-guide.md](references/debugging-guide.md)
- Performance investigation playbook: [references/performance-investigation-playbook.md](references/performance-investigation-playbook.md)
- Failure modes and incident patterns: [references/failure-modes-and-incident-patterns.md](references/failure-modes-and-incident-patterns.md)
- Database and SQLX: [references/database-and-sqlx.md](references/database-and-sqlx.md)
- Error taxonomy and API mapping: [references/error-taxonomy-and-api-mapping.md](references/error-taxonomy-and-api-mapping.md)
- HTTP service patterns: [references/http-service-patterns.md](references/http-service-patterns.md)
- Service boundaries and modular monoliths: [references/service-boundaries-and-modular-monoliths.md](references/service-boundaries-and-modular-monoliths.md)
- Multi-tenant and resource isolation: [references/multi-tenant-and-resource-isolation.md](references/multi-tenant-and-resource-isolation.md)
- Observability: [references/observability.md](references/observability.md)
- Capacity planning and load shedding: [references/capacity-planning-and-load-shedding.md](references/capacity-planning-and-load-shedding.md)
- Queue consumers and idempotency: [references/queue-consumers-and-idempotency.md](references/queue-consumers-and-idempotency.md)
- Reliability: [references/reliability.md](references/reliability.md)
- SLO, error budgets, and governance: [references/slo-error-budgets-and-governance.md](references/slo-error-budgets-and-governance.md)
- Service rollouts and compatibility: [references/service-rollouts-and-compatibility.md](references/service-rollouts-and-compatibility.md)
- Multi-region failover and disaster posture: [references/multi-region-failover-and-disaster-posture.md](references/multi-region-failover-and-disaster-posture.md)
- Streaming and backpressure: [references/streaming-and-backpressure.md](references/streaming-and-backpressure.md)
- Operational smells and red flags: [references/operational-smells-and-red-flags.md](references/operational-smells-and-red-flags.md)
- Outage triage, first 15 minutes: [references/outage-triage-first-15-minutes.md](references/outage-triage-first-15-minutes.md)
- Principal backend code review playbook: [references/principal-backend-code-review-playbook.md](references/principal-backend-code-review-playbook.md)
- Async concurrency and shutdown: [references/async-concurrency-and-shutdown.md](references/async-concurrency-and-shutdown.md)
- Queue poison message and dead-letter playbook: [references/queue-poison-message-and-dead-letter-playbook.md](references/queue-poison-message-and-dead-letter-playbook.md)
- Review checklists by change type: [references/review-checklists-by-change-type.md](references/review-checklists-by-change-type.md)
- Safe rollout patterns for high-traffic services: [references/safe-rollout-patterns-for-high-traffic-services.md](references/safe-rollout-patterns-for-high-traffic-services.md)
- Service decomposition and boundary decisions: [references/service-decomposition-and-boundary-decisions.md](references/service-decomposition-and-boundary-decisions.md)
- Senior habits and idioms: [references/senior-habits.md](references/senior-habits.md)
- Tenant fairness and noisy neighbor playbook: [references/tenant-fairness-and-noisy-neighbor-playbook.md](references/tenant-fairness-and-noisy-neighbor-playbook.md)
- Trusted libraries: [references/trusted-libraries.md](references/trusted-libraries.md)
- Production readiness and operations: [references/production-readiness.md](references/production-readiness.md)
- Zero-downtime schema and contract migrations: [references/zero-downtime-schema-and-contract-migrations.md](references/zero-downtime-schema-and-contract-migrations.md)

## Scripts & Assets

- `scripts/scaffold_project.py` - bootstrap a Rust project skeleton.
- `assets/github-ci.yml` - CI baseline for GitHub Actions.
