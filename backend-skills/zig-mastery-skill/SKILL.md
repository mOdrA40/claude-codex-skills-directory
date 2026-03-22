---
name: zig-principal-engineer
description: |
  Principal/Senior-level Zig playbook for backend services, systems-aware APIs, memory management, concurrency, performance, reliability, observability, and production operations.
  Use when: building or reviewing Zig services, designing low-level backend components, optimizing latency-sensitive services, hardening memory-sensitive systems, debugging allocator issues, or preparing Zig applications for production.
---

# Zig Mastery (Senior → Principal)

## Operate

- Start by confirming: Zig version, target OS/arch, deployment model, latency and memory goals, FFI requirements, concurrency model, and the definition of done.
- Prefer small vertical slices with explicit ownership, allocator strategy, and error paths.
- Keep the design boring and operable: simple boundaries, measurable behavior, and predictable cleanup beat clever abstractions.
- Treat memory, failure handling, and observability as first-class design constraints, not afterthoughts.

> The goal is not just "fast Zig". The goal is a backend that stays correct under pressure, exposes predictable failure modes, and remains maintainable by the next engineer.

## Default Standards

- Keep transport, domain, and infrastructure boundaries explicit.
- Choose allocator strategy intentionally and document ownership at boundaries.
- Propagate errors with context; do not hide them behind catch-all defaults.
- Prefer explicit resource lifecycle management with `defer` and `errdefer`.
- Avoid hidden global state; inject dependencies and configuration explicitly.
- Treat outbound IO, parsing, and serialization as untrusted work: set limits, timeouts, and validation rules.
- Use the standard library first unless a dependency clearly improves correctness or delivery speed.

## “Bad vs Good” (common production pitfalls)

```zig
// ❌ BAD: allocator choice is implicit and cleanup is forgotten.
const data = try fetchUsers();
process(data);

// ✅ GOOD: allocator ownership and cleanup are explicit.
var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
defer arena.deinit();
const allocator = arena.allocator();
const data = try fetchUsers(allocator);
try process(data);
```

```zig
// ❌ BAD: panic-style behavior for recoverable backend failures.
const body = request.reader().readAllAlloc(allocator, max_size) catch unreachable;

// ✅ GOOD: map recoverable errors explicitly.
const body = request.reader().readAllAlloc(allocator, max_size) catch |err| switch (err) {
    error.StreamTooLong => return AppError.payload_too_large,
    else => return AppError.bad_request,
};
```

```zig
// ❌ BAD: fire-and-forget thread with no owner or shutdown path.
_ = try std.Thread.spawn(.{}, runWorker, .{});

// ✅ GOOD: thread lifecycle belongs to a supervisor.
const worker = try std.Thread.spawn(.{}, runWorker, .{shutdown_signal});
defer worker.join();
```

## Workflow (Feature / Refactor / Bug)

1. Reproduce behavior or codify it with a failing test.
2. Decide boundaries: protocol, orchestration, domain logic, persistence, and external integrations.
3. Define allocator strategy, ownership, and cleanup rules.
4. Implement the smallest end-to-end slice.
5. Validate tests, formatting, release behavior, and operational guardrails.
6. Review latency, allocations, failure modes, and shutdown behavior before release.

## Validation Commands

- Run `zig fmt .`.
- Run `zig test src/main.zig` or the relevant test targets.
- Run `zig build test` if the project uses a build graph.
- Run `zig build -Doptimize=ReleaseSafe` before release validation.
- Run `zig build -Doptimize=ReleaseFast` only for benchmark or release candidate comparison, not as the first debugging posture.
- Run smoke tests against the compiled binary in a production-like environment.
- Run container build validation if the service is deployed via Docker.

## Recommended Service Shape

```text
src/
├── main.zig
├── app/
│   ├── config.zig
│   ├── bootstrap.zig
│   └── shutdown.zig
├── transport/
│   ├── http.zig
│   └── dto.zig
├── domain/
│   ├── user_service.zig
│   └── errors.zig
├── adapters/
│   ├── postgres.zig
│   ├── redis.zig
│   └── outbound_http.zig
└── observability/
    ├── logging.zig
    └── metrics.zig
```

- Keep `main.zig` limited to startup, dependency wiring, and shutdown choreography.
- Separate protocol parsing from business rules so malformed transport input never pollutes domain logic.
- Keep allocator ownership visible at module boundaries, especially in adapters and parsing-heavy code.
- Prefer a modular monolith layout until there is real evidence for more process or package fragmentation.

## Backend Architecture Guardrails

- Prefer a modular monolith before inventing service sprawl.
- Keep HTTP or TCP transport thin; map protocol concerns at the edge.
- Make timeouts, retries, backoff, and circuit breaking explicit for outbound calls.
- Bound request size, parsing depth, and connection counts.
- Treat memory pressure as an operational event: measure allocations and cap untrusted workloads.
- Every background thread or async task needs an owner, stop condition, and failure-reporting path.

## Reliability and Operations

- Expose `/health` and `/ready` style endpoints if the service runs behind orchestration.
- Use structured logs with request IDs and stable fields.
- Implement graceful shutdown: stop accepting traffic, drain in-flight work, release resources, and join workers.
- Prefer idempotent handlers where retries may occur.
- Benchmark hot paths before micro-optimizing.
- Make overload behavior explicit: reject, queue, degrade, or shed load instead of letting memory pressure decide for you.
- Distinguish programmer bugs from recoverable runtime faults; not every failure should terminate the process.

## Debugging and Performance Playbook

- Reproduce correctness issues under `Debug` or `ReleaseSafe` before chasing peak throughput.
- Investigate allocator churn, buffer growth, serialization overhead, and lock contention before rewriting architecture.
- Capture representative request sizes and concurrency levels; microbenchmarks without production-like inputs are misleading.
- When debugging leaks or ownership mistakes, trace who allocates, who frees, and what the failure path does under `errdefer`.
- Use production-safe logging fields and counters so incidents can answer: what failed, for whom, under what load, and against which dependency.

## Security Checklist (Minimum)

- Validate all untrusted input lengths, counts, and encodings.
- Use allowlists for outbound destinations in sensitive environments.
- Never log secrets, credentials, or raw sensitive payloads.
- Use parameterized queries and least-privilege credentials for data stores.
- Keep unsafe FFI boundaries isolated and well-tested.

## Code Review Checklist

- Confirm allocator ownership is obvious for every non-trivial buffer or parsed payload.
- Confirm all outbound IO has timeout, retry, and cancellation posture defined.
- Confirm background workers and threads have a clear owner, stop signal, and error-reporting path.
- Confirm transport handlers validate size and shape before touching domain logic.
- Confirm logs, metrics, and health endpoints are sufficient for first-response incident debugging.
- Confirm `unsafe`-like FFI boundaries are isolated and do not leak undefined behavior into the wider service.

## Decision Heuristics

- Use Zig for backend services when control over memory, latency, binary shape, or C interoperability is central to the problem.
- Prefer Zig when the system has strict limits, high fan-in/fan-out efficiency concerns, or sensitive allocator behavior that must stay explicit.
- Avoid Zig when the main value comes from rapid CRUD delivery, mature batteries-included frameworks, or a large pool of application engineers with no systems background.
- Prefer Zig selectively for critical components if the rest of the platform benefits from a more conventional application stack.

## References

- Architecture and dependency direction: [references/architecture.md](references/architecture.md)
- Service architecture decision framework: [references/service-architecture-decision-framework.md](references/service-architecture-decision-framework.md)
- Agent instructions for backend tasks: [references/agent-instructions-for-backend-tasks.md](references/agent-instructions-for-backend-tasks.md)
- API and event schema compatibility matrix: [references/api-event-schema-compatibility-matrix.md](references/api-event-schema-compatibility-matrix.md)
- Backend cost and performance tradeoffs: [references/backend-cost-performance-tradeoffs.md](references/backend-cost-performance-tradeoffs.md)
- Dependency failure decision tree: [references/dependency-failure-decision-tree.md](references/dependency-failure-decision-tree.md)
- Operational smells and red flags: [references/operational-smells-and-red-flags.md](references/operational-smells-and-red-flags.md)
- Anti-patterns and operational smells: [references/anti-patterns-and-operational-smells.md](references/anti-patterns-and-operational-smells.md)
- Allocator strategy and memory ownership: [references/allocators-and-memory.md](references/allocators-and-memory.md)
- Allocator debugging and memory leaks: [references/allocator-debugging-and-memory-leaks.md](references/allocator-debugging-and-memory-leaks.md)
- Data contracts and versioning: [references/data-contracts-and-versioning.md](references/data-contracts-and-versioning.md)
- HTTP service design and reliability: [references/http-and-reliability.md](references/http-and-reliability.md)
- FFI and unsafe boundary control: [references/ffi-and-unsafe-boundaries.md](references/ffi-and-unsafe-boundaries.md)
- Request sizing and overload control: [references/request-sizing-and-overload-control.md](references/request-sizing-and-overload-control.md)
- SLO, error budgets, and service governance: [references/slo-error-budgets-and-service-governance.md](references/slo-error-budgets-and-service-governance.md)
- Tenant isolation and fairness: [references/tenant-isolation-and-fairness.md](references/tenant-isolation-and-fairness.md)
- Workers and background jobs: [references/workers-and-background-jobs.md](references/workers-and-background-jobs.md)
- Observability and incidents: [references/observability-and-incidents.md](references/observability-and-incidents.md)
- Outage triage, first 15 minutes: [references/outage-triage-first-15-minutes.md](references/outage-triage-first-15-minutes.md)
- Backend incident postmortem patterns: [references/backend-incident-postmortem-patterns.md](references/backend-incident-postmortem-patterns.md)
- High-risk change preflight checklist: [references/high-risk-change-preflight-checklist.md](references/high-risk-change-preflight-checklist.md)
- Principal backend code review playbook: [references/principal-backend-code-review-playbook.md](references/principal-backend-code-review-playbook.md)
- Queue poison message and dead-letter playbook: [references/queue-poison-message-and-dead-letter-playbook.md](references/queue-poison-message-and-dead-letter-playbook.md)
- Request path vs background path separation: [references/request-path-vs-background-path-separation.md](references/request-path-vs-background-path-separation.md)
- Review checklists by change type: [references/review-checklists-by-change-type.md](references/review-checklists-by-change-type.md)
- Rollouts and release safety: [references/rollouts-and-release-safety.md](references/rollouts-and-release-safety.md)
- Safe rollout patterns for high-traffic services: [references/safe-rollout-patterns-for-high-traffic-services.md](references/safe-rollout-patterns-for-high-traffic-services.md)
- Service decomposition and boundary decisions: [references/service-decomposition-and-boundary-decisions.md](references/service-decomposition-and-boundary-decisions.md)
- Stateful vs stateless service decisions: [references/stateful-vs-stateless-service-decisions.md](references/stateful-vs-stateless-service-decisions.md)
- Tenant fairness and noisy neighbor playbook: [references/tenant-fairness-and-noisy-neighbor-playbook.md](references/tenant-fairness-and-noisy-neighbor-playbook.md)
- Testing and debugging: [references/testing-and-debugging.md](references/testing-and-debugging.md)
- Zero-downtime schema and contract migrations: [references/zero-downtime-schema-and-contract-migrations.md](references/zero-downtime-schema-and-contract-migrations.md)
