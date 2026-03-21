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
- Run smoke tests against the compiled binary in a production-like environment.
- Run container build validation if the service is deployed via Docker.

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

## Security Checklist (Minimum)

- Validate all untrusted input lengths, counts, and encodings.
- Use allowlists for outbound destinations in sensitive environments.
- Never log secrets, credentials, or raw sensitive payloads.
- Use parameterized queries and least-privilege credentials for data stores.
- Keep unsafe FFI boundaries isolated and well-tested.

## Decision Heuristics

```text
Choose Zig when:
- predictable memory behavior matters
- latency and binary size matter
- you need more control than Go/JavaScript typically provide
- C interop is part of the system boundary

Prefer another backend stack when:
- the team needs a richer web ecosystem immediately
- delivery speed depends on mature framework conventions
- the problem is primarily CRUD with little systems-level pressure
```

## References

- Architecture and dependency direction: [references/architecture.md](references/architecture.md)
- Allocator strategy and memory ownership: [references/allocators-and-memory.md](references/allocators-and-memory.md)
- HTTP service design and reliability: [references/http-and-reliability.md](references/http-and-reliability.md)
- FFI and unsafe boundary control: [references/ffi-and-unsafe-boundaries.md](references/ffi-and-unsafe-boundaries.md)
- Testing and debugging: [references/testing-and-debugging.md](references/testing-and-debugging.md)
