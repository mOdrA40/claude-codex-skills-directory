# Zig Backend Architecture

## Purpose

This guide is for designing production Zig backends that must remain explicit under operational pressure. The main value of Zig is not just performance. It is the ability to make ownership, allocation, cleanup, failure behavior, and interop visible in code.

## Core Architectural Position

For backend systems, prefer this dependency direction:

`transport -> application/use-case -> domain -> ports -> adapters`

Use Zig to make the boundaries hard to ignore:

- Transport handles protocol details, parsing, and response mapping.
- Application orchestrates workflows and deadlines.
- Domain owns invariants and business rules.
- Ports define what the domain needs from storage or external systems.
- Adapters implement HTTP clients, databases, queues, files, or FFI.

Do not let transport structs or wire formats leak into the domain unless the application is intentionally thin.

## Design Rules

- Keep request parsing at the edge.
- Keep domain functions allocator-aware when they produce owned data.
- Avoid hidden singleton state for configuration, metrics, or clients.
- Prefer boring composition over inheritance-style abstraction emulation.
- Treat every external dependency as failure-prone.

## Recommended Layout

```text
src/
├── main.zig
├── app.zig
├── config.zig
├── transport/
│   ├── http.zig
│   ├── dto.zig
│   └── errors.zig
├── application/
│   ├── create_user.zig
│   └── list_orders.zig
├── domain/
│   ├── user.zig
│   ├── order.zig
│   └── errors.zig
├── ports/
│   ├── user_repository.zig
│   └── event_publisher.zig
├── adapters/
│   ├── postgres_user_repository.zig
│   ├── http_payment_client.zig
│   └── stdout_metrics.zig
└── support/
    ├── allocators.zig
    ├── logging.zig
    └── shutdown.zig
```

## Boundary Mapping

### Transport

Transport code should:

- decode input
- validate shape and limits
- attach request metadata
- call an application service
- map typed errors to protocol semantics

Transport code should not:

- hold business rules
- directly talk to multiple downstream dependencies
- decide retry policies for domain operations

### Application Layer

Application services coordinate:

- deadlines and cancellation
- transactions
- calling repositories and publishers
- idempotency behavior
- audit or metrics hooks

### Domain Layer

Domain logic should be:

- deterministic
- testable without sockets or databases
- independent from protocol details
- explicit about invalid states

## Error Taxonomy

You should define a small and stable taxonomy. Example:

```zig
pub const AppError = error{
    InvalidInput,
    Unauthorized,
    Forbidden,
    NotFound,
    Conflict,
    DependencyUnavailable,
    Timeout,
    Internal,
};
```

The key is not the exact names. The key is consistency across transport, logs, and runbooks.

## Bad vs Good: Boundary Leakage

```zig
// ❌ BAD: domain function depends on HTTP request concepts.
pub fn createUser(req: HttpRequest) !HttpResponse {
    if (req.headers.get("x-admin") == null) return error.Forbidden;
    // ...
}
```

```zig
// ✅ GOOD: transport translates protocol details into application input.
pub const CreateUserInput = struct {
    actor_role: []const u8,
    email: []const u8,
};

pub fn createUser(input: CreateUserInput) !User {
    if (!std.mem.eql(u8, input.actor_role, "admin")) return error.Forbidden;
    // ...
}
```

## Concurrency Ownership

Every spawned thread or long-lived worker must answer:

- Who owns it?
- How does it stop?
- What happens on failure?
- What metrics expose unhealthy behavior?

If you cannot answer these four questions, the concurrency model is not production-ready.

## Outbound Dependency Policy

For any database, HTTP call, queue, or file system dependency:

- set explicit timeout behavior
- bound payload sizes
- classify retryable vs non-retryable failures
- define idempotency expectations
- record latency and error rates

## Principal-Level Review Questions

- Which layer owns retries?
- Which layer owns error translation?
- Can this operation be replayed safely?
- Which objects own memory and for how long?
- Where does backpressure show up first?
- How does the service degrade under dependency slowness?
- What do operators need to know during an incident?
