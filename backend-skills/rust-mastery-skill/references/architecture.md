# Architecture (Rust Services and Crates)

This guide focuses on boundary discipline, failure containment, and pragmatic crate/service design.

## Rule 0

Most architecture failures are boundary failures:

- domain logic leaking into transport handlers,
- async tasks with no ownership,
- database concerns mixed into use-case logic,
- or error translation happening in multiple places.

## Choose the Smallest Viable Shape

## Single binary / modular monolith

Prefer when:

- one team owns the system,
- deploy cadence matters more than independent scaling,
- operational maturity is still growing.

Use:

- `main.rs` for bootstrap only,
- internal modules for transport, use-cases, domain, adapters,
- one typed configuration surface.

## Workspace / multiple crates

Prefer when:

- multiple binaries or shared libraries are real, not hypothetical,
- compile boundaries improve ownership,
- separate release cadence is meaningful.

Avoid splitting crates only to mimic enterprise diagrams.

## Dependency Direction

Recommended flow:

- transport (`axum`, CLI, queue consumer)
- use-cases / application services
- domain types and invariants
- ports / traits when needed by consumers
- adapters (`sqlx`, `reqwest`, Redis, broker)

Keep framework types out of core business logic.

## Error Boundaries

- Domain errors should model business failures.
- Adapter errors should preserve technical causes.
- Transport should map errors once into HTTP/gRPC/CLI status and public payloads.
- Avoid mixing `anyhow` deep in domain code when typed outcomes matter.

## Async Boundaries

- Every spawned task has an owner.
- Every queue or channel has bounded capacity or explicit backpressure behavior.
- Every outbound call has a timeout.
- Cancellation is part of correctness, not an afterthought.

## Persistence Boundaries

- Repositories should express domain intent, not only CRUD mechanics.
- Use transactions for multi-write invariants.
- Enforce idempotency with unique constraints or state-machine transitions.
- Prefer SQL that is observable and reviewable over opaque abstraction layers.

## Service Skeleton

A practical service layout:

```text
src/
├── main.rs
├── lib.rs
├── config/
├── transport/
├── application/
├── domain/
├── ports/
└── adapters/
```

Not every project needs every folder. Add structure only when it clarifies ownership.

## Principal Review Lens

Before approving an architecture, ask:

- Where is business policy expressed?
- Where do timeouts and retries live?
- How are duplicate side effects prevented?
- What can panic, block, or deadlock?
- Which boundary owns error translation and observability fields?
