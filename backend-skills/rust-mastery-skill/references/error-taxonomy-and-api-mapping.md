# Error Taxonomy and API Mapping in Rust

## Principle

Typed errors are valuable only when the service maps them consistently at boundaries.

## Core Categories

Use a stable taxonomy such as:

- invalid input
- unauthorized
- forbidden
- not found
- conflict
- dependency timeout
- dependency unavailable
- internal failure

## Bad vs Good

```text
❌ BAD
Handlers invent different status codes and payloads for similar failures.

✅ GOOD
Domain and adapter errors are translated once into stable API semantics.
```

## Rules

- use `thiserror` for domain and adapter errors
- use `anyhow` mainly at application edges and tooling
- avoid leaking driver or SQL details directly to clients
- log operator detail separately from client payload detail

## Review Questions

- Will clients see stable semantics if internals change?
- Can operators distinguish timeout, conflict, and dependency failure quickly?
- Is the error contract documented in code and tests?
