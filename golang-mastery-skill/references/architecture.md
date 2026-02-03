# Architecture (Pick by Use-case)

## Rule 0

Most “architecture” bugs are really missing boundaries: unclear ownership of data, timeouts, retries, idempotency, and error taxonomy.

## Decision matrix (pragmatic)

### Single service / small team / fast iteration

Prefer:
- modular monolith
- `cmd/` + `internal/` boundaries
- transport thin, domain/use-cases testable without HTTP

Avoid:
- microservices without strong operational maturity

### Latency-sensitive API

Prefer:
- explicit timeouts and budgets (every dependency call)
- caching with clear invalidation rules (or don’t cache)
- performance profiling in CI for hotspots

### High reliability / money-moving systems

Prefer:
- idempotency keys for side effects
- outbox pattern (DB transaction → event publish) where needed
- optimistic concurrency (`ETag` / version columns)
- strict audit logging (separate from app logs)

### Data-heavy / complex queries

Prefer:
- SQL-first approach with migrations
- query codegen (e.g., SQLC) if it reduces bugs
- read models/materialized views for hot reads

## Boundary layout (clean but not dogmatic)

Recommended dependency direction:

transport (Gin/Fiber/Beego/net/http)
→ use-cases (orchestration, policies)
→ domain (types + invariants; minimal deps)
→ ports (interfaces)
→ adapters (db/cache/queue implementations)

Keep “business meaning” out of handlers. Handlers should:
- parse/validate
- call use-case
- map result/error to HTTP response

## Concurrency architecture

- Prefer bounded worker pools for heavy work (avoid unbounded goroutines).
- Every goroutine must have:
  - an owner
  - a stop condition
  - a bounded queue or backpressure strategy

## Deployment architecture

- Put limits at the edge: body size, request timeouts, rate limits.
- Keep config in env; validate on startup; fail fast.

