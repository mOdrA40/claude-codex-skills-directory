---
name: nodejs-principal-engineer
description: |
  Principal/Senior-level Node.js playbook for backend APIs, asynchronous systems, service boundaries, reliability, security, observability, and production operations.
  Use when: building or reviewing Node.js services, designing Express/Fastify/Nest-style backends, debugging event-loop latency, hardening APIs, scaling worker and queue systems, or preparing JavaScript/TypeScript services for production.
---

# Node.js Mastery (Senior → Principal)

## Operate

- Start by confirming: Node.js version, package manager, framework choice, TypeScript policy, deployment target, traffic shape, latency/SLO targets, and the definition of done.
- Prefer small end-to-end slices with explicit boundaries between transport, orchestration, domain logic, and adapters.
- Use the platform deliberately: event loop behavior, streaming, cancellation, and resource cleanup are design constraints, not implementation details.
- Optimize for operability: timeouts, request limits, structured logs, health/readiness, graceful shutdown, and dependency failure behavior belong in the baseline.

> The goal is not just to ship JavaScript that works. The goal is a backend that stays predictable under load, easy to debug in incidents, and safe to evolve.

## Default Standards

- Keep `src/index.ts` or `src/main.ts` as bootstrap only; put business logic in services/use-cases.
- Validate config, request payloads, and external responses at boundaries.
- Prefer explicit error classes and one global error-mapping strategy.
- Treat all IO as failure-prone: set timeouts, retries only where justified, and degradation behavior.
- Avoid hidden global state; inject configuration and dependencies explicitly.
- Use TypeScript strict mode by default for production services.
- Prefer idempotent handling for side-effecting operations that may be retried.

## “Bad vs Good” (common production pitfalls)

```typescript
// ❌ BAD: controller does parsing, business logic, and persistence in one place.
app.post('/users', async (req, res) => {
  const user = await db.insert(req.body)
  res.json(user)
})

// ✅ GOOD: validate at the edge and delegate to a service boundary.
app.post('/users', async (req, res, next) => {
  try {
    const input = createUserSchema.parse(req.body)
    const user = await createUserService.execute(input)
    res.status(201).json(user)
  } catch (error) {
    next(error)
  }
})
```

```typescript
// ❌ BAD: fire-and-forget work in request path.
void sendWebhook(order)

// ✅ GOOD: explicit ownership and failure handling.
await webhookPublisher.publish(order, { timeoutMs: 2_000 })
```

```typescript
// ❌ BAD: no timeout on outbound dependency.
const response = await fetch(url)

// ✅ GOOD: bound the dependency.
const response = await fetch(url, { signal: AbortSignal.timeout(2_000) })
```

## Workflow (Feature / Refactor / Bug)

1. Reproduce the behavior or encode it in tests.
2. Decide boundaries: transport, service/use-case, domain, adapters, persistence.
3. Define failure semantics: timeout, retry, idempotency, overload, shutdown.
4. Implement the smallest end-to-end slice.
5. Validate formatting, type checks, tests, and production guardrails.
6. Review observability, rollout safety, and incident ergonomics before release.

## Validation Commands

- Run `npm test` or the repository test command.
- Run `npx tsc --noEmit` when TypeScript is in use.
- Run `npm run lint` if linting is configured.
- Run `npm run build` before release validation.
- Run smoke tests against health/readiness endpoints.
- Run container build validation if Docker is part of delivery.

## Runtime and Service Guardrails

- Set request body size limits and header limits.
- Bound outbound IO with `AbortController` or equivalent cancellation.
- Expose `/health` and `/ready` endpoints when deployed behind orchestration.
- Implement graceful shutdown: stop accepting traffic, drain in-flight requests, close pools and queues, and exit with a deadline.
- Measure event-loop delay for latency-sensitive services.
- Separate online request traffic from expensive background or batch work.

## Security Checklist (Minimum)

- Validate all untrusted input and external response contracts.
- Never leak stack traces, secrets, or internal dependency details to clients.
- Use least-privilege credentials for databases and queues.
- Apply allowlists and SSRF protections for outbound network access when relevant.
- Treat file upload, archive parsing, and webhook endpoints as high-risk boundaries.

## Decision Heuristics

```text
Choose Node.js when:
- the team needs strong JavaScript/TypeScript productivity
- IO-bound service work dominates over CPU-heavy compute
- ecosystem maturity matters for delivery speed
- streaming and event-driven integration are common

Prefer another backend stack when:
- strict memory predictability matters more than ecosystem speed
- CPU-bound heavy computation dominates request handling
- ultra-low-latency systems-level control is the main requirement
```

## References

- Architecture and dependency direction: [references/architecture.md](references/architecture.md)
- Architecture decision framework: [references/architecture-decision-framework.md](references/architecture-decision-framework.md)
- Event loop, concurrency, and cancellation: [references/event-loop-and-concurrency.md](references/event-loop-and-concurrency.md)
- Event-driven and outbox patterns: [references/event-driven-and-outbox-patterns.md](references/event-driven-and-outbox-patterns.md)
- HTTP APIs and framework guardrails: [references/http-api-and-frameworks.md](references/http-api-and-frameworks.md)
- Database boundaries and transaction design: [references/database-boundaries-and-transaction-design.md](references/database-boundaries-and-transaction-design.md)
- NestJS boundaries: [references/nestjs-boundaries.md](references/nestjs-boundaries.md)
- Fastify production patterns: [references/fastify-production-patterns.md](references/fastify-production-patterns.md)
- Schema validation and contract testing: [references/schema-validation-and-contract-testing.md](references/schema-validation-and-contract-testing.md)
- Multi-tenant isolation patterns: [references/multi-tenant-and-isolation.md](references/multi-tenant-and-isolation.md)
- Streaming and backpressure: [references/streaming-and-backpressure.md](references/streaming-and-backpressure.md)
- Worker threads and CPU-bound work: [references/worker-threads-and-cpu-work.md](references/worker-threads-and-cpu-work.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Release rollout and incidents: [references/release-rollout-and-incidents.md](references/release-rollout-and-incidents.md)
- Observability and debugging playbook: [references/observability-debugging-playbook.md](references/observability-debugging-playbook.md)
- Failure stories and anti-patterns: [references/failure-stories-and-anti-patterns.md](references/failure-stories-and-anti-patterns.md)
- Security and outbound IO: [references/security-and-outbound-io.md](references/security-and-outbound-io.md)
- Webhook safety: [references/webhook-safety.md](references/webhook-safety.md)
- Queues, jobs, and background work: [references/queues-and-background-jobs.md](references/queues-and-background-jobs.md)
- Testing and incident debugging: [references/testing-and-debugging.md](references/testing-and-debugging.md)
